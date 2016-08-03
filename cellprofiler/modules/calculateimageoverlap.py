import cellprofiler.icons
from cellprofiler.gui.help import PROTIP_RECOMEND_ICON

__doc__ = '''
<b>Calculate Image Overlap </b> calculates how much overlap occurs between the white portions of two black and white images
<hr>
This module calculates overlap by determining a set of statistics that measure the closeness of an image or object
to its' true value.  One image/object is considered the "ground truth" (possibly the result of hand-segmentation) and the other
is the "test" image/object; the images are determined to overlap most completely when the test image matches the ground
truth perfectly.  If using images, the module requires binary (black and white) input, where the foreground is white and
the background is black.  If you segment your images in CellProfiler using <b>IdentifyPrimaryObjects</b>,
you can create such an image using <b>ConvertObjectsToImage</b> by selecting <i>Binary</i> as the color type.

If your images have been segmented using other image processing software, or you have hand-segmented them in software
such as Photoshop, you may need to use one or more of the following to prepare the images for this module:
<ul>
<li> <b>ImageMath</b>: If the objects are black and the background is white, you must invert the intensity using this module.</li>
<li> <b>ApplyThreshold</b>: If the image is grayscale, you must make it binary using this module, or alternately use an <b>Identify</b> module followed by <b>ConvertObjectsToImage</b> as described above. </li>
<li> <b>ColorToGray</b>: If the image is in color, you must first convert it to grayscale using this module, and then use <b>ApplyThreshold</b> to generate a binary image. </li>
</ul>

In the test image, any foreground (white) pixels that overlap with the foreground of the ground
truth will be considered "true positives", since they are correctly labeled as foreground.  Background (black)
pixels that overlap with the background of the ground truth image are considered "true negatives",
since they are correctly labeled as background.  A foreground pixel in the test image that overlaps with the background in the ground truth image will
be considered a "false positive" (since it should have been labeled as part of the background),
while a background pixel in the test image that overlaps with foreground in the ground truth will be considered a "false negative"
(since it was labeled as part of the background, but should not be).

<h4>Available measurements</h4>
<ul>
<li><b>For images and objects:</b>
<ul>
<li><i>True positive rate:</i> Total number of true positive pixels / total number of actual positive pixels.</li>
<li><i>False positive rate:</i> Total number of false positive pixels / total number of actual negative pixels </li>
<li><i>True negative rate:</i> Total number of true negative pixels / total number of actual negative pixels.</li>
<li><i>False negative rate:</i> Total number of false negative pixels / total number of actual postive pixels </li>
<li><i>Precision:</i> Number of true positive pixels / (number of true positive pixels + number of false positive pixels) </li>
<li><i>Recall:</i> Number of true positive pixels/ (number of true positive pixels + number of false negative pixels) </li>
<li><i>F-factor:</i> 2 &times; (precision &times; recall)/(precision + recall). Also known as F<sub>1</sub> score, F-score or F-measure.</li>
<li><i>Earth mover's distance:</i>The minimum distance required to move each
foreground pixel in the test image to some corresponding foreground pixel in the
reference image.</li>
</ul>
</li>
<li><b>For objects:</b>
<ul>
<li><i>Rand index:</i> A measure of the similarity between two data clusterings. Perfectly random clustering returns the minimum
score of 0, perfect clustering returns the maximum score of 1.</li>
<li><i>Adjusted Rand index:</i> A variation of the Rand index which takes into account the fact that random chance will cause some
objects to occupy the same clusters, so the Rand Index will never actually be zero. Can return a value between -1 and +1.</li>
</ul>
</li>
</ul>

<h4>References</h4>
<ul>
<li>Collins LM, Dent CW (1998) "Omega: A general formulation of the Rand Index of cluster
recovery suitable for non-disjoint solutions", <i>Multivariate Behavioral
Research</i>, 23, 231-242 <a href="http://dx.doi.org/10.1207/s15327906mbr2302_6">(link)</a></li>
<li>Pele O, Werman M (2009) "Fast and Robust Earth Mover's Distances",
<i>2009 IEEE 12th International Conference on Computer Vision</i></li>
</ul>
'''

import numpy as np

from scipy.ndimage import label, distance_transform_edt
from scipy.sparse import coo_matrix

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
import centrosome.cpmorphology as morph
from centrosome.index import Indexes
from centrosome.fastemd import emd_hat_int32
from centrosome.propagate import propagate
from centrosome.filter import poisson_equation

C_IMAGE_OVERLAP = "Overlap"
FTR_F_FACTOR = "Ffactor"
FTR_PRECISION = "Precision"
FTR_RECALL = "Recall"
FTR_TRUE_POS_RATE = "TruePosRate"
FTR_FALSE_POS_RATE = "FalsePosRate"
FTR_FALSE_NEG_RATE = "FalseNegRate"
FTR_TRUE_NEG_RATE = "TrueNegRate"
FTR_RAND_INDEX = "RandIndex"
FTR_ADJUSTED_RAND_INDEX = "AdjustedRandIndex"
FTR_EARTH_MOVERS_DISTANCE = "EarthMoversDistance"

FTR_ALL = [FTR_F_FACTOR, FTR_PRECISION, FTR_RECALL,
           FTR_TRUE_POS_RATE, FTR_TRUE_NEG_RATE,
           FTR_FALSE_POS_RATE, FTR_FALSE_NEG_RATE,
           FTR_RAND_INDEX, FTR_ADJUSTED_RAND_INDEX]

O_OBJ = "Segmented objects"
O_IMG = "Foreground/background segmentation"
O_ALL = [O_OBJ, O_IMG]

L_LOAD = "Loaded from a previous run"
L_CP = "From this CP pipeline"

DM_KMEANS = "K Means"
DM_SKEL = "Skeleton"


class CalculateImageOverlap(cpm.Module):
    category = "Measurement"
    variable_revision_number = 4
    module_name = "CalculateImageOverlap"

    def create_settings(self):
        self.obj_or_img = cps.Choice(
                "Compare segmented objects, or foreground/background?", O_ALL)

        self.ground_truth = cps.ImageNameSubscriber(
                "Select the image to be used as the ground truth basis for calculating the amount of overlap",
                cps.NONE, doc="""
            <i>(Used only when comparing foreground/background)</i> <br>
            This binary (black and white) image is known as the "ground truth" image.  It can be the product of segmentation performed by hand, or
            the result of another segmentation algorithm whose results you would like to compare.""")

        self.test_img = cps.ImageNameSubscriber(
                "Select the image to be used to test for overlap",
                cps.NONE, doc="""
            <i>(Used only when comparing foreground/background)</i> <br>
            This binary (black and white) image is what you will compare with the ground truth image. It is known as the "test image".""")

        self.object_name_GT = cps.ObjectNameSubscriber(
                "Select the objects to be used as the ground truth basis for calculating the amount of overlap",
                cps.NONE, doc="""
            <i>(Used only when comparing segmented objects)</i> <br>
            Choose which set of objects will used as the "ground truth" objects. It can be the product of segmentation performed by hand, or
            the result of another segmentation algorithm whose results you would like to compare. See the <b>Load</b> modules for more details
            on loading objects.""")

        self.object_name_ID = cps.ObjectNameSubscriber(
                "Select the objects to be tested for overlap against the ground truth",
                cps.NONE, doc="""
            <i>(Used only when comparing segmented objects)</i> <br>
            This set of objects is what you will compare with the ground truth objects. It is known as the "test object." """)
        self.wants_emd = cps.Binary(
                "Calculate earth mover's distance?", False,
                doc="""The earth mover's distance computes the shortest distance
            that would have to be travelled to move each foreground pixel in the
            test image to some foreground pixel in the reference image.
            "Earth mover's" refers to an analogy: the pixels are "earth" that
            has to be moved by some machine at the smallest possible cost.
            <br>
            It would take too much memory and processing time to compute the
            exact earth mover's distance, so <b>CalculateImageOverlap</b>
            chooses representative foreground pixels in each image and
            assigns each foreground pixel to its closest representative. The
            earth mover's distance is then computed for moving the foreground
            pixels associated with each representative in the test image to
            those in the reference image.
            """)
        self.max_points = cps.Integer(
                "Maximum # of points", value=250,
                minval=100,
                doc="""
            <i>(Used only when computing the earth mover's distance)</i> <br>
            This is the number of representative points that will be taken
            from the foreground of the test image and from the foreground of
            the reference image using the point selection method (see below).
            """)
        self.decimation_method = cps.Choice(
                "Point selection method",
                choices=[DM_KMEANS, DM_SKEL],
                doc="""
            <i>(Used only when computing the earth mover's distance)</i> <br>
            The point selection setting determines how the
            representative points are chosen.
            <ul>
            <li><i>%(DM_KMEANS)s:</i> Select to pick representative points using a
            K-Means clustering technique. The foregrounds of both images are combined
            and representatives are picked that minimize the distance to the nearest
            representative. The same representatives are then used for the test and
            reference images.</li>
            <li><i>%(DM_SKEL)s:</i> Select to skeletonize the image and pick
            points eqidistant along the skeleton. </li>
            </ul>
            <dl>
            <dd><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;
            <i>%(DM_KMEANS)s</i> is a
            choice that's generally applicable to all images. <i>%(DM_SKEL)s</i>
            is best suited to long, skinny objects such as worms or neurites.</dd>
            </dl>
            """ % globals())
        self.max_distance = cps.Integer(
                "Maximum distance", value=250, minval=1,
                doc="""
            <i>(Used only when computing the earth mover's distance)</i> <br>
            This setting sets an upper bound to the distance penalty
            assessed during the movement calculation. As an example, the score
            for moving 10 pixels from one location to a location that is
            100 pixels away is 10*100, but if the maximum distance were set
            to 50, the score would be 10*50 instead.
            <br>
            The maximum distance should be set to the largest reasonable
            distance that pixels could be expected to move from one image
            to the next.
            """)
        self.penalize_missing = cps.Binary(
                "Penalize missing pixels", value=False,
                doc="""
            <i>(Used only when computing the earth mover's distance)</i> <br>
            If one image has more foreground pixels than the other, the
            earth mover's distance is not well-defined because there is
            no destination for the extra source pixels or vice-versa.
            It's reasonable to assess a penalty for the discrepancy when
            comparing the accuracy of a segmentation because the discrepancy
            represents an error. It's also reasonable to assess no penalty
            if the goal is to compute the cost of movement, for example between
            two frames in a time-lapse movie, because the discrepancy is
            likely caused by noise or artifacts in segmentation.

            Set this setting to "Yes" to assess a penalty equal to the
            maximum distance times the absolute difference in number of
            foreground pixels in the two images. Set this setting to "No"
            to assess no penalty.
            """)

    def settings(self):
        result = [self.obj_or_img, self.ground_truth, self.test_img,
                  self.object_name_GT, self.object_name_ID,
                  self.wants_emd, self.max_points, self.decimation_method,
                  self.max_distance, self.penalize_missing]
        return result

    def visible_settings(self):
        result = [self.obj_or_img]
        if self.obj_or_img == O_IMG:
            result += [self.ground_truth, self.test_img]
        elif self.obj_or_img == O_OBJ:
            result += [self.object_name_GT, self.object_name_ID]
        result += [self.wants_emd]
        if self.wants_emd:
            result += [self.max_points, self.decimation_method,
                       self.max_distance, self.penalize_missing]
        return result

    def run(self, workspace):
        if self.obj_or_img == O_IMG:
            self.measure_image(workspace)
        elif self.obj_or_img == O_OBJ:
            self.measure_objects(workspace)

    def measure_image(self, workspace):
        '''Add the image overlap measurements'''

        image_set = workspace.image_set
        ground_truth_image = image_set.get_image(self.ground_truth.value,
                                                 must_be_binary=True)
        test_image = image_set.get_image(self.test_img.value,
                                         must_be_binary=True)
        ground_truth_pixels = ground_truth_image.pixel_data
        ground_truth_pixels = test_image.crop_image_similarly(ground_truth_pixels)
        mask = ground_truth_image.mask
        mask = test_image.crop_image_similarly(mask)
        if test_image.has_mask:
            mask = mask & test_image.mask
        test_pixels = test_image.pixel_data

        false_positives = test_pixels & ~ ground_truth_pixels
        false_positives[~ mask] = False
        false_negatives = (~ test_pixels) & ground_truth_pixels
        false_negatives[~ mask] = False
        true_positives = test_pixels & ground_truth_pixels
        true_positives[~ mask] = False
        true_negatives = (~ test_pixels) & (~ ground_truth_pixels)
        true_negatives[~ mask] = False

        false_positive_count = np.sum(false_positives)
        true_positive_count = np.sum(true_positives)

        false_negative_count = np.sum(false_negatives)
        true_negative_count = np.sum(true_negatives)

        labeled_pixel_count = true_positive_count + false_positive_count
        true_count = true_positive_count + false_negative_count

        ##################################
        #
        # Calculate the F-Factor
        #
        # 2 * precision * recall
        # -----------------------
        # precision + recall
        #
        # precision = true positives / labeled
        # recall = true positives / true count
        #
        ###################################

        if labeled_pixel_count == 0:
            precision = 1.0
        else:
            precision = float(true_positive_count) / float(labeled_pixel_count)
        if true_count == 0:
            recall = 1.0
        else:
            recall = float(true_positive_count) / float(true_count)
        if (precision + recall) == 0:
            f_factor = 0.0  # From http://en.wikipedia.org/wiki/F1_score
        else:
            f_factor = 2.0 * precision * recall / (precision + recall)
        negative_count = false_positive_count + true_negative_count
        if negative_count == 0:
            false_positive_rate = 0.0
            true_negative_rate = 1.0
        else:
            false_positive_rate = (float(false_positive_count) /
                                   float(negative_count))
            true_negative_rate = (float(true_negative_count) /
                                  float(negative_count))
        if true_count == 0:
            false_negative_rate = 0.0
            true_positive_rate = 1.0
        else:
            false_negative_rate = (float(false_negative_count) /
                                   float(true_count))
            true_positive_rate = (float(true_positive_count) /
                                  float(true_count))
        ground_truth_labels, ground_truth_count = label(
                ground_truth_pixels & mask, np.ones((3, 3), bool))
        test_labels, test_count = label(
                test_pixels & mask, np.ones((3, 3), bool))
        rand_index, adjusted_rand_index = self.compute_rand_index(
                test_labels, ground_truth_labels, mask)

        m = workspace.measurements
        m.add_image_measurement(self.measurement_name(FTR_F_FACTOR), f_factor)
        m.add_image_measurement(self.measurement_name(FTR_PRECISION),
                                precision)
        m.add_image_measurement(self.measurement_name(FTR_RECALL), recall)
        m.add_image_measurement(self.measurement_name(FTR_TRUE_POS_RATE),
                                true_positive_rate)
        m.add_image_measurement(self.measurement_name(FTR_FALSE_POS_RATE),
                                false_positive_rate)
        m.add_image_measurement(self.measurement_name(FTR_TRUE_NEG_RATE),
                                true_negative_rate)
        m.add_image_measurement(self.measurement_name(FTR_FALSE_NEG_RATE),
                                false_negative_rate)
        m.add_image_measurement(self.measurement_name(FTR_RAND_INDEX),
                                rand_index)
        m.add_image_measurement(self.measurement_name(FTR_ADJUSTED_RAND_INDEX),
                                adjusted_rand_index)

        if self.wants_emd:
            test_objects = cpo.Objects()
            test_objects.segmented = test_labels
            ground_truth_objects = cpo.Objects()
            ground_truth_objects.segmented = ground_truth_labels
            emd = self.compute_emd(test_objects, ground_truth_objects)
            m.add_image_measurement(
                    self.measurement_name(FTR_EARTH_MOVERS_DISTANCE), emd)

        if self.show_window:
            workspace.display_data.true_positives = true_positives
            workspace.display_data.true_negatives = true_negatives
            workspace.display_data.false_positives = false_positives
            workspace.display_data.false_negatives = false_negatives
            workspace.display_data.rand_index = rand_index
            workspace.display_data.adjusted_rand_index = adjusted_rand_index
            workspace.display_data.statistics = [
                (FTR_F_FACTOR, f_factor),
                (FTR_PRECISION, precision),
                (FTR_RECALL, recall),
                (FTR_FALSE_POS_RATE, false_positive_rate),
                (FTR_FALSE_NEG_RATE, false_negative_rate),
                (FTR_RAND_INDEX, rand_index),
                (FTR_ADJUSTED_RAND_INDEX, adjusted_rand_index)
            ]
            if self.wants_emd:
                workspace.display_data.statistics.append(
                        (FTR_EARTH_MOVERS_DISTANCE, emd))

    def measure_objects(self, workspace):
        image_set = workspace.image_set
        object_name_GT = self.object_name_GT.value
        objects_GT = workspace.get_objects(object_name_GT)
        iGT, jGT, lGT = objects_GT.ijv.transpose()
        object_name_ID = self.object_name_ID.value
        objects_ID = workspace.get_objects(object_name_ID)
        iID, jID, lID = objects_ID.ijv.transpose()
        ID_obj = 0 if len(lID) == 0 else max(lID)
        GT_obj = 0 if len(lGT) == 0 else max(lGT)

        xGT, yGT = objects_GT.shape
        xID, yID = objects_ID.shape
        GT_pixels = np.zeros((xGT, yGT))
        ID_pixels = np.zeros((xID, yID))
        total_pixels = xGT * yGT

        GT_pixels[iGT, jGT] = 1
        ID_pixels[iID, jID] = 1

        GT_tot_area = len(iGT)
        if len(iGT) == 0 and len(iID) == 0:
            intersect_matrix = np.zeros((0, 0), int)
        else:
            #
            # Build a matrix with rows of i, j, label and a GT/ID flag
            #
            all_ijv = np.column_stack(
                    (np.hstack((iGT, iID)),
                     np.hstack((jGT, jID)),
                     np.hstack((lGT, lID)),
                     np.hstack((np.zeros(len(iGT)), np.ones(len(iID))))))
            #
            # Order it so that runs of the same i, j are consecutive
            #
            order = np.lexsort((all_ijv[:, -1], all_ijv[:, 0], all_ijv[:, 1]))
            all_ijv = all_ijv[order, :]
            # Mark the first at each i, j != previous i, j
            first = np.where(np.hstack(
                    ([True],
                     ~ np.all(all_ijv[:-1, :2] == all_ijv[1:, :2], 1),
                     [True])))[0]
            # Count # at each i, j
            count = first[1:] - first[:-1]
            # First indexer - mapping from i,j to index in all_ijv
            all_ijv_map = Indexes([count])
            # Bincount to get the # of ID pixels per i,j
            id_count = np.bincount(all_ijv_map.rev_idx,
                                   all_ijv[:, -1]).astype(int)
            gt_count = count - id_count
            # Now we can create an indexer that has NxM elements per i,j
            # where N is the number of GT pixels at that i,j and M is
            # the number of ID pixels. We can then use the indexer to pull
            # out the label values for each to populate a sparse array.
            #
            cross_map = Indexes([id_count, gt_count])
            off_gt = all_ijv_map.fwd_idx[cross_map.rev_idx] + cross_map.idx[0]
            off_id = all_ijv_map.fwd_idx[cross_map.rev_idx] + cross_map.idx[1] + \
                     id_count[cross_map.rev_idx]
            intersect_matrix = coo_matrix(
                    (np.ones(len(off_gt)),
                     (all_ijv[off_id, 2], all_ijv[off_gt, 2])),
                    shape=(ID_obj + 1, GT_obj + 1)).toarray()[1:, 1:]

        gt_areas = objects_GT.areas
        id_areas = objects_ID.areas
        FN_area = gt_areas[np.newaxis, :] - intersect_matrix
        all_intersecting_area = np.sum(intersect_matrix)

        dom_ID = []

        for i in range(0, ID_obj):
            indices_jj = np.nonzero(lID == i)
            indices_jj = indices_jj[0]
            id_i = iID[indices_jj]
            id_j = jID[indices_jj]
            ID_pixels[id_i, id_j] = 1

        for i in intersect_matrix:  # loop through the GT objects first
            if len(i) == 0 or max(i) == 0:
                id = -1  # we missed the object; arbitrarily assign -1 index
            else:
                id = np.where(i == max(i))[0][0]  # what is the ID of the max pixels?
            dom_ID += [id]  # for ea GT object, which is the dominating ID?

        dom_ID = np.array(dom_ID)

        for i in range(0, len(intersect_matrix.T)):
            if len(np.where(dom_ID == i)[0]) > 1:
                final_id = np.where(intersect_matrix.T[i] == max(intersect_matrix.T[i]))
                final_id = final_id[0][0]
                all_id = np.where(dom_ID == i)[0]
                nonfinal = [x for x in all_id if x != final_id]
                for n in nonfinal:  # these others cannot be candidates for the corr ID now
                    intersect_matrix.T[i][n] = 0
            else:
                continue

        TP = 0
        FN = 0
        FP = 0
        for i in range(0, len(dom_ID)):
            d = dom_ID[i]
            if d == -1:
                tp = 0
                fn = id_areas[i]
                fp = 0
            else:
                fp = np.sum(intersect_matrix[i][0:d]) + np.sum(intersect_matrix[i][(d + 1)::])
                tp = intersect_matrix[i][d]
                fn = FN_area[i][d]
            TP += tp
            FN += fn
            FP += fp

        TN = max(0, total_pixels - TP - FN - FP)

        def nan_divide(numerator, denominator):
            if denominator == 0:
                return np.nan
            return float(numerator) / float(denominator)

        accuracy = nan_divide(TP, all_intersecting_area)
        recall = nan_divide(TP, GT_tot_area)
        precision = nan_divide(TP, (TP + FP))
        F_factor = nan_divide(2 * (precision * recall), (precision + recall))
        true_positive_rate = nan_divide(TP, (FN + TP))
        false_positive_rate = nan_divide(FP, (FP + TN))
        false_negative_rate = nan_divide(FN, (FN + TP))
        true_negative_rate = nan_divide(TN, (FP + TN))
        shape = np.maximum(np.maximum(
                np.array(objects_GT.shape), np.array(objects_ID.shape)),
                np.ones(2, int))
        rand_index, adjusted_rand_index = self.compute_rand_index_ijv(
                objects_GT.ijv, objects_ID.ijv, shape)
        m = workspace.measurements
        m.add_image_measurement(self.measurement_name(FTR_F_FACTOR), F_factor)
        m.add_image_measurement(self.measurement_name(FTR_PRECISION),
                                precision)
        m.add_image_measurement(self.measurement_name(FTR_RECALL), recall)
        m.add_image_measurement(self.measurement_name(FTR_TRUE_POS_RATE),
                                true_positive_rate)
        m.add_image_measurement(self.measurement_name(FTR_FALSE_POS_RATE),
                                false_positive_rate)
        m.add_image_measurement(self.measurement_name(FTR_TRUE_NEG_RATE),
                                true_negative_rate)
        m.add_image_measurement(self.measurement_name(FTR_FALSE_NEG_RATE),
                                false_negative_rate)
        m.add_image_measurement(self.measurement_name(FTR_RAND_INDEX),
                                rand_index)
        m.add_image_measurement(self.measurement_name(FTR_ADJUSTED_RAND_INDEX),
                                adjusted_rand_index)

        def subscripts(condition1, condition2):
            x1, y1 = np.where(GT_pixels == condition1)
            x2, y2 = np.where(ID_pixels == condition2)
            mask = set(zip(x1, y1)) & set(zip(x2, y2))
            return list(mask)

        TP_mask = subscripts(1, 1)
        FN_mask = subscripts(1, 0)
        FP_mask = subscripts(0, 1)
        TN_mask = subscripts(0, 0)

        TP_pixels = np.zeros((xGT, yGT))
        FN_pixels = np.zeros((xGT, yGT))
        FP_pixels = np.zeros((xGT, yGT))
        TN_pixels = np.zeros((xGT, yGT))

        def maskimg(mask, img):
            for ea in mask:
                img[ea] = 1
            return img

        TP_pixels = maskimg(TP_mask, TP_pixels)
        FN_pixels = maskimg(FN_mask, FN_pixels)
        FP_pixels = maskimg(FP_mask, FP_pixels)
        TN_pixels = maskimg(TN_mask, TN_pixels)
        if self.wants_emd:
            emd = self.compute_emd(objects_ID, objects_GT)
            m.add_image_measurement(
                    self.measurement_name(FTR_EARTH_MOVERS_DISTANCE), emd)

        if self.show_window:
            workspace.display_data.true_positives = TP_pixels
            workspace.display_data.true_negatives = TN_pixels
            workspace.display_data.false_positives = FP_pixels
            workspace.display_data.false_negatives = FN_pixels
            workspace.display_data.statistics = [
                (FTR_F_FACTOR, F_factor),
                (FTR_PRECISION, precision),
                (FTR_RECALL, recall),
                (FTR_FALSE_POS_RATE, false_positive_rate),
                (FTR_FALSE_NEG_RATE, false_negative_rate),
                (FTR_RAND_INDEX, rand_index),
                (FTR_ADJUSTED_RAND_INDEX, adjusted_rand_index)
            ]
            if self.wants_emd:
                workspace.display_data.statistics.append(
                        (FTR_EARTH_MOVERS_DISTANCE, emd))

    def compute_rand_index(self, test_labels, ground_truth_labels, mask):
        """Calculate the Rand Index

        http://en.wikipedia.org/wiki/Rand_index

        Given a set of N elements and two partitions of that set, X and Y

        A = the number of pairs of elements in S that are in the same set in
            X and in the same set in Y
        B = the number of pairs of elements in S that are in different sets
            in X and different sets in Y
        C = the number of pairs of elements in S that are in the same set in
            X and different sets in Y
        D = the number of pairs of elements in S that are in different sets
            in X and the same set in Y

        The rand index is:   A + B
                             -----
                            A+B+C+D


        The adjusted rand index is the rand index adjusted for chance
        so as not to penalize situations with many segmentations.

        Jorge M. Santos, Mark Embrechts, "On the Use of the Adjusted Rand
        Index as a Metric for Evaluating Supervised Classification",
        Lecture Notes in Computer Science,
        Springer, Vol. 5769, pp. 175-184, 2009. Eqn # 6

        ExpectedIndex = best possible score

        ExpectedIndex = sum(N_i choose 2) * sum(N_j choose 2)

        MaxIndex = worst possible score = 1/2 (sum(N_i choose 2) + sum(N_j choose 2)) * total

        A * total - ExpectedIndex
        -------------------------
        MaxIndex - ExpectedIndex

        returns a tuple of the Rand Index and the adjusted Rand Index
        """
        ground_truth_labels = ground_truth_labels[mask].astype(np.uint64)
        test_labels = test_labels[mask].astype(np.uint64)
        if len(test_labels) > 0:
            #
            # Create a sparse matrix of the pixel labels in each of the sets
            #
            # The matrix, N(i,j) gives the counts of all of the pixels that were
            # labeled with label I in the ground truth and label J in the
            # test set.
            #
            N_ij = coo_matrix((np.ones(len(test_labels)),
                               (ground_truth_labels, test_labels))).toarray()

            def choose2(x):
                '''Compute # of pairs of x things = x * (x-1) / 2'''
                return x * (x - 1) / 2

            #
            # Each cell in the matrix is a count of a grouping of pixels whose
            # pixel pairs are in the same set in both groups. The number of
            # pixel pairs is n * (n - 1), so A = sum(matrix * (matrix - 1))
            #
            A = np.sum(choose2(N_ij))
            #
            # B is the sum of pixels that were classified differently by both
            # sets. But the easier calculation is to find A, C and D and get
            # B by subtracting A, C and D from the N * (N - 1), the total
            # number of pairs.
            #
            # For C, we take the number of pixels classified as "i" and for each
            # "j", subtract N(i,j) from N(i) to get the number of pixels in
            # N(i,j) that are in some other set = (N(i) - N(i,j)) * N(i,j)
            #
            # We do the similar calculation for D
            #
            N_i = np.sum(N_ij, 1)
            N_j = np.sum(N_ij, 0)
            C = np.sum((N_i[:, np.newaxis] - N_ij) * N_ij) / 2
            D = np.sum((N_j[np.newaxis, :] - N_ij) * N_ij) / 2
            total = choose2(len(test_labels))
            # an astute observer would say, why bother computing A and B
            # when all we need is A+B and C, D and the total can be used to do
            # that. The calculations aren't too expensive, though, so I do them.
            B = total - A - C - D
            rand_index = (A + B) / total
            #
            # Compute adjusted Rand Index
            #
            expected_index = np.sum(choose2(N_i)) * np.sum(choose2(N_j))
            max_index = (np.sum(choose2(N_i)) + np.sum(choose2(N_j))) * total / 2

            adjusted_rand_index = \
                (A * total - expected_index) / (max_index - expected_index)
        else:
            rand_index = adjusted_rand_index = np.nan
        return rand_index, adjusted_rand_index

    def compute_rand_index_ijv(self, gt_ijv, test_ijv, shape):
        '''Compute the Rand Index for an IJV matrix

        This is in part based on the Omega Index:
        Collins, "Omega: A General Formulation of the Rand Index of Cluster
        Recovery Suitable for Non-disjoint Solutions", Multivariate Behavioral
        Research, 1988, 23, 231-242

        The basic idea of the paper is that a pair should be judged to
        agree only if the number of clusters in which they appear together
        is the same.
        '''
        #
        # The idea here is to assign a label to every pixel position based
        # on the set of labels given to that position by both the ground
        # truth and the test set. We then assess each pair of labels
        # as agreeing or disagreeing as to the number of matches.
        #
        # First, add the backgrounds to the IJV with a label of zero
        #
        gt_bkgd = np.ones(shape, bool)
        gt_bkgd[gt_ijv[:, 0], gt_ijv[:, 1]] = False
        test_bkgd = np.ones(shape, bool)
        test_bkgd[test_ijv[:, 0], test_ijv[:, 1]] = False
        gt_ijv = np.vstack([
            gt_ijv,
            np.column_stack([np.argwhere(gt_bkgd),
                             np.zeros(np.sum(gt_bkgd), gt_bkgd.dtype)])])
        test_ijv = np.vstack([
            test_ijv,
            np.column_stack([np.argwhere(test_bkgd),
                             np.zeros(np.sum(test_bkgd), test_bkgd.dtype)])])
        #
        # Create a unified structure for the pixels where a fourth column
        # tells you whether the pixels came from the ground-truth or test
        #
        u = np.vstack([
            np.column_stack([gt_ijv, np.zeros(gt_ijv.shape[0], gt_ijv.dtype)]),
            np.column_stack([test_ijv, np.ones(test_ijv.shape[0], test_ijv.dtype)])])
        #
        # Sort by coordinates, then by identity
        #
        order = np.lexsort([u[:, 2], u[:, 3], u[:, 0], u[:, 1]])
        u = u[order, :]
        # Get rid of any duplicate labelings (same point labeled twice with
        # same label.
        #
        first = np.hstack([[True], np.any(u[:-1, :] != u[1:, :], 1)])
        u = u[first, :]
        #
        # Create a 1-d indexer to point at each unique coordinate.
        #
        first_coord_idxs = np.hstack([
            [0],
            np.argwhere((u[:-1, 0] != u[1:, 0]) |
                        (u[:-1, 1] != u[1:, 1])).flatten() + 1,
            [u.shape[0]]])
        first_coord_counts = first_coord_idxs[1:] - first_coord_idxs[:-1]
        indexes = Indexes([first_coord_counts])
        #
        # Count the number of labels at each point for both gt and test
        #
        count_test = np.bincount(indexes.rev_idx, u[:, 3]).astype(np.int64)
        count_gt = first_coord_counts - count_test
        #
        # For each # of labels, pull out the coordinates that have
        # that many labels. Count the number of similarly labeled coordinates
        # and record the count and labels for that group.
        #
        labels = []
        for i in range(1, np.max(count_test) + 1):
            for j in range(1, np.max(count_gt) + 1):
                match = ((count_test[indexes.rev_idx] == i) &
                         (count_gt[indexes.rev_idx] == j))
                if not np.any(match):
                    continue
                #
                # Arrange into an array where the rows are coordinates
                # and the columns are the labels for that coordinate
                #
                lm = u[match, 2].reshape(np.sum(match) / (i + j), i + j)
                #
                # Sort by label.
                #
                order = np.lexsort(lm.transpose())
                lm = lm[order, :]
                #
                # Find indices of unique and # of each
                #
                lm_first = np.hstack([
                    [0],
                    np.argwhere(np.any(lm[:-1, :] != lm[1:, :], 1)).flatten() + 1,
                    [lm.shape[0]]])
                lm_count = lm_first[1:] - lm_first[:-1]
                for idx, count in zip(lm_first[:-1], lm_count):
                    labels.append((count,
                                   lm[idx, :j],
                                   lm[idx, j:]))
        #
        # We now have our sets partitioned. Do each against each to get
        # the number of true positive and negative pairs.
        #
        max_t_labels = reduce(max, [len(t) for c, t, g in labels], 0)
        max_g_labels = reduce(max, [len(g) for c, t, g in labels], 0)
        #
        # tbl is the contingency table from Table 4 of the Collins paper
        # It's a table of the number of pairs which fall into M sets
        # in the ground truth case and N in the test case.
        #
        tbl = np.zeros(((max_t_labels + 1), (max_g_labels + 1)))
        for i, (c1, tobject_numbers1, gobject_numbers1) in enumerate(labels):
            for j, (c2, tobject_numbers2, gobject_numbers2) in \
                    enumerate(labels[i:]):
                nhits_test = np.sum(
                        tobject_numbers1[:, np.newaxis] ==
                        tobject_numbers2[np.newaxis, :])
                nhits_gt = np.sum(
                        gobject_numbers1[:, np.newaxis] ==
                        gobject_numbers2[np.newaxis, :])
                if j == 0:
                    N = c1 * (c1 - 1) / 2
                else:
                    N = c1 * c2
                tbl[nhits_test, nhits_gt] += N

        N = np.sum(tbl)
        #
        # Equation 13 from the paper
        #
        min_JK = min(max_t_labels, max_g_labels) + 1
        rand_index = np.sum(tbl[:min_JK, :min_JK] * np.identity(min_JK)) / N
        #
        # Equation 15 from the paper, the expected index
        #
        e_omega = np.sum(np.sum(tbl[:min_JK, :min_JK], 0) *
                         np.sum(tbl[:min_JK, :min_JK], 1)) / N ** 2
        #
        # Equation 16 is the adjusted index
        #
        adjusted_rand_index = (rand_index - e_omega) / (1 - e_omega)
        return rand_index, adjusted_rand_index

    def compute_emd(self, src_objects, dest_objects):
        '''Compute the earthmovers distance between two sets of objects

        src_objects - move pixels from these objects

        dest_objects - move pixels to these objects

        returns the earth mover's distance
        '''
        #
        # if either foreground set is empty, the emd is the penalty.
        #
        for angels, demons in ((src_objects, dest_objects),
                               (dest_objects, src_objects)):
            if angels.count == 0:
                if self.penalize_missing:
                    return np.sum(demons.areas) * self.max_distance.value
                else:
                    return 0
        if self.decimation_method == DM_KMEANS:
            isrc, jsrc = self.get_kmeans_points(src_objects, dest_objects)
            idest, jdest = isrc, jsrc
        else:
            isrc, jsrc = self.get_skeleton_points(src_objects)
            idest, jdest = self.get_skeleton_points(dest_objects)
        src_weights, dest_weights = [
            self.get_weights(i, j, self.get_labels_mask(objects))
            for i, j, objects in ((isrc, jsrc, src_objects),
                                  (idest, jdest, dest_objects))]
        ioff, joff = [src[:, np.newaxis] - dest[np.newaxis, :]
                      for src, dest in ((isrc, idest), (jsrc, jdest))]
        c = np.sqrt(ioff * ioff + joff * joff).astype(np.int32)
        c[c > self.max_distance.value] = self.max_distance.value
        extra_mass_penalty = \
            self.max_distance.value if self.penalize_missing else 0
        return emd_hat_int32(
                src_weights.astype(np.int32),
                dest_weights.astype(np.int32),
                c,
                extra_mass_penalty=extra_mass_penalty)

    def get_labels_mask(self, obj):
        labels_mask = np.zeros(obj.shape, bool)
        for labels, indexes in obj.labels():
            labels_mask = labels_mask | labels > 0
        return labels_mask

    def get_skeleton_points(self, obj):
        '''Get points by skeletonizing the objects and decimating'''
        ii = []
        jj = []
        total_skel = np.zeros(obj.shape, bool)
        for labels, indexes in obj.labels():
            colors = morph.color_labels(labels)
            for color in range(1, np.max(colors) + 1):
                labels_mask = colors == color
                skel = morph.skeletonize(
                        labels_mask,
                        ordering=distance_transform_edt(labels_mask) *
                                 poisson_equation(labels_mask))
                total_skel = total_skel | skel
        n_pts = np.sum(total_skel)
        if n_pts == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        i, j = np.where(total_skel)
        if n_pts > self.max_points.value:
            #
            # Decimate the skeleton by finding the branchpoints in the
            # skeleton and propagating from those.
            #
            markers = np.zeros(total_skel.shape, np.int32)
            branchpoints = \
                morph.branchpoints(total_skel) | morph.endpoints(total_skel)
            markers[branchpoints] = np.arange(np.sum(branchpoints)) + 1
            #
            # We compute the propagation distance to that point, then impose
            # a slightly arbitarary order to get an unambiguous ordering
            # which should number the pixels in a skeleton branch monotonically
            #
            ts_labels, distances = propagate(np.zeros(markers.shape),
                                             markers, total_skel, 1)
            order = np.lexsort((j, i, distances[i, j], ts_labels[i, j]))
            #
            # Get a linear space of self.max_points elements with bounds at
            # 0 and len(order)-1 and use that to select the points.
            #
            order = order[
                np.linspace(0, len(order) - 1, self.max_points.value).astype(int)]
            return i[order], j[order]
        return i, j

    def get_kmeans_points(self, src_obj, dest_obj):
        '''Get representative points in the objects using K means

        src_obj - get some of the foreground points from the source objects
        dest_obj - get the rest of the foreground points from the destination
                   objects

        returns a vector of i coordinates of representatives and a vector
                of j coordinates
        '''
        from sklearn.cluster import KMeans

        ijv = np.vstack((src_obj.ijv, dest_obj.ijv))
        if len(ijv) <= self.max_points.value:
            return ijv[:, 0], ijv[:, 1]
        random_state = np.random.RandomState()
        random_state.seed(ijv.astype(int).flatten())
        kmeans = KMeans(n_clusters=self.max_points.value, tol=2,
                        random_state=random_state)
        kmeans.fit(ijv[:, :2])
        return kmeans.cluster_centers_[:, 0].astype(np.uint32), \
               kmeans.cluster_centers_[:, 1].astype(np.uint32)

    def get_weights(self, i, j, labels_mask):
        '''Return the weights to assign each i,j point

        Assign each pixel in the labels mask to the nearest i,j and return
        the number of pixels assigned to each i,j
        '''
        #
        # Create a mapping of chosen points to their index in the i,j array
        #
        total_skel = np.zeros(labels_mask.shape, int)
        total_skel[i, j] = np.arange(1, len(i) + 1)
        #
        # Compute the distance from each chosen point to all others in image,
        # return the nearest point.
        #
        ii, jj = distance_transform_edt(
                total_skel == 0,
                return_indices=True,
                return_distances=False)
        #
        # Filter out all unmasked points
        #
        ii, jj = [x[labels_mask] for x in ii, jj]
        if len(ii) == 0:
            return np.zeros(0, np.int32)
        #
        # Use total_skel to look up the indices of the chosen points and
        # bincount the indices.
        #
        result = np.zeros(len(i), np.int32)
        bc = np.bincount(total_skel[ii, jj])[1:]
        result[:len(bc)] = bc
        return result

    def display(self, workspace, figure):
        '''Display the image confusion matrix & statistics'''
        figure.set_subplots((3, 2))
        for x, y, image, label in (
                (0, 0, workspace.display_data.true_positives, "True positives"),
                (0, 1, workspace.display_data.false_positives, "False positives"),
                (1, 0, workspace.display_data.false_negatives, "False negatives"),
                (1, 1, workspace.display_data.true_negatives, "True negatives")):
            figure.subplot_imshow_bw(x, y, image, title=label,
                                     sharexy=figure.subplot(0, 0))

        figure.subplot_table(2, 0,
                             workspace.display_data.statistics,
                             col_labels=("Measurement", "Value"),
                             n_rows=2)

    def measurement_name(self, feature):
        if self.obj_or_img == O_IMG:
            name = '_'.join((C_IMAGE_OVERLAP, feature, self.test_img.value))
        if self.obj_or_img == O_OBJ:
            name = '_'.join((C_IMAGE_OVERLAP, feature,
                             self.object_name_GT.value,
                             self.object_name_ID.value))
        return name

    def get_categories(self, pipeline, object_name):
        '''Return the measurement categories for an object'''
        if object_name == cpmeas.IMAGE:
            return [C_IMAGE_OVERLAP]
        return []

    def get_measurements(self, pipeline, object_name, category):
        '''Return the measurements made for a category'''
        if object_name == cpmeas.IMAGE and category == C_IMAGE_OVERLAP:
            return self.all_features()
        return []

    def get_measurement_images(self, pipeline, object_name, category,
                               measurement):
        '''Return the images that were used when making the measurement'''
        if measurement in self.get_measurements(pipeline, object_name, category) \
                and self.obj_or_img == O_IMG:
            return [self.test_img.value]
        return []

    def get_measurement_scales(
            self, pipeline, object_name, category, measurement, image_name):
        '''Return a "scale" that captures the measurement parameters

        pipeline - the module's pipeline

        object_name - should be "Images"

        category - measurement category

        measurement - measurement feature name

        image_name - ignored

        The "scale" in this case is the combination of ground-truth objects and
        test objects.
        '''
        if (object_name == cpmeas.IMAGE and category == C_IMAGE_OVERLAP and
                    measurement in FTR_ALL and self.obj_or_img == O_OBJ):
            return ["_".join((self.object_name_GT.value,
                              self.object_name_ID.value))]
        return []

    def all_features(self):
        '''Return a list of all the features measured by this module'''
        all_features = list(FTR_ALL)
        if self.wants_emd:
            all_features.append(FTR_EARTH_MOVERS_DISTANCE)
        return all_features

    def get_measurement_columns(self, pipeline):
        '''Return database column information for each measurement'''
        return [(cpmeas.IMAGE,
                 self.measurement_name(feature),
                 cpmeas.COLTYPE_FLOAT)
                for feature in self.all_features()]

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab:
            # Variable revision # wasn't in Matlab file
            # All settings were identical to CP 2.0 v 1
            from_matlab = False
            variable_revision_number = 1
        if variable_revision_number == 1:
            # no object choice before rev 2
            old_setting_values = setting_values
            setting_values = [
                O_IMG, old_setting_values[0], old_setting_values[1],
                "None", "None", "None", "None"]
            variable_revision_number = 2
        if variable_revision_number == 2:
            #
            # Removed images associated with objects from the settings
            #
            setting_values = setting_values[:4] + setting_values[5:6]
            variable_revision_number = 3

        if variable_revision_number == 3:
            #
            # Added earth mover's distance
            #
            setting_values = setting_values + [
                cps.NO,  # wants_emd
                250,  # max points
                DM_KMEANS,  # decimation method
                250,  # max distance
                cps.NO  # penalize missing
            ]
            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab
