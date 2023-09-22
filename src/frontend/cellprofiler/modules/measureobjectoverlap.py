"""
MeasureObjectOverlap
====================

**MeasureObjectOverlap** calculates how much overlap occurs between
objects.

This module calculates object overlap by determining a set of statistics
that measure the closeness of an object to its true value. One
object is considered the “ground truth” (possibly the result of
hand-segmentation) and the other is the “test” object; the objects
are determined to overlap most completely when the test object matches
the ground truth perfectly. The module requires input to be objects obtained
after "IdentifyPrimaryObjects", "IdentifySecondaryObjects" or "IdentifyTertiaryObjects".
If your images have been segmented using other image processing software,
or you have hand-segmented them in software such as Photoshop, you will
need to use "Object Processing" modules such as "IdentifyPrimaryObjects" to identify
"ground truth" objects.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *True positive rate:* Total number of true positive pixels / total number of actual positive pixels.

-  *False positive rate:* Total number of false positive pixels / total number of actual negative pixels

-  *True negative rate:* Total number of true negative pixels / total number of actual negative pixels.

-  *False negative rate:* Total number of false negative pixels / total number of actual positive pixels

-  *Precision:* Number of true positive pixels / (number of true positive pixels + number of false positive pixels)

-  *Recall:* Number of true positive pixels/ (number of true positive pixels + number of false negative pixels)

-  *F-factor:* 2 × (precision × recall)/(precision + recall). Also known as F\ :sub:`1` score, F-score or F-measure.

-  *Earth mover’s distance:* The minimum distance required to move each foreground
   pixel in the test object to some corresponding foreground pixel in the reference object.

-  *Rand index:* A measure of the similarity between two data clusterings. Perfectly random clustering
   returns the minimum score of 0, perfect clustering returns the maximum score of 1.

-  *Adjusted Rand index:* A variation of the Rand index which considers a correction for chance.

References
^^^^^^^^^^

-  Collins LM, Dent CW (1988) “Omega: A general formulation of the Rand
   Index of cluster recovery suitable for non-disjoint solutions”,
   *Multivariate Behavioral Research*, 23, 231-242 `(link)`_

-  Pele O, Werman M (2009) “Fast and Robust Earth Mover’s Distances”,
   *2009 IEEE 12th International Conference on Computer Vision*

.. _(link): https://doi.org/10.1207/s15327906mbr2302_6
"""

from functools import reduce

import centrosome.cpmorphology
import centrosome.fastemd
import centrosome.filter
import centrosome.index
import centrosome.propagate
import numpy
import scipy.ndimage
import scipy.sparse
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Integer

from cellprofiler.modules import _help

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

FTR_ALL = [
    FTR_F_FACTOR,
    FTR_PRECISION,
    FTR_RECALL,
    FTR_TRUE_POS_RATE,
    FTR_TRUE_NEG_RATE,
    FTR_FALSE_POS_RATE,
    FTR_FALSE_NEG_RATE,
    FTR_RAND_INDEX,
    FTR_ADJUSTED_RAND_INDEX,
]

O_OBJ = "Segmented objects"

L_LOAD = "Loaded from a previous run"
L_CP = "From this CP pipeline"

DM_KMEANS = "K Means"
DM_SKEL = "Skeleton"


class MeasureObjectOverlap(Module):
    category = "Measurement"
    variable_revision_number = 2
    module_name = "MeasureObjectOverlap"

    def create_settings(self):
        self.object_name_GT = LabelSubscriber(
            "Select the objects to be used as the ground truth basis for calculating the amount of overlap",
            "None",
            doc="""\
Choose which set of objects will used as the “ground truth” objects. It
can be the product of segmentation performed by hand, or the result of
another segmentation algorithm whose results you would like to compare.
See the **Load** modules for more details on loading objects.""",
        )

        self.object_name_ID = LabelSubscriber(
            "Select the objects to be tested for overlap against the ground truth",
            "None",
            doc="""\
This set of objects is what you will compare with the ground truth
objects. It is known as the “test object.”""",
        )

        self.wants_emd = Binary(
            "Calculate earth mover's distance?",
            False,
            doc="""\
The earth mover’s distance computes the shortest distance that would
have to be travelled to move each foreground pixel in the test object to
some foreground pixel in the reference object. “Earth mover’s” refers to
an analogy: the pixels are “earth” that has to be moved by some machine
at the smallest possible cost.
It would take too much memory and processing time to compute the exact
earth mover’s distance, so **MeasureObjectOverlap** chooses
representative foreground pixels in each object and assigns each
foreground pixel to its closest representative. The earth mover’s
distance is then computed for moving the foreground pixels associated
with each representative in the test object to those in the reference
object.""",
        )

        self.max_points = Integer(
            "Maximum # of points",
            value=250,
            minval=100,
            doc="""\
*(Used only when computing the earth mover’s distance)*

This is the number of representative points that will be taken from the
foreground of the test objects and from the foreground of the reference
objects using the point selection method (see below).""",
        )

        self.decimation_method = Choice(
            "Point selection method",
            choices=[DM_KMEANS, DM_SKEL],
            doc="""\
*(Used only when computing the earth mover’s distance)*

The point selection setting determines how the representative points
are chosen.

-  *{DM_KMEANS}:* Select to pick representative points using a K-Means
   clustering technique. The foregrounds of both objects are combined and
   representatives are picked that minimize the distance to the nearest
   representative. The same representatives are then used for the test
   and reference objects.
-  *{DM_SKEL}:* Select to skeletonize the object and pick points
   equidistant along the skeleton.

|image0|  *{DM_KMEANS}* is a choice that’s generally applicable to all
images. *{DM_SKEL}* is best suited to long, skinny objects such as
worms or neurites.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{
                    "DM_KMEANS": DM_KMEANS,
                    "DM_SKEL": DM_SKEL,
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                }
            ),
        )

        self.max_distance = Integer(
            "Maximum distance",
            value=250,
            minval=1,
            doc="""\
*(Used only when computing the earth mover’s distance)*

This setting sets an upper bound to the distance penalty assessed during
the movement calculation. As an example, the score for moving 10 pixels
from one location to a location that is 100 pixels away is 10\*100, but
if the maximum distance were set to 50, the score would be 10\*50
instead.

The maximum distance should be set to the largest reasonable distance
that pixels could be expected to move from one object to the next.""",
        )

        self.penalize_missing = Binary(
            "Penalize missing pixels",
            value=False,
            doc="""\
*(Used only when computing the earth mover’s distance)*

If one object has more foreground pixels than the other, the earth
mover’s distance is not well-defined because there is no destination for
the extra source pixels or vice-versa. It’s reasonable to assess a
penalty for the discrepancy when comparing the accuracy of a
segmentation because the discrepancy represents an error. It’s also
reasonable to assess no penalty if the goal is to compute the cost of
movement, for example between two frames in a time-lapse movie, because
the discrepancy is likely caused by noise or artifacts in segmentation.
Set this setting to “Yes” to assess a penalty equal to the maximum
distance times the absolute difference in number of foreground pixels in
the two objects. Set this setting to “No” to assess no penalty.""",
        )

    def settings(self):
        return [
            self.object_name_GT,
            self.object_name_ID,
            self.wants_emd,
            self.max_points,
            self.decimation_method,
            self.max_distance,
            self.penalize_missing,
        ]

    def visible_settings(self):
        visible_settings = [self.object_name_GT, self.object_name_ID, self.wants_emd]

        if self.wants_emd:
            visible_settings += [
                self.max_points,
                self.decimation_method,
                self.max_distance,
                self.penalize_missing,
            ]

        return visible_settings

    def run(self, workspace):
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
        GT_pixels = numpy.zeros((xGT, yGT))
        ID_pixels = numpy.zeros((xID, yID))
        total_pixels = xGT * yGT

        GT_pixels[iGT, jGT] = 1
        ID_pixels[iID, jID] = 1

        GT_tot_area = len(iGT)
        if len(iGT) == 0 and len(iID) == 0:
            intersect_matrix = numpy.zeros((0, 0), int)
        else:
            #
            # Build a matrix with rows of i, j, label and a GT/ID flag
            #
            all_ijv = numpy.column_stack(
                (
                    numpy.hstack((iGT, iID)),
                    numpy.hstack((jGT, jID)),
                    numpy.hstack((lGT, lID)),
                    numpy.hstack((numpy.zeros(len(iGT)), numpy.ones(len(iID)))),
                )
            )
            #
            # Order it so that runs of the same i, j are consecutive
            #
            order = numpy.lexsort((all_ijv[:, -1], all_ijv[:, 0], all_ijv[:, 1]))
            all_ijv = all_ijv[order, :]
            # Mark the first at each i, j != previous i, j
            first = numpy.where(
                numpy.hstack(
                    ([True], ~numpy.all(all_ijv[:-1, :2] == all_ijv[1:, :2], 1), [True])
                )
            )[0]
            # Count # at each i, j
            count = first[1:] - first[:-1]
            # First indexer - mapping from i,j to index in all_ijv
            all_ijv_map = centrosome.index.Indexes([count])
            # Bincount to get the # of ID pixels per i,j
            id_count = numpy.bincount(all_ijv_map.rev_idx, all_ijv[:, -1]).astype(int)
            gt_count = count - id_count
            # Now we can create an indexer that has NxM elements per i,j
            # where N is the number of GT pixels at that i,j and M is
            # the number of ID pixels. We can then use the indexer to pull
            # out the label values for each to populate a sparse array.
            #
            cross_map = centrosome.index.Indexes([id_count, gt_count])
            off_gt = all_ijv_map.fwd_idx[cross_map.rev_idx] + cross_map.idx[0]
            off_id = (
                all_ijv_map.fwd_idx[cross_map.rev_idx]
                + cross_map.idx[1]
                + id_count[cross_map.rev_idx]
            )
            intersect_matrix = scipy.sparse.coo_matrix(
                (numpy.ones(len(off_gt)), (all_ijv[off_id, 2], all_ijv[off_gt, 2])),
                shape=(ID_obj + 1, GT_obj + 1),
            ).toarray()[1:, 1:]

        gt_areas = objects_GT.areas
        id_areas = objects_ID.areas
        FN_area = gt_areas[numpy.newaxis, :] - intersect_matrix
        all_intersecting_area = numpy.sum(intersect_matrix)

        dom_ID = []

        for i in range(0, ID_obj):
            indices_jj = numpy.nonzero(lID == i)
            indices_jj = indices_jj[0]
            id_i = iID[indices_jj]
            id_j = jID[indices_jj]
            ID_pixels[id_i, id_j] = 1

        for i in intersect_matrix:  # loop through the GT objects first
            if len(i) == 0 or max(i) == 0:
                id = -1  # we missed the object; arbitrarily assign -1 index
            else:
                id = numpy.where(i == max(i))[0][0]  # what is the ID of the max pixels?
            dom_ID += [id]  # for ea GT object, which is the dominating ID?

        dom_ID = numpy.array(dom_ID)

        for i in range(0, len(intersect_matrix.T)):
            if len(numpy.where(dom_ID == i)[0]) > 1:
                final_id = numpy.where(
                    intersect_matrix.T[i] == max(intersect_matrix.T[i])
                )
                final_id = final_id[0][0]
                all_id = numpy.where(dom_ID == i)[0]
                nonfinal = [x for x in all_id if x != final_id]
                for (
                    n
                ) in nonfinal:  # these others cannot be candidates for the corr ID now
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
                fp = numpy.sum(intersect_matrix[i][0:d]) + numpy.sum(
                    intersect_matrix[i][(d + 1) : :]
                )
                tp = intersect_matrix[i][d]
                fn = FN_area[i][d]
            TP += tp
            FN += fn
            FP += fp

        TN = max(0, total_pixels - TP - FN - FP)

        def nan_divide(numerator, denominator):
            if denominator == 0:
                return numpy.nan
            return float(numerator) / float(denominator)

        accuracy = nan_divide(TP, all_intersecting_area)
        recall = nan_divide(TP, GT_tot_area)
        precision = nan_divide(TP, (TP + FP))
        F_factor = nan_divide(2 * (precision * recall), (precision + recall))
        true_positive_rate = nan_divide(TP, (FN + TP))
        false_positive_rate = nan_divide(FP, (FP + TN))
        false_negative_rate = nan_divide(FN, (FN + TP))
        true_negative_rate = nan_divide(TN, (FP + TN))
        shape = numpy.maximum(
            numpy.maximum(numpy.array(objects_GT.shape), numpy.array(objects_ID.shape)),
            numpy.ones(2, int),
        )
        rand_index, adjusted_rand_index = self.compute_rand_index_ijv(
            objects_GT.ijv, objects_ID.ijv, shape
        )
        m = workspace.measurements
        m.add_image_measurement(self.measurement_name(FTR_F_FACTOR), F_factor)
        m.add_image_measurement(self.measurement_name(FTR_PRECISION), precision)
        m.add_image_measurement(self.measurement_name(FTR_RECALL), recall)
        m.add_image_measurement(
            self.measurement_name(FTR_TRUE_POS_RATE), true_positive_rate
        )
        m.add_image_measurement(
            self.measurement_name(FTR_FALSE_POS_RATE), false_positive_rate
        )
        m.add_image_measurement(
            self.measurement_name(FTR_TRUE_NEG_RATE), true_negative_rate
        )
        m.add_image_measurement(
            self.measurement_name(FTR_FALSE_NEG_RATE), false_negative_rate
        )
        m.add_image_measurement(self.measurement_name(FTR_RAND_INDEX), rand_index)
        m.add_image_measurement(
            self.measurement_name(FTR_ADJUSTED_RAND_INDEX), adjusted_rand_index
        )

        def subscripts(condition1, condition2):
            x1, y1 = numpy.where(GT_pixels == condition1)
            x2, y2 = numpy.where(ID_pixels == condition2)
            mask = set(zip(x1, y1)) & set(zip(x2, y2))
            return list(mask)

        TP_mask = subscripts(1, 1)
        FN_mask = subscripts(1, 0)
        FP_mask = subscripts(0, 1)
        TN_mask = subscripts(0, 0)

        TP_pixels = numpy.zeros((xGT, yGT))
        FN_pixels = numpy.zeros((xGT, yGT))
        FP_pixels = numpy.zeros((xGT, yGT))
        TN_pixels = numpy.zeros((xGT, yGT))

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
                self.measurement_name(FTR_EARTH_MOVERS_DISTANCE), emd
            )

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
                (FTR_ADJUSTED_RAND_INDEX, adjusted_rand_index),
            ]
            if self.wants_emd:
                workspace.display_data.statistics.append(
                    (FTR_EARTH_MOVERS_DISTANCE, emd)
                )

    # def compute_rand_index(self, test_labels, ground_truth_labels, mask):
    #     """Calculate the Rand Index
    #
    #     http://en.wikipedia.org/wiki/Rand_index
    #
    #     Given a set of N elements and two partitions of that set, X and Y
    #
    #     A = the number of pairs of elements in S that are in the same set in
    #         X and in the same set in Y
    #     B = the number of pairs of elements in S that are in different sets
    #         in X and different sets in Y
    #     C = the number of pairs of elements in S that are in the same set in
    #         X and different sets in Y
    #     D = the number of pairs of elements in S that are in different sets
    #         in X and the same set in Y
    #
    #     The rand index is:   A + B
    #                          -----
    #                         A+B+C+D
    #
    #
    #     The adjusted rand index is the rand index adjusted for chance
    #     so as not to penalize situations with many segmentations.
    #
    #     Jorge M. Santos, Mark Embrechts, "On the Use of the Adjusted Rand
    #     Index as a Metric for Evaluating Supervised Classification",
    #     Lecture Notes in Computer Science,
    #     Springer, Vol. 5769, pp. 175-184, 2009. Eqn # 6
    #
    #     ExpectedIndex = best possible score
    #
    #     ExpectedIndex = sum(N_i choose 2) * sum(N_j choose 2)
    #
    #     MaxIndex = worst possible score = 1/2 (sum(N_i choose 2) + sum(N_j choose 2)) * total
    #
    #     A * total - ExpectedIndex
    #     -------------------------
    #     MaxIndex - ExpectedIndex
    #
    #     returns a tuple of the Rand Index and the adjusted Rand Index
    #     """
    #     ground_truth_labels = ground_truth_labels[mask].astype(numpy.uint64)
    #     test_labels = test_labels[mask].astype(numpy.uint64)
    #     if len(test_labels) > 0:
    #         #
    #         # Create a sparse matrix of the pixel labels in each of the sets
    #         #
    #         # The matrix, N(i,j) gives the counts of all of the pixels that were
    #         # labeled with label I in the ground truth and label J in the
    #         # test set.
    #         #
    #         N_ij = scipy.sparse.coo_matrix((numpy.ones(len(test_labels)),
    #                                         (ground_truth_labels, test_labels))).toarray()
    #
    #         def choose2(x):
    #             '''Compute # of pairs of x things = x * (x-1) / 2'''
    #             return x * (x - 1) / 2
    #
    #         #
    #         # Each cell in the matrix is a count of a grouping of pixels whose
    #         # pixel pairs are in the same set in both groups. The number of
    #         # pixel pairs is n * (n - 1), so A = sum(matrix * (matrix - 1))
    #         #
    #         A = numpy.sum(choose2(N_ij))
    #         #
    #         # B is the sum of pixels that were classified differently by both
    #         # sets. But the easier calculation is to find A, C and D and get
    #         # B by subtracting A, C and D from the N * (N - 1), the total
    #         # number of pairs.
    #         #
    #         # For C, we take the number of pixels classified as "i" and for each
    #         # "j", subtract N(i,j) from N(i) to get the number of pixels in
    #         # N(i,j) that are in some other set = (N(i) - N(i,j)) * N(i,j)
    #         #
    #         # We do the similar calculation for D
    #         #
    #         N_i = numpy.sum(N_ij, 1)
    #         N_j = numpy.sum(N_ij, 0)
    #         C = numpy.sum((N_i[:, numpy.newaxis] - N_ij) * N_ij) / 2
    #         D = numpy.sum((N_j[numpy.newaxis, :] - N_ij) * N_ij) / 2
    #         total = choose2(len(test_labels))
    #         # an astute observer would say, why bother computing A and B
    #         # when all we need is A+B and C, D and the total can be used to do
    #         # that. The calculations aren't too expensive, though, so I do them.
    #         B = total - A - C - D
    #         rand_index = (A + B) / total
    #         #
    #         # Compute adjusted Rand Index
    #         #
    #         expected_index = numpy.sum(choose2(N_i)) * numpy.sum(choose2(N_j))
    #         max_index = (numpy.sum(choose2(N_i)) + numpy.sum(choose2(N_j))) * total / 2
    #
    #         adjusted_rand_index = \
    #             (A * total - expected_index) / (max_index - expected_index)
    #     else:
    #         rand_index = adjusted_rand_index = numpy.nan
    #     return rand_index, adjusted_rand_index

    def compute_rand_index_ijv(self, gt_ijv, test_ijv, shape):
        """Compute the Rand Index for an IJV matrix

        This is in part based on the Omega Index:
        Collins, "Omega: A General Formulation of the Rand Index of Cluster
        Recovery Suitable for Non-disjoint Solutions", Multivariate Behavioral
        Research, 1988, 23, 231-242

        The basic idea of the paper is that a pair should be judged to
        agree only if the number of clusters in which they appear together
        is the same.
        """
        #
        # The idea here is to assign a label to every pixel position based
        # on the set of labels given to that position by both the ground
        # truth and the test set. We then assess each pair of labels
        # as agreeing or disagreeing as to the number of matches.
        #
        # First, add the backgrounds to the IJV with a label of zero
        #
        gt_bkgd = numpy.ones(shape, bool)
        gt_bkgd[gt_ijv[:, 0], gt_ijv[:, 1]] = False
        test_bkgd = numpy.ones(shape, bool)
        test_bkgd[test_ijv[:, 0], test_ijv[:, 1]] = False
        gt_ijv = numpy.vstack(
            [
                gt_ijv,
                numpy.column_stack(
                    [
                        numpy.argwhere(gt_bkgd),
                        numpy.zeros(numpy.sum(gt_bkgd), gt_bkgd.dtype),
                    ]
                ),
            ]
        )
        test_ijv = numpy.vstack(
            [
                test_ijv,
                numpy.column_stack(
                    [
                        numpy.argwhere(test_bkgd),
                        numpy.zeros(numpy.sum(test_bkgd), test_bkgd.dtype),
                    ]
                ),
            ]
        )
        #
        # Create a unified structure for the pixels where a fourth column
        # tells you whether the pixels came from the ground-truth or test
        #
        u = numpy.vstack(
            [
                numpy.column_stack(
                    [gt_ijv, numpy.zeros(gt_ijv.shape[0], gt_ijv.dtype)]
                ),
                numpy.column_stack(
                    [test_ijv, numpy.ones(test_ijv.shape[0], test_ijv.dtype)]
                ),
            ]
        )
        #
        # Sort by coordinates, then by identity
        #
        order = numpy.lexsort([u[:, 2], u[:, 3], u[:, 0], u[:, 1]])
        u = u[order, :]
        # Get rid of any duplicate labellings (same point labeled twice with
        # same label.
        #
        first = numpy.hstack([[True], numpy.any(u[:-1, :] != u[1:, :], 1)])
        u = u[first, :]
        #
        # Create a 1-d indexer to point at each unique coordinate.
        #
        first_coord_idxs = numpy.hstack(
            [
                [0],
                numpy.argwhere(
                    (u[:-1, 0] != u[1:, 0]) | (u[:-1, 1] != u[1:, 1])
                ).flatten()
                + 1,
                [u.shape[0]],
            ]
        )
        first_coord_counts = first_coord_idxs[1:] - first_coord_idxs[:-1]
        indexes = centrosome.index.Indexes([first_coord_counts])
        #
        # Count the number of labels at each point for both gt and test
        #
        count_test = numpy.bincount(indexes.rev_idx, u[:, 3]).astype(numpy.int64)
        count_gt = first_coord_counts - count_test
        #
        # For each # of labels, pull out the coordinates that have
        # that many labels. Count the number of similarly labeled coordinates
        # and record the count and labels for that group.
        #
        labels = []
        for i in range(1, numpy.max(count_test) + 1):
            for j in range(1, numpy.max(count_gt) + 1):
                match = (count_test[indexes.rev_idx] == i) & (
                    count_gt[indexes.rev_idx] == j
                )
                if not numpy.any(match):
                    continue
                #
                # Arrange into an array where the rows are coordinates
                # and the columns are the labels for that coordinate
                #
                lm = u[match, 2].reshape(numpy.sum(match) // (i + j), i + j)
                #
                # Sort by label.
                #
                order = numpy.lexsort(lm.transpose())
                lm = lm[order, :]
                #
                # Find indices of unique and # of each
                #
                lm_first = numpy.hstack(
                    [
                        [0],
                        numpy.argwhere(numpy.any(lm[:-1, :] != lm[1:, :], 1)).flatten()
                        + 1,
                        [lm.shape[0]],
                    ]
                )
                lm_count = lm_first[1:] - lm_first[:-1]
                for idx, count in zip(lm_first[:-1], lm_count):
                    labels.append((count, lm[idx, :j], lm[idx, j:]))
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
        tbl = numpy.zeros(((max_t_labels + 1), (max_g_labels + 1)))
        for i, (c1, tobject_numbers1, gobject_numbers1) in enumerate(labels):
            for j, (c2, tobject_numbers2, gobject_numbers2) in enumerate(labels[i:]):
                nhits_test = numpy.sum(
                    tobject_numbers1[:, numpy.newaxis]
                    == tobject_numbers2[numpy.newaxis, :]
                )
                nhits_gt = numpy.sum(
                    gobject_numbers1[:, numpy.newaxis]
                    == gobject_numbers2[numpy.newaxis, :]
                )
                if j == 0:
                    N = c1 * (c1 - 1) / 2
                else:
                    N = c1 * c2
                tbl[nhits_test, nhits_gt] += N

        N = numpy.sum(tbl)
        #
        # Equation 13 from the paper
        #
        min_JK = min(max_t_labels, max_g_labels) + 1
        rand_index = numpy.sum(tbl[:min_JK, :min_JK] * numpy.identity(min_JK)) / N
        #
        # Equation 15 from the paper, the expected index
        #
        e_omega = (
            numpy.sum(
                numpy.sum(tbl[:min_JK, :min_JK], 0)
                * numpy.sum(tbl[:min_JK, :min_JK], 1)
            )
            / N ** 2
        )
        #
        # Equation 16 is the adjusted index
        #
        adjusted_rand_index = (rand_index - e_omega) / (1 - e_omega)
        return rand_index, adjusted_rand_index

    def compute_emd(self, src_objects, dest_objects):
        """Compute the earthmovers distance between two sets of objects

        src_objects - move pixels from these objects

        dest_objects - move pixels to these objects

        returns the earth mover's distance
        """
        #
        # if either foreground set is empty, the emd is the penalty.
        #
        for angels, demons in (
            (src_objects, dest_objects),
            (dest_objects, src_objects),
        ):
            if angels.count == 0:
                if self.penalize_missing:
                    return numpy.sum(demons.areas) * self.max_distance.value
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
            for i, j, objects in (
                (isrc, jsrc, src_objects),
                (idest, jdest, dest_objects),
            )
        ]
        ioff, joff = [
            src[:, numpy.newaxis] - dest[numpy.newaxis, :]
            for src, dest in ((isrc, idest), (jsrc, jdest))
        ]
        c = numpy.sqrt(ioff * ioff + joff * joff).astype(numpy.int32)
        c[c > self.max_distance.value] = self.max_distance.value
        extra_mass_penalty = self.max_distance.value if self.penalize_missing else 0
        return centrosome.fastemd.emd_hat_int32(
            src_weights.astype(numpy.int32),
            dest_weights.astype(numpy.int32),
            c,
            extra_mass_penalty=extra_mass_penalty,
        )

    def get_labels_mask(self, obj):
        labels_mask = numpy.zeros(obj.shape, bool)
        for labels, indexes in obj.get_labels():
            labels_mask = labels_mask | labels > 0
        return labels_mask

    def get_skeleton_points(self, obj):
        """Get points by skeletonizing the objects and decimating"""
        ii = []
        jj = []
        total_skel = numpy.zeros(obj.shape, bool)
        for labels, indexes in obj.get_labels():
            colors = centrosome.cpmorphology.color_labels(labels)
            for color in range(1, numpy.max(colors) + 1):
                labels_mask = colors == color
                skel = centrosome.cpmorphology.skeletonize(
                    labels_mask,
                    ordering=scipy.ndimage.distance_transform_edt(labels_mask)
                    * centrosome.filter.poisson_equation(labels_mask),
                )
                total_skel = total_skel | skel
        n_pts = numpy.sum(total_skel)
        if n_pts == 0:
            return numpy.zeros(0, numpy.int32), numpy.zeros(0, numpy.int32)
        i, j = numpy.where(total_skel)
        if n_pts > self.max_points.value:
            #
            # Decimate the skeleton by finding the branchpoints in the
            # skeleton and propagating from those.
            #
            markers = numpy.zeros(total_skel.shape, numpy.int32)
            branchpoints = centrosome.cpmorphology.branchpoints(
                total_skel
            ) | centrosome.cpmorphology.endpoints(total_skel)
            markers[branchpoints] = numpy.arange(numpy.sum(branchpoints)) + 1
            #
            # We compute the propagation distance to that point, then impose
            # a slightly arbitarary order to get an unambiguous ordering
            # which should number the pixels in a skeleton branch monotonically
            #
            ts_labels, distances = centrosome.propagate.propagate(
                numpy.zeros(markers.shape), markers, total_skel, 1
            )
            order = numpy.lexsort((j, i, distances[i, j], ts_labels[i, j]))
            #
            # Get a linear space of self.max_points elements with bounds at
            # 0 and len(order)-1 and use that to select the points.
            #
            order = order[
                numpy.linspace(0, len(order) - 1, self.max_points.value).astype(int)
            ]
            return i[order], j[order]
        return i, j

    def get_kmeans_points(self, src_obj, dest_obj):
        """Get representative points in the objects using K means

        src_obj - get some of the foreground points from the source objects
        dest_obj - get the rest of the foreground points from the destination
                   objects

        returns a vector of i coordinates of representatives and a vector
                of j coordinates
        """
        from sklearn.cluster import KMeans

        ijv = numpy.vstack((src_obj.ijv, dest_obj.ijv))
        if len(ijv) <= self.max_points.value:
            return ijv[:, 0], ijv[:, 1]
        random_state = numpy.random.RandomState()
        random_state.seed(ijv.astype(int).flatten())
        kmeans = KMeans(
            n_clusters=self.max_points.value, tol=2, random_state=random_state
        )
        kmeans.fit(ijv[:, :2])
        return (
            kmeans.cluster_centers_[:, 0].astype(numpy.uint32),
            kmeans.cluster_centers_[:, 1].astype(numpy.uint32),
        )

    def get_weights(self, i, j, labels_mask):
        """Return the weights to assign each i,j point

        Assign each pixel in the labels mask to the nearest i,j and return
        the number of pixels assigned to each i,j
        """
        #
        # Create a mapping of chosen points to their index in the i,j array
        #
        total_skel = numpy.zeros(labels_mask.shape, int)
        total_skel[i, j] = numpy.arange(1, len(i) + 1)
        #
        # Compute the distance from each chosen point to all others in image,
        # return the nearest point.
        #
        ii, jj = scipy.ndimage.distance_transform_edt(
            total_skel == 0, return_indices=True, return_distances=False
        )
        #
        # Filter out all unmasked points
        #
        ii, jj = [x[labels_mask] for x in (ii, jj)]
        if len(ii) == 0:
            return numpy.zeros(0, numpy.int32)
        #
        # Use total_skel to look up the indices of the chosen points and
        # bincount the indices.
        #
        result = numpy.zeros(len(i), numpy.int32)
        bc = numpy.bincount(total_skel[ii, jj])[1:]
        result[: len(bc)] = bc
        return result

    def display(self, workspace, figure):
        """Display the image confusion matrix & statistics"""
        figure.set_subplots((3, 2))

        for x, y, image, label in (
            (0, 0, workspace.display_data.true_positives, "True positives"),
            (0, 1, workspace.display_data.false_positives, "False positives"),
            (1, 0, workspace.display_data.false_negatives, "False negatives"),
            (1, 1, workspace.display_data.true_negatives, "True negatives"),
        ):
            figure.subplot_imshow_bw(
                x, y, image, title=label, sharexy=figure.subplot(0, 0)
            )

        figure.subplot_table(
            2,
            0,
            workspace.display_data.statistics,
            col_labels=("Measurement", "Value"),
            n_rows=2,
        )

    def measurement_name(self, feature):
        return "_".join(
            (
                C_IMAGE_OVERLAP,
                feature,
                self.object_name_GT.value,
                self.object_name_ID.value,
            )
        )

    def get_categories(self, pipeline, object_name):
        if object_name == "Image":
            return [C_IMAGE_OVERLAP]

        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == "Image" and category == C_IMAGE_OVERLAP:
            return self.all_features()

        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.test_img.value]

        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, measurement, image_name
    ):
        if (
            object_name == "Image"
            and category == C_IMAGE_OVERLAP
            and measurement in FTR_ALL
        ):
            return ["_".join((self.object_name_GT.value, self.object_name_ID.value))]

        return []

    def all_features(self):
        all_features = list(FTR_ALL)

        if self.wants_emd:
            all_features.append(FTR_EARTH_MOVERS_DISTANCE)

        return all_features

    def get_measurement_columns(self, pipeline):
        return [
            ("Image", self.measurement_name(feature), COLTYPE_FLOAT,)
            for feature in self.all_features()
        ]
