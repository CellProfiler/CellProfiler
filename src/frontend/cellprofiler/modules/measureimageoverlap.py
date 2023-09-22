"""
MeasureImageOverlap
===================

**MeasureImageOverlap** calculates how much overlap occurs between
the white portions of two black and white images

This module calculates overlap by determining a set of statistics that
measure the closeness of an image to its true value. One
image is considered the “ground truth” (possibly the result of
hand-segmentation) and the other is the “test” image; the images
are determined to overlap most completely when the test image matches
the ground truth perfectly. The module requires binary
(black and white) input, where the foreground of the images is white and the
background is black. If you segment your images in CellProfiler using
**IdentifyPrimaryObjects**, you can create such an image using
**ConvertObjectsToImage** by selecting *Binary* as the color type. If
your images have been segmented using other image processing software,
or you have hand-segmented them in software such as Photoshop, you may
need to use one or more of the following to prepare the images for this
module:

-  **ImageMath**: If the objects are black and the background is white,
   you must invert the intensity using this module.

-  **Threshold**: If the image is grayscale, you must make it
   binary using this module, or alternately use an **Identify** module
   followed by **ConvertObjectsToImage** as described above.

-  **ColorToGray**: If the image is in color, you must first convert it
   to grayscale using this module, and then use **Threshold** to
   generate a binary image.

In the test image, any foreground (white) pixels that overlap with the
foreground of the ground truth will be considered “true positives”,
since they are correctly labeled as foreground. Background (black)
pixels that overlap with the background of the ground truth image are
considered “true negatives”, since they are correctly labeled as
background. A foreground pixel in the test image that overlaps with the
background in the ground truth image will be considered a “false
positive” (since it should have been labeled as part of the background),
while a background pixel in the test image that overlaps with foreground
in the ground truth will be considered a “false negative” (since it was
labeled as part of the background, but should not be).

For 3D images, all image planes are concatenated into one large XY image and 
the overlap is computed on the transformed image. 

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *True positive rate:* Total number of true positive pixels / total number of actual positive pixels.

-  *False positive rate:* Total number of false positive pixels / total number of actual negative pixels.

-  *True negative rate:* Total number of true negative pixels / total number of actual negative pixels.

-  *False negative rate:* Total number of false negative pixels / total number of actual positive pixels.

-  *Precision:* Number of true positive pixels / (number of true positive pixels + number of false positive pixels).

-  *Recall:* Number of true positive pixels/ (number of true positive pixels + number of false negative pixels).

-  *F-factor:* 2 × (precision × recall)/(precision + recall). Also known as F\ :sub:`1` score, F-score or F-measure.

-  *Earth mover’s distance:* The minimum distance required to move each foreground pixel in the test image to
   some corresponding foreground pixel in the reference image.

-  *Rand index:* A measure of the similarity between two data clusterings. Perfectly random clustering
   returns the minimum score of 0, perfect clustering returns the maximum score of 1.

-  *Adjusted Rand index:* A variation of the Rand index which considers a correction for chance.

References
^^^^^^^^^^

-  Collins LM, Dent CW (1988) “Omega: A general formulation of the Rand
   Index of cluster recovery suitable for non-disjoint solutions”,
   *Multivariate Behavioral Research*, 23, 231-242. `(link) <https://doi.org/10.1207/s15327906mbr2302_6>`__
-  Pele O, Werman M (2009) “Fast and Robust Earth Mover’s Distances”,
   *2009 IEEE 12th International Conference on Computer Vision*.
"""

import cellprofiler_core.object
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
from cellprofiler_core.setting.subscriber import ImageSubscriber
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
O_IMG = "Foreground/background segmentation"

L_LOAD = "Loaded from a previous run"
L_CP = "From this CP pipeline"

DM_KMEANS = "K Means"
DM_SKEL = "Skeleton"


class MeasureImageOverlap(Module):
    category = "Measurement"
    variable_revision_number = 5
    module_name = "MeasureImageOverlap"

    def create_settings(self):
        self.ground_truth = ImageSubscriber(
            "Select the image to be used as the ground truth basis for calculating the amount of overlap",
            "None",
            doc="""\
This binary (black and white) image is known as the “ground truth”
image. It can be the product of segmentation performed by hand, or the
result of another segmentation algorithm whose results you would like to
compare.""",
        )

        self.test_img = ImageSubscriber(
            "Select the image to be used to test for overlap",
            "None",
            doc="""\
This binary (black and white) image is what you will compare with the
ground truth image. It is known as the “test image”.""",
        )

        self.wants_emd = Binary(
            "Calculate earth mover's distance?",
            False,
            doc="""\
The earth mover’s distance computes the shortest distance that would
have to be travelled to move each foreground pixel in the test image to
some foreground pixel in the reference image. “Earth mover’s” refers to
an analogy: the pixels are “earth” that has to be moved by some machine
at the smallest possible cost.
It would take too much memory and processing time to compute the exact
earth mover’s distance, so **MeasureImageOverlap** chooses
representative foreground pixels in each image and assigns each
foreground pixel to its closest representative. The earth mover’s
distance is then computed for moving the foreground pixels associated
with each representative in the test image to those in the reference
image.""",
        )

        self.max_points = Integer(
            "Maximum # of points",
            value=250,
            minval=100,
            doc="""\
*(Used only when computing the earth mover’s distance)*

This is the number of representative points that will be taken from the
foreground of the test image and from the foreground of the reference
image using the point selection method (see below).""",
        )

        self.decimation_method = Choice(
            "Point selection method",
            choices=[DM_KMEANS, DM_SKEL],
            doc="""\
*(Used only when computing the earth mover’s distance)*

The point selection setting determines how the representative points
are chosen.

-  *{DM_KMEANS}:* Select to pick representative points using a K-Means
   clustering technique. The foregrounds of both images are combined and
   representatives are picked that minimize the distance to the nearest
   representative. The same representatives are then used for the test
   and reference images.
-  *{DM_SKEL}:* Select to skeletonize the image and pick points
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
that pixels could be expected to move from one image to the next.""",
        )

        self.penalize_missing = Binary(
            "Penalize missing pixels",
            value=False,
            doc="""\
*(Used only when computing the earth mover’s distance)*

If one image has more foreground pixels than the other, the earth
mover’s distance is not well-defined because there is no destination for
the extra source pixels or vice-versa. It’s reasonable to assess a
penalty for the discrepancy when comparing the accuracy of a
segmentation because the discrepancy represents an error. It’s also
reasonable to assess no penalty if the goal is to compute the cost of
movement, for example between two frames in a time-lapse movie, because
the discrepancy is likely caused by noise or artifacts in segmentation.
Set this setting to “Yes” to assess a penalty equal to the maximum
distance times the absolute difference in number of foreground pixels in
the two images. Set this setting to “No” to assess no penalty.""",
        )

    def settings(self):
        return [
            self.ground_truth,
            self.test_img,
            self.wants_emd,
            self.max_points,
            self.decimation_method,
            self.max_distance,
            self.penalize_missing,
        ]

    def visible_settings(self):
        visible_settings = [self.ground_truth, self.test_img, self.wants_emd]

        if self.wants_emd:
            visible_settings += [
                self.max_points,
                self.decimation_method,
                self.max_distance,
                self.penalize_missing,
            ]

        return visible_settings

    def run(self, workspace):
        image_set = workspace.image_set

        ground_truth_image = image_set.get_image(
            self.ground_truth.value, must_be_binary=True
        )

        test_image = image_set.get_image(self.test_img.value, must_be_binary=True)

        ground_truth_pixels = ground_truth_image.pixel_data

        ground_truth_pixels = test_image.crop_image_similarly(ground_truth_pixels)

        mask = ground_truth_image.mask

        mask = test_image.crop_image_similarly(mask)

        if test_image.has_mask:
            mask = mask & test_image.mask

        test_pixels = test_image.pixel_data

        # In volumetric case the 3D image stack gets converted to a long 2D image and gets analyzed
        if ground_truth_image.volumetric:
            ground_truth_pixels = ground_truth_pixels.reshape(
                -1, ground_truth_pixels.shape[-1]
            )

            mask = mask.reshape(-1, mask.shape[-1])

            test_pixels = test_pixels.reshape(-1, test_pixels.shape[-1])

        false_positives = test_pixels & ~ground_truth_pixels

        false_positives[~mask] = False

        false_negatives = (~test_pixels) & ground_truth_pixels

        false_negatives[~mask] = False

        true_positives = test_pixels & ground_truth_pixels

        true_positives[~mask] = False

        true_negatives = (~test_pixels) & (~ground_truth_pixels)

        true_negatives[~mask] = False

        false_positive_count = numpy.sum(false_positives)

        true_positive_count = numpy.sum(true_positives)

        false_negative_count = numpy.sum(false_negatives)

        true_negative_count = numpy.sum(true_negatives)

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
            false_positive_rate = float(false_positive_count) / float(negative_count)

            true_negative_rate = float(true_negative_count) / float(negative_count)
        if true_count == 0:
            false_negative_rate = 0.0

            true_positive_rate = 1.0
        else:
            false_negative_rate = float(false_negative_count) / float(true_count)

            true_positive_rate = float(true_positive_count) / float(true_count)

        ground_truth_labels, ground_truth_count = scipy.ndimage.label(
            ground_truth_pixels & mask, numpy.ones((3, 3), bool)
        )

        test_labels, test_count = scipy.ndimage.label(
            test_pixels & mask, numpy.ones((3, 3), bool)
        )

        rand_index, adjusted_rand_index = self.compute_rand_index(
            test_labels, ground_truth_labels, mask
        )

        m = workspace.measurements

        m.add_image_measurement(self.measurement_name(FTR_F_FACTOR), f_factor)

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

        if self.wants_emd:
            test_objects = cellprofiler_core.object.Objects()

            test_objects.segmented = test_labels

            ground_truth_objects = cellprofiler_core.object.Objects()

            ground_truth_objects.segmented = ground_truth_labels

            emd = self.compute_emd(test_objects, ground_truth_objects)

            m.add_image_measurement(
                self.measurement_name(FTR_EARTH_MOVERS_DISTANCE), emd
            )

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
                (FTR_ADJUSTED_RAND_INDEX, adjusted_rand_index),
            ]

            if self.wants_emd:
                workspace.display_data.statistics.append(
                    (FTR_EARTH_MOVERS_DISTANCE, emd)
                )

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
        ground_truth_labels = ground_truth_labels[mask].astype(numpy.uint32)
        test_labels = test_labels[mask].astype(numpy.uint32)
        if len(test_labels) > 0:
            #
            # Create a sparse matrix of the pixel labels in each of the sets
            #
            # The matrix, N(i,j) gives the counts of all of the pixels that were
            # labeled with label I in the ground truth and label J in the
            # test set.
            #
            N_ij = scipy.sparse.coo_matrix(
                (numpy.ones(len(test_labels)), (ground_truth_labels, test_labels))
            ).toarray()

            def choose2(x):
                """Compute # of pairs of x things = x * (x-1) / 2"""
                return x * (x - 1) / 2

            #
            # Each cell in the matrix is a count of a grouping of pixels whose
            # pixel pairs are in the same set in both groups. The number of
            # pixel pairs is n * (n - 1), so A = sum(matrix * (matrix - 1))
            #
            A = numpy.sum(choose2(N_ij))
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
            N_i = numpy.sum(N_ij, 1)
            N_j = numpy.sum(N_ij, 0)
            C = numpy.sum((N_i[:, numpy.newaxis] - N_ij) * N_ij) / 2
            D = numpy.sum((N_j[numpy.newaxis, :] - N_ij) * N_ij) / 2
            total = choose2(len(test_labels))
            # an astute observer would say, why bother computing A and B
            # when all we need is A+B and C, D and the total can be used to do
            # that. The calculations aren't too expensive, though, so I do them.
            B = total - A - C - D
            rand_index = (A + B) / total
            #
            # Compute adjusted Rand Index
            #
            expected_index = numpy.sum(choose2(N_i)) * numpy.sum(choose2(N_j))
            max_index = (numpy.sum(choose2(N_i)) + numpy.sum(choose2(N_j))) * total / 2

            adjusted_rand_index = (A * total - expected_index) / (
                max_index - expected_index
            )
        else:
            rand_index = adjusted_rand_index = numpy.nan
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
            # a slightly arbitrary order to get an unambiguous ordering
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
        return "_".join((C_IMAGE_OVERLAP, feature, self.test_img.value))

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

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # no object choice before rev 2
            old_setting_values = setting_values
            setting_values = [
                O_IMG,
                old_setting_values[0],
                old_setting_values[1],
                "None",
                "None",
                "None",
                "None",
            ]
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
                "No",  # wants_emd
                250,  # max points
                DM_KMEANS,  # decimation method
                250,  # max distance
                "No",  # penalize missing
            ]
            variable_revision_number = 4

        if variable_revision_number == 4:
            obj_or_img = setting_values[0]

            if obj_or_img == O_OBJ:
                raise RuntimeError(
                    """\
MeasureImageOverlap does not compute object measurements.

Please update your pipeline to use MeasureObjectOverlap to compute object measurements.
"""
                )

            setting_values = setting_values[1:]
            variable_revision_number = 5

        return setting_values, variable_revision_number

    def volumetric(self):
        return True
