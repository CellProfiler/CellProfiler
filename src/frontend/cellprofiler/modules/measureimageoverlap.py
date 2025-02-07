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

from cellprofiler.modules import _help

from cellprofiler_library.modules import measureimageoverlap
from cellprofiler_library.opts.measureimageoverlap import DM
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Integer

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
    FTR_FALSE_POS_RATE,
    FTR_FALSE_NEG_RATE,
    FTR_TRUE_NEG_RATE,
    FTR_RAND_INDEX,
    FTR_ADJUSTED_RAND_INDEX,
]

O_OBJ = "Segmented objects"
O_IMG = "Foreground/background segmentation"

L_LOAD = "Loaded from a previous run"
L_CP = "From this CP pipeline"


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
            choices=DM,
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
                    "DM_KMEANS": DM.KMEANS.value,
                    "DM_SKEL": DM.SKELETON.value,
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

        data = measureimageoverlap(
            ground_truth_pixels, 
            test_pixels, 
            mask=mask,
            calculate_emd=self.wants_emd,
            decimation_method=self.decimation_method.enum_member,
            max_distance=self.max_distance.value,
            max_points=self.max_points.value,
            penalize_missing=self.penalize_missing
            )

        m = workspace.measurements

        m.add_image_measurement(self.measurement_name(FTR_F_FACTOR), data[FTR_F_FACTOR])

        m.add_image_measurement(self.measurement_name(FTR_PRECISION), data[FTR_PRECISION])

        m.add_image_measurement(self.measurement_name(FTR_RECALL), data[FTR_RECALL])

        m.add_image_measurement(
            self.measurement_name(FTR_TRUE_POS_RATE), data[FTR_TRUE_POS_RATE]
        )

        m.add_image_measurement(
            self.measurement_name(FTR_FALSE_POS_RATE), data[FTR_FALSE_POS_RATE]
        )

        m.add_image_measurement(
            self.measurement_name(FTR_TRUE_NEG_RATE), data[FTR_TRUE_NEG_RATE]
        )

        m.add_image_measurement(
            self.measurement_name(FTR_FALSE_NEG_RATE), data[FTR_FALSE_NEG_RATE]
        )

        m.add_image_measurement(self.measurement_name(FTR_RAND_INDEX), data[FTR_RAND_INDEX])

        m.add_image_measurement(
            self.measurement_name(FTR_ADJUSTED_RAND_INDEX), data[FTR_ADJUSTED_RAND_INDEX]
        )

        if self.wants_emd:

            m.add_image_measurement(
                self.measurement_name(FTR_EARTH_MOVERS_DISTANCE), data[FTR_EARTH_MOVERS_DISTANCE]
            )

        if self.show_window:
           
            workspace.display_data.dimensions = test_image.dimensions
           
            workspace.display_data.true_positives = data["true_positives"]

            workspace.display_data.true_negatives = data["true_negatives"]

            workspace.display_data.false_positives = data["false_positives"]

            workspace.display_data.false_negatives = data["false_negatives"]

            workspace.display_data.rand_index = data[FTR_RAND_INDEX]

            workspace.display_data.adjusted_rand_index = data[FTR_ADJUSTED_RAND_INDEX]

            workspace.display_data.statistics = [
                (FTR_F_FACTOR, data[FTR_F_FACTOR]),
                (FTR_PRECISION, data[FTR_PRECISION]),
                (FTR_RECALL, data[FTR_RECALL]),
                (FTR_FALSE_POS_RATE, data[FTR_FALSE_POS_RATE]),
                (FTR_FALSE_NEG_RATE, data[FTR_FALSE_NEG_RATE]),
                (FTR_RAND_INDEX, data[FTR_RAND_INDEX]),
                (FTR_ADJUSTED_RAND_INDEX, data[FTR_ADJUSTED_RAND_INDEX]),
            ]

            if self.wants_emd:
                workspace.display_data.statistics.append(
                    (FTR_EARTH_MOVERS_DISTANCE, data[FTR_EARTH_MOVERS_DISTANCE])
                )


    def display(self, workspace, figure):
        """Display the image confusion matrix & statistics"""
        figure.set_subplots((3, 2), dimensions=workspace.display_data.dimensions)

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
                DM.KMEANS.value,  # decimation method
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
