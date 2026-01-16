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

import numpy
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Integer

from cellprofiler_library.modules._measureobjectoverlap import measure_object_overlap
from cellprofiler_library.opts.measureobjectoverlap import Feature, ALL_FEATURES, C_IMAGE_OVERLAP, DecimationMethod
from cellprofiler.modules import _help

O_OBJ = "Segmented objects"

L_LOAD = "Loaded from a previous run"
L_CP = "From this CP pipeline"

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
            choices=[DecimationMethod.KMEANS, DecimationMethod.SKELETON],
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
                    "DM_KMEANS": DecimationMethod.KMEANS.value,
                    "DM_SKEL": DecimationMethod.SKELETON.value,
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
        object_name_ID = self.object_name_ID.value

        objects_GT = workspace.get_objects(object_name_GT)
        objects_ID = workspace.get_objects(object_name_ID)

        objects_GT_labelset = objects_GT.get_labels()
        objects_ID_labelset = objects_ID.get_labels()

        result = measure_object_overlap(
            objects_GT_labelset,
            objects_ID_labelset,
            objects_GT.shape,
            objects_ID.shape,
            object_name_GT=object_name_GT,
            object_name_ID=object_name_ID,
            calcualte_emd=self.wants_emd.value,
            decimation_method=self.decimation_method.value,
            max_distance=self.max_distance.value,
            max_points=self.max_points.value,
            penalize_missing=self.penalize_missing.value,
            return_visualization_data=self.show_window
        )
        
        # Unpack result based on whether visualization data was requested
        if self.show_window:
            lib_measurements, GT_pixels, ID_pixels, xGT, yGT = result
        else:
            lib_measurements = result
        
        m = workspace.measurements
        for feature_name, value in lib_measurements.image.items():
            m.add_image_measurement(feature_name, value)

        if self.show_window:
            def get_val(feature):
                name = self.measurement_name(feature)
                return lib_measurements.image.get(name)

            F_factor = get_val(Feature.F_FACTOR)
            precision = get_val(Feature.PRECISION)
            recall = get_val(Feature.RECALL)
            false_positive_rate = get_val(Feature.FALSE_POS_RATE)
            false_negative_rate = get_val(Feature.FALSE_NEG_RATE)
            rand_index = get_val(Feature.RAND_INDEX)
            adjusted_rand_index = get_val(Feature.ADJUSTED_RAND_INDEX)
            emd = get_val(Feature.EARTH_MOVERS_DISTANCE) if self.wants_emd.value else None

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
            
            workspace.display_data.true_positives = TP_pixels
            workspace.display_data.true_negatives = TN_pixels
            workspace.display_data.false_positives = FP_pixels
            workspace.display_data.false_negatives = FN_pixels
            workspace.display_data.statistics = [
                (Feature.F_FACTOR.value, F_factor),
                (Feature.PRECISION.value, precision),
                (Feature.RECALL.value, recall),
                (Feature.FALSE_POS_RATE.value, false_positive_rate),
                (Feature.FALSE_NEG_RATE.value, false_negative_rate),
                (Feature.RAND_INDEX.value, rand_index),
                (Feature.ADJUSTED_RAND_INDEX.value, adjusted_rand_index),
            ]
            if self.wants_emd:
                assert emd is not None, "Earth Movers Distance was not calculated"
                workspace.display_data.statistics.append(
                    (Feature.EARTH_MOVERS_DISTANCE.value, emd)
                )


    def get_labels_mask(self, obj_labels, obj_shape):
        labels_mask = numpy.zeros(obj_shape, bool)
        for labels, indexes in obj_labels:
            labels_mask = labels_mask | labels > 0
        return labels_mask

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
            and measurement in ALL_FEATURES
        ):
            return ["_".join((self.object_name_GT.value, self.object_name_ID.value))]

        return []

    def all_features(self):
        all_features = list(ALL_FEATURES)

        if self.wants_emd:
            all_features.append(Feature.EARTH_MOVERS_DISTANCE)

        return all_features

    def get_measurement_columns(self, pipeline):
        return [
            ("Image", self.measurement_name(feature), COLTYPE_FLOAT,)
            for feature in self.all_features()
        ]
