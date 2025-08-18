import logging

import cellprofiler_core.workspace
import numpy
import scipy.ndimage
import skimage.morphology
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, Binary, ValidationError
from cellprofiler_core.setting.subscriber import (
    ImageListSubscriber,
    LabelListSubscriber,
)
from cellprofiler_core.setting.text import Float, Integer
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix

from cellprofiler.gui.help.content import image_resource
from cellprofiler_library.modules._measuregranularity import measure_granularity, ObjectRecord

LOGGER = logging.getLogger(__name__)

__doc__ = """\
MeasureGranularity
==================
**MeasureGranularity** outputs spectra of size measurements of the
textures in the image.

Image granularity is a texture measurement that tries to fit a series of
structure elements of increasing size into the texture of the image and outputs a spectrum of measures
based on how well they fit.
Granularity is measured as described by Ilya Ravkin (references below).

Basically, MeasureGranularity:
1 - Downsamples the image (if you tell it to). This is set in
**Subsampling factor for granularity measurements** or **Subsampling factor for background reduction**.
2 - Background subtracts anything larger than the radius in pixels set in
**Radius of structuring element.**
3 - For as many times as you set in **Range of the granular spectrum**, it gets rid of bright areas
that are only 1 pixel across, reports how much signal was lost by doing that, then repeats.
i.e. The first time it removes one pixel from all bright areas in the image,
(effectively deleting those that are only 1 pixel in size) and then reports what % of the signal was lost.
It then takes the first-iteration image and repeats the removal and reporting (effectively reporting
the amount of signal that is two pixels in size). etc.

|MeasureGranularity_example|

As of **CellProfiler 4.0** the settings for this module have been changed to simplify
configuration. A single set of parameters is now applied to all images and objects within the module,
rather than each image needing individual configuration.
Pipelines from older versions will be converted to match this format. If multiple sets of parameters
were defined CellProfiler will apply the first set from the older pipeline version.
Specifying multiple sets of parameters can still be achieved by running multiple copies of this module.


|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Granularity:* The module returns one measurement for each instance
   of the granularity spectrum set in **Range of the granular spectrum**.

References
^^^^^^^^^^

-  Serra J. (1989) *Image Analysis and Mathematical Morphology*, Vol. 1.
   Academic Press, London
-  Maragos P. “Pattern spectrum and multiscale shape representation”,
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11,
   N 7, pp. 701-716, 1989
-  Vincent L. (2000) “Granulometries and Opening Trees”, *Fundamenta
   Informaticae*, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
-  Vincent L. (1992) “Morphological Area Opening and Closing for
   Grayscale Images”, *Proc. NATO Shape in Picture Workshop*,
   Driebergen, The Netherlands, pp. 197-208.
-  Ravkin I, Temov V. (1988) “Bit representation techniques and image
   processing”, *Applied Informatics*, v.14, pp. 41-90, Finances and
   Statistics, Moskow, (in Russian)

.. |MeasureGranularity_example| image:: {MEASUREGRANULARITY_EXAMPLE}
""".format(
    **{"MEASUREGRANULARITY_EXAMPLE": image_resource("MeasureGranularity_example.png")}
)


"Granularity category"
C_GRANULARITY = "Granularity_%s_%s"

IMAGE_SETTING_COUNT_V2 = 5
IMAGE_SETTING_COUNT_V3 = 6
IMAGE_SETTING_COUNT = IMAGE_SETTING_COUNT_V3

OBJECTS_SETTING_COUNT_V3 = 1
OBJECTS_SETTING_COUNT = OBJECTS_SETTING_COUNT_V3


class MeasureGranularity(Module):
    module_name = "MeasureGranularity"
    category = "Measurement"
    variable_revision_number = 4

    def create_settings(self):
        self.images_list = ImageListSubscriber(
            "Select images to measure",
            [],
            doc="""Select images in which to measure the granularity.""",
        )

        self.divider_top = Divider(line=True)

        self.wants_objects = Binary(
            "Measure within objects?",
            False,
            doc="""\
        Press this button to capture granularity measurements for objects, such as
        those identified by a prior **IdentifyPrimaryObjects** module.
        **MeasureGranularity** will measure the image’s granularity within each
        object at the requested scales.""",
        )

        self.objects_list = LabelListSubscriber(
            "Select objects to measure",
            [],
            doc="""\
        *(Used only when "Measure within objects" is enabled)*

        Select the objects within which granularity will be measured.""",
        )

        self.divider_bottom = Divider(line=True)
        self.subsample_size = Float(
            "Subsampling factor for granularity measurements",
            0.25,
            minval=numpy.finfo(float).eps,
            maxval=1,
            doc="""\
        If the textures of interest are larger than a few pixels, we recommend
        you subsample the image with a factor <1 to speed up the processing.
        Downsampling the image will let you detect larger structures with a
        smaller sized structure element. A factor >1 might increase the accuracy
        but also require more processing time. Images are typically of higher
        resolution than is required for granularity measurements, so the default
        value is 0.25. For low-resolution images, increase the subsampling
        fraction; for high-resolution images, decrease the subsampling fraction.
        Subsampling by 1/4 reduces computation time by (1/4) :sup:`3` because the
        size of the image is (1/4) :sup:`2` of original and the range of granular
        spectrum can be 1/4 of original. Moreover, the results are sometimes
        actually a little better with subsampling, which is probably because
        with subsampling the individual granular spectrum components can be used
        as features, whereas without subsampling a feature should be a sum of
        several adjacent granular spectrum components. The recommendation on the
        numerical value cannot be determined in advance; an analysis as in this
        reference may be required before running the whole set. See this `pdf`_,
        slides 27-31, 49-50.

        .. _pdf: http://www.ravkin.net/presentations/Statistical%20properties%20of%20algorithms%20for%20analysis%20of%20cell%20images.pdf""",
        )

        self.image_sample_size = Float(
            "Subsampling factor for background reduction",
            0.25,
            minval=numpy.finfo(float).eps,
            maxval=1,
            doc="""\
        It is important to remove low frequency image background variations as
        they will affect the final granularity measurement. Any method can be
        used as a pre-processing step prior to this module; we have chosen to
        simply subtract a highly open image. To do it quickly, we subsample the
        image first. The subsampling factor for background reduction is usually
        [0.125 – 0.25]. This is highly empirical, but a small factor should be
        used if the structures of interest are large. The significance of
        background removal in the context of granulometry is that image volume
        at certain granular size is normalized by total image volume, which
        depends on how the background was removed.""",
        )

        self.element_size = Integer(
            "Radius of structuring element",
            10,
            minval=1,
            doc="""\
        This radius should correspond to the radius of the textures of interest
        *after* subsampling; i.e., if textures in the original image scale have
        a radius of 40 pixels, and a subsampling factor of 0.25 is used, the
        structuring element size should be 10 or slightly smaller, and the range
        of the spectrum defined below will cover more sizes.""",
        )

        self.granular_spectrum_length = Integer(
            "Range of the granular spectrum",
            16,
            minval=1,
            doc="""\
        You may need a trial run to see which granular
        spectrum range yields informative measurements. Start by using a wide spectrum and
        narrow it down to the informative range to save time.""",
        )

    def validate_module(self, pipeline):
        """Make sure settings are compatible. In particular, we make sure that no measurements are duplicated"""
        if len(self.images_list.value) == 0:
            raise ValidationError("No images selected", self.images_list)

        if self.wants_objects.value:
            if len(self.objects_list.value) == 0:
                raise ValidationError("No object sets selected", self.objects_list)

        measurements, sources = self.get_measurement_columns(
            pipeline, return_sources=True
        )
        d = {}
        for m, s in zip(measurements, sources):
            if m in d:
                raise ValidationError("Measurement %s made twice." % (m[1]), s[0])
            d[m] = True

    def settings(self):
        result = [
            self.images_list,
            self.wants_objects,
            self.objects_list,
            self.subsample_size,
            self.image_sample_size,
            self.element_size,
            self.granular_spectrum_length,
        ]
        return result

    def visible_settings(self):
        result = [self.images_list, self.divider_top, self.wants_objects]
        if self.wants_objects.value:
            result += [self.objects_list]
        result += [
            self.divider_bottom,
            self.subsample_size,
            self.image_sample_size,
            self.element_size,
            self.granular_spectrum_length,
        ]
        return result

    def run(self, workspace):
        col_labels = ["Image name"] + [
            "GS%d" % n for n in range(1, self.granular_spectrum_length.value + 1)
        ]
        statistics = []
        for image_name in self.images_list.value:
            object_records = [
        (objects_name, workspace.object_set.get_objects(objects_name).segmented) for objects_name in self.objects_list.value
    ]
            statistic = self.run_on_image_setting(workspace, image_name)
            statistics.append(statistic)
        if self.show_window:
            workspace.display_data.statistics = statistics
            workspace.display_data.col_labels = col_labels

    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics
        col_labels = workspace.display_data.col_labels
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0,
            0,
            statistics,
            col_labels=col_labels,
            title="If individual objects were measured, use an Export module to view their results",
        )

    def add_names_to_measurements(self, measurements_arr, image_measurements_arr, statistics, image_name, granular_spectrum_length):
        _statistics = [image_name] + statistics
        _measurements_arr = []
        _image_measurements_arr = []
        ng = granular_spectrum_length
        for i in range(1, ng+1):
            feature = C_GRANULARITY % (i, image_name)
            _image_measurements_arr += [(feature, image_measurements_arr[i-1])]
            if i < len(measurements_arr):
                if measurements_arr[i]:
                    _measurements_arr += [(measurements_arr[i-1][0], feature, measurements_arr[i-1][1])]
        return _measurements_arr, _image_measurements_arr, _statistics

    def run_on_image_setting(self, workspace, image_name):
        im = workspace.image_set.get_image(image_name, must_be_grayscale=True)
        im_pixel_data = im.pixel_data
        im_mask = im.mask
        subsample_size = self.subsample_size.value
        image_sample_size = self.image_sample_size.value
        element_size = self.element_size.value
        object_records = [
            ObjectRecord(objects_name, workspace.object_set.get_objects(objects_name).segmented, im_mask, im_pixel_data) for objects_name in self.objects_list.value
        ]
        granular_spectrum_length = self.granular_spectrum_length.value
        image_name = image_name
        dimensions = im.dimensions

        measurements_arr, image_measurements_arr, statistics = measure_granularity(
            im_pixel_data, 
            im_mask, 
            subsample_size, 
            image_sample_size, 
            element_size, 
            object_records, 
            granular_spectrum_length, 
            dimensions
        )
        measurements_arr, image_measurements_arr, statistics = self.add_names_to_measurements(measurements_arr, image_measurements_arr, statistics, image_name, granular_spectrum_length)

        for packed_measurements in measurements_arr:
            workspace.measurements.add_measurement(*packed_measurements)
        for packed_measurements in image_measurements_arr:
            workspace.measurements.add_image_measurement(*packed_measurements)

        return statistics

    def get_measurement_columns(self, pipeline, return_sources=False):
        result = []
        sources = []
        for image_name in self.images_list.value:
            gslength = self.granular_spectrum_length.value
            for i in range(1, gslength + 1):
                result += [
                    ("Image", self.granularity_feature(i, image_name), COLTYPE_FLOAT,)
                ]
                sources += [(image_name, self.granularity_feature(i, image_name))]
            for object_name in self.objects_list.value:
                for i in range(1, gslength + 1):
                    result += [
                        (
                            object_name,
                            self.granularity_feature(i, image_name),
                            COLTYPE_FLOAT,
                        )
                    ]
                    sources += [(object_name, self.granularity_feature(i, image_name))]

        if return_sources:
            return result, sources
        else:
            return result

    def get_matching_images(self, object_name):
        """Return all image records that match the given object name

        object_name - name of an object or IMAGE to match all
        """
        if object_name == "Image":
            return self.images_list.value
        return [
            image_name
            for image_name in self.images_list.value
            if object_name in self.objects_list.value
        ]

    def get_categories(self, pipeline, object_name):
        """Return the categories supported by this module for the given object

        object_name - name of the measured object or IMAGE
        """
        if object_name in self.objects_list.value and self.wants_objects.value:
            return ["Granularity"]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        max_length = 0
        if category == "Granularity":
            max_length = max(max_length, self.granular_spectrum_length.value)
        return [str(i) for i in range(1, max_length + 1)]

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        result = []
        if category == "Granularity":
            try:
                length = int(measurement)
                if length <= 0:
                    return []
            except ValueError:
                return []
            if self.granular_spectrum_length.value >= length:
                for image_name in self.images_list.value:
                    result.append(image_name)
        return result

    def granularity_feature(self, length, image_name):
        return C_GRANULARITY % (length, image_name)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # changed to use cellprofiler_core.setting.SettingsGroup() but did not change the
            # ordering of any of the settings
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Changed to add objects and explicit image numbers
            image_count = int(len(setting_values) / IMAGE_SETTING_COUNT_V2)
            new_setting_values = [str(image_count)]
            for i in range(image_count):
                # Object setting count = 0
                new_setting_values += ["0"]
                new_setting_values += setting_values[:IMAGE_SETTING_COUNT_V2]
                setting_values = setting_values[IMAGE_SETTING_COUNT_V2:]
            setting_values = new_setting_values
            variable_revision_number = 3
        if variable_revision_number == 3:
            n_images = int(setting_values[0])
            grouplist = setting_values[1:]
            images_list = []
            objects_list = []
            setting_groups = []
            while grouplist:
                n_objects = int(grouplist[0])
                images_list += [grouplist[1]]
                setting_groups.append(tuple(grouplist[2:6]))
                if grouplist[6 : 6 + n_objects] != "None":
                    objects_list += grouplist[6 : 6 + n_objects]
                if len(grouplist) > 6 + n_objects:
                    grouplist = grouplist[6 + n_objects :]
                else:
                    grouplist = False
            images_set = set(images_list)
            objects_set = set(objects_list)
            settings_set = set(setting_groups)
            if "None" in images_set:
                images_set.remove("None")
            if len(settings_set) > 1:
                LOGGER.warning(
                    "The pipeline you loaded was converted from an older version of CellProfiler.\n"
                    "The MeasureGranularity module no longer supports different settings for each image.\n"
                    "Instead, all selected images and objects will be analysed together with the same settings.\n"
                    "If you want to perform analysis with additional settings, please use a second "
                    "copy of the module."
                )
            if len(objects_set) > len(objects_list):
                LOGGER.warning(
                    "The pipeline you loaded was converted from an older version of CellProfiler.\n"
                    "The MeasureGranularity module now analyses all images and object sets together.\n"
                    "Specific pairs of images and objects are no longer supported.\n"
                    "If you want to restrict analysis to specific image/object sets, please use a second "
                    "copy of the module."
                )
            if len(objects_set) > 0:
                wants_objects = True
            else:
                wants_objects = False
            images_string = ", ".join(map(str, images_set))
            objects_string = ", ".join(map(str, objects_set))
            setting_values = [images_string, wants_objects, objects_string] + list(
                setting_groups[0]
            )
            variable_revision_number = 4
        return setting_values, variable_revision_number

    def volumetric(self):
        return True
