import cellprofiler.gui.help.content
import cellprofiler.icons

__doc__ = """\
MeasureTexture
==============

**MeasureTexture** measures the degree and nature of textures within
images and objects to quantify their roughness and smoothness.

This module measures intensity variations in grayscale images. An object or
entire image without much texture has a smooth appearance; an object or
image with a lot of texture will appear rough and show a wide variety of
pixel intensities.

Note that any input objects specified will have their texture measured
against *all* input images specified, which may lead to image-object
texture combinations that are unnecessary. If you do not want this
behavior, use multiple **MeasureTexture** modules to specify the
particular image-object measures that you want.

Note also that CellProfiler in all 2.X versions increased speed by binning 
the image into only 8 grayscale levels before calculating Haralick features;
in all 3.X CellProfiler versions the images were binned into 256 grayscale
levels. CellProfiler 4 allows you to select your own preferred number of
grayscale levels, but note that since we use a slightly different
implementation than CellProfiler 2 we do not guarantee concordance with
CellProfiler 2.X-generated texture values.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Haralick Features:* Haralick texture features are derived from the
   co-occurrence matrix, which contains information about how image
   intensities in pixels with a certain position in relation to each
   other occur together. **MeasureTexture** can measure textures at
   different scales; the scale you choose determines how the
   co-occurrence matrix is constructed. For example, if you choose a
   scale of 2, each pixel in the image (excluding some border pixels)
   will be compared against the one that is two pixels to the right.

   Thirteen measurements are then calculated for the image by performing
   mathematical operations on the co-occurrence matrix (the formulas can
   be found `here`_):

   -  *AngularSecondMoment:* Measure of image homogeneity. A higher
      value of this feature indicates that the intensity varies less in
      an image. Has a value of 1 for a uniform image.
   -  *Contrast:* Measure of local variation in an image, with 0 for a
      uniform image and a high value indicating a high degree of local
      variation.
   -  *Correlation:* Measure of linear dependency of intensity values in
      an image. For an image with large areas of similar intensities,
      correlation is much higher than for an image with noisier,
      uncorrelated intensities. Has a value of 1 or -1 for a perfectly
      positively or negatively correlated image, respectively.
   -  *Variance:* Measure of the variation of image intensity values.
      For an image with uniform intensity, the texture variance would be
      zero.
   -  *InverseDifferenceMoment:* Another feature to represent image
      contrast. Has a low value for inhomogeneous images, and a
      relatively higher value for homogeneous images.
   -  *SumAverage:* The average of the normalized grayscale image in the
      spatial domain.
   -  *SumVariance:* The variance of the normalized grayscale image in
      the spatial domain.
   -  *SumEntropy:* A measure of randomness within an image.
   -  *Entropy:* An indication of the complexity within an image. A
      complex image produces a high entropy value.
   -  *DifferenceVariance:* The image variation in a normalized
      co-occurrence matrix.
   -  *DifferenceEntropy:* Another indication of the amount of
      randomness in an image.
   -  *InfoMeas1:* A measure of the total amount of information contained
      within a region of pixels derived from the recurring spatial
      relationship between specific intensity values.
   -  *InfoMeas2:* An additional measure of the total amount of information
      contained within a region of pixels derived from the recurring spatial
      relationship between specific intensity values. It is a complementary
      value to InfoMeas1 and is on a different scale.

**Note**: each of the above measurements are computed for different 
'directions' in the image, specified by a series of correspondence vectors. 
These are indicated in the results table in the *scale* column as n_00, n_01,
n_02... for each scale *n*. In 2D, the directions and correspondence vectors *(y, x)* 
for each measurement are given below:

- _00 = horizontal -, 0 degrees   (0, 1)
- _01 = diagonal \\\\, 135 degrees or NW-SE   (1, 1)
- _02 = vertical \|, 90 degrees   (1, 0)
- _03 = diagonal /, 45 degrees or NE-SW  (1, -1)

When analyzing 3D images, there are 13 correspondence vectors *(y, x, z)*:

- (1, 0, 0)
- (1, 1, 0)
- (0, 1, 0)
- (1,-1, 0)
- (0, 0, 1)
- (1, 0, 1)
- (0, 1, 1)
- (1, 1, 1)
- (1,-1, 1)
- (1, 0,-1)
- (0, 1,-1)
- (1, 1,-1)
- (1,-1,-1)

In this case, an image makes understanding their directions easier. 
Imagine the origin (0, 0, 0) is at the upper left corner of the first image
in your z-stack. Yellow vectors fall along the axes, and pairs of vectors with 
matching colors are reflections of each other across the x axis. The two
images represent two views of the same vectors. Images made in `GeoGebra`_.

|MT_image0| |MT_image1|

Technical notes
^^^^^^^^^^^^^^^

To calculate the Haralick features, **MeasureTexture** normalizes the
co-occurrence matrix at the per-object level by basing the intensity
levels of the matrix on the maximum and minimum intensity observed
within each object. This is beneficial for images in which the maximum
intensities of the objects vary substantially because each object will
have the full complement of levels.

References
^^^^^^^^^^

-  Haralick RM, Shanmugam K, Dinstein I. (1973), “Textural Features for
   Image Classification” *IEEE Transaction on Systems Man, Cybernetics*,
   SMC-3(6):610-621. `(link) <https://doi.org/10.1109/TSMC.1973.4309314>`__

.. _here: http://murphylab.web.cmu.edu/publications/boland/boland_node26.html
.. _GeoGebra: https://www.geogebra.org/ 
.. |MT_image0| image:: {MEASURE_TEXTURE_3D_INFO}
.. |MT_image1| image:: {MEASURE_TEXTURE_3D_INFO2}
""".format(
    **{
        "MEASURE_TEXTURE_3D_INFO": cellprofiler.gui.help.content.image_resource(
            "Measure_texture_3D_correspondences_1.png"
        ),
        "MEASURE_TEXTURE_3D_INFO2": cellprofiler.gui.help.content.image_resource(
            "Measure_texture_3D_correspondences_2.png"
        )
    }
)

import mahotas.features
import numpy
import skimage.exposure
import skimage.measure
import skimage.util
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.setting import (
    HiddenCount,
    Divider,
    SettingsGroup,
    ValidationError,
)
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import (
    ImageListSubscriber,
    LabelListSubscriber,
)
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.utilities.core.object import size_similarly

TEXTURE = "Texture"

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()

IO_IMAGES = "Images"
IO_OBJECTS = "Objects"
IO_BOTH = "Both"


class MeasureTexture(Module):
    module_name = "MeasureTexture"

    variable_revision_number = 7

    category = "Measurement"

    def create_settings(self):
        self.images_list = ImageListSubscriber(
            "Select images to measure",
            [],
            doc="""Select the grayscale images whose intensity you want to measure.""",
        )

        self.objects_list = LabelListSubscriber(
            "Select objects to measure",
            [],
            doc="""\
        Select the objects whose texture you want to measure. If you only want
        to measure the texture for the image overall, you can remove all objects
        using the “Remove this object” button.

        Objects specified here will have their texture measured against *all*
        images specified above, which may lead to image-object combinations that
        are unnecessary. If you do not want this behavior, use multiple
        **MeasureTexture** modules to specify the particular image-object
        measures that you want.
        """,
        )

        self.gray_levels = Integer(
            "Enter how many gray levels to measure the texture at",
            256,
            2,
            256,
            doc="""\
        Enter the number of gray levels (ie, total possible values of intensity) 
        you want to measure texture at.  Measuring at more levels gives you 
        _potentially_ more detailed information about your image, but at the cost
        of somewhat decreased processing speed.  

        Before processing, your image will be rescaled from its current pixel values
        to 0 - [gray levels - 1]. The texture features will then be calculated. 

        In all CellProfiler 2 versions, this value was fixed at 8; in all 
        CellProfiler 3 versions it was fixed at 256.  The minimum number of levels is
        2, the maximum is 256.
        """,
        )

        self.scale_groups = []

        self.scale_count = HiddenCount(self.scale_groups)

        self.image_divider = Divider()

        self.object_divider = Divider()

        self.add_scale(removable=False)

        self.add_scales = DoSomething(
            callback=self.add_scale,
            label="Add another scale",
            text="",
            doc="""\
            Add an additional texture scale to measure. Useful when you
            want to measure texture features of different sizes.
            """,
        )

        self.images_or_objects = Choice(
            "Measure whole images or objects?",
            [IO_IMAGES, IO_OBJECTS, IO_BOTH],
            value=IO_BOTH,
            doc="""\
This setting determines whether the module computes image-wide
measurements, per-object measurements or both.

-  *{IO_IMAGES}:* Select if you only want to measure the texture
   across entire images.
-  *{IO_OBJECTS}:* Select if you want to measure the texture
   on a per-object basis only.
-  *{IO_BOTH}:* Select to make both image and object measurements.
""".format(
                **{"IO_IMAGES": IO_IMAGES, "IO_OBJECTS": IO_OBJECTS, "IO_BOTH": IO_BOTH}
            ),
        )

    def settings(self):
        settings = [
            self.images_list,
            self.objects_list,
            self.gray_levels,
            self.scale_count,
            self.images_or_objects,
        ]

        for group in self.scale_groups:
            settings += [getattr(group, "scale")]

        return settings

    def prepare_settings(self, setting_values):
        counts_and_sequences = [
            (int(setting_values[3]), self.scale_groups, self.add_scale),
        ]

        for count, sequence, fn in counts_and_sequences:
            del sequence[count:]
            while len(sequence) < count:
                fn()

    def visible_settings(self):
        visible_settings = [
            self.images_list,
            self.image_divider,
            self.images_or_objects,
        ]

        if self.wants_object_measurements():
            visible_settings += [self.objects_list]
        visible_settings += [self.object_divider]

        visible_settings += [self.gray_levels]

        for group in self.scale_groups:
            visible_settings += group.visible_settings()

        visible_settings += [self.add_scales]

        return visible_settings

    def wants_image_measurements(self):
        return self.images_or_objects in (IO_IMAGES, IO_BOTH)

    def wants_object_measurements(self):
        return self.images_or_objects in (IO_OBJECTS, IO_BOTH)

    def add_scale(self, removable=True):
        """

        Add a scale to the scale_groups collection

        :param removable: set this to False to keep from showing the "remove" button for scales that must be present.

        """
        group = SettingsGroup()

        if removable:
            group.append("divider", Divider(line=False))

        scale = Integer(
            doc="""\
You can specify the scale of texture to be measured, in pixel units; the
texture scale is the distance between correlated intensities in the
image. A higher number for the scale of texture measures larger patterns
of texture whereas smaller numbers measure more localized patterns of
texture. It is best to measure texture on a scale smaller than your
objects’ sizes, so be sure that the value entered for scale of texture
is smaller than most of your objects. For very small objects (smaller
than the scale of texture you are measuring), the texture cannot be
measured and will result in a undefined value in the output file.
""",
            text="Texture scale to measure",
            value=len(self.scale_groups) + 3,
        )

        group.append("scale", scale)

        if removable:
            remove_setting = RemoveSettingButton(
                entry=group, label="Remove this scale", list=self.scale_groups, text=""
            )

            group.append("remover", remove_setting)

        self.scale_groups.append(group)

    def validate_module(self, pipeline):
        images = set()
        if len(self.images_list.value) == 0:
            raise ValidationError("No images selected", self.images_list)
        for image_name in self.images_list.value:
            if image_name in images:
                raise ValidationError(
                    "%s has already been selected" % image_name, image_name
                )
            images.add(image_name)

        if self.wants_object_measurements():
            objects = set()
            if len(self.objects_list.value) == 0:
                raise ValidationError("No objects selected", self.objects_list)
            for object_name in self.objects_list.value:
                if object_name in objects:
                    raise ValidationError(
                        "%s has already been selected" % object_name, object_name
                    )
                objects.add(object_name)

        scales = set()
        for group in self.scale_groups:
            if group.scale.value in scales:
                raise ValidationError(
                    "{} has already been selected".format(group.scale.value),
                    group.scale,
                )

            scales.add(group.scale.value)

    def get_categories(self, pipeline, object_name):
        object_name_exists = object_name in self.objects_list.value

        if self.wants_object_measurements() and object_name_exists:
            return [TEXTURE]

        if self.wants_image_measurements() and object_name == "Image":
            return [TEXTURE]

        return []

    def get_features(self):
        return F_HARALICK

    def get_measurements(self, pipeline, object_name, category):
        if category in self.get_categories(pipeline, object_name):
            return self.get_features()

        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        measurements = self.get_measurements(pipeline, object_name, category)

        if measurement in measurements:
            return self.images_list.value

        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, measurement, image_name
    ):
        def format_measurement(scale_group):
            return [
                "{:d}_{:02d}_{:d}".format(scale_group.scale.value, angle,self.gray_levels.value)
                for angle in range(13 if pipeline.volumetric() else 4)
            ]

        if (
            len(
                self.get_measurement_images(
                    pipeline, object_name, category, measurement
                )
            )
            > 0
        ):
            return sum(
                [format_measurement(scale_group) for scale_group in self.scale_groups],
                [],
            )

        return []

    # TODO: fix nested loops
    def get_measurement_columns(self, pipeline):
        columns = []

        if self.wants_image_measurements():
            for feature in self.get_features():
                for image_name in self.images_list.value:
                    for scale_group in self.scale_groups:
                        for angle in range(13 if pipeline.volumetric() else 4):
                            columns += [
                                (
                                    "Image",
                                    "{}_{}_{}_{:d}_{:02d}_{:d}".format(
                                        TEXTURE,
                                        feature,
                                        image_name,
                                        scale_group.scale.value,
                                        angle,
                                        self.gray_levels.value,
                                    ),
                                    COLTYPE_FLOAT,
                                )
                            ]

        if self.wants_object_measurements():
            for object_name in self.objects_list.value:
                for feature in self.get_features():
                    for image_name in self.images_list.value:
                        for scale_group in self.scale_groups:
                            for angle in range(13 if pipeline.volumetric() else 4):
                                columns += [
                                    (
                                        object_name,
                                        "{}_{}_{}_{:d}_{:02d}_{:d}".format(
                                            TEXTURE,
                                            feature,
                                            image_name,
                                            scale_group.scale.value,
                                            angle,
                                            self.gray_levels.value,
                                        ),
                                        COLTYPE_FLOAT,
                                    )
                                ]

        return columns

    def run(self, workspace):
        workspace.display_data.col_labels = [
            "Image",
            "Object",
            "Measurement",
            "Scale",
            "Value",
        ]

        statistics = []

        for image_name in self.images_list.value:
            for scale_group in self.scale_groups:
                scale = scale_group.scale.value

                if self.wants_image_measurements():
                    statistics += self.run_image(image_name, scale, workspace)

                if self.wants_object_measurements():
                    for object_name in self.objects_list.value:
                        statistics += self.run_one(
                            image_name, object_name, scale, workspace
                        )

        if self.show_window:
            workspace.display_data.statistics = statistics

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        if self.wants_object_measurements():
            helptext = "default"
        else:
            helptext = None
        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
            title=helptext,
        )

    def run_one(self, image_name, object_name, scale, workspace):
        statistics = []

        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)

        objects = workspace.get_objects(object_name)
        labels = objects.segmented

        gray_levels = int(self.gray_levels.value)

        unique_labels = numpy.unique(labels)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        n_directions = 13 if objects.volumetric else 4

        if len(unique_labels) == 0:
            for direction in range(n_directions):
                for feature_name in F_HARALICK:
                    statistics += self.record_measurement(
                        image=image_name,
                        feature=feature_name,
                        obj=object_name,
                        result=numpy.zeros((0,)),
                        scale="{:d}_{:02d}".format(scale, direction),
                        workspace=workspace,
                        gray_levels="{:d}".format(gray_levels),
                    )

            return statistics

        # IMG-961: Ensure image and objects have the same shape.
        try:
            mask = (
                image.mask
                if image.has_mask
                else numpy.ones_like(image.pixel_data, dtype=bool)
            )
            pixel_data = objects.crop_image_similarly(image.pixel_data)
        except ValueError:
            pixel_data, m1 = size_similarly(labels, image.pixel_data)

            if numpy.any(~m1):
                if image.has_mask:
                    mask, m2 = size_similarly(labels, image.mask)
                    mask[~m2] = False
                else:
                    mask = m1

        pixel_data[~mask] = 0
        # mahotas.features.haralick bricks itself when provided a dtype larger than uint8 (version 1.4.3)
        pixel_data = skimage.util.img_as_ubyte(pixel_data)
        if gray_levels != 256:
            pixel_data = skimage.exposure.rescale_intensity(
                pixel_data, in_range=(0, 255), out_range=(0, gray_levels - 1)
            ).astype(numpy.uint8)
        props = skimage.measure.regionprops(labels, pixel_data)

        features = numpy.empty((n_directions, 13, len(unique_labels)))

        for index, prop in enumerate(props):
            label_data = prop["intensity_image"]
            try:
                features[:, :, index] = mahotas.features.haralick(
                    label_data, distance=scale, ignore_zeros=True
                )
            except ValueError:
                features[:, :, index] = numpy.nan

        for direction, direction_features in enumerate(features):
            for feature_name, feature in zip(F_HARALICK, direction_features):
                statistics += self.record_measurement(
                    image=image_name,
                    feature=feature_name,
                    obj=object_name,
                    result=feature,
                    scale="{:d}_{:02d}".format(scale, direction),
                    workspace=workspace,
                    gray_levels="{:d}".format(gray_levels),
                )

        return statistics

    def run_image(self, image_name, scale, workspace):
        statistics = []

        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)

        # mahotas.features.haralick bricks itself when provided a dtype larger than uint8 (version 1.4.3)
        gray_levels = int(self.gray_levels.value)
        pixel_data = skimage.util.img_as_ubyte(image.pixel_data)
        if gray_levels != 256:
            pixel_data = skimage.exposure.rescale_intensity(
                pixel_data, in_range=(0, 255), out_range=(0, gray_levels - 1)
            ).astype(numpy.uint8)

        features = mahotas.features.haralick(pixel_data, distance=scale)

        for direction, direction_features in enumerate(features):
            object_name = "{:d}_{:02d}".format(scale, direction)

            for feature_name, feature in zip(F_HARALICK, direction_features):
                statistics += self.record_image_measurement(
                    feature_name=feature_name,
                    image_name=image_name,
                    result=feature,
                    scale=object_name,
                    workspace=workspace,
                    gray_levels="{:d}".format(gray_levels),
                )

        return statistics

    def record_measurement(
        self, workspace, image, obj, scale, feature, result, gray_levels
    ):
        result[~numpy.isfinite(result)] = 0

        workspace.add_measurement(
            obj,
            "{}_{}_{}_{}_{}".format(TEXTURE, feature, image, str(scale), gray_levels),
            result,
        )

        # TODO: get outta crazee towne
        functions = [
            ("min", numpy.min),
            ("max", numpy.max),
            ("mean", numpy.mean),
            ("median", numpy.median),
            ("std dev", numpy.std),
        ]

        # TODO: poop emoji
        statistics = [
            [
                image,
                obj,
                "{} {}".format(aggregate, feature),
                scale,
                "{:.2}".format(fn(result)) if len(result) else "-",
            ]
            for aggregate, fn in functions
        ]

        return statistics

    def record_image_measurement(
        self, workspace, image_name, scale, feature_name, result, gray_levels
    ):
        # TODO: this is very concerning
        if not numpy.isfinite(result):
            result = 0

        feature = "{}_{}_{}_{}_{}".format(
            TEXTURE, feature_name, image_name, str(scale), gray_levels
        )

        workspace.measurements.add_image_measurement(feature, result)

        statistics = [
            image_name,
            "-",
            feature_name,
            scale,
            "{:.2}".format(float(result)),
        ]

        return [statistics]

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            #
            # Added "wants_gabor"
            #
            setting_values = setting_values[:-1] + ["Yes"] + setting_values[-1:]

            variable_revision_number = 2

        if variable_revision_number == 2:
            #
            # Added angles
            #
            image_count = int(setting_values[0])

            object_count = int(setting_values[1])

            scale_count = int(setting_values[2])

            scale_offset = 3 + image_count + object_count

            new_setting_values = setting_values[:scale_offset]

            for scale in setting_values[scale_offset : scale_offset + scale_count]:
                new_setting_values += [scale, "Horizontal"]

            new_setting_values += setting_values[scale_offset + scale_count :]

            setting_values = new_setting_values

            variable_revision_number = 3

        if variable_revision_number == 3:
            #
            # Added image / objects choice
            #
            setting_values = setting_values + [IO_BOTH]

            variable_revision_number = 4

        if variable_revision_number == 4:
            #
            #  Removed angles
            #
            image_count, object_count, scale_count = setting_values[:3]
            scale_offset = 3 + int(image_count) + int(object_count)
            scales = setting_values[scale_offset::2][: int(scale_count)]
            new_setting_values = setting_values[:scale_offset] + scales

            #
            # Removed "wants_gabor", and "gabor_angles"
            #
            new_setting_values += setting_values[-1:]

            setting_values = new_setting_values
            variable_revision_number = 5
        if variable_revision_number == 5:
            num_images = int(setting_values[0])
            num_objects = int(setting_values[1])
            num_scales = setting_values[2]
            div_img = 3 + num_images
            div_obj = div_img + num_objects
            images_set = set(setting_values[3:div_img])
            objects_set = set(setting_values[div_img:div_obj])
            scales_list = setting_values[div_obj:-1]

            if "None" in images_set:
                images_set.remove("None")
            if "None" in objects_set:
                objects_set.remove("None")
            images_string = ", ".join(map(str, images_set))
            objects_string = ", ".join(map(str, objects_set))

            module_mode = setting_values[-1]
            setting_values = [
                images_string,
                objects_string,
                num_scales,
                module_mode,
            ] + scales_list
            variable_revision_number = 6

        if variable_revision_number == 6:
            setting_values = setting_values[:2] + ["256"] + setting_values[2:]
            variable_revision_number = 7

        return setting_values, variable_revision_number

    def volumetric(self):
        return True
