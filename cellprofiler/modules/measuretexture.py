# coding=utf-8

"""
**Measure Texture** measures the degree and nature of textures within
objects (versus smoothness).

This module measures the variations in grayscale images. An object (or
entire image) without much texture has a smooth appearance; an object or
image with a lot of texture will appear rough and show a wide variety of
pixel intensities.

This module can also measure textures of objects against grayscale
images. Any input objects specified will have their texture measured
against *all* input images specified, which may lead to image-object
texture combinations that are unneccesary. If you do not want this
behavior, use multiple **MeasureTexture** modules to specify the
particular image-object measures that you want.

Available measurements
^^^^^^^^^^^^^^^^^^^^^^

-  *Haralick Features:* Haralick texture features are derived from the
   co-occurrence matrix, which contains information about how image
   intensities in pixels with a certain position in relation to each
   other occur together. **MeasureTexture** can measure textures at
   different scales; the scale you choose determines how the
   co-occurrence matrix is constructed. For example, if you choose a
   scale of 2, each pixel in the image (excluding some border pixels)
   will be compared against the one that is two pixels to the right.
   **MeasureTexture** quantizes the image into eight intensity levels.
   There are then 8x8 possible ways to categorize a pixel with its
   scale-neighbor. **MeasureTexture** forms the 8x8 co-occurrence matrix
   by counting how many pixels and neighbors have each of the 8x8
   intensity combinations.

   Thirteen measurements are then calculated for the image by performing
   mathematical operations on the co-occurrence matrix (the formulas can
   be found `here`_):

   -  *AngularSecondMoment:* Measure of image homogeneity. A higher
      value of this feature indicates that the intensity varies less in
      an image. Has a value of 1 for a uniform image.
   -  *Contrast:* Measure of local variation in an image. A high
      contrast value indicates a high degree of local variation, and is
      0 for a uniform image.
   -  *Correlation:* Measure of linear dependency of intensity values in
      an image. For an image with large areas of similar intensities,
      correlation is much higher than for an image with noisier,
      uncorrelated intensities. Has a value of 1 or -1 for a perfectly
      positively or negatively correlated image.
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
      co-occurance matrix.
   -  *DifferenceEntropy:* Another indication of the amount of
      randomness in an image.
   -  *InfoMeas1*
   -  *InfoMeas2*

   Each measurement is suffixed with the direction of the offset used
   between pixels in the co-occurrence matrix:

   -  *0:* Horizontal
   -  *90:* Vertical
   -  *45:* Diagonal
   -  *135:* Anti-diagonal

.. _here: http://murphylab.web.cmu.edu/publications/boland/boland_node26.html

Technical notes
^^^^^^^^^^^^^^^

To calculate the Haralick features, **MeasureTexture** normalizes the
co-occurence matrix at the per-object level by basing the intensity
levels of the matrix on the maximum and minimum intensity observed
within each object. This is beneficial for images in which the maximum
intensities of the objects vary substantially because each object will
have the full complement of levels.

References
^^^^^^^^^^

-  Haralick RM, Shanmugam K, Dinstein I. (1973), “Textural Features for
   Image Classification” *IEEE Transaction on Systems Man, Cybernetics*,
   SMC-3(6):610-621. `(link)`_

.. _(link): http://dx.doi.org/10.1109/TSMC.1973.4309314
"""

import centrosome.cpmorphology
import centrosome.filter
import centrosome.haralick
import numpy
import scipy.ndimage

import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting

TEXTURE = "Texture"

OG_NAME = "name"

OG_REMOVE = "remove"

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()

H_HORIZONTAL = "Horizontal"
A_HORIZONTAL = "0"
H_VERTICAL = "Vertical"
A_VERTICAL = "90"
H_DIAGONAL = "Diagonal"
A_DIAGONAL = "45"
H_ANTIDIAGONAL = "Anti-diagonal"
A_ANTIDIAGONAL = "135"
H_ALL = [
    H_HORIZONTAL,
    H_VERTICAL,
    H_DIAGONAL,
    H_ANTIDIAGONAL
]

H_TO_A = {
    H_HORIZONTAL: A_HORIZONTAL,
    H_VERTICAL: A_VERTICAL,
    H_DIAGONAL: A_DIAGONAL,
    H_ANTIDIAGONAL: A_ANTIDIAGONAL
}

IO_IMAGES = "Images"
IO_OBJECTS = "Objects"
IO_BOTH = "Both"


class MeasureTexture(cellprofiler.module.Module):
    module_name = "MeasureTexture"

    variable_revision_number = 5

    category = "Measurement"

    def create_settings(self):
        self.image_groups = []

        self.object_groups = []

        self.scale_groups = []

        self.image_count = cellprofiler.setting.HiddenCount(self.image_groups)

        self.object_count = cellprofiler.setting.HiddenCount(self.object_groups)

        self.scale_count = cellprofiler.setting.HiddenCount(self.scale_groups)

        self.add_image(removable=False)

        self.add_images = cellprofiler.setting.DoSomething(
            callback=self.add_image,
            label="Add another image",
            text=""
        )

        self.image_divider = cellprofiler.setting.Divider()

        self.add_object(removable=True)

        self.add_objects = cellprofiler.setting.DoSomething(
            callback=self.add_object,
            label="Add another object",
            text=""
        )

        self.object_divider = cellprofiler.setting.Divider()

        self.add_scale(removable=False)

        self.add_scales = cellprofiler.setting.DoSomething(
            callback=self.add_scale,
            label="Add another scale",
            text=""
        )

        self.scale_divider = cellprofiler.setting.Divider()

        self.images_or_objects = cellprofiler.setting.Choice(
            "Measure images or objects?",
            [
                IO_IMAGES,
                IO_OBJECTS,
                IO_BOTH
            ],
            value=IO_BOTH,
            doc="""
            This setting determines whether the module computes image-wide measurements, per-object
            measurements or both.
            <ul>
                <li><i>{IO_IMAGES}:</i> Select if you only want to measure the texture of objects.</li>
                <li><i>{IO_OBJECTS}:</i> Select if your pipeline does not contain objects or if you only want
                to make per-image measurements.</li>
                <li><i>{IO_BOTH}:</i> Select to make both image and object measurements.</li>
            </ul>
            """.format(**{
                "IO_IMAGES": IO_IMAGES,
                "IO_OBJECTS": IO_OBJECTS,
                "IO_BOTH": IO_BOTH
            })
        )

    def settings(self):
        settings = [
            self.image_count,
            self.object_count,
            self.scale_count
        ]

        groups = [
            self.image_groups,
            self.object_groups,
            self.scale_groups
        ]

        elements = [
            ["image_name"],
            ["object_name"],
            ["scale", "angles"]
        ]

        for groups, elements in zip(groups, elements):
            for group in groups:
                for element in elements:
                    settings += [getattr(group, element)]

        settings += [
            self.images_or_objects
        ]

        return settings

    def prepare_settings(self, setting_values):
        counts_and_sequences = [
            (int(setting_values[0]), self.image_groups, self.add_image),
            (int(setting_values[1]), self.object_groups, self.add_object),
            (int(setting_values[2]), self.scale_groups, self.add_scale)
        ]

        for count, sequence, fn in counts_and_sequences:
            del sequence[count:]

            while len(sequence) < count:
                fn()

    def visible_settings(self):
        visible_settings = []

        if self.wants_object_measurements():
            vs_groups = [
                (self.image_groups, self.add_images, self.image_divider),
                (self.object_groups, self.add_objects, self.object_divider),
                (self.scale_groups, self.add_scales, self.scale_divider)
            ]
        else:
            vs_groups = [
                (self.image_groups, self.add_images, self.image_divider),
                (self.scale_groups, self.add_scales, self.scale_divider)
            ]

        for groups, add_button, div in vs_groups:
            for group in groups:
                visible_settings += group.visible_settings()

            visible_settings += [
                add_button,
                div
            ]

            if groups == self.image_groups:
                visible_settings += [self.images_or_objects]

        return visible_settings

    def wants_image_measurements(self):
        return self.images_or_objects in (IO_IMAGES, IO_BOTH)

    def wants_object_measurements(self):
        return self.images_or_objects in (IO_OBJECTS, IO_BOTH)

    def add_image(self, removable=True):
        """

        Add an image to the image_groups collection

        :param removable: set this to False to keep from showing the "remove" button for images that must be present.

        """
        group = cellprofiler.setting.SettingsGroup()

        if removable:
            divider = cellprofiler.setting.Divider(
                line=False
            )

            group.append("divider", divider)

        image = cellprofiler.setting.ImageNameSubscriber(
            doc="Select the grayscale images whose texture you want to measure.",
            text="Select an image to measure",
            value=cellprofiler.setting.NONE
        )

        group.append('image_name', image)

        if removable:
            remove_setting = cellprofiler.setting.RemoveSettingButton(
                entry=group,
                label="Remove this image",
                list=self.image_groups,
                text=""
            )

            group.append("remover", remove_setting)

        self.image_groups.append(group)

    def add_object(self, removable=True):
        """

        Add an object to the object_groups collection

        :param removable: set this to False to keep from showing the "remove" button for objects that must be present.

        """
        group = cellprofiler.setting.SettingsGroup()

        if removable:
            divider = cellprofiler.setting.Divider(line=False)

            group.append("divider", divider)

        object_subscriber = cellprofiler.setting.ObjectNameSubscriber(
            doc="""
            <p>Select the objects whose texture you want to measure. If you only want to measure the texture for
            the image overall, you can remove all objects using the "Remove this object" button.</p>

            <p>Objects specified here will have their texture measured against <i>all</i> images specified
            above, which may lead to image-object combinations that are unneccesary. If you do not want this
            behavior, use multiple <b>MeasureTexture</b> modules to specify the particular image-object
            measures that you want.</p>
            """,
            text="Select objects to measure",
            value=cellprofiler.setting.NONE
        )

        group.append("object_name", object_subscriber)

        if removable:
            remove_setting = cellprofiler.setting.RemoveSettingButton(
                entry=group,
                label="Remove this object",
                list=self.object_groups,
                text=""
            )

            group.append("remover", remove_setting)

        self.object_groups.append(group)

    def add_scale(self, removable=True):
        """

        Add a scale to the scale_groups collection

        :param removable: set this to False to keep from showing the "remove" button for scales that must be present.

        """
        group = cellprofiler.setting.SettingsGroup()

        if removable:
            group.append("divider", cellprofiler.setting.Divider(line=False))

        scale = cellprofiler.setting.Integer(
            doc="""
            <p>You can specify the scale of texture to be measured, in pixel units; the texture scale is the
            distance between correlated intensities in the image. A higher number for the scale of texture
            measures larger patterns of texture whereas smaller numbers measure more localized patterns of
            texture. It is best to measure texture on a scale smaller than your objects' sizes, so be sure that
            the value entered for scale of texture is smaller than most of your objects. For very small objects
            (smaller than the scale of texture you are measuring), the texture cannot be measured and will
            result in a undefined value in the output file.</p>
            """,
            text="Texture scale to measure",
            value=len(self.scale_groups) + 3
        )

        group.append("scale", scale)

        angles = cellprofiler.setting.MultiChoice(
            choices=H_ALL,
            doc="""
            <p>The Haralick texture measurements are based on the correlation between pixels offset by the scale
            in one of four directions:</p>

            <ul>
                <li><i>{H_HORIZONTAL}</i> - the correlated pixel is "scale" pixels to the right of the pixel
                of interest.</li>
                <li><i>{H_VERTICAL}</i> - the correlated pixel is "scale" pixels below the pixel of
                interest.</li>
                <li><i>{H_DIAGONAL}</i> - the correlated pixel is "scale" pixels to the right and "scale"
                pixels below the pixel of interest.</li>
                <li><i>{H_ANTIDIAGONAL}</i> - the correlated pixel is "scale" pixels to the left and "scale"
                pixels below the pixel of interest.</li>
            </ul>

            <p>Choose one or more directions to measure.</p>
            """.format(**{
                "H_ANTIDIAGONAL": H_ANTIDIAGONAL,
                "H_DIAGONAL": H_DIAGONAL,
                "H_HORIZONTAL": H_HORIZONTAL,
                "H_VERTICAL": H_VERTICAL
            }),
            text="Angles to measure",
            value=H_ALL
        )

        group.append("angles", angles)

        if removable:
            remove_setting = cellprofiler.setting.RemoveSettingButton(
                entry=group,
                label="Remove this scale",
                list=self.scale_groups,
                text=""
            )

            group.append("remover", remove_setting)

        self.scale_groups.append(group)

    def validate_module(self, pipeline):
        images = set()

        for group in self.image_groups:
            if group.image_name.value in images:
                raise cellprofiler.setting.ValidationError(
                    u"{} has already been selected".format(group.image_name.value),
                    group.image_name
                )

            images.add(group.image_name.value)

        if self.wants_object_measurements():
            objects = set()

            for group in self.object_groups:
                if group.object_name.value in objects:
                    raise cellprofiler.setting.ValidationError(
                        u"{} has already been selected".format(group.object_name.value),
                        group.object_name
                    )

                objects.add(group.object_name.value)

        scales = set()

        for group in self.scale_groups:
            if group.scale.value in scales:
                raise cellprofiler.setting.ValidationError(
                    u"{} has already been selected".format(group.scale.value),
                    group.scale
                )

            scales.add(group.scale.value)

    def get_categories(self, pipeline, object_name):
        object_name_exists = any([object_name == object_group.object_name for object_group in self.object_groups])

        if self.wants_object_measurements() and object_name_exists:
            return [TEXTURE]

        if self.wants_image_measurements() and object_name == cellprofiler.measurement.IMAGE:
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
            return [x.image_name.value for x in self.image_groups]

        return []

    def get_measurement_scales(self, pipeline, object_name, category, measurement, image_name):
        def format_measurement(scale_group):
            return [
                "{:d}_{}".format(
                    scale_group.scale.value,
                    H_TO_A[angle]
                ) for angle in scale_group.angles.get_selections()
            ]

        if len(self.get_measurement_images(pipeline, object_name, category, measurement)) > 0:
            return sum([format_measurement(scale_group) for scale_group in self.scale_groups], [])

        return []

    # TODO: fix nested loops
    def get_measurement_columns(self, pipeline):
        columns = []

        if self.wants_image_measurements():
            for feature in self.get_features():
                for image_group in self.image_groups:
                    for scale_group in self.scale_groups:
                        for angle in scale_group.angles.get_selections():
                            columns += [
                                (
                                    cellprofiler.measurement.IMAGE,
                                    "{}_{}_{}_{:d}_{}".format(
                                        TEXTURE,
                                        feature,
                                        image_group.image_name.value,
                                        scale_group.scale.value,
                                        H_TO_A[angle]
                                    ),
                                    cellprofiler.measurement.COLTYPE_FLOAT
                                )
                            ]

        if self.wants_object_measurements():
            for object_group in self.object_groups:
                for feature in self.get_features():
                    for image_group in self.image_groups:
                        for scale_group in self.scale_groups:
                            for angle in scale_group.angles.get_selections():
                                columns += [
                                    (
                                        object_group.object_name.value,
                                        "{}_{}_{}_{:d}_{}".format(
                                            TEXTURE,
                                            feature,
                                            image_group.image_name.value,
                                            scale_group.scale.value,
                                            H_TO_A[angle]
                                        ),
                                        cellprofiler.measurement.COLTYPE_FLOAT
                                    )
                                ]

        return columns

    def run(self, workspace):
        workspace.display_data.col_labels = [
            "Image",
            "Object",
            "Measurement",
            "Scale",
            "Value"
        ]

        statistics = []

        for image_group in self.image_groups:
            image_name = image_group.image_name.value

            for scale_group in self.scale_groups:
                scale = scale_group.scale.value

                if self.wants_image_measurements():
                    for angle in scale_group.angles.get_selections():
                        statistics += self.run_image(image_name, scale, angle, workspace)

                if self.wants_object_measurements():
                    for object_group in self.object_groups:
                        object_name = object_group.object_name.value

                        for angle in scale_group.angles.get_selections():
                            statistics += self.run_one(image_name, object_name, scale, angle, workspace)

        if self.show_window:
            workspace.display_data.statistics = statistics

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        figure.subplot_table(0, 0, workspace.display_data.statistics, col_labels=workspace.display_data.col_labels)

    def run_one(self, image_name, object_name, scale, angle, workspace):
        statistics = []

        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)

        objects = workspace.get_objects(object_name)

        pixel_data = image.pixel_data

        if image.has_mask:
            mask = image.mask
        else:
            mask = None

        labels = objects.segmented

        try:
            pixel_data = objects.crop_image_similarly(pixel_data)
        except ValueError:
            pixel_data, m1 = cellprofiler.object.size_similarly(labels, pixel_data)

            if numpy.any(~m1):
                if mask is None:
                    mask = m1
                else:
                    mask, m2 = cellprofiler.object.size_similarly(labels, mask)

                    mask[~m2] = False

        if numpy.all(labels == 0):
            for name in F_HARALICK:
                statistics += self.record_measurement(
                    workspace,
                    image_name,
                    object_name,
                    str(scale) + "_" + H_TO_A[angle], name, numpy.zeros((0,))
                )
        else:
            scale_i, scale_j = self.get_angle_ij(angle, scale)

            for name, value in zip(F_HARALICK, centrosome.haralick.Haralick(pixel_data, labels, scale_i, scale_j, mask=mask).all()):
                statistics += self.record_measurement(
                    workspace,
                    image_name,
                    object_name,
                    str(scale) + "_" + H_TO_A[angle],
                    name,
                    value
                )

        return statistics

    def get_angle_ij(self, angle, scale):
        if angle == H_VERTICAL:
            return scale, 0

        if angle == H_HORIZONTAL:
            return 0, scale

        if angle == H_DIAGONAL:
            return scale, scale

        if angle == H_ANTIDIAGONAL:
            return scale, -scale

    def run_image(self, image_name, scale, angle, workspace):
        statistics = []

        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)

        pixel_data = image.pixel_data

        image_labels = numpy.ones(pixel_data.shape, int)

        if image.has_mask:
            image_labels[~ image.mask] = 0

        scale_i, scale_j = self.get_angle_ij(angle, scale)

        names_and_values = zip(F_HARALICK, centrosome.haralick.Haralick(pixel_data, image_labels, scale_i, scale_j).all())

        for name, value in names_and_values:
            object_name = str(scale) + "_" + H_TO_A[angle]

            statistics += self.record_image_measurement(workspace, image_name, object_name, name, value)

        return statistics

    def record_measurement(self, workspace, image, obj, scale, feature, result):
        data = centrosome.cpmorphology.fixup_scipy_ndimage_result(result)

        data[~numpy.isfinite(data)] = 0

        workspace.add_measurement(
            obj,
            "{}_{}_{}_{}".format(TEXTURE, feature, image, str(scale)),
            data
        )

        # TODO: get outta crazee towne
        functions = [
            ("min", numpy.min),
            ("max", numpy.max),
            ("mean", numpy.mean),
            ("median", numpy.median),
            ("std dev", numpy.std)
        ]

        # TODO: poop emoji
        statistics = [
            [
                image,
                obj,
                "{} {}".format(aggregate, feature), scale, "{:.2}".format(fn(data)) if len(data) else "-"
            ] for aggregate, fn in functions
        ]

        return statistics

    def record_image_measurement(self, workspace, image_name, scale, feature_name, result):
        # TODO: this is very concerning
        if not numpy.isfinite(result):
            result = 0

        feature = "{}_{}_{}_{}".format(TEXTURE, feature_name, image_name, str(scale))

        workspace.measurements.add_image_measurement(feature, result)

        statistics = [image_name, "-", feature_name, scale, "{:.2}".format(float(result))]

        return [statistics]

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if variable_revision_number == 1:
            #
            # Added "wants_gabor"
            #
            setting_values = setting_values[:-1] + [cellprofiler.setting.YES] + setting_values[-1:]

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

            for scale in setting_values[scale_offset:scale_offset + scale_count]:
                new_setting_values += [
                    scale,
                    H_HORIZONTAL
                ]

            new_setting_values += setting_values[scale_offset + scale_count:]

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
            # Removed "wants_gabor", and "gabor_angles"
            #
            setting_values = setting_values[:-3] + setting_values[-1:]

            variable_revision_number = 5

        return setting_values, variable_revision_number, False
