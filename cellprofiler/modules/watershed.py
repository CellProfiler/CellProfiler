"""

<strong>Watershed</strong>

Watershed is a segmentation algorithm. It is used to separate different objects in an image.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import scipy.ndimage
import skimage.feature
import skimage.measure
import skimage.morphology


class Watershed(cellprofiler.module.ImageSegmentation):
    module_name = "Watershed"

    variable_revision_number = 2

    def create_settings(self):
        super(Watershed, self).create_settings()

        self.operation = cellprofiler.setting.Choice(
            "Generate from",
            [
                "Distance",
                "Markers"
            ],
            "Distance",
            doc="""Select a method of inputs for the watershed algorithm:
            <ul>
                <li>
                    <i>Distance</i> (default): This is classical nuclei segmentation using watershed. Your "Input" image
                    should be a binary image. Markers and other inputs for the watershed algorithm will be
                    automatically generated.
                </li>
                <br>
                <li>
                    <i>Markers</i>: Use manually generated markers and supply an optional mask for watershed. Watershed
                    works best when the "Input" image has high intensity surrounding regions of interest and low intensity
                    inside regions of interest. Refer to the documentation for the other available options for more
                    information.
                </li>
            </ul>
            """
        )

        self.markers_name = cellprofiler.setting.ImageNameSubscriber(
            "Markers",
            doc="An image marking the approximate centers of the objects for segmentation."
        )

        self.mask_name = cellprofiler.setting.ImageNameSubscriber(
            "Mask",
            can_be_blank=True,
            doc="Optional. Only regions not blocked by the mask will be segmented."
        )

        self.size = cellprofiler.setting.Integer(
            "Minimum object diameter",
            value=1,
            minval=1,
            doc="Minimum size of objects (in pixels) in the image."
        )

    def settings(self):
        __settings__ = super(Watershed, self).settings()

        return __settings__ + [
            self.operation,
            self.markers_name,
            self.mask_name,
            self.size
        ]

    def visible_settings(self):
        __settings__ = super(Watershed, self).settings()

        __settings__ = __settings__ + [
            self.operation
        ]

        if self.operation.value == "Distance":
            __settings__ = __settings__ + [
                self.size
            ]
        else:
            __settings__ = __settings__ + [
                self.markers_name,
                self.mask_name
            ]

        return __settings__

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        if self.operation.value == "Distance":
            distance_data = scipy.ndimage.distance_transform_edt(x_data)

            # http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max says:
            #       Peaks are the local maxima in a region of 2 * min_distance + 1
            #       (i.e. peaks are separated by at least min_distance).
            #
            # We observe under-segmentation when passing min_distance=self.size.value. Better segmentation
            # occurs when self.size.value = 2 * min_distance + 1.
            min_distance = max(1, (self.size.value - 1) / 2)

            local_maximums = skimage.feature.peak_local_max(
                distance_data,
                indices=False,
                min_distance=min_distance,
                labels=x_data
            )

            data = -1 * distance_data

            markers_data = skimage.measure.label(local_maximums)

            mask_data = x_data
        else:
            markers_name = self.markers_name.value

            markers = images.get_image(markers_name)

            data = x_data

            markers_data = markers.pixel_data

            mask_data = None

            if not self.mask_name.is_blank:
                mask_name = self.mask_name.value

                mask = images.get_image(mask_name)

                mask_data = mask.pixel_data

        y_data = skimage.morphology.watershed(
            image=data,
            markers=markers_data,
            mask=mask_data
        )

        y_data = skimage.measure.label(y_data)

        objects = cellprofiler.object.Objects()

        objects.segmented = y_data

        workspace.object_set.add_objects(objects, y_name)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if variable_revision_number == 1:
            setting_values = setting_values[:6] + [1] + setting_values[6:]
            variable_revision_number = 2

        if variable_revision_number == 2:
            setting_values = setting_values[0] + setting_values[2:]
            variable_revision_number = 3

        return super(Watershed, self).upgrade_settings(setting_values, variable_revision_number, module_name, from_matlab)
