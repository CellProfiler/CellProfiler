"""

<strong>Watershed</strong>

Watershed is a segmentation algorithm. It is used to separate different objects in an image.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import skimage.measure
import skimage.morphology


class Watershed(cellprofiler.module.ImageSegmentation):
    module_name = "Watershed"

    variable_revision_number = 1

    def create_settings(self):
        super(Watershed, self).create_settings()

        self.markers_name = cellprofiler.setting.ImageNameSubscriber(
            "Markers",
            doc="An image marking the approximate centers of the objects for segmentation."
        )

        self.mask_name = cellprofiler.setting.ImageNameSubscriber(
            "Mask",
            can_be_blank=True,
            doc="Optional. Only regions not blocked by the mask will be segmented."
        )

    def settings(self):
        __settings__ = super(Watershed, self).settings()

        return __settings__ + [
            self.markers_name,
            self.mask_name
        ]

    def visible_settings(self):
        __settings__ = super(Watershed, self).settings()

        return __settings__ + [
            self.markers_name,
            self.mask_name
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        markers_name = self.markers_name.value

        markers = images.get_image(markers_name)

        markers_data = markers.pixel_data

        mask_data = None

        if not self.mask_name.is_blank:
            mask_name = self.mask_name.value

            mask = images.get_image(mask_name)

            mask_data = mask.pixel_data

        y_data = skimage.morphology.watershed(
            image=x_data,
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
