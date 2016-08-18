"""

Watershed

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import numpy
import scipy.ndimage
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology


class Watershed(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "Watershed"
    variable_revision_number = 1

    def create_settings(self):
        self.image = cellprofiler.setting.ImageNameSubscriber(
            "Image"
        )

        self.object = cellprofiler.setting.ObjectNameProvider(
            "Object",
            ""
        )

        self.markers = cellprofiler.setting.ImageNameSubscriber(
            "Markers"
        )

        self.markers = cellprofiler.setting.ImageNameSubscriber(
            "Markers"
        )

        self.mask = cellprofiler.setting.ImageNameSubscriber(
            "Mask"
        )

    def settings(self):
        return [
            self.image,
            self.object,
            self.markers,
            self.mask
        ]

    def visible_settings(self):
        return [
            self.image,
            self.object,
            self.markers,
            self.mask
        ]

    def run(self, workspace):
        images = workspace.image_set

        image = images.get_image(self.image.value).pixel_data

        markers = images.get_image(self.markers.value).pixel_data

        mask = images.get_image(self.mask.value).pixel_data

        labels = skimage.morphology.watershed(image, markers, mask=mask)

        labels = skimage.measure.label(labels)

        objects = cellprofiler.object.Objects()

        objects.segmented = labels

        workspace.object_set.add_objects(objects, self.object.value)

        if self.show_window:
            workspace.display_data.image = image

            workspace.display_data.labels = labels

    def display(self, workspace, figure):
        dimensions = (2, 1)

        image = workspace.display_data.image[16]

        labels = workspace.display_data.labels[16]

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            image,
            colormap="gray"
        )

        figure.subplot_imshow_labels(
            1,
            0,
            labels
        )
