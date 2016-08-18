"""

Watershed

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import numpy
import skimage.exposure
import skimage.filters
import skimage.measure
import skimage.morphology


class Watershed(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "Watershed"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.object_name = cellprofiler.setting.ObjectNameProvider(
            "Object name",
            ""
        )

        self.markers = cellprofiler.setting.ImageNameSubscriber(
            "Markers"
        )

        self.connectivity = cellprofiler.setting.ImageNameSubscriber(
            "Connectivity"
        )

        self.mask = cellprofiler.setting.ImageNameSubscriber(
            "Mask"
        )


    def settings(self):
        return [
            self.x_name,
            self.object_name,
            self.markers,
            self.connectivity,
            self.mask
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.object_name,
            self.markers,
            self.connectivity,
            self.mask
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        object_name = self.object_name.value

        markers_name = self.markers.value

        connectivity_name = self.connectivity.value

        mask_name = self.mask.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        markers = images.get_image(markers_name)

        markers_data = markers.pixel_data

        connectivity = images.get_image(connectivity_name)

        connectivity_data = connectivity.pixel_data

        mask = images.get_image(mask_name)

        mask_data = mask.pixel_data

        y_data = numpy.zeros_like(x_data)

        segmentation = skimage.morphology.watershed(
            image=x_data,
            markers=markers_data
        )

        labels = skimage.measure.label(segmentation)

        objects = cellprofiler.object.Objects()

        objects.segmented = labels

        workspace.object_set.add_objects(objects, object_name)

        if self.show_window:
            workspace.display_data.x_data = x_data

    def display(self, workspace, figure):
        dimensions = (1, 1)

        x_data = workspace.display_data.x_data[16]

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            x_data,
            colormap="gray"
        )
