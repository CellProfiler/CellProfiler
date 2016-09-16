"""

Graph partition

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import skimage.measure
import skimage.segmentation


class GraphPartition(cellprofiler.module.Module):
    category = "Image segmentation"
    module_name = "Graph partition"
    variable_revision_number = 1

    def create_settings(self):
        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Image"
        )

        self.object_name = cellprofiler.setting.ObjectNameProvider(
            "Object",
            ""
        )

        self.markers_name = cellprofiler.setting.ImageNameSubscriber(
            "Markers"
        )

    def settings(self):
        return [
            self.image_name,
            self.object_name,
            self.markers_name
        ]

    def visible_settings(self):
        return [
            self.image_name,
            self.object_name,
            self.markers_name
        ]

    def run(self, workspace):
        image_name = self.image_name.value

        object_name = self.object_name.value

        images = workspace.image_set

        image = images.get_image(image_name)

        dimensions = image.dimensions

        image_data = image.pixel_data

        markers_name = self.markers_name.value

        markers = images.get_image(markers_name)

        markers_data = markers.pixel_data

        label_data = skimage.segmentation.random_walker(
            data=image_data,
            labels=markers_data,
            beta=130,
            mode="bf",
            tol=0.001,
            copy=True,
            multichannel=False,
            return_full_prob=False,
            spacing=None
        )

        label_data = skimage.measure.label(label_data)

        objects = cellprofiler.object.Objects()

        objects.segmented = label_data

        workspace.object_set.add_objects(objects, object_name)

        if self.show_window:
            workspace.display_data.image_data = image_data

            workspace.display_data.label_data = label_data

            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        figure.set_subplots((1, 2))

        figure.subplot_imshow(0, 0, workspace.display_data.image_data, dimensions=workspace.display_data.dimensions)

        figure.subplot_imshow(0, 1, workspace.display_data.label_data, dimensions=workspace.display_data.dimensions)
