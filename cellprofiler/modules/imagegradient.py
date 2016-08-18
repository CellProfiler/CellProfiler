"""

Image gradient

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.exposure
import skimage.filters
import skimage.morphology


class ImageGradient(cellprofiler.module.Module):
    category = "Image processing"
    module_name = "Image gradient"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "OutputImage"
        )

        self.structure = cellprofiler.setting.Choice(
            "Structuing element",
            [
                "Ball"
            ]
        )

        self.radius = cellprofiler.setting.Integer(
            "Radius",
            1
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structure,
            self.radius
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.structure,
            self.radius
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        radius = self.radius.value

        structure = self.structure.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        disk = skimage.morphology.disk(radius)

        y_data = numpy.zeros_like(x_data)

        for z, image in enumerate(x_data):
            y_data[z] = skimage.filters.rank.gradient(image, disk)

        y = cellprofiler.image.Image(
            dimensions=3,
            parent_image=x
        )

        y.pixel_data = y_data

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        dimensions = (2, 1)

        x_data = workspace.display_data.x_data[16]
        y_data = workspace.display_data.y_data[16]

        figure.set_subplots(dimensions)

        figure.imshow(
            0,
            0,
            x_data,
            "gray"
        )

        figure.imshow(
            1,
            0,
            y_data,
            "gray"
        )
