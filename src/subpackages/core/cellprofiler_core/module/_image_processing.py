from ._module import Module
from ..image import Image
from ..setting.subscriber import ImageSubscriber
from ..setting.text import ImageName


class ImageProcessing(Module):
    category = "Image Processing"

    def create_settings(self):
        self.x_name = ImageSubscriber(
            "Select the input image", doc="Select the image you want to use."
        )

        self.y_name = ImageName(
            "Name the output image",
            self.__class__.__name__,
            doc="Enter the name you want to call the image produced by this module.",
        )

    def display(self, workspace, figure, cmap=None):
        if cmap is None:
            cmap = ["gray", "gray"]
        layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap=cmap[0],
            image=workspace.display_data.x_data,
            title=self.x_name.value,
            x=0,
            y=0,
        )

        figure.subplot_imshow(
            colormap=cmap[1],
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        args = (setting.value for setting in self.settings()[2:])

        y_data = self.function(x_data, *args)

        y = Image(dimensions=dimensions, image=y_data, parent_image=x, convert=False)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def settings(self):
        return [self.x_name, self.y_name]

    def visible_settings(self):
        return [self.x_name, self.y_name]

    def volumetric(self):
        return True
