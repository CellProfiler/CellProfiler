"""

Open

"""

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting
import loadimages
import os
import skimage.io
import numpy


class Open(cellprofiler.module.Module):
    category = "File Processing"

    module_name = "Open"

    variable_revision_number = 1

    def create_settings(self):
        self.name = cellprofiler.setting.Text(
            "Name",
            ""
        )

        self.pathname = cellprofiler.setting.Pathname(
            "Pathname",
            cellprofiler.setting.NONE
        )

        self.spacing = cellprofiler.setting.Float(
            "Spacing",
            1.0
        )

        self.three_dimensional = cellprofiler.setting.Binary(
            "Three-dimensional",
            False
        )

        self.time_varying = cellprofiler.setting.Binary(
            "Time varying",
            False
        )

    def run(self, workspace):
        provider = ThreeDimensionalImageProvider(self.name.value, self.pathname.value, self.pathname.value)

        workspace.image_set.providers.append(provider)

        name = provider.get_name()

        image = provider.provide_image(workspace.measurements)

        checksum = provider.get_md5_hash(workspace.measurements)

        workspace.measurements["Image", "MD5Digest_" + name] = checksum

        x, y = image.pixel_data.shape[1], image.pixel_data.shape[0]

        workspace.measurements["Image", "Width_" + name] = x
        workspace.measurements["Image", "Height_" + name] = y

        scaling = provider.scale

        workspace.measurements["Image", "Scaling_" + name] = scaling

        x_data = image.pixel_data

        dimensions = 3

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        layout = (1, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions,
            subplots=layout
        )

        figure.subplot_imshow(
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.x_data,
            x=0,
            y=0
        )

    def settings(self):
        return [
            self.name,
            self.pathname,
            self.spacing,
            self.three_dimensional,
            self.time_varying
        ]

    def visible_settings(self):
        __settings__ = [
            self.pathname,
            self.name
        ]

        if self.three_dimensional.value is True:
            __settings__ = __settings__ + [
                self.spacing
            ]

        __settings__ = __settings__ + [
            self.three_dimensional
        ]

        if self.three_dimensional.value is True:
            __settings__ = __settings__ + [
                self.spacing
            ]

        __settings__ = __settings__ + [
            self.time_varying
        ]

        return __settings__


class ThreeDimensionalImageProvider(loadimages.LoadImagesImageProviderBase):
    def __init__(self, name, pathname, filename):
        super(ThreeDimensionalImageProvider, self).__init__(name, pathname, filename)

        self.scale = 1

    def provide_image(self, image_set):
        pathname = self.get_pathname()

        data = skimage.io.imread(
            fname=pathname
        )

        image = cellprofiler.image.Image(
            dimensions=3,
            file_name=pathname,
            image=data,
            path_name=pathname
        )

        return image
