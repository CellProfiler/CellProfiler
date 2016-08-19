# coding=utf-8
"""

Corner detection

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.feature


class CornerDetection(cellprofiler.module.Module):
    category = "Feature detection"
    module_name = "Corner detection"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            u"Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            u"Output"
        )

        self.minimum_distance = cellprofiler.setting.Float(
            u"Minimum distance",
            1
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.minimum_distance
        ]

    def visible_settings(self):
        settings = [
            self.x_name,
            self.y_name,
            self.minimum_distance
        ]

        return settings

    def run(self, workspace):
        x_name = self.x_name.value
        x_data = workspace.image_set.get_image(x_name).pixel_data

        minimum_distance = self.minimum_distance.value

        coordinates = [skimage.feature.corner_peaks(skimage.feature.corner_harris(data), min_distance=minimum_distance) for data in x_data]

        corners = [skimage.feature.corner_subpix(data, point, window_size=25) for data, point in zip(x_data, coordinates)]

        # Corners are floats
        print(corners[0])

        # y_data = numpy.zeros_like(x_data)
        # y_data[corners[:, 1], corners[:, 0]] = 1

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.corners = corners

    def display(self, workspace, figure):
        dimensions = (1, 1)

        x_data = workspace.display_data.x_data[0]
        corners = workspace.display_data.corners[0]

        figure.set_subplots(dimensions)

        figure.plot(
            0,
            0,
            x_data,
            corners
        )

