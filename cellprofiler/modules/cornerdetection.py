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

        self.method = cellprofiler.setting.Choice(
            u"Harris corner method",
            [
                "Sensitivity factor",  # k -- default
                "Normalization factor"  # eps
            ],
            "Sensitivity factor"
        )

        self.k = cellprofiler.setting.Float(
            u"Sensitivity factor",
            0.05
        )

        self.eps = cellprofiler.setting.Float(
            u"Normalization factor",
            1e-06
        )

        self.minimum_distance = cellprofiler.setting.Float(
            u"Minimum distance",
            1
        )

        self.exclude_border = cellprofiler.setting.Binary(
            u"Exclude peaks from boarder?",
            False
        )

        self.exclude_border_distance = cellprofiler.setting.Integer(
            u"Distance from boarder",
            0
        )

        self.num_peaks = cellprofiler.setting.Integer(
            u"Number of peaks",
            0
        )

        self.window_size = cellprofiler.setting.Integer(
            u"Window size",
            13
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.method,
            self.minimum_distance,
            self.exclude_border,
            self.exclude_border_distance,
            self.num_peaks,
            self.window_size
        ]

    def visible_settings(self):
        settings = [
            self.x_name,
            self.y_name,
            self.minimum_distance,
            self.num_peaks,
            self.window_size,
            self.method
        ]

        if self.method.value == "Sensitivity factor":
            settings = settings + [
                self.k
            ]

        if self.method.value == "Normalization factor":
            settings = settings + [
                self.eps
            ]

        settings = settings + [self.exclude_border]

        if self.exclude_border.value:
            settings = settings + [
                self.exclude_border_distance
            ]

        return settings

    def run(self, workspace):
        x_name = self.x_name.value
        x_data = workspace.image_set.get_image(x_name).pixel_data

        minimum_distance = self.minimum_distance.value

        coordinates = [
            skimage.feature.corner_peaks(
                skimage.feature.corner_harris(
                    data,
                    method=self.__method(),
                    k=self.k.value,
                    eps=self.eps.value
                ),
                min_distance=minimum_distance,
                exclude_border=self.exclude_border_distance.value,
                num_peaks=self.__num_peaks()
            ) for data in x_data
        ]

        corners = [
            skimage.feature.corner_subpix(
                data,
                point,
                window_size=self.window_size.value
            ) for data, point in zip(x_data, coordinates)
        ]

        # TODO: Save corners as an... image? segmentation?
        # Convert corners from float to integers?
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

    def __method(self):
        if self.method.value == "Sensitivity factor":
            return "k"

        return "eps"

    def __num_peaks(self):
        if self.num_peaks.value == 0:
            return numpy.inf

        return self.num_peaks.value