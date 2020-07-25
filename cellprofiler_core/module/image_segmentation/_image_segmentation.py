import numpy

import cellprofiler_core.measurement
import cellprofiler_core.object
import cellprofiler_core.setting
import cellprofiler_core.setting.text.alphanumeric.name._label
from .._module import Module


class ImageSegmentation(Module):
    category = "Image Segmentation"

    def add_measurements(self, workspace, object_name=None):
        if object_name is None:
            object_name = self.y_name.value

        objects = workspace.object_set.get_objects(object_name)

        centers = objects.center_of_mass()

        if len(centers) == 0:
            center_z, center_y, center_x = [], [], []
        else:
            if objects.volumetric:
                center_z, center_y, center_x = centers.transpose()
            else:
                center_z = [0] * len(centers)

                center_y, center_x = centers.transpose()

        workspace.measurements.add_measurement(
            object_name, cellprofiler_core.measurement.M_LOCATION_CENTER_X, center_x
        )

        workspace.measurements.add_measurement(
            object_name, cellprofiler_core.measurement.M_LOCATION_CENTER_Y, center_y
        )

        workspace.measurements.add_measurement(
            object_name, cellprofiler_core.measurement.M_LOCATION_CENTER_Z, center_z
        )

        workspace.measurements.add_measurement(
            object_name,
            cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER,
            numpy.arange(1, objects.count + 1),
        )

        workspace.measurements.add_measurement(
            cellprofiler_core.measurement.IMAGE,
            cellprofiler_core.measurement.FF_COUNT % object_name,
            numpy.array([objects.count], dtype=float),
        )

    def create_settings(self):
        self.x_name = cellprofiler_core.setting.ImageNameSubscriber(
            "Select the input image", doc="Select the image you want to use."
        )

        self.y_name = cellprofiler_core.setting.text.alphanumeric.name._label.Label(
            "Name the output object",
            self.__class__.__name__,
            doc="Enter the name you want to call the object produced by this module.",
        )

    def display(self, workspace, figure):
        layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow(
            colormap="gray",
            image=workspace.display_data.x_data,
            title=self.x_name.value,
            x=0,
            y=0,
        )

        figure.subplot_imshow_labels(
            background_image=workspace.display_data.x_data,
            image=workspace.display_data.y_data,
            sharexy=figure.subplot(0, 0),
            title=self.y_name.value,
            x=1,
            y=0,
        )

    def get_categories(self, pipeline, object_name):
        if object_name == cellprofiler_core.measurement.IMAGE:
            return [cellprofiler_core.measurement.C_COUNT]

        if object_name == self.y_name.value:
            return [
                cellprofiler_core.measurement.C_LOCATION,
                cellprofiler_core.measurement.C_NUMBER,
            ]

        return []

    def get_measurement_columns(self, pipeline, object_name=None):
        if object_name is None:
            object_name = self.y_name.value

        return [
            (
                object_name,
                cellprofiler_core.measurement.M_LOCATION_CENTER_X,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                object_name,
                cellprofiler_core.measurement.M_LOCATION_CENTER_Y,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                object_name,
                cellprofiler_core.measurement.M_LOCATION_CENTER_Z,
                cellprofiler_core.measurement.COLTYPE_FLOAT,
            ),
            (
                object_name,
                cellprofiler_core.measurement.M_NUMBER_OBJECT_NUMBER,
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
            (
                cellprofiler_core.measurement.IMAGE,
                cellprofiler_core.measurement.FF_COUNT % object_name,
                cellprofiler_core.measurement.COLTYPE_INTEGER,
            ),
        ]

    def get_measurements(self, pipeline, object_name, category):
        if (
            object_name == cellprofiler_core.measurement.IMAGE
            and category == cellprofiler_core.measurement.C_COUNT
        ):
            return [self.y_name.value]

        if object_name == self.y_name.value:
            if category == cellprofiler_core.measurement.C_LOCATION:
                return [
                    cellprofiler_core.measurement.FTR_CENTER_X,
                    cellprofiler_core.measurement.FTR_CENTER_Y,
                    cellprofiler_core.measurement.FTR_CENTER_Z,
                ]

            if category == cellprofiler_core.measurement.C_NUMBER:
                return [cellprofiler_core.measurement.FTR_OBJECT_NUMBER]

        return []

    def run(self, workspace):
        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = x.pixel_data

        args = (setting.value for setting in self.settings()[2:])

        y_data = self.function(x_data, *args)

        y = cellprofiler_core.object.Objects()

        y.segmented = y_data

        y.parent_image = x.parent_image

        objects = workspace.object_set

        objects.add_objects(y, y_name)

        self.add_measurements(workspace)

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
