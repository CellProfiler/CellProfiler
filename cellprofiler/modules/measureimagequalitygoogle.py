import numpy
import scipy.ndimage
import skimage.segmentation
import skimage.util

import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.setting

__doc__ = """
<p>This module can collect measurements indicating possible image abberations,
e.g. blur (poor focus), intensity, saturation (i.e., the percentage
of pixels in the image that are minimal and maximal). Details and guidance for
each of these measures is provided in the settings help.
</p>
"""

C_IMAGE_QUALITY = "ImageQuality"
F_SCORE = "Score"

def score(image):
    return 0

class MeasureImageQualityGoogle(cellprofiler.module.Module):
    category = "Measurement"

    module_name = "MeasureImageQualityGoogle"

    variable_revision_number = 1

    def create_settings(self):
        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Image",
            doc="""
            The name of an image.
            """
        )

    def settings(self):
        return [
            self.image_name
        ]

    def display(self, workspace, figure=None):
        layout = (2, 1)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions,
            subplots=layout
        )

        figure.subplot_imshow(
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.image,
            title=C_IMAGE_QUALITY,
            x=0,
            y=0
        )

        figure.subplot_table(
            col_labels=workspace.display_data.names,
            dimensions=workspace.display_data.dimensions,
            statistics=workspace.display_data.statistics,
            title="Measurement",
            x=0,
            y=1
        )

    def get_categories(self, pipeline, object_name):
        if object_name == cellprofiler.measurement.IMAGE:
            return [
                C_IMAGE_QUALITY
            ]

        return []

    def get_feature_name(self, name):
        image = self.image_name.value

        return C_IMAGE_QUALITY + "_{}_{}".format(name, image)

    def get_measurements(self, pipeline, object_name, category):
        name = self.image_name.value

        if object_name == cellprofiler.measurement.IMAGE and category == C_IMAGE_QUALITY:
            return [
                C_IMAGE_QUALITY + "_" + F_SCORE + "_{}".format(name)
            ]

        return []

    def get_measurement_columns(self, pipeline):
        image = cellprofiler.measurement.IMAGE

        features = [
            self.get_measurement_name(F_SCORE)
        ]

        column_type = cellprofiler.measurement.COLTYPE_INTEGER

        return [(image, feature, column_type) for feature in features]

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.image_name.value]

        return []

    def get_measurement_name(self, name):
        feature = self.get_feature_name(name)

        return feature

    def measure(self, image, workspace):
        data = image.pixel_data

        measurements = workspace.measurements

        measurement_name = self.image_name.value

        statistics = []

        name = C_IMAGE_QUALITY + "_{}".format(measurement_name)

        value = score(data)

        statistics.append(value)

        measurements.add_image_measurement(name, value)

        return [statistics]

    def volumetric(self):
        return True