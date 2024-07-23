"""
MeasureImageSkeleton
====================

**MeasureImageSkeleton** measures the number of branches and endpoints in a
skeletonized structure such as neurons, roots, or vasculature.

This module can analyze the number of total branches and endpoints for
branching objects in an image. A branch is a pixel with more than two
neighbors and an endpoint is a pixel with only one neighbor.

You can create a morphological skeleton with the **MorphologicalSkeleton**
module from the *Advanced* category.

See also **MeasureObjectSkeleton**.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- *Branches*: Total number of pixels with more than two neighbors.

- *Endpoints*: Total number of pixels with only one neighbor.
"""

import numpy
import scipy.ndimage
import skimage.segmentation
import skimage.util
from cellprofiler_core.module import Module
from cellprofiler_core.setting.subscriber import ImageSubscriber


def _neighbors(image):
    """

    Counts the neighbor pixels for each pixel of an image:

            x = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]

            _neighbors(x)

            [
                [0, 3, 0],
                [3, 4, 3],
                [0, 3, 0]
            ]

    :type image: numpy.ndarray

    :param image: A two-or-three dimensional image

    :return: neighbor pixels for each pixel of an image

    """
    padding = numpy.pad(image, 1, "constant")

    mask = padding > 0

    padding = padding.astype(float)

    if image.ndim == 2:
        response = 3 ** 2 * scipy.ndimage.uniform_filter(padding) - 1

        labels = (response * mask)[1:-1, 1:-1]

        return labels.astype(numpy.uint16)
    elif image.ndim == 3:
        response = 3 ** 3 * scipy.ndimage.uniform_filter(padding) - 1

        labels = (response * mask)[1:-1, 1:-1, 1:-1]

        return labels.astype(numpy.uint16)


def branches(image):
    return _neighbors(image) > 2


def endpoints(image):
    return _neighbors(image) == 1


class MeasureImageSkeleton(Module):
    category = "Measurement"

    module_name = "MeasureImageSkeleton"

    variable_revision_number = 1

    def create_settings(self):
        self.skeleton_name = ImageSubscriber(
            "Select an image to measure",
            doc="""\
Select the morphological skeleton image you wish to measure.
You can create a morphological skeleton with the
**MorphologicalSkeleton** module from the *Advanced* category.
""",
        )

    def settings(self):
        return [self.skeleton_name]

    def run(self, workspace):
        names = ["Branches", "Endpoints"]

        input_image_name = self.skeleton_name.value

        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name, must_be_grayscale=True)

        dimensions = input_image.dimensions

        pixels = input_image.pixel_data

        pixels = pixels > 0

        branch_nodes = branches(pixels)

        endpoint_nodes = endpoints(pixels)

        statistics = self.measure(input_image, workspace)

        if self.show_window:
            workspace.display_data.skeleton = pixels

            a = numpy.copy(branch_nodes).astype(numpy.uint16)
            b = numpy.copy(endpoint_nodes).astype(numpy.uint16)

            a[a == 1] = 1
            b[b == 1] = 2

            nodes = skimage.segmentation.join_segmentations(a, b)

            workspace.display_data.nodes = nodes

            workspace.display_data.dimensions = dimensions

            workspace.display_data.names = names

            workspace.display_data.statistics = statistics

    def display(self, workspace, figure=None):
        layout = (2, 2)

        cmap = figure.return_cmap()

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.skeleton, title="Skeleton", x=0, y=0, colormap=cmap,
        )

        figure.subplot_imshow_labels(
            image=workspace.display_data.nodes,
            title="Nodes",
            x=1,
            y=0,
            sharexy=figure.subplot(0, 0),
            colormap=cmap,

        )

        figure.subplot_table(
            col_labels=workspace.display_data.names,
            statistics=workspace.display_data.statistics,
            title="Measurement",
            x=0,
            y=1,
        )

    def get_categories(self, pipeline, object_name):
        if object_name == "Image":
            return ["Skeleton"]

        return []

    def get_feature_name(self, name):
        image = self.skeleton_name.value

        return "Skeleton_{}_{}".format(name, image)

    def get_measurements(self, pipeline, object_name, category):
        name = self.skeleton_name.value

        if object_name == "Image" and category == "Skeleton":
            return [
                "Branches",
                "Endpoints"
            ]

        return []

    def get_measurement_columns(self, pipeline):
        image = "Image"

        features = [
            self.get_measurement_name("Branches"),
            self.get_measurement_name("Endpoints"),
        ]

        column_type = "integer"

        return [(image, feature, column_type) for feature in features]

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.skeleton_name.value]

        return []

    def get_measurement_name(self, name):
        feature = self.get_feature_name(name)

        return feature

    def measure(self, image, workspace):
        data = image.pixel_data

        data = data.astype(bool)

        measurements = workspace.measurements

        measurement_name = self.skeleton_name.value

        statistics = []

        name = "Skeleton_Branches_{}".format(measurement_name)

        value = numpy.count_nonzero(branches(data))

        statistics.append(value)

        measurements.add_image_measurement(name, value)

        name = "Skeleton_Endpoints_{}".format(measurement_name)

        value = numpy.count_nonzero(endpoints(data))

        statistics.append(value)

        measurements.add_image_measurement(name, value)

        return [statistics]

    def volumetric(self):
        return True
