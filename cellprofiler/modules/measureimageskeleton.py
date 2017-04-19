import numpy
import scipy.ndimage
import skimage.measure
import skimage.segmentation
import skimage.util

import cellprofiler.module
import cellprofiler.setting


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
    padding = skimage.util.pad(image, 1, "constant")

    mask = padding > 0

    padding = padding.astype(numpy.float)

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


class MeasureImageSkeleton(cellprofiler.module.Module):
    category = "Measurement"

    module_name = "MeasureImageSkeleton"

    variable_revision_number = 1

    def create_settings(self):
        self.skeleton_name = cellprofiler.setting.ImageNameSubscriber(
            "Skeleton"
        )

    def settings(self):
        return [
            self.skeleton_name
        ]

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

            a = numpy.copy(branch_nodes)
            b = numpy.copy(endpoint_nodes)

            a[a == 1] = 1
            b[b == 1] = 2

            nodes = skimage.segmentation.join_segmentations(a, b)

            workspace.display_data.nodes = nodes

            workspace.display_data.dimensions = dimensions

            workspace.display_data.names = names

            workspace.display_data.statistics = statistics

    def display(self, workspace, figure=None):
        layout = (2, 2)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions,
            subplots=layout
        )

        figure.subplot_imshow(
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.skeleton,
            title="Skeleton",
            x=0,
            y=0
        )

        figure.subplot_imshow(
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.nodes,
            title="Nodes",
            x=1,
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

    def measure(self, image, workspace):
        data = image.pixel_data

        data = data.astype(numpy.bool)

        measurements = workspace.measurements

        measurement_name = self.skeleton_name.value

        statistics = []

        name = "Skeleton_Branches_{}".format(measurement_name)

        value = numpy.count_nonzero(skimage.measure.label(branches(data)))

        statistics.append(value)

        measurements.add_image_measurement(name, value)

        name = "Skeleton_Endpoints_{}".format(measurement_name)

        value = numpy.count_nonzero(skimage.measure.label(endpoints(data)))

        statistics.append(value)

        measurements.add_image_measurement(name, value)

        return [statistics]
