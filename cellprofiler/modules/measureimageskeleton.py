import cellprofiler.module
import cellprofiler.setting
import numpy
import scipy.ndimage
import skimage.segmentation
import skimage.util


def _neighbors(image):
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

        self.branches_name = cellprofiler.setting.ImageNameProvider(
            "Branches",
            "MeasureSkeletonBranches"
        )

        self.endpoints_name = cellprofiler.setting.ImageNameProvider(
            "Endpoints",
            "MeasureSkeletonEndpoints"
        )

    def settings(self):
        return [
            self.skeleton_name,
            self.branches_name,
            self.endpoints_name
        ]

    def run(self, workspace):
        input_image_name = self.skeleton_name.value

        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name, must_be_grayscale=True)

        dimensions = input_image.dimensions

        pixels = input_image.pixel_data

        pixels = pixels > 0

        branch_nodes = branches(pixels)

        endpoint_nodes = endpoints(pixels)

        if self.show_window:
            workspace.display_data.skeleton = pixels

            a = numpy.copy(branch_nodes)
            b = numpy.copy(endpoint_nodes)

            a[a == 1] = 1
            b[b == 1] = 2

            nodes = skimage.segmentation.join_segmentations(a, b)

            workspace.display_data.nodes = nodes

            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure=None):
        layout = (2, 1)

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
