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
from cellprofiler_library.modules._measureimageskeleton import measure_image_skeleton
from cellprofiler_library.opts.measureimageskeleton import SkeletonMeasurements



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

        branch_nodes, endpoint_nodes, measurements = measure_image_skeleton(pixels, im_name=self.skeleton_name.value)

        for object_name, features in measurements.items():
            if object_name == "Image":
                for feature_name, value in features.items():
                    workspace.measurements.add_image_measurement(feature_name, value)

        name_branches = SkeletonMeasurements.BRANCHES.value.format(input_image_name)
        name_endpoints = SkeletonMeasurements.ENDPOINTS.value.format(input_image_name)

        num_branches = measurements["Image"][name_branches]
        num_endpoints = measurements["Image"][name_endpoints]
        
        statistics = [[num_branches, num_endpoints]]

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


    def volumetric(self):
        return True
