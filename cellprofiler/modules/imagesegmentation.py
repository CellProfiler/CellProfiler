"""

Image segmentation

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import numpy
import skimage.filters
import skimage.morphology
import skimage.segmentation


class ImageSegmentation(cellprofiler.module.Module):
    module_name = "ImageSegmentation"
    category = "Volumetric"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
            "Input image name:",
            cellprofiler.setting.NONE
        )

        self.object_name = cellprofiler.setting.ObjectNameProvider(
            "Object name",
            ""
        )

        self.method = cellprofiler.setting.Choice(
            "Method",
            [
                "Active contour model",
                "Graph partition",
                "Partial differential equation (PDE)",
                "Region growing",
                "Watershed"
            ]
        )

        self.active_contour_model_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Chan-Vese"
            ]
        )

        self.chan_vese_mask = cellprofiler.setting.ImageNameSubscriber(
            "Mask",
            cellprofiler.setting.NONE
        )

        self.chan_vese_iterations = cellprofiler.setting.Integer(
            "Iterations",
            200
        )

        self.graph_partition_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Random walker algorithm"
            ]
        )

        self.partial_differential_equation_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Level set method (LSM)"
            ]
        )

        self.region_growing_implementation = cellprofiler.setting.Choice(
            "Implementation",
            [
                "Simple Linear Iterative Clustering (SLIC)"
            ]
        )

        self.simple_linear_iterative_clustering_segments = cellprofiler.setting.Integer(
            "Segments",
            200
        )

        self.simple_linear_iterative_clustering_compactness = cellprofiler.setting.Float(
            "Compactness",
            10.0
        )

        self.simple_linear_iterative_clustering_iterations = cellprofiler.setting.Integer(
            "Iterations",
            10
        )

        self.simple_linear_iterative_clustering_sigma = cellprofiler.setting.Float(
            "Sigma",
            0
        )

    def settings(self):
        return [
            self.input_image_name,
            self.object_name,
            self.method,
            self.active_contour_model_implementation,
            self.graph_partition_implementation,
            self.partial_differential_equation_implementation
        ]

    def visible_settings(self):
        settings = [
            self.input_image_name,
            self.object_name,
            self.method
        ]

        if self.method.value == "Active contour model":
            settings.append(self.active_contour_model_implementation)

            if self.active_contour_model_implementation == "Chan-Vese":
                settings.append(self.chan_vese_mask)

                settings.append(self.chan_vese_iterations)

        if self.method.value == "Graph partition":
            settings.append(self.graph_partition_implementation)

        if self.method.value == "Partial differential equation (PDE)":
            settings.append(self.partial_differential_equation_implementation)

        if self.method.value == "Region growing":
            settings.append(self.region_growing_implementation)

            if self.region_growing_implementation == "Simple Linear Iterative Clustering (SLIC)":
                settings.append(self.simple_linear_iterative_clustering_segments)

                settings.append(self.simple_linear_iterative_clustering_compactness)

                settings.append(self.simple_linear_iterative_clustering_iterations)

                settings.append(self.simple_linear_iterative_clustering_sigma)

        return settings

    def run(self, workspace):
        input_image_name = self.input_image_name.value

        image_set = workspace.image_set
        input_image = image_set.get_image(input_image_name)
        pixels = input_image.pixel_data

        if self.method.value == "Region growing":
            if self.region_growing_implementation == "Simple Linear Iterative Clustering (SLIC)":
                segmentation = skimage.segmentation.slic(
                    pixels,
                    self.simple_linear_iterative_clustering_segments.value,
                    self.simple_linear_iterative_clustering_compactness.value,
                    self.simple_linear_iterative_clustering_iterations.value,
                    self.simple_linear_iterative_clustering_sigma.value
                )


    def display(self, workspace, figure):
        pass
        # figure.set_subplots((2, 1))
        #
        # figure.subplot_imshow_grayscale(
        #     0,
        #     0,
        #     workspace.display_data.input_pixels[16],
        #     title=self.input_image_name.value
        # )
        #
        # figure.subplot_imshow_grayscale(
        #     1,
        #     0,
        #     workspace.display_data.output_pixels[16],
        #     title=self.output_object.value
        # )
