"""

Image segmentation

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import skimage.color
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

        self.random_walker_algorithm_labels = cellprofiler.setting.ImageNameSubscriber(
            "Labels",
            cellprofiler.setting.NONE
        )

        self.random_walker_algorithm_beta = cellprofiler.setting.Float(
            "Beta",
            130.0
        )

        self.random_walker_algorithm_mode = cellprofiler.setting.Choice(
            "Mode",
            [
                "Brute force",
                "Conjugate gradient",
                "Conjugate gradient with multigrid preconditioner"
            ]
        )

        self.random_walker_algorithm_tolerance = cellprofiler.setting.Float(
            "Tolerance",
            0.001
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
            settings = settings + [
                self.active_contour_model_implementation
            ]

            if self.active_contour_model_implementation == "Chan-Vese":
                settings = settings + [
                    self.chan_vese_mask,
                    self.chan_vese_iterations
                ]

        if self.method.value == "Graph partition":
            settings = settings + [
                self.graph_partition_implementation
            ]

            if self.graph_partition_implementation == "Random walker algorithm":
                settings = settings + [
                    self.random_walker_algorithm_beta,
                    self.random_walker_algorithm_labels,
                    self.random_walker_algorithm_mode,
                    self.random_walker_algorithm_tolerance
                ]

        if self.method.value == "Partial differential equation (PDE)":
            settings = settings + [
                self.partial_differential_equation_implementation
            ]

        if self.method.value == "Region growing":
            settings = settings + [
                self.region_growing_implementation
            ]

            if self.region_growing_implementation == "Simple Linear Iterative Clustering (SLIC)":
                settings = settings + [
                    self.simple_linear_iterative_clustering_segments,
                    self.simple_linear_iterative_clustering_compactness,
                    self.simple_linear_iterative_clustering_iterations,
                    self.simple_linear_iterative_clustering_sigma
                ]

        return settings

    def run(self, workspace):
        name = self.input_image_name.value

        images = workspace.image_set

        image = images.get_image(name)

        data = image.pixel_data

        if self.method.value == "Region growing":
            if self.region_growing_implementation == "Simple Linear Iterative Clustering (SLIC)":
                segments = self.simple_linear_iterative_clustering_segments.value

                compactness = self.simple_linear_iterative_clustering_compactness.value

                iterations = self.simple_linear_iterative_clustering_iterations.value

                sigma = self.simple_linear_iterative_clustering_sigma.value

                segmentation = skimage.segmentation.slic(data, segments, compactness, iterations, sigma)

                segmentation = skimage.color.label2rgb(segmentation, data, kind="avg")

        if self.show_window:
            workspace.display_data.image = image

            workspace.display_data.segmentation = segmentation

    def display(self, workspace, figure):
        figure.set_subplots((2, 1))

        figure.subplot_imshow_grayscale(
            0,
            0,
            workspace.display_data.image[16],
            ""
        )

        figure.subplot_imshow_grayscale(
            1,
            0,
            workspace.display_data.image[16],
            ""
        )
