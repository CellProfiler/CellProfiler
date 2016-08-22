"""

Morphological skeleton

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.exposure
import skimage.morphology
import skimage.util


class MorphologicalSkeleton(cellprofiler.module.Module):
    category = "Volumetric"
    module_name = "MorphologicalSkeleton"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "OutputImage"
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        x_data = skimage.exposure.rescale_intensity(x_data * 1.0)

        y_data = skimage.morphology.skeletonize_3d(x_data)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        figure.gridspec((1, 2), (3, 3))

        figure.add_grid(0, workspace.display_data.x_data)

        figure.add_grid(1, workspace.display_data.y_data)
