"""

<strong>Morphological skeleton<strong> thins an image into a single-pixel wide skeleton.

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import skimage.morphology


class MorphologicalSkeleton(cellprofiler.module.Module):
    category = "Mathematical morphology"
    module_name = "Morphological skeleton"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "MorphologicalSkeleton"
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

        dimensions = x.dimensions

        x_data = x.pixel_data

        y_data = skimage.morphology.skeletonize_3d(x_data)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x,
            dimensions=dimensions
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data

            workspace.display_data.y_data = y_data
            
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        layout = (1, 2)

        figure.set_subplots(layout)

        figure.subplot_imshow(
            colormap="gray",
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.x_data,
            x=0,
            y=0
        )

        figure.subplot_imshow(
            colormap="gray",
            dimensions=workspace.display_data.dimensions,
            image=workspace.display_data.y_data,
            x=1,
            y=0
        )
