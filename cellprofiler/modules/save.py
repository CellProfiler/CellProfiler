"""

Save

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import imageio


class Save(cellprofiler.module.Module):
    category = "Input/output (I/O)"
    module_name = "Save"
    variable_revision_number = 1

    def create_settings(self):
        self.image = cellprofiler.setting.ImageNameSubscriber(
            "Image"
        )

        self.filename = cellprofiler.setting.Text(
            "Filename",
            ""
        )

    def settings(self):
        return [
            self.image,
            self.filename
        ]

    def visible_settings(self):
        return [
            self.image,
            self.filename
        ]

    def run(self, workspace):
        filename = self.filename.value

        images = workspace.image_set

        image = images.get_image(self.image.value).pixel_data

        imageio.mimsave(filename, image)

    def display(self, workspace, figure):
        pass
