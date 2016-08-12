import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting


class Open(cellprofiler.module.Module):
    module_name = "Open"

    category = "File Processing"

    variable_revision_number = 1

    def create_settings(self):
        self.directory = cellprofiler.setting.DirectoryPath("Input image file location", support_urls=True, doc="")

    def settings(self):
        return [
            self.directory
        ]

    def run(self, workspace):
        pass

    def display(self, workspace, **kwargs):
        pass
