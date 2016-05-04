"""

"""

import cellprofiler.cpimage
import cellprofiler.cpmodule
import cellprofiler.settings


class Open(cellprofiler.cpmodule.CPModule):
    module_name = "Open"

    category = "File Processing"

    variable_revision_number = 1

    def create_settings(self):
        self.directory = cellprofiler.settings.DirectoryPath("Input image file location", support_urls=True, doc="")

    def settings(self):
        return [
            self.directory
        ]

    def run(self, workspace):
        pass

    def display(self, workspace, **kwargs):
        pass
