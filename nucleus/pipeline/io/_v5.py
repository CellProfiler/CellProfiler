import io
import re
import os
import tempfile
import urllib.parse
import urllib.request

import nucleus.measurement
import nucleus.pipeline


def dump(pipeline, fp, save_image_plane_details):
    if len(pipeline.file_list) == 0:
        save_image_plane_details = False

    date_revision = int(re.sub(r"\.|rc\d", "", nucleus.__version__))
    module_count = len(pipeline.modules())

    fp.write("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    fp.write(f"Version:{nucleus.pipeline.NATIVE_VERSION:d}\n")
    fp.write(f"DateRevision:{date_revision:d}\n")
    fp.write(f"GitHash:{''}\n")
    fp.write(f"ModuleCount:{module_count:d}\n")
    fp.write(f"HasImagePlaneDetails:{save_image_plane_details}\n")

    default_module_attributes = (
        "module_num",
        "svn_version",
        "variable_revision_number",
        "show_window",
        "notes",
        "batch_state",
        "enabled",
        "wants_pause",
    )

    for module in pipeline.modules():
        fp.write("\n")

        module_attributes = []

        for default_module_attribute in default_module_attributes:
            attribute = repr(getattr(module, default_module_attribute))

            module_attributes += [f"{default_module_attribute}:{attribute}"]

        fp.write(f"{module.module_name}:[{'|'.join(module_attributes)}]\n")

        for setting in module.settings():
            fp.write(f"    {setting.text}:{setting.unicode_value}\n")

    if save_image_plane_details:
        fp.write("\n")

        nucleus.pipeline.write_file_list(fp, pipeline.file_list)
