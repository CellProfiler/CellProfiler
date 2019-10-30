import re

import nucleus.pipeline


def dump_v5(pipeline, fp, save_image_plane_details=True):
    if hasattr(fp, "write"):
        fd = fp
        needs_close = False
    else:
        fd = open(fp, "wt")
        needs_close = True

    # Don't write image plane details if we don't have any
    if len(pipeline.file_list) == 0:
        save_image_plane_details = False

    date_revision = int(re.sub(r"\.|rc\d", "", nucleus.__version__))
    module_count = len(pipeline.modules())

    fd.write("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    fd.write(f"Version:{nucleus.pipeline.NATIVE_VERSION:d}\n")
    fd.write(f"DateRevision:{date_revision:d}\n")
    fd.write(f"GitHash:{''}\n")
    fd.write(f"ModuleCount:{module_count:d}\n")
    fd.write(f"HasImagePlaneDetails:{save_image_plane_details}\n")

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
        fd.write("\n")

        module_attributes = []

        for default_module_attribute in default_module_attributes:
            attribute = repr(getattr(module, default_module_attribute))

            module_attributes += [f"{default_module_attribute}:{attribute}"]

        fd.write(f"{module.module_name}:[{'|'.join(module_attributes)}]\n")

        for setting in module.settings():
            fd.write(f"    {setting.text}:{setting.unicode_value}\n")

    if save_image_plane_details:
        fd.write("\n")

        nucleus.pipeline.write_file_list(fd, pipeline.file_list)

    if needs_close:
        fd.close()
