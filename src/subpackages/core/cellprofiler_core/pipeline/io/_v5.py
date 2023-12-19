from packaging.version import Version

import cellprofiler_core.constants.pipeline
import cellprofiler_core.measurement
import cellprofiler_core.pipeline
import cellprofiler_core.utilities.core.pipeline
from cellprofiler_core import __version__ as core_version


def dump(pipeline, fp, save_image_plane_details, sanitize=False):
    if len(pipeline.file_list) == 0:
        save_image_plane_details = False

    sem_ver = Version(core_version)

    ver = int(f"{sem_ver.major}{sem_ver.minor}{sem_ver.micro}")
    module_count = len(pipeline.modules(False))

    fp.write("CellProfiler Pipeline: http://www.cellprofiler.org\n")
    fp.write(f"Version:{cellprofiler_core.constants.pipeline.NATIVE_VERSION :d}\n")
    fp.write(f"DateRevision:{ver:d}\n")
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

    for module in pipeline.modules(False):
        fp.write("\n")

        module_attributes = []

        for default_module_attribute in default_module_attributes:
            attribute = repr(getattr(module, default_module_attribute))

            module_attributes += [f"{default_module_attribute}:{attribute}"]

        fp.write(f"{module.module_name}:[{'|'.join(module_attributes)}]\n")

        for setting in module.settings():
            if sanitize and any(
                phrase in setting.text.lower()
                for phrase in ("username", "password", "host")
            ):
                fp.write(f"    [SensitiveSetting]:*****\n")
            else:
                fp.write(f"    {setting.text}:{setting.unicode_value}\n")

    if save_image_plane_details:
        fp.write("\n")

        cellprofiler_core.utilities.core.pipeline.write_file_list(
            fp, pipeline.file_list
        )
