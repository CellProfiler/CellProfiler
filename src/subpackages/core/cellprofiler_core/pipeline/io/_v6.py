import json
import logging
from packaging.version import Version

from cellprofiler_core import __version__ as core_version
from cellprofiler_core.constants.pipeline import (
    IMAGE_PLANE_DESCRIPTOR_VERSION,
    H_PLANE_COUNT,
)

LOGGER = logging.getLogger(__name__)

def dump(pipeline, fp, save_image_plane_details):
    """Serializes pipeline into JSON"""
    modules = []

    for module in pipeline.modules(False):
        settings = []
        for setting in module.settings():
            settings.append(setting.to_dict())

        modules += [{"attributes": module.to_dict(), "settings": settings}]

    if len(pipeline.file_list) == 0:
        save_image_plane_details = False
        
    sem_ver = Version(core_version)

    content = {
        "has_image_plane_details": save_image_plane_details,
        "date_revision": int(f"{sem_ver.major}{sem_ver.minor}{sem_ver.micro}"),
        "module_count": len(pipeline.modules(False)),
        "modules": modules,
        "version": "v6",
    }

    if save_image_plane_details:
        urls = [url for url in pipeline.file_list]
        file_list = {
            "version": '"%s":"%d","%s":"%d"'
            % (
                "Version",
                IMAGE_PLANE_DESCRIPTOR_VERSION,
                H_PLANE_COUNT,
                len(pipeline.file_list),
            ),
            "urls": urls,
        }
        content["file_list"] = file_list

    json.dump(content, fp, indent=4)


def load(pipeline, fd):
    pipeline_dict = json.load(fd)
    sem_ver = Version(core_version)
    cp_version = int(f"{sem_ver.major}{sem_ver.minor}{sem_ver.micro}")
    if cp_version != pipeline_dict['date_revision']:
        LOGGER.warning(f"Pipeline file is from a different version of CellProfiler. "
                        f"Current:v{cp_version} File:v{pipeline_dict['date_revision']}."
                        f" Will attempt to upgrade settings.")
    pipeline_modules = pipeline.modules(False)
    pipeline_modules.clear()
    for module in pipeline_dict["modules"]:
        module_name = module["attributes"]["module_name"]
        settings = [setting_dict for setting_dict in module["settings"]]
        new_module = pipeline.instantiate_module(module_name)
        new_module.from_dict(settings, module["attributes"])
        pipeline_modules.append(new_module)

