"""
Functions to read and write CellProfiler pipelines.

pipeline._io’s API matches the APIs exposed by the json, marshal, and pickle
standard library modules.
"""

from ._v5 import dump as dump_v5
from ._v6 import dump as dump_v6


def dump(pipeline, fp, save_image_plane_details=True, version=5, sanitize=False):
    if version == 5:
        dump_v5(pipeline, fp, save_image_plane_details, sanitize)
    elif version == 6:
        dump_v6(pipeline, fp, save_image_plane_details)
    else:
        raise NotImplementedError("Invalid pipeline version")
