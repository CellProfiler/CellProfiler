"""
Functions to read and write CellProfiler pipelines.

pipeline._io’s API matches the APIs exposed by the json, marshal, and pickle
standard library modules.
"""

from ._v5 import dump as dump_v5
from ._v5 import load as load_v5


def dump(pipeline, fp, save_image_plane_details=True, version=5):
    if version == 5:
        dump_v5(pipeline, fp, save_image_plane_details)
