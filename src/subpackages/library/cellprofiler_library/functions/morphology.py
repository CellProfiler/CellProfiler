import inspect

import numpy
import skimage.morphology


_HAS_MAX_SIZE = "max_size" in inspect.signature(
    skimage.morphology.remove_small_holes
).parameters


def get_footprint(shape: str, size: int):
    if shape in ("square", "cube"):
        dimensions = 2 if shape == "square" else 3
        return skimage.morphology.footprint_rectangle((size,) * dimensions)
    return getattr(skimage.morphology, shape)(size)


def binary_erosion(image, footprint):
    footprint = skimage.morphology.pad_footprint(footprint, pad_end=True)
    return skimage.morphology.erosion(
        numpy.asarray(image, dtype=bool), footprint, mode="ignore"
    )


def binary_dilation(image, footprint):
    footprint = skimage.morphology.pad_footprint(footprint, pad_end=True)
    footprint = skimage.morphology.mirror_footprint(footprint)
    return skimage.morphology.dilation(
        numpy.asarray(image, dtype=bool), footprint, mode="ignore"
    )


def remove_small_holes(image, size):
    if _HAS_MAX_SIZE:
        return skimage.morphology.remove_small_holes(
            image, max_size=size
        )
    return skimage.morphology.remove_small_holes(image, area_threshold=size)
