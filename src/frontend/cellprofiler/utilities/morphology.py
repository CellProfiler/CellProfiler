"""
Shared mophology methods used by multiple modules
"""

import numpy
import scipy.ndimage
import skimage.morphology


def dilation(x_data, structuring_element):
    is_strel_2d = structuring_element.ndim == 2

    is_img_2d = x_data.ndim == 2

    if is_strel_2d and not is_img_2d:
        y_data = numpy.zeros_like(x_data)

        for index, plane in enumerate(x_data):

            y_data[index] = skimage.morphology.dilation(plane, structuring_element)

        return y_data

    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )

    y_data = skimage.morphology.dilation(x_data, structuring_element)

    return y_data


def erosion(x_data, structuring_element):
    is_strel_2d = structuring_element.ndim == 2

    is_img_2d = x_data.ndim == 2

    if is_strel_2d and not is_img_2d:
        y_data = numpy.zeros_like(x_data)

        for index, plane in enumerate(x_data):

            y_data[index] = skimage.morphology.erosion(plane, structuring_element)

        return y_data

    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )

    y_data = skimage.morphology.erosion(x_data, structuring_element)

    return y_data


def binary_erosion(x_data, structuring_element):
    is_strel_2d = structuring_element.ndim == 2

    is_img_2d = x_data.ndim == 2

    if is_strel_2d and not is_img_2d:
        y_data = numpy.zeros_like(x_data)

        for index, plane in enumerate(x_data):

            y_data[index] = skimage.morphology.binary_erosion(
                plane, structuring_element
            )

        return y_data

    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )

    y_data = skimage.morphology.binary_erosion(x_data, structuring_element)

    return y_data


def morphological_gradient(x_data, structuring_element):
    is_strel_2d = structuring_element.ndim == 2

    is_img_2d = x_data.ndim == 2

    if is_strel_2d and not is_img_2d:
        y_data = numpy.zeros_like(x_data)

        for index, plane in enumerate(x_data):
            y_data[index] = scipy.ndimage.morphological_gradient(
                plane, footprint=structuring_element
            )

        return y_data

    if not is_strel_2d and is_img_2d:
        raise NotImplementedError(
            "A 3D structuring element cannot be applied to a 2D image."
        )

    y_data = scipy.ndimage.morphological_gradient(x_data, footprint=structuring_element)

    return y_data
