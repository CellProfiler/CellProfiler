from multiprocessing import Value
from tkinter import ALL
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Annotated, Type
from pydantic import Field, AfterValidator, validate_call, ConfigDict
import numpy as np
import numpy.typing as npt
from numbers import Number

""" TODO: update this docstring
Image Types can be classified into:
3D/2D: "3D" or "2D"
Single channel/multi-channel: "single" or "multi"
Data type: uint8, uint16, float32, float64, bool
Tiled or single: "tiled" or "single"
"""

def create_type_validator(is_3d: bool, is_multi_channel: bool, is_tiled: bool, dtype, shape_override: Optional[Iterable[int]] = None):
    def validate(input_image: npt.NDArray[Any]):
        print(f"running validate with args: {input_image.shape}, {is_3d}, {is_multi_channel}, {is_tiled}, {dtype}, {shape_override}")
        # TODO: edit all error strings to be more descriptive
        if shape_override is not None:
            if input_image.shape != shape_override:
                raise ValueError(f"Expected an array of shape {shape_override}, got {input_image.shape}") 
        else:
            if is_3d:
                # TODO: Confirm if 3D images are inputted as 5D arrays or 3D arrays (tests seem to indicate 3D)
                # # check cztyx shape
                # if input_image.ndim != 5:
                #     raise ValueError(f"Expected a 5D array (cztyx), got {input_image.ndim}D")
                # if input_image.shape[1] <= 1:
                #     raise ValueError(f"Expected a 5D array with at least 2 z channels, got {input_image.shape[1]}")
                # if is_multi_channel:
                #     if input_image.shape[0] <= 1:
                #         raise ValueError(f"Expected a 5D array with at least 2 c channels, got {input_image.shape[0]}")
                if input_image.ndim < 3 or input_image.ndim > 4:
                    raise ValueError(f"Expected a 3D or 4D array ([c], z, y, x), got {input_image.ndim}D")
                if input_image.ndim == 3 and input_image.shape[0] <= 1:
                    raise ValueError(f"Expected a 3D array with at least 2 z channels, got {input_image.shape[1]}")
                if input_image.ndim == 4:
                    if input_image.shape[1] <= 1:
                        raise ValueError(f"Expected a 4D array with at least 2 z channels, got {input_image.shape[2]}")
                    if input_image.shape[0] <= 1:
                        raise ValueError(f"Expected a 4D array with at least 2 c channels, got {input_image.shape[1]}")
                if is_multi_channel:
                    if input_image.shape[0] <= 1:
                        raise ValueError(f"Expected a 5D array with at least 2 c channels, got {input_image.shape[0]}")
                    
            else: # 2d image. 
                if is_multi_channel: # color
                    if input_image.ndim != 3:
                        raise ValueError(f"Expected a 3D array (cyx),got {input_image.ndim}D")
                else: # grayscale
                    if input_image.ndim != 2:
                        raise ValueError(f"Expected a 2D array (yx),got {input_image.ndim}D")
        if '__args__' not in dir(dtype):
            if input_image.dtype != dtype:
                raise ValueError(f"Expected an array of type {dtype}, got {input_image.dtype}")
        elif input_image.dtype not in dtype.__args__:
            raise ValueError(f"Expected an array of type {dtype}, got {input_image.dtype}")
        if is_tiled:
            pass
        return input_image
    return validate

def validate_object_labels_dense(input_image: npt.NDArray[np.generic]) -> np.ndarray:
    if input_image.ndim != 6:
        raise ValueError(f"Expected an array of shape (n, c, z, t, y, x), got {input_image.shape}")
    return input_image  

def validate_object_label_set(label_set: list) -> list:
    # label set is a list of 2 tuples
    for label in label_set:
        if type(label) != tuple or len(label) != 2:
            raise ValueError(f"Expected a list of tuples of length 2, got {label}")
    return label_set

# # TODO: Is there a better way to do this?
Image2DColor = Annotated[np.ndarray, Field(description="2D image with multiple channels of type float32"), AfterValidator(create_type_validator(False, True, False, Union[np.float32, np.float64]))]
Image2DColorMask = Annotated[np.ndarray, Field(description="2D color mask"), AfterValidator(create_type_validator(False, True, False, np.bool_))]
Image2DGrayscale = Annotated[np.ndarray, Field(description="2D grayscale image of type float32"), AfterValidator(create_type_validator(False, False, False, Union[np.float32, np.float64]))]
Image2DGrayscaleMask = Annotated[np.ndarray, Field(description="2D grayscale mask"), AfterValidator(create_type_validator(False, False, False, np.bool_))]

Image3DColor = Annotated[np.ndarray, Field(description="3D image with multiple channels of type float32"), AfterValidator(create_type_validator(True, True, False, Union[np.float32, np.float64]))]
Image3DColorMask = Annotated[np.ndarray, Field(description="3D image with multiple channels of type float32"), AfterValidator(create_type_validator(True, True, False, np.bool_))]
Image3DGrayscale = Annotated[np.ndarray, Field(description="3D grayscale image of type float32"), AfterValidator(create_type_validator(True, False, False, Union[np.float32, np.float64]))]
Image3DGrayscaleMask = Annotated[np.ndarray, Field(description="3D grayscale mask"), AfterValidator(create_type_validator(True, False, False, np.bool_))]

ObjectLabelsDense = Annotated[np.ndarray, Field(description="Dense array of object labels"), AfterValidator(validate_object_labels_dense)]
ObjectLabelSet = Annotated[list, Field(description="List of object labels"), AfterValidator(validate_object_label_set)]

ImageGrayscale = Union[Image2DGrayscale, Image3DGrayscale]
ImageGrayscaleMask = Union[Image2DGrayscaleMask, Image3DGrayscaleMask]

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def sum_up_im(im: Annotated[Image2DColor, Field(description="2D image with multiple channels of type float32")]) -> float:
    return im.sum()