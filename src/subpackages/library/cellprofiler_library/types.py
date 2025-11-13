from typing import Any, Iterable, Optional, Union, Annotated, get_origin, get_args, Sequence, Tuple
from pydantic import Field, AfterValidator
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


class CellProfilerInputValidationError(ValueError):
    def __init__(self, message: str):
        self.message = f"CellProfiler Input Validation Error: {message}"
        super().__init__(self.message)

    def __str__(self):
        return self.message
    
""" 
Image Types can be classified into:
3D/2D: "3D" or "2D"
Single channel/multi-channel: "single" or "multi"
Data type: uint8, uint16, float32, float64, bool
Tiled or single: "tiled" or "single"
"""
def create_type_validator(is_3d: bool, is_multi_channel: bool, is_tiled: bool, dtype: type, shape_override: Optional[Iterable[int]] = None):
    def validate(input_image: npt.NDArray[Any]):
        if shape_override is not None:
            if input_image.shape != shape_override:
                raise CellProfilerInputValidationError(f"Expected an array of shape {shape_override}, got {input_image.shape}") 
        else:
            if is_3d:
                if input_image.ndim < 3 or input_image.ndim > 4: # not 3D or 3D multi-channel
                    raise CellProfilerInputValidationError(f"Expected a 3D or 4D array ([c], z, y, x), got {input_image.ndim}D")
                if input_image.ndim == 3 and input_image.shape[0] <= 1: # Only one z channel
                    raise CellProfilerInputValidationError(f"Expected a 3D array with at least 2 z channels, got {input_image.shape[1]}")
                if input_image.ndim == 4:
                    if input_image.shape[1] <= 1:
                        raise CellProfilerInputValidationError(f"Expected a 4D array with at least 2 z channels, got {input_image.shape[2]}")
                    if input_image.shape[0] <= 1:
                        raise CellProfilerInputValidationError(f"Expected a 4D array with at least 2 c channels, got {input_image.shape[1]}")
                if is_multi_channel:
                    if input_image.shape[0] <= 1:
                        raise CellProfilerInputValidationError(f"Expected a 5D array with at least 2 c channels, got {input_image.shape[0]}")
                    
            else: # 2d image. 
                if is_multi_channel: # color
                    if input_image.ndim != 3:
                        raise CellProfilerInputValidationError(f"Expected a 3D array (cyx),got {input_image.ndim}D")
                else: # grayscale
                    if input_image.ndim != 2:
                        raise CellProfilerInputValidationError(f"Expected a 2D array (yx),got {input_image.ndim}D")
        if get_origin(dtype) is Union:
            if input_image.dtype not in get_args(dtype):
                raise CellProfilerInputValidationError(f"Expected an array of type {dtype}, got {input_image.dtype}")
        elif input_image.dtype != dtype:
            raise CellProfilerInputValidationError(f"Expected an array of type {dtype}, got {input_image.dtype}")
        if is_tiled:
            pass
        return input_image
    return validate

def validate_object_labels_dense(input_image: npt.NDArray[np.generic]) -> NDArray[np.generic]:
    if input_image.ndim != 6:
        raise ValueError(f"Expected ObjectLabelsDense as an array of shape (n, c, z, t, y, x), got {input_image.shape}")
    return input_image

def validate_object_label_set(label_set: Sequence[Tuple[NDArray[np.int32], NDArray[np.int32]]]) -> Sequence[Tuple[NDArray[np.int32], NDArray[np.int32]]]:
    # label set is a list of 2 tuples
    for label in label_set:
        if type(label) != tuple or len(label) != 2:
            raise ValueError(f"Expected a list of tuples of length 2, got {label}")
        if type(label[0]) != np.ndarray or type(label[1]) != np.ndarray:
            raise ValueError(f"Expected a list of tuples of ndarrays, got {label}")
        if len(label[0].shape) > 3:
            # see cellprofiler_library.functions.segmentation._validate_labels
            raise ValueError(f"Expected labels of shape (y, x) or (z, y, x), got {label[0].shape}")
    return label_set

Pixel = Annotated[Union[np.float32, np.float64], Field(description="Pixel value")]
ObjectLabel =           Annotated[Union[np.int8, np.int16,np.int32], Field(description="Object label")]

Image2DColor =          Annotated[NDArray[Pixel],       Field(description="2D image with multiple channels of type float32"), AfterValidator(create_type_validator(False, True, False, Union[np.float32, np.float64]))]
Image2DColorMask =      Annotated[NDArray[np.bool_],    Field(description="2D color mask"),        AfterValidator(create_type_validator(False, True, False, np.bool_))]
Image2DGrayscale =      Annotated[NDArray[Pixel],       Field(description="2D grayscale image of type float32"), AfterValidator(create_type_validator(False, False, False, Union[np.float32, np.float64]))]
Image2DGrayscaleMask =  Annotated[NDArray[np.bool_],    Field(description="2D grayscale mask"),    AfterValidator(create_type_validator(False, False, False, np.bool_))]
Image2DBinary =         Annotated[NDArray[np.bool_],    Field(description="2D binary image"),      AfterValidator(create_type_validator(False, False, False, np.bool_))]
Image2DBinaryMask =     Annotated[NDArray[np.bool_],    Field(description="2D binary mask"),       AfterValidator(create_type_validator(False, False, False, np.bool_))]
Image2DInt =            Annotated[NDArray[np.int32],    Field(description="2D int32 image"),       AfterValidator(create_type_validator(False, False, False, np.int32))]

Image3DColor =          Annotated[NDArray[Pixel],       Field(description="3D image with multiple channels of type float32"), AfterValidator(create_type_validator(True, True, False, Union[np.float32, np.float64]))]
Image3DColorMask =      Annotated[NDArray[np.bool_],    Field(description="3D image with multiple channels of type float32"), AfterValidator(create_type_validator(True, True, False, np.bool_))]
Image3DGrayscale =      Annotated[NDArray[Pixel],       Field(description="3D grayscale image of type float32"), AfterValidator(create_type_validator(True, False, False, Union[np.float32, np.float64]))]
Image3DGrayscaleMask =  Annotated[NDArray[np.bool_],    Field(description="3D grayscale mask"),    AfterValidator(create_type_validator(True, False, False, np.bool_))]
Image3DBinary =         Annotated[NDArray[np.bool_],    Field(description="3D binary image"),      AfterValidator(create_type_validator(True, False, False, np.bool_))]
Image3DBinaryMask =     Annotated[NDArray[np.bool_],    Field(description="3D binary mask"),       AfterValidator(create_type_validator(True, False, False, np.bool_))]
Image3DInt =            Annotated[NDArray[np.int32],    Field(description="3D int32 image"),       AfterValidator(create_type_validator(True, False, False, np.int32))]

# see cellprofiler_library.functions.segmentation._validate_<type> for more details 
ObjectLabelsDense =     Annotated[NDArray[ObjectLabel], Field(description="Dense array of object labels"), AfterValidator(validate_object_labels_dense)]
ObjectLabelSet =        Annotated[Sequence[Tuple[NDArray[ObjectLabel], NDArray[np.int32]]], Field(description="List of Tuples of object labels and object numbers in each label matrix"), AfterValidator(validate_object_label_set)]
ObjectSegmentation =    Annotated[NDArray[ObjectLabel], Field(description="Object segmentation")]

ImageGrayscale =        Union[Image2DGrayscale, Image3DGrayscale]
ImageGrayscaleMask =    Union[Image2DGrayscaleMask, Image3DGrayscaleMask]

ImageColor =            Union[Image2DColor, Image3DColor]
ImageColorMask =        Union[Image2DColorMask, Image3DColorMask]

ImageBinary =           Union[Image2DBinary, Image3DBinary]
ImageBinaryMask =       Union[Image2DBinaryMask, Image3DBinaryMask]

ImageAny =              Union[Image2DColor, Image3DColor, Image2DGrayscale, Image3DGrayscale, Image2DBinary, Image3DBinary, Image2DInt, Image3DInt]
ImageAnyMask =          Union[Image2DColorMask, Image3DColorMask, Image2DGrayscaleMask, Image3DGrayscaleMask, Image2DBinaryMask, Image3DBinaryMask]

Image2D =               Union[Image2DColor, Image2DGrayscale, Image2DBinary]
Image2DMask =           Union[Image2DColorMask, Image2DGrayscaleMask]

Image3D =               Union[Image3DColor, Image3DGrayscale, Image3DBinary]
Image3DMask =           Union[Image3DColorMask, Image3DGrayscaleMask]

ImageInt =              Union[Image2DInt, Image3DInt]
