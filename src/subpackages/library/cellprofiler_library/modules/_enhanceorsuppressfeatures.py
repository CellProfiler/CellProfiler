import numpy
import skimage
from cellprofiler_library.functions.image_processing import enhance_speckles, enhance_neurites, enhance_circles, enhance_texture, enhance_dark_holes, enhance_dic, suppress
from pydantic import Field, ConfigDict, validate_call
from typing import Annotated, Optional
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask
from ..opts.enhanceorsuppressfeatures import OperationMethod, EnhanceMethod, SpeckleAccuracy, NeuriteMethod
from numpy.typing import NDArray

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def enhance_or_suppress_features(
        im_pixel_data:          Annotated[ImageGrayscale, Field(description="Image pixel data")],  
        im_mask:                Annotated[ImageGrayscaleMask, Field(description="Image mask")],
        im_volumetric:          Annotated[bool, Field(description="Image is volumetric")] = False,
        im_spacing:             Annotated[tuple[float, ...], Field(description="Image spacing")] = (1.0, 1.0, 1.0),
        radius:                 Annotated[float, Field(description="Feature size")] = 10,
        method:                 Annotated[OperationMethod, Field(description="Operation method")] = OperationMethod.ENHANCE,
        enhance_method:         Annotated[EnhanceMethod, Field(description="Feature type")] = EnhanceMethod.SPECKLES,
        speckle_accuracy:       Annotated[SpeckleAccuracy, Field(description="Speed and accuracy")] = SpeckleAccuracy.FAST,
        neurite_choice:         Annotated[NeuriteMethod, Field(description="Neurite choice")] = NeuriteMethod.GRADIENT,
        neurite_rescale:        Annotated[bool, Field(description="Rescale result image")] = False,
        dark_hole_radius_min:   Annotated[int, Field(description="Dark hole radius min")] = 1,
        dark_hole_radius_max:   Annotated[int, Field(description="Dark hole radius max")] = 10,
        smoothing_value:        Annotated[float, Field(description="Smoothing value")] = 2.0,
        dic_angle:              Annotated[float, Field(description="Angle")] = 0.0,
        dic_decay:              Annotated[float, Field(description="Decay")] = 0.95,
        ):
    if method == OperationMethod.ENHANCE:
        if enhance_method == EnhanceMethod.SPECKLES:
            result = enhance_speckles(im_pixel_data, im_mask, im_volumetric, radius, speckle_accuracy)

        elif enhance_method == EnhanceMethod.NEURITES:
            result = enhance_neurites(im_pixel_data, im_mask, im_volumetric, im_spacing, smoothing_value, radius, neurite_choice, neurite_rescale)
            
        elif enhance_method == EnhanceMethod.DARK_HOLES:
            result = enhance_dark_holes(im_pixel_data, im_mask, im_volumetric, dark_hole_radius_min, dark_hole_radius_max)

        elif enhance_method == EnhanceMethod.CIRCLES:
            result = enhance_circles(im_pixel_data, im_mask, im_volumetric, radius)

        elif enhance_method == EnhanceMethod.TEXTURE:
            result = enhance_texture(im_pixel_data, im_mask, smoothing_value)

        elif enhance_method == EnhanceMethod.DIC:
            result = enhance_dic(im_pixel_data, im_volumetric, dic_angle, dic_decay, smoothing_value)
        
        else:
            raise NotImplementedError("Unimplemented enhance method: %s" % enhance_method)
        
    elif method == OperationMethod.SUPPRESS:
        result = suppress(im_pixel_data, im_mask, im_volumetric, radius)

    else:
        raise ValueError("Unknown filtering method: %s" % method)
    
    return result


