from cellprofiler.library.functions.object_processing import (
    watershed_distance, watershed_markers, watershed_advanced
)
import skimage

def watershed(
    input_image,
    use_advanced=False,
    markers=None,
    mask=None,
    intensity_image=None,
    method="distance",
    declump_method="shape",
    footprint=8,
    downsample=1,
    connectivity=1,
    compactness=0,
    watershed_line=False,
    structuring_element="disk",
    structuring_element_size=1,
    gaussian_sigma=1,
    min_distance=1,
    min_intensity=0,
    exclude_border=0,
    max_seeds=-1
    ):
    # Make sure all input arrays are of equal shape
    inputs = [input_image, markers, mask, intensity_image]
    inputs = [i for i in inputs if i is not None]
    if not all(arr.shape==inputs[0].shape for arr in inputs):
        shapes = [i.shape for i in inputs]
        raise ValueError(f"""Shape of input arrays do not match:
        {shapes}""")

    if not use_advanced:
        if method.casefold() == "distance":
            y_data = watershed_distance(
                input_image,
                footprint=footprint,
                downsample=downsample
            )
        elif method.casefold() == "markers":
            y_data = watershed_markers(
                input_image,
                markers=markers,
                mask=mask,
                connectivity=connectivity,
                compactness=compactness,
                watershed_line=watershed_line
            )
        else:
            raise NotImplementedError(f"""Watershed method {method} does not exist""")
    else:
        y_data = watershed_advanced(
            input_image=input_image,
            markers=markers,
            mask=mask,
            intensity_image=intensity_image,
            method=method,
            declump_method=declump_method,
            footprint=footprint,
            downsample=downsample,
            connectivity=connectivity,
            compactness=compactness,
            watershed_line=watershed_line,
            structuring_element=structuring_element,
            structuring_element_size=structuring_element_size,
            gaussian_sigma=gaussian_sigma,
            min_distance=min_distance,
            min_intensity=min_intensity,
            exclude_border=exclude_border,
            max_seeds=max_seeds
        )

    # finalize and convert watershed to objects to export
    y_data = skimage.measure.label(y_data)

    return y_data