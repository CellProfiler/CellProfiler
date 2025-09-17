from typing import Literal, Annotated, Optional
import numpy
import skimage
import scipy
import centrosome
import centrosome.propagate
from pydantic import validate_call, ConfigDict, Field
from ..types import Image2DGrayscale, Image2DGrayscaleMask, ObjectSegmentation
from cellprofiler_library.modules import threshold
from cellprofiler_library.functions.object_processing import filter_labels
from cellprofiler_library.opts.identifysecondaryobjects import SecondaryObjectMethod
import cellprofiler_library.opts.threshold as ThresholdOpts

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def identifysecondaryobjects(
    image:                      Annotated[Image2DGrayscale, Field(description="Input image")],
    objects:                    Annotated[ObjectSegmentation, Field(description="Object segmentations")],
    unedited_objects:           Annotated[Optional[ObjectSegmentation], Field(description="Unedited object segmentations")] = None,
    mask:                       Annotated[Optional[Image2DGrayscaleMask], Field(description="Input mask")] = None,
    secondary_object_method:    Annotated[SecondaryObjectMethod , Field(description="Method to determine edges of secondary objects")] = SecondaryObjectMethod.PROPAGATION,
    threshold_method:           Annotated[str, Field(description="Thresholding method")] = ThresholdOpts.Method.MINIMUM_CROSS_ENTROPY, #TODO: change type to enum from threshold module
    threshold_scope:            Annotated[str, Field(description="Thresholding scope")] = ThresholdOpts.Scope.GLOBAL, #TODO: change type to enum from threshold module
    assign_middle_to_foreground:Annotated[str, Field(description="Assign middle to foreground")] = ThresholdOpts.Assignment.BACKGROUND, #TODO: change type to enum from threshold module
    log_transform:              Annotated[bool, Field(description="Apply log transform to image before thresholding")] = False,
    threshold_correction_factor:Annotated[float, Field(description="Multiply threshold by this factor")] = 1.0,
    threshold_min:              Annotated[float, Field(description="Minimum threshold value")] = 0.0,
    threshold_max:              Annotated[float, Field(description="Maximum threshold value")] = 1.0,
    window_size:                Annotated[int, Field(description="Size of window for thresholding")] = 50,
    threshold_smoothing:        Annotated[float, Field(description="Smoothing factor for thresholding")] = 0.0,
    lower_outlier_fraction:     Annotated[float, Field(description="Fraction of pixels to use for lower outlier detection")] = 0.05,
    upper_outlier_fraction:     Annotated[float, Field(description="Fraction of pixels to use for upper outlier detection")] = 0.05,
    averaging_method:           Annotated[str, Field(description="Averaging method for thresholding")] = ThresholdOpts.AveragingMethod.MEAN, #TODO: change type to enum from threshold module
    variance_method:            Annotated[str, Field(description="Variance method for thresholding")] = ThresholdOpts.VarianceMethod.STANDARD_DEVIATION, #TODO: change type to enum from threshold module
    number_of_deviations:       Annotated[int, Field(description="Number of deviations for thresholding")] = 2,
    predefined_threshold:       Annotated[Optional[float], Field(description="Predefined threshold value")] = None,
    distance_to_dilate:         Annotated[int, Field(description="Number of pixels by which to expand the primary objects")] = 10,
    fill_holes:                 Annotated[bool, Field(description="Fill holes in identified objects?")] = True,
    discard_edge:               Annotated[bool, Field(description="Discard objects touching the edge of the image")] = False,
    regularization_factor:      Annotated[float, Field(description="Regularization factor")] = 0.05,
    return_cp_output:           Annotated[bool, Field(description="Return CellProfiler output")] = False,
):
    if image.shape != objects.shape:
        raise ValueError(
            f"""
            The input image shape {image.shape} does not match the input objects
            shape {objects.shape} If they are paired correctly you may want to
            use the Resize, ResizeObjects or Crop module(s) to make them the
            same size.
        """
        )

    if secondary_object_method != SecondaryObjectMethod.DISTANCE_N:
        (
            final_threshold,
            orig_threshold,
            guide_threshold,
            binary_image, # thresholded_image
            sigma,
        ) = threshold(
            image=image,
            mask=mask,
            threshold_scope=threshold_scope,
            threshold_method=threshold_method,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            threshold_correction_factor=threshold_correction_factor,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            window_size=window_size,
            smoothing=threshold_smoothing,
            lower_outlier_fraction=lower_outlier_fraction,
            upper_outlier_fraction=upper_outlier_fraction,
            averaging_method=averaging_method,
            variance_method=variance_method,
            number_of_deviations=number_of_deviations,
            volumetric=False,  # IDSecondary does not support 3D
            predefined_threshold=predefined_threshold,
        )
    else:
        final_threshold = None
        orig_threshold = None
        guide_threshold = None
        binary_image = None
        sigma = None

    # Input primary objects that will be altered on edge discard
    segmented_labels = objects

    #
    # Get the following labels:
    # * all edited labels
    # * labels touching the edge, including small removed
    #
    if unedited_objects is None:
        labels_in = objects
    else:
        labels_in = unedited_objects.copy()
    labels_touching_edge = numpy.hstack(
        (labels_in[0, :], labels_in[-1, :], labels_in[:, 0], labels_in[:, -1])
    )
    labels_touching_edge = numpy.unique(labels_touching_edge)
    is_touching = numpy.zeros(numpy.max(labels_in) + 1, bool)
    is_touching[labels_touching_edge] = True
    is_touching = is_touching[labels_in]

    labels_in[(~is_touching) & (objects == 0)] = 0
    #
    # Stretch the input labels to match the image size. If there's no
    # label matrix, then there's no label in that area.
    #
    if tuple(labels_in.shape) != tuple(image.shape):
        tmp = numpy.zeros(image.shape, labels_in.dtype)
        i_max = min(image.shape[0], labels_in.shape[0])
        j_max = min(image.shape[1], labels_in.shape[1])
        tmp[:i_max, :j_max] = labels_in[:i_max, :j_max]
        labels_in = tmp

    if secondary_object_method in (SecondaryObjectMethod.DISTANCE_B, SecondaryObjectMethod.DISTANCE_N):
        if secondary_object_method == SecondaryObjectMethod.DISTANCE_N:
            distances, (i, j) = scipy.ndimage.distance_transform_edt(
                labels_in == 0, return_indices=True
            )
            labels_out = numpy.zeros(labels_in.shape, int)
            dilate_mask = distances <= distance_to_dilate
            labels_out[dilate_mask] = labels_in[i[dilate_mask], j[dilate_mask]] 
        else:
            labels_out, distances = centrosome.propagate.propagate(
                image, labels_in, binary_image, 1.0
            )
            labels_out[distances > distance_to_dilate] = 0
            labels_out[labels_in > 0] = labels_in[labels_in > 0]
        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out
        #
        # Create the final output labels by removing labels in the
        # output matrix that are missing from the segmented image
        #

        segmented_out = filter_labels(
            small_removed_segmented_out, objects, mask=mask, discard_edge=discard_edge
        )
    elif secondary_object_method == SecondaryObjectMethod.PROPAGATION:
        labels_out, distance = centrosome.propagate.propagate(
            image, labels_in, binary_image, regularization_factor
        )
        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out.copy()
        segmented_out = filter_labels(
            small_removed_segmented_out, objects, mask=mask, discard_edge=discard_edge
        )
    elif secondary_object_method == SecondaryObjectMethod.WATERSHED_GRADIENT:
        #
        # First, apply the sobel filter to the image (both horizontal
        # and vertical). The filter measures gradient.
        #
        sobel_image = numpy.abs(scipy.ndimage.sobel(image))
        #
        # Combine the image mask and threshold to mask the watershed
        #
        watershed_mask = numpy.logical_or(binary_image, labels_in > 0)
        watershed_mask = numpy.logical_and(watershed_mask, mask)

        #
        # Perform the first watershed
        #
        labels_out = skimage.segmentation.watershed(
            connectivity=numpy.ones((3, 3), bool),
            image=sobel_image,
            markers=labels_in,
            mask=watershed_mask,
        )
        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out.copy()
        segmented_out = filter_labels(
            small_removed_segmented_out, objects, mask=mask, discard_edge=discard_edge
        )
    elif secondary_object_method == SecondaryObjectMethod.WATERSHED_IMAGE:
        #
        # invert the image so that the maxima are filled first
        # and the cells compete over what's close to the threshold
        #
        inverted_img = 1 - image
        #
        # Same as above, but perform the watershed on the original image
        #
        watershed_mask = numpy.logical_or(binary_image, labels_in > 0)
        watershed_mask = numpy.logical_and(watershed_mask, mask)
        #
        # Perform the watershed
        #
        labels_out = skimage.segmentation.watershed(
            connectivity=numpy.ones((3, 3), bool),
            image=inverted_img,
            markers=labels_in,
            mask=watershed_mask,
        )
        if fill_holes:
            label_mask = labels_out == 0
            small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                labels_out, mask=label_mask
            )
        else:
            small_removed_segmented_out = labels_out
        segmented_out = filter_labels(
            small_removed_segmented_out, objects, mask=mask, discard_edge=discard_edge
        )
    else:
        raise ValueError(f"Method {secondary_object_method} is not supported")
    
    if discard_edge:
        lookup = scipy.ndimage.maximum(
            segmented_out,
            objects,
            list(range(numpy.max(objects) + 1)),
        )
        lookup = centrosome.cpmorphology.fixup_scipy_ndimage_result(lookup)
        lookup[0] = 0
        lookup[lookup != 0] = numpy.arange(numpy.sum(lookup != 0)) + 1
        segmented_labels = lookup[objects]
        segmented_out = lookup[segmented_out]

    if return_cp_output:
        return (
            final_threshold,
            orig_threshold,
            guide_threshold,
            binary_image,
            sigma, 
            # Input primary objects, but if discard_edge==True, will be primary
            # objects that have been discarded due to edge touching secondary
            # objects following edge discard
            segmented_labels,
            segmented_out, # The actual output objects
            small_removed_segmented_out
        )
    else:
        return segmented_out
