from typing import Literal
import numpy
import skimage
import scipy
import centrosome
import centrosome.propagate

from cellprofiler_library.modules import threshold
from cellprofiler_library.functions.object_processing import filter_labels


def identifysecondaryobjects(
    image: numpy.ndarray,
    objects: numpy.ndarray,
    unedited_objects: numpy.ndarray = None,
    mask: numpy.ndarray = None,
    secondary_object_method: Literal[
        "propagation",
        "watershed_gradient",
        "watershed_image",
        "distance_n",
        "distance_b",
    ] = "propagation",
    threshold_method: Literal[
        "minimum_cross_entropy", "otsu", "multiotsu", "robust_background"
    ] = "minimum_cross_entropy",
    threshold_scope: Literal["global", "adaptive"] = "global",
    assign_middle_to_foreground: Literal["foreground", "background"] = "background",
    log_transform: bool = False,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    window_size: int = 50,
    threshold_smoothing: float = 0.0,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: Literal["mean", "median", "mode"] = "mean",
    variance_method: Literal[
        "standard_deviation", "median_absolute_deviation"
    ] = "standard_deviation",
    number_of_deviations: int = 2,
    predefined_threshold: float = None,
    distance_to_dilate: int = 10,
    fill_holes: bool = True,
    discard_edge: bool = False,
    regularization_factor: float = 0.05,
    return_cp_output: bool = False,
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

    if secondary_object_method.casefold() != "distance_n":
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

    if secondary_object_method.casefold() in ("distance_b", "distance_n"):
        if secondary_object_method.casefold() == "distance_n":
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
    elif secondary_object_method.casefold() == "propagation":
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
    elif secondary_object_method.casefold() == "watershed_gradient":
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
    elif secondary_object_method.casefold() == "watershed_image":
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
