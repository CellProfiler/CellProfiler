from typing import Optional, Annotated, Union, Tuple, Dict, Any
import numpy
import numpy as np
import scipy.ndimage
import scipy.sparse
import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
from pydantic import validate_call, ConfigDict, Field
from numpy.typing import NDArray

from cellprofiler_library.measurement_model import LibraryMeasurements
from cellprofiler_library.functions.measurement import (
    compute_normalized_distance,
    compute_radial_intensity_distribution,
    compute_radial_cv,
    compute_zernike_measurements,
)
from cellprofiler_library.opts.measureobjectintensitydistribution import (
    ALL_TEMPLATE_MEASUREMENT_FEATURES,
    ALL_TEMPLATE_OVERFLOW_FEATURES,
    CenterChoice,
    Feature,
    IntensityZernike,
    TemplateMeasurementFeature,
    TemplateOverflowFeature,
    TemplateZernikeFeature,
)

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_intensity_distribution(
    pixel_data:         Annotated[NDArray[np.float64], Field(description="Image pixel data (grayscale)")],
    labels:             Annotated[NDArray[np.int32], Field(description="Object label matrix")],
    image_name:         Annotated[str, Field(description="Name of the image")],
    object_name:        Annotated[str, Field(description="Name of the objects")],
    bin_count:          Annotated[int, Field(description="Number of radial bins", ge=2)],
    wants_scaled:       Annotated[bool, Field(description="Scale bins by object size?")] = True,
    maximum_radius:     Annotated[int, Field(description="Maximum radius for unscaled bins", ge=1)] = 100,
    center_object_name: Annotated[Optional[str], Field(description="Name of centering objects")] = None,
    center_labels:      Annotated[Optional[NDArray[np.int32]], Field(description="Labels of centering objects")] = None,
    center_choice:      Annotated[str, Field(description="Center choice (C_SELF, C_CENTERS_OF_OTHER, C_EDGES_OF_OTHER)")] = CenterChoice.SELF.value,
    return_heatmap_data: Annotated[bool, Field(description="Return heatmap data for display")] = False,
) -> Union[LibraryMeasurements, Tuple[LibraryMeasurements, Dict[str, Any]]]:
    """
    Measure object intensity distribution.
    
    Computes radial distribution of intensities from object centers to edges,
    optionally including Zernike moments.
    
    Args:
        pixel_data: Image pixel intensities (H, W)
        labels: Object label matrix (H, W)
        image_name: Name of the image for measurement keys
        object_name: Name of the objects for measurement keys
        bin_count: Number of radial bins
        wants_scaled: If True, scale bins by object size; if False, use fixed radius
        maximum_radius: Maximum radius in pixels (for unscaled bins)
        center_object_name: Name of centering objects (if using centers of other objects)
        center_labels: Label matrix of centering objects (if using centers of other objects)
        center_choice: One of "These objects", "Centers of other objects", "Edges of other objects"
        return_heatmap_data: If True, return heatmap data for display
        
    Returns:
        LibraryMeasurements if return_heatmap_data=False, else
        Tuple[LibraryMeasurements, heatmap_data]
    """
    measurements = LibraryMeasurements()
    
    nobjects = numpy.max(labels)
    
    # Handle empty case
    if nobjects == 0:
        for bin_index in range(1, bin_count + 1):
            for feature in ALL_TEMPLATE_MEASUREMENT_FEATURES:
                feature_name = feature % (image_name, bin_index, bin_count)
                measurements.add_measurement(object_name, feature_name, numpy.zeros(0))
        
        if not wants_scaled:
            for feature in ALL_TEMPLATE_OVERFLOW_FEATURES:
                feature_name = feature % image_name
                measurements.add_measurement(object_name, feature_name, numpy.zeros(0))
        
        if return_heatmap_data:
            return measurements, {}
        return measurements
    
    # Compute centers and distances
    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    #
    # Use the center of the centering objects to assign a center
    # to each labeled pixel using propagation
    #
    if center_object_name is not None and center_labels is not None:
        #
        # Use the center of the centering objects to assign a center
        # to each labeled pixel using propagation
        #
        pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.sum(
                numpy.ones(center_labels.shape),
                center_labels,
                numpy.arange(1, numpy.max(center_labels) + 1, dtype=numpy.int32),
            )
        )
        
        good = pixel_counts > 0
        i, j = (centrosome.cpmorphology.centers_of_labels(center_labels) + 0.5).astype(int)
        ig = i[good]
        jg = j[good]
        lg = numpy.arange(1, len(i) + 1)[good]
        
        if center_choice == CenterChoice.CENTERS_OF_OTHER.value:
            #
            # Reduce the propagation labels to the centers of
            # the centering objects
            #
            center_labels_reduced = numpy.zeros(center_labels.shape, int)
            center_labels_reduced[ig, jg] = lg
            center_labels = center_labels_reduced
        
        cl, d_from_center = centrosome.propagate.propagate(
            numpy.zeros(center_labels.shape), center_labels, labels != 0, 1
        )
        
        #
        # Erase the centers that fall outside of labels
        #
        cl[labels == 0] = 0
        
        #
        # If objects are hollow or crescent-shaped, there may be
        # objects without center labels. As a backup, find the
        # center that is the closest to the center of mass.
        #
        missing_mask = (labels != 0) & (cl == 0)
        missing_labels = numpy.unique(labels[missing_mask])
        
        if len(missing_labels):
            all_centers = centrosome.cpmorphology.centers_of_labels(labels)
            missing_i_centers, missing_j_centers = all_centers[:, missing_labels - 1]
            
            di = missing_i_centers[:, numpy.newaxis] - ig[numpy.newaxis, :]
            dj = missing_j_centers[:, numpy.newaxis] - jg[numpy.newaxis, :]
            missing_best = lg[numpy.argsort(di * di + dj * dj)[:, 0]]
            
            best = numpy.zeros(numpy.max(labels) + 1, int)
            best[missing_labels] = missing_best
            cl[missing_mask] = best[labels[missing_mask]]
            
            #
            # Now compute the crow-flies distance to the centers
            # of these pixels from whatever center was assigned to
            # the object.
            #
            iii, jjj = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
            di = iii[missing_mask] - i[cl[missing_mask] - 1]
            dj = jjj[missing_mask] - j[cl[missing_mask] - 1]
            d_from_center[missing_mask] = numpy.sqrt(di * di + dj * dj)
    else:
        #
        # Find the point in each object farthest away from the edge.
        # This does better than the centroid:
        # * The center is within the object
        # * The center tends to be an interesting point, like the
        #   center of the nucleus or the center of one or the other
        #   of two touching cells.
        #
        i, j = centrosome.cpmorphology.maximum_position_of_labels(
            d_to_edge, labels, numpy.arange(1, nobjects + 1)
        )
        
        center_labels = numpy.zeros(labels.shape, int)
        center_labels[i, j] = labels[i, j]
        
        #
        # Use the coloring trick here to process touching objects
        # in separate operations
        #
        colors = centrosome.cpmorphology.color_labels(labels)
        ncolors = numpy.max(colors)
        
        d_from_center = numpy.zeros(labels.shape)
        cl = numpy.zeros(labels.shape, int)
        
        for color in range(1, ncolors + 1):
            mask = colors == color
            l, d = centrosome.propagate.propagate(
                numpy.zeros(center_labels.shape), center_labels, mask, 1
            )
            d_from_center[mask] = d[mask]
            cl[mask] = l[mask]
    
    good_mask = cl > 0
    
    if center_choice == CenterChoice.EDGES_OF_OTHER.value and center_labels is not None:
        # Exclude pixels within the centering objects
        # when performing calculations from the centers
        good_mask = good_mask & (center_labels == 0)
    
    i_center = numpy.zeros(cl.shape)
    i_center[good_mask] = i[cl[good_mask] - 1]
    
    j_center = numpy.zeros(cl.shape)
    j_center[good_mask] = j[cl[good_mask] - 1]
    
    # Compute normalized distance
    normalized_distance = compute_normalized_distance(
        labels, d_from_center, d_to_edge, wants_scaled, maximum_radius, good_mask
    )
    
    # Compute bin indexes
    bin_indexes = (normalized_distance * bin_count).astype(int)
    bin_indexes[bin_indexes > bin_count] = bin_count
    
    # Compute radial distributions
    fraction_at_distance, mean_pixel_fraction, number_at_distance = \
        compute_radial_intensity_distribution(
            pixel_data, labels, good_mask, bin_indexes, nobjects, bin_count
        )
    
    # Prepare heatmap data if requested
    heatmaps = {}
    if return_heatmap_data:
        heatmaps[Feature.FRAC_AT_D.value] = numpy.zeros(labels.shape)
        heatmaps[Feature.MEAN_FRAC.value] = numpy.zeros(labels.shape)
        heatmaps[Feature.RADIAL_CV.value] = numpy.zeros(labels.shape)
    
    # Process each bin
    for bin in range(bin_count + (0 if wants_scaled else 1)):
        # Compute radial CV
        radial_cv = compute_radial_cv(
            pixel_data, labels, good_mask, bin_indexes, bin,
            i_center, j_center, nobjects
        )
        
        # Add measurements
        if bin == bin_count:
            # Overflow bin
            frac_name = TemplateOverflowFeature.FRAC_AT_D % image_name
            mean_name = TemplateOverflowFeature.MEAN_FRAC % image_name
            cv_name = TemplateOverflowFeature.RADIAL_CV % image_name
        else:
            # Regular bin
            frac_name = TemplateMeasurementFeature.FRAC_AT_D % (image_name, bin + 1, bin_count)
            mean_name = TemplateMeasurementFeature.MEAN_FRAC % (image_name, bin + 1, bin_count)
            cv_name = TemplateMeasurementFeature.RADIAL_CV % (image_name, bin + 1, bin_count)
        
        measurements.add_measurement(object_name, frac_name, fraction_at_distance[:, bin])
        measurements.add_measurement(object_name, mean_name, mean_pixel_fraction[:, bin])
        measurements.add_measurement(object_name, cv_name, radial_cv)
        
        # Update heatmaps
        if return_heatmap_data:
            bin_mask = good_mask & (bin_indexes == bin)
            bin_labels = labels[bin_mask]
            
            heatmaps[Feature.FRAC_AT_D.value][bin_mask] = fraction_at_distance[:, bin][bin_labels - 1]
            heatmaps[Feature.MEAN_FRAC.value][bin_mask] = mean_pixel_fraction[:, bin][bin_labels - 1]
            heatmaps[Feature.RADIAL_CV.value][bin_mask] = radial_cv[bin_labels - 1]
    
    if return_heatmap_data:
        return measurements, heatmaps
    return measurements


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_zernike_features(
    pixel_data:     Annotated[NDArray[np.float64], Field(description="Image pixel data (grayscale)")],
    image_mask:     Annotated[NDArray[np.bool_], Field(description="Image mask")],
    objects_ijv:    Annotated[NDArray, Field(description="Objects in IJV format")],
    object_count:   Annotated[int, Field(description="Number of objects")],
    object_indices: Annotated[NDArray[np.int32], Field(description="Object indices")],
    image_name:     Annotated[str, Field(description="Name of the image")],
    object_name:    Annotated[str, Field(description="Name of the objects")],
    zernike_degree: Annotated[int, Field(description="Maximum Zernike degree", ge=1, le=20)] = 9,
    wants_phase:    Annotated[bool, Field(description="Calculate phase?")] = False,
) -> LibraryMeasurements:
    """
    Measure Zernike features for object intensity distribution.
    
    Args:
        pixel_data: Image pixel intensities
        image_mask: Image mask
        objects_ijv: Object coordinates in IJV format
        object_count: Number of objects
        object_indices: Array of object indices
        image_name: Name of the image for measurement keys
        object_name: Name of the objects for measurement keys
        zernike_degree: Maximum Zernike degree to compute
        wants_phase: If True, also compute phase
        
    Returns:
        LibraryMeasurements with Zernike features
    """
    measurements = LibraryMeasurements()
    
    if object_count == 0:
        zernike_indexes = centrosome.zernike.get_zernike_indexes(zernike_degree + 1)
        for n, m in zernike_indexes:
            mag_name = TemplateZernikeFeature.MAGNITUDE % (image_name, n, m)
            measurements.add_measurement(object_name, mag_name, numpy.zeros(0))
            
            if wants_phase:
                phase_name = TemplateZernikeFeature.PHASE % (image_name, n, m)
                measurements.add_measurement(object_name, phase_name, numpy.zeros(0))
        
        return measurements
    
    # Compute Zernike measurements
    magnitudes, phases, zernike_indexes = compute_zernike_measurements(
        objects_ijv, pixel_data, image_mask, object_count, object_indices, zernike_degree
    )
    
    # Add measurements
    for idx, (n, m) in enumerate(zernike_indexes):
        mag_name = TemplateZernikeFeature.MAGNITUDE % (image_name, n, m)
        measurements.add_measurement(object_name, mag_name, magnitudes[:, idx])
        
        if wants_phase:
            phase_name = TemplateZernikeFeature.PHASE % (image_name, n, m)
            measurements.add_measurement(object_name, phase_name, phases[:, idx])
    
    return measurements
