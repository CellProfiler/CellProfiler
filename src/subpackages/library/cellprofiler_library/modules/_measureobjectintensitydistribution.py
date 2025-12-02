import numpy
import scipy.ndimage
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import scipy.sparse
from cellprofiler_library.functions.segmentation import convert_label_set_to_ijv, count_from_ijv, indices_from_ijv
from cellprofiler_library.functions.object_processing import size_similarly
from cellprofiler_library.opts.measureobjectintensitydistribution import (
    CenterChoice,
    IntensityZernike,
    Feature, 
    FullFeature,
    MeasurementFeature,
    OverflowFeature,
    MeasurementAlias,
    C_ALL,
    Z_ALL,
    M_CATEGORY,
    F_ALL,
    MEASUREMENT_CHOICES, 
    MEASUREMENT_ALIASES,
    FF_SCALE,
    FF_GENERIC
)


def assign_centers_using_centering_objects(
        center_labels,
        center_choice, 
        labels
    ):
    pixel_counts = fix(scipy.ndimage.sum(
        numpy.ones(center_labels.shape),
        center_labels,
        numpy.arange(1, numpy.max(center_labels) + 1, dtype=numpy.int32),
    ))

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
        center_labels = numpy.zeros(center_labels.shape, int)

        center_labels[ig, jg] = lg

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

    return (i, j), d_from_center, cl

# TODO: Is this the right name?
def assign_centers_automatically(
        d_to_edge, 
        labels, 
        objects_indices
    ):
    #
    # Find the point in each object farthest away from the edge.
    # This does better than the centroid:
    # * The center is within the object
    # * The center tends to be an interesting point, like the
    #   center of the nucleus or the center of one or the other
    #   of two touching cells.
    #
    i, j = centrosome.cpmorphology.maximum_position_of_labels(d_to_edge, labels, objects_indices)

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
        l, d = centrosome.propagate.propagate(numpy.zeros(center_labels.shape), center_labels, mask, 1)
        d_from_center[mask] = d[mask]
        cl[mask] = l[mask]

    return (i, j), d_from_center, cl

# TODO: Find a better name for all functions
def get_normalized_distance(
        good_mask, 
        d_from_center, 
        labels, 
        wants_scaled, 
        d_to_edge, 
        maximum_radius
    ):
    normalized_distance = numpy.zeros(labels.shape)

    if wants_scaled:
        total_distance = d_from_center + d_to_edge
        normalized_distance[good_mask] = d_from_center[good_mask] / (total_distance[good_mask] + 0.001)
    else:
        normalized_distance[good_mask] = d_from_center[good_mask] / maximum_radius

    return normalized_distance

def get_fraction_at_distance(
        histogram, 
        bin_count
    ):
    sum_by_object = numpy.sum(histogram, 1)
    sum_by_object_per_bin = numpy.dstack([sum_by_object] * (bin_count + 1))[0]
    fraction_at_distance = histogram / sum_by_object_per_bin
    return fraction_at_distance

def get_fraction_at_bin(
        number_at_distance, 
        bin_count
    ):
    return get_fraction_at_distance(number_at_distance, bin_count)

def get_radial_index(
        labels, 
        good_mask, 
        i_center, 
        j_center
    ):
    # Anisotropy calculation.  Split each cell into eight wedges, then
    # compute coefficient of variation of the wedges' mean intensities
    # in each ring.
    #
    # Compute each pixel's delta from the center object's centroid
    i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
    imask = i[good_mask] > i_center[good_mask]
    jmask = j[good_mask] > j_center[good_mask]
    absmask = abs(i[good_mask] - i_center[good_mask]) > abs(j[good_mask] - j_center[good_mask])

    radial_index = (imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4)
    return radial_index

def get_bin_measurements(
        good_mask, 
        labels, 
        normalized_distance, 
        bin_count, 
        pixel_data, 
        nobjects
    ):
    ngood_pixels = numpy.sum(good_mask)
    good_labels = labels[good_mask]

    bin_indexes = (normalized_distance * bin_count).astype(int)
    bin_indexes[bin_indexes > bin_count] = bin_count

    labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

    histogram = scipy.sparse.coo_matrix((pixel_data[good_mask], labels_and_bins), (nobjects, bin_count + 1)).toarray()
    number_at_distance = scipy.sparse.coo_matrix((numpy.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count + 1)).toarray()
    object_mask = number_at_distance > 0

    fraction_at_bin = get_fraction_at_bin(number_at_distance, bin_count)
    fraction_at_distance = get_fraction_at_distance(histogram, bin_count)
    mean_pixel_fraction = fraction_at_distance / (fraction_at_bin + numpy.finfo(float).eps)

    masked_fraction_at_distance = numpy.ma.masked_array(fraction_at_distance, ~object_mask)
    masked_mean_pixel_fraction = numpy.ma.masked_array(mean_pixel_fraction, ~object_mask)

    return bin_indexes, fraction_at_distance, mean_pixel_fraction, masked_fraction_at_distance, masked_mean_pixel_fraction

def get_radial_cv(
        good_mask, 
        bin_indexes, 
        labels, 
        radial_index, 
        pixel_data, 
        nobjects, 
        bin
    ):
    bin_mask = good_mask & (bin_indexes == bin)
    bin_pixels = numpy.sum(bin_mask)
    bin_labels = labels[bin_mask]

    bin_radial_index = radial_index[bin_indexes[good_mask] == bin]
    labels_and_radii = (bin_labels - 1, bin_radial_index)

    radial_values = scipy.sparse.coo_matrix((pixel_data[bin_mask], labels_and_radii), (nobjects, 8)).toarray()
    pixel_count = scipy.sparse.coo_matrix((numpy.ones(bin_pixels), labels_and_radii), (nobjects, 8)).toarray()

    mask = pixel_count == 0
    radial_means = numpy.ma.masked_array(radial_values / pixel_count, mask)
    radial_cv = numpy.std(radial_means, 1) / numpy.mean(radial_means, 1)
    radial_cv[numpy.sum(~mask, 1) == 0] = 0

    return bin_mask, bin_labels, radial_cv, mask

def get_i_j_centers(
        cl, 
        good_mask, 
        i,
        j
    ):
    i_center = numpy.zeros(cl.shape)
    j_center = numpy.zeros(cl.shape)
    i_center[good_mask] = i[cl[good_mask] - 1]
    j_center[good_mask] = j[cl[good_mask] - 1]
    return i_center, j_center

def get_normalized_distance_centers_and_good_mask(
        labels, 
        center_object_segmentations, 
        objects_indices, 
        center_choice, 
        wants_scaled, 
        maximum_radius
    ):

    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    if center_object_segmentations is not None:
        #
        # Use the center of the centering objects to assign a center
        # to each labeled pixel using propagation
        #
        center_labels, _ = size_similarly(labels, center_object_segmentations)
        (i, j), d_from_center, cl = assign_centers_using_centering_objects(center_labels, center_choice, labels)
    else:
        (i, j), d_from_center, cl = assign_centers_automatically(d_to_edge, labels, objects_indices)

    good_mask = cl > 0
    if center_choice == CenterChoice.EDGES_OF_OTHER.value:
        # Exclude pixels within the centering objects
        # when performing calculations from the centers
        good_mask = good_mask & (center_labels == 0)

    i_center, j_center = get_i_j_centers(cl, good_mask, i, j)
    normalized_distance = get_normalized_distance(good_mask, d_from_center, labels, wants_scaled, d_to_edge, maximum_radius)

    return normalized_distance, i_center, j_center, good_mask



def get_positions_within_unit_circle(
        objects_label_set, 
        zernike_indexes, 
        objects_ijv=None
    ):
    #
    # First, get a table of centers and radii of minimum enclosing
    # circles per object
    #
    if objects_ijv is None:
        objects_ijv = convert_label_set_to_ijv(objects_label_set, validate=True)
    objects_count = count_from_ijv(objects_ijv, validate=False)
    ij = numpy.zeros((objects_count + 1, 2))

    r = numpy.zeros(objects_count + 1)

    for labels, indexes in objects_label_set:
        ij_, r_ = centrosome.cpmorphology.minimum_enclosing_circle(
            labels, indexes
        )

        ij[indexes] = ij_

        r[indexes] = r_

    #
    # Then compute x and y, the position of each labeled pixel
    # within a unit circle around the object
    #

    l = objects_ijv[:, 2]

    yx = (objects_ijv[:, :2] - ij[l, :]) / r[l, numpy.newaxis]

    z = centrosome.zernike.construct_zernike_polynomials(
        yx[:, 1], yx[:, 0], zernike_indexes
    )
    return l, yx, z

def calculate_zernikes_for_image(
        image_pixel_data, 
        image_mask, 
        image_name,
        ijv, 
        objects_indices,
        zernike_opts,
        zernike_indexes,
        l, 
        yx, 
        z, 
        objects_labels = None, 
    ):
    if l is None or yx is None or z is None:
        assert objects_labels is not None
        l, yx, z = get_positions_within_unit_circle(objects_labels, zernike_indexes)
    measurements_dict_for_image = {}
    pixels = image_pixel_data
    mask = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])
    mask[mask] = image_mask[ijv[mask, 0], ijv[mask, 1]]

    yx_ = yx[mask, :]
    l_ = l[mask]
    z_ = z[mask, :]

    if len(l_) == 0:
        for i, (n, m) in enumerate(zernike_indexes):
            ftr = get_zernike_magnitude_name(image_name, n, m)

            measurements_dict_for_image[ftr] = numpy.zeros(0)

            if zernike_opts == IntensityZernike.MAGNITUDES_AND_PHASE.value:
                ftr = get_zernike_phase_name(image_name, n, m)

                measurements_dict_for_image[ftr] = numpy.zeros(0)

    else:
        areas = scipy.ndimage.sum(
            numpy.ones(l_.shape, int), labels=l_, index=objects_indices
        )

        for i, (n, m) in enumerate(zernike_indexes):
            vr = scipy.ndimage.sum(
                pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].real,
                labels=l_,
                index=objects_indices,
            )

            vi = scipy.ndimage.sum(
                pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].imag,
                labels=l_,
                index=objects_indices,
            )

            magnitude = numpy.sqrt(vr * vr + vi * vi) / areas

            ftr = get_zernike_magnitude_name(image_name, n, m)

            measurements_dict_for_image[ftr] = magnitude

            if zernike_opts == IntensityZernike.MAGNITUDES_AND_PHASE.value:
                phase = numpy.arctan2(vr, vi)

                ftr = get_zernike_phase_name(image_name, n, m)

                measurements_dict_for_image[ftr] = phase
    return measurements_dict_for_image

def calculate_object_intensiry_zernikes(objects_names_and_label_sets, zernike_degree, image_name_data_mask_list, zernike_opts):
    measurements_dict = {}
    zernike_indexes = centrosome.zernike.get_zernike_indexes(
            zernike_degree + 1
        )
    for object_name, objects_labels in objects_names_and_label_sets:
        measurements_dict_for_image = {} if object_name not in measurements_dict else measurements_dict[object_name]
        objects_ijv = convert_label_set_to_ijv(objects_labels)
        objects_indices = indices_from_ijv(objects_ijv)
        
        l, yx, z = get_positions_within_unit_circle(objects_labels, zernike_indexes)
        for image_name, image_pixel_data, image_mask in image_name_data_mask_list:
            measurements_dict_for_image = calculate_zernikes_for_image(
                image_pixel_data, 
                image_mask, 
                image_name,
                objects_ijv, 
                objects_indices,
                zernike_opts,
                zernike_indexes,
                l, 
                yx, 
                z, 
            )
            measurements_dict[object_name] = measurements_dict_for_image
    return measurements_dict

def get_zernike_magnitude_name(image_name, n, m):
    """The feature name of the magnitude of a Zernike moment

    image_name - the name of the image being measured
    n - the radial moment of the Zernike
    m - the azimuthal moment of the Zernike
    """
    return "_".join((M_CATEGORY, FullFeature.ZERNIKE_MAGNITUDE.value, image_name, str(n), str(m)))

def get_zernike_phase_name(image_name, n, m):
    """The feature name of the phase of a Zernike moment

    image_name - the name of the image being measured
    n - the radial moment of the Zernike
    m - the azimuthal moment of the Zernike
    """
    return "_".join((M_CATEGORY, FullFeature.ZERNIKE_PHASE.value, image_name, str(n), str(m)))

def get_object_intensity_distribution_measurements(
        object_name, 
        center_object_name, 
        heatmap_dict, 
        labels, # these are obtained after `cropping similarly` to the image
        center_object_segmented, 
        center_choice,
        wants_scaled, 
        maximum_radius,
        bin_count,
        pixel_data,
        nobjects,
        image_name,
        heatmaps,
        objects_indices
    ):
    name = (
        object_name
        if center_object_name is None
        else "{}_{}".format(object_name, center_object_name)
    )
    # objects_ijv = convert_label_set_to_ijv(labels)
    # objects_indices = indices_from_ijv(objects_ijv)

    if name in heatmap_dict:
        normalized_distance, i_center, j_center, good_mask = heatmap_dict[name]
    else:
        if center_object_name is not None:
            center_object_segmentations = center_object_segmented
        else:
            center_object_segmentations = None

        normalized_distance, i_center, j_center, good_mask = get_normalized_distance_centers_and_good_mask(labels, center_object_segmentations, objects_indices, center_choice, wants_scaled, maximum_radius)
        heatmap_dict[name] = [normalized_distance, i_center, j_center, good_mask]
    (
        bin_indexes, 
        fraction_at_distance, 
        mean_pixel_fraction, 
        masked_fraction_at_distance, 
        masked_mean_pixel_fraction, 
    ) = get_bin_measurements(good_mask, labels, normalized_distance, bin_count, pixel_data, nobjects)
    
    radial_index = get_radial_index(labels, good_mask, i_center, j_center)

    statistics = []
    measurements = []

    for bin in range(bin_count + (0 if wants_scaled else 1)):
        (
            bin_mask, 
            bin_labels, 
            radial_cv, 
            mask
        ) = get_radial_cv(good_mask, bin_indexes, labels, radial_index, pixel_data, nobjects, bin)

        for measurement, feature, overflow_feature in (
            (fraction_at_distance[:, bin], MeasurementFeature.FRAC_AT_D.value, OverflowFeature.FRAC_AT_D.value),
            (mean_pixel_fraction[:, bin], MeasurementFeature.MEAN_FRAC.value, OverflowFeature.MEAN_FRAC.value),
            (numpy.array(radial_cv), MeasurementFeature.RADIAL_CV.value, OverflowFeature.RADIAL_CV.value),
        ):
            if bin == bin_count:
                measurement_name = overflow_feature % image_name
            else:
                measurement_name = feature % (image_name, bin + 1, bin_count)

            measurements.append([object_name, measurement_name, measurement])

            if feature in heatmaps:
                heatmaps[feature][bin_mask] = measurement[bin_labels - 1]

        radial_cv.mask = numpy.sum(~mask, 1) == 0

        bin_name = str(bin + 1) if bin < bin_count else "Overflow"

        statistics += [
            (
                image_name,
                object_name,
                bin_name,
                str(bin_count),
                numpy.round(numpy.mean(masked_fraction_at_distance[:, bin]), 4),
                numpy.round(numpy.mean(masked_mean_pixel_fraction[:, bin]), 4),
                numpy.round(numpy.mean(radial_cv), 4),
            )
        ]

    return statistics, measurements
