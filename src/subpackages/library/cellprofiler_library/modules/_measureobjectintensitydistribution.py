from typing import Annotated, Any, Dict, List, Optional, Tuple

import centrosome.cpmorphology
import centrosome.propagate
import centrosome.zernike
import numpy
import numpy.ma
import scipy.ndimage
import scipy.sparse
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, validate_call

from cellprofiler_library.functions.object_processing import size_similarly
from cellprofiler_library.opts.measureobjectintensitydistribution import (
    C_RADIAL_DISTRIBUTION,
    CenterChoice,
    Feature,
    FF_GENERIC,
    FF_OVERFLOW,
    F_ALL,
    TemplateMeasurementFormat,
)
from cellprofiler_library.types import ImageGrayscale, ImageGrayscaleMask


def get_zernike_magnitude_name(image_name, n, m):
    return "_".join(
        (
            C_RADIAL_DISTRIBUTION,
            Feature.ZERNIKE_MAGNITUDE.value,
            image_name,
            str(n),
            str(m),
        )
    )


def get_zernike_phase_name(image_name, n, m):
    return "_".join(
        (
            C_RADIAL_DISTRIBUTION,
            Feature.ZERNIKE_PHASE.value,
            image_name,
            str(n),
            str(m),
        )
    )


def record_empty_object_measurements(image_name, object_name, bin_count, wants_scaled):
    measurement_pairs = []

    for bin_index in range(1, bin_count + 1):
        for feature in F_ALL:
            feature_name = (feature + FF_GENERIC) % (
                image_name,
                bin_index,
                bin_count,
            )

            measurement_pairs.append(
                (
                    "_".join([C_RADIAL_DISTRIBUTION, feature_name]),
                    numpy.zeros(0),
                )
            )

            if not wants_scaled:
                overflow_name = "_".join(
                    [C_RADIAL_DISTRIBUTION, feature, image_name, FF_OVERFLOW]
                )

                measurement_pairs.append((overflow_name, numpy.zeros(0)))

    stats_row = (image_name, object_name, "no objects", "-", "-", "-", "-")

    return stats_row, measurement_pairs


def compute_center_distances(
    labels,
    objects_indices,
    center_objects_segmented,
    center_choice,
    wants_scaled,
    maximum_radius,
):
    d_to_edge = centrosome.cpmorphology.distance_to_edge(labels)

    if center_objects_segmented is not None:
        #
        # Use the center of the centering objects to assign a center
        # to each labeled pixel using propagation
        #
        center_labels, cmask = size_similarly(labels, center_objects_segmented)

        pixel_counts = centrosome.cpmorphology.fixup_scipy_ndimage_result(
            scipy.ndimage.sum(
                numpy.ones(center_labels.shape),
                center_labels,
                numpy.arange(
                    1, numpy.max(center_labels) + 1, dtype=numpy.int32
                ),
            )
        )

        good = pixel_counts > 0

        i, j = (
            centrosome.cpmorphology.centers_of_labels(center_labels) + 0.5
        ).astype(int)

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

            missing_i_centers, missing_j_centers = all_centers[
                :, missing_labels - 1
            ]

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
        # Find the point in each object farthest away from the edge.
        # This does better than the centroid:
        # * The center is within the object
        # * The center tends to be an interesting point, like the
        #   center of the nucleus or the center of one or the other
        #   of two touching cells.
        #
        i, j = centrosome.cpmorphology.maximum_position_of_labels(
            d_to_edge, labels, objects_indices
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

    if center_choice == CenterChoice.EDGES_OF_OTHER.value:
        # Exclude pixels within the centering objects
        # when performing calculations from the centers
        good_mask = good_mask & (center_labels == 0)

    i_center = numpy.zeros(cl.shape)

    i_center[good_mask] = i[cl[good_mask] - 1]

    j_center = numpy.zeros(cl.shape)

    j_center[good_mask] = j[cl[good_mask] - 1]

    normalized_distance = numpy.zeros(labels.shape)

    if wants_scaled:
        total_distance = d_from_center + d_to_edge

        normalized_distance[good_mask] = d_from_center[good_mask] / (
            total_distance[good_mask] + 0.001
        )
    else:
        normalized_distance[good_mask] = (
            d_from_center[good_mask] / maximum_radius
        )

    return normalized_distance, i_center, j_center, good_mask


def compute_per_bin_distributions(
    labels, pixel_data, good_mask, normalized_distance, bin_count, nobjects
):
    ngood_pixels = numpy.sum(good_mask)

    good_labels = labels[good_mask]

    bin_indexes = (normalized_distance * bin_count).astype(int)

    bin_indexes[bin_indexes > bin_count] = bin_count

    labels_and_bins = (good_labels - 1, bin_indexes[good_mask])

    histogram = scipy.sparse.coo_matrix(
        (pixel_data[good_mask], labels_and_bins), (nobjects, bin_count + 1)
    ).toarray()

    sum_by_object = numpy.sum(histogram, 1)

    sum_by_object_per_bin = numpy.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_distance = histogram / sum_by_object_per_bin

    number_at_distance = scipy.sparse.coo_matrix(
        (numpy.ones(ngood_pixels), labels_and_bins), (nobjects, bin_count + 1)
    ).toarray()

    object_mask = number_at_distance > 0

    sum_by_object = numpy.sum(number_at_distance, 1)

    sum_by_object_per_bin = numpy.dstack([sum_by_object] * (bin_count + 1))[0]

    fraction_at_bin = number_at_distance / sum_by_object_per_bin

    mean_pixel_fraction = fraction_at_distance / (
        fraction_at_bin + numpy.finfo(float).eps
    )

    masked_fraction_at_distance = numpy.ma.masked_array(
        fraction_at_distance, ~object_mask
    )

    masked_mean_pixel_fraction = numpy.ma.masked_array(
        mean_pixel_fraction, ~object_mask
    )

    return (
        bin_indexes,
        fraction_at_distance,
        mean_pixel_fraction,
        masked_fraction_at_distance,
        masked_mean_pixel_fraction,
    )


def compute_radial_indexes(labels, i_center, j_center, good_mask):
    # Anisotropy calculation.  Split each cell into eight wedges, then
    # compute coefficient of variation of the wedges' mean intensities
    # in each ring.
    #
    # Compute each pixel's delta from the center object's centroid
    i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]

    imask = i[good_mask] > i_center[good_mask]

    jmask = j[good_mask] > j_center[good_mask]

    absmask = abs(i[good_mask] - i_center[good_mask]) > abs(
        j[good_mask] - j_center[good_mask]
    )

    radial_index = (
        imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
    )

    return radial_index


def record_bin_measurements(
    image_name,
    object_name,
    bin_count,
    wants_scaled,
    labels,
    pixel_data,
    nobjects,
    normalized_distance,
    i_center,
    j_center,
    good_mask,
    heatmaps,
):
    (
        bin_indexes,
        fraction_at_distance,
        mean_pixel_fraction,
        masked_fraction_at_distance,
        masked_mean_pixel_fraction,
    ) = compute_per_bin_distributions(
        labels, pixel_data, good_mask, normalized_distance, bin_count, nobjects
    )

    radial_index = compute_radial_indexes(labels, i_center, j_center, good_mask)

    statistics = []
    measurement_pairs = []

    for bin in range(bin_count + (0 if wants_scaled else 1)):
        bin_mask = good_mask & (bin_indexes == bin)

        bin_pixels = numpy.sum(bin_mask)

        bin_labels = labels[bin_mask]

        bin_radial_index = radial_index[bin_indexes[good_mask] == bin]

        labels_and_radii = (bin_labels - 1, bin_radial_index)

        radial_values = scipy.sparse.coo_matrix(
            (pixel_data[bin_mask], labels_and_radii), (nobjects, 8)
        ).toarray()

        pixel_count = scipy.sparse.coo_matrix(
            (numpy.ones(bin_pixels), labels_and_radii), (nobjects, 8)
        ).toarray()

        mask = pixel_count == 0

        radial_means = numpy.ma.masked_array(radial_values / pixel_count, mask)

        radial_cv = numpy.std(radial_means, 1) / numpy.mean(radial_means, 1)

        radial_cv[numpy.sum(~mask, 1) == 0] = 0

        for measurement, feature, overflow_feature in (
            (
                fraction_at_distance[:, bin],
                TemplateMeasurementFormat.RD_FRAC_AT_D,
                TemplateMeasurementFormat.RD_OVERFLOW_FRAC_AT_D,
            ),
            (
                mean_pixel_fraction[:, bin],
                TemplateMeasurementFormat.RD_MEAN_FRAC,
                TemplateMeasurementFormat.RD_OVERFLOW_MEAN_FRAC,
            ),
            (
                numpy.array(radial_cv),
                TemplateMeasurementFormat.RD_RADIAL_CV,
                TemplateMeasurementFormat.RD_OVERFLOW_RADIAL_CV,
            ),
        ):
            if bin == bin_count:
                measurement_name = overflow_feature % image_name
            else:
                measurement_name = feature % (image_name, bin + 1, bin_count)

            measurement_pairs.append((measurement_name, measurement))

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

    return statistics, measurement_pairs


def compute_minimum_enclosing_circles(label_indexes_pairs, n_objects):
    #
    # First, get a table of centers and radii of minimum enclosing
    # circles per object
    #
    ij = numpy.zeros((n_objects + 1, 2))

    r = numpy.zeros(n_objects + 1)

    for labels, indexes in label_indexes_pairs:
        ij_, r_ = centrosome.cpmorphology.minimum_enclosing_circle(
            labels, indexes
        )

        ij[indexes] = ij_

        r[indexes] = r_

    return ij, r


def calculate_zernikes_for_image(
    image_name,
    indices,
    pixels,
    image_mask,
    ijv,
    l,
    z,
    zernike_indexes,
    wants_phase,
):
    mask = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])

    mask[mask] = image_mask[ijv[mask, 0], ijv[mask, 1]]

    l_ = l[mask]

    z_ = z[mask, :]

    measurement_pairs = []

    if len(l_) == 0:
        for i, (n, m) in enumerate(zernike_indexes):
            measurement_pairs.append(
                (get_zernike_magnitude_name(image_name, n, m), numpy.zeros(0))
            )

            if wants_phase:
                measurement_pairs.append(
                    (get_zernike_phase_name(image_name, n, m), numpy.zeros(0))
                )

        return measurement_pairs

    areas = scipy.ndimage.sum(
        numpy.ones(l_.shape, int), labels=l_, index=indices
    )

    for i, (n, m) in enumerate(zernike_indexes):
        vr = scipy.ndimage.sum(
            pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].real,
            labels=l_,
            index=indices,
        )

        vi = scipy.ndimage.sum(
            pixels[ijv[mask, 0], ijv[mask, 1]] * z_[:, i].imag,
            labels=l_,
            index=indices,
        )

        magnitude = numpy.sqrt(vr * vr + vi * vi) / areas

        measurement_pairs.append(
            (get_zernike_magnitude_name(image_name, n, m), magnitude)
        )

        if wants_phase:
            phase = numpy.arctan2(vr, vi)

            measurement_pairs.append(
                (get_zernike_phase_name(image_name, n, m), phase)
            )

    return measurement_pairs


CenterDistanceCache = Tuple[
    NDArray[numpy.float_],
    NDArray[numpy.float_],
    NDArray[numpy.float_],
    NDArray[numpy.bool_],
]

StatsRow = Tuple[str, str, str, str, Any, Any, Any]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measureobjectintensitydistribution(
    pixel_data: Annotated[
        ImageGrayscale,
        Field(description="Pixel array for the image, cropped to the object label matrix."),
    ],
    image_name: Annotated[
        str, Field(description="Name of the image being measured.")
    ],
    object_labels: Annotated[
        NDArray[numpy.int_],
        Field(
            description="2D label matrix of objects to measure, cropped to the pixel array shape."
        ),
    ],
    object_name: Annotated[
        str, Field(description="Name of the object set being measured.")
    ],
    objects_indices: Annotated[
        NDArray[numpy.int_],
        Field(description="1D array of object label indices present in object_labels."),
    ],
    center_choice: Annotated[
        CenterChoice,
        Field(description="How to choose object centers: SELF, CENTERS_OF_OTHER, or EDGES_OF_OTHER."),
    ],
    bin_count: Annotated[
        int, Field(description="Number of radial bins.", ge=1)
    ],
    wants_scaled: Annotated[
        bool,
        Field(
            description="If True, scale bins to each object's radius. If False, use fixed-width bins up to maximum_radius."
        ),
    ],
    maximum_radius: Annotated[
        int,
        Field(
            description="Maximum radius (in pixels) for unscaled binning. Beyond this distance, pixels are counted in an overflow bin.",
            ge=1,
        ),
    ],
    center_object_labels: Annotated[
        Optional[NDArray[numpy.int_]],
        Field(
            description="2D label matrix of centering objects (for CENTERS_OF_OTHER / EDGES_OF_OTHER). None when using SELF."
        ),
    ] = None,
    heatmap_feature_templates: Annotated[
        Optional[List[str]],
        Field(
            description="List of TemplateMeasurementFormat values (e.g. RD_FRAC_AT_D) for which to populate heatmap arrays."
        ),
    ] = None,
    cached_center_distances: Annotated[
        Optional[CenterDistanceCache],
        Field(
            description="Pre-computed (normalized_distance, i_center, j_center, good_mask) tuple to skip distance recomputation across repeated calls for the same object set."
        ),
    ] = None,
) -> Tuple[
    List[StatsRow],
    List[Tuple[str, NDArray]],
    Dict[str, NDArray[numpy.float_]],
    Optional[CenterDistanceCache],
]:
    """Compute the radial intensity distribution for one (image, object, bin_count) triple.

    Returns a 4-tuple:
      - statistics: display rows, one per bin (+ overflow if unscaled).
      - measurement_pairs: list of (feature_name, value_array) to be added to
        the measurements store under `object_name`.
      - heatmap_arrays: dict keyed by the requested feature templates,
        mapping each to a 2D heatmap array (zeros if no objects).
      - center_distances: the cache tuple to feed back as
        `cached_center_distances` on the next call sharing this
        (object, center) pair.
    """
    if heatmap_feature_templates is None:
        heatmap_feature_templates = []

    heatmap_arrays = {
        template: numpy.zeros(object_labels.shape)
        for template in heatmap_feature_templates
    }

    nobjects = int(numpy.max(object_labels))

    if nobjects == 0:
        stats_row, measurement_pairs = record_empty_object_measurements(
            image_name, object_name, bin_count, wants_scaled
        )
        return [stats_row], measurement_pairs, heatmap_arrays, cached_center_distances

    if cached_center_distances is None:
        cached_center_distances = compute_center_distances(
            object_labels,
            objects_indices,
            center_object_labels,
            center_choice,
            wants_scaled,
            maximum_radius,
        )

    normalized_distance, i_center, j_center, good_mask = cached_center_distances

    statistics, measurement_pairs = record_bin_measurements(
        image_name,
        object_name,
        bin_count,
        wants_scaled,
        object_labels,
        pixel_data,
        nobjects,
        normalized_distance,
        i_center,
        j_center,
        good_mask,
        heatmap_arrays,
    )

    return statistics, measurement_pairs, heatmap_arrays, cached_center_distances
