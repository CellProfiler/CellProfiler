from typing import Annotated, Any, Dict, List, Optional, Tuple

import numpy
import numpy.ma
import scipy.ndimage
import scipy.sparse
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, validate_call

from cellprofiler_library.functions.measurement import (
    compute_center_distances as _compute_center_distances,
    compute_minimum_enclosing_circles,
    compute_per_bin_distributions,
    compute_radial_indexes,
)
from cellprofiler_library.measurement_model import LibraryMeasurements
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


def _center_choice_to_flags(center_choice):
    """Translate a CenterChoice setting value into the two booleans that
    the pure-math ``compute_center_distances`` helper expects."""
    return (
        center_choice == CenterChoice.CENTERS_OF_OTHER.value,
        center_choice == CenterChoice.EDGES_OF_OTHER.value,
    )


def compute_center_distances(
    labels,
    objects_indices,
    center_objects_segmented,
    center_choice,
    wants_scaled,
    maximum_radius,
):
    """Layer-2 wrapper around the pure-math ``compute_center_distances``.

    Translates the ``CenterChoice`` enum value into the two boolean flags
    consumed by the helper, so callers (including the frontend) can keep
    passing the human-readable setting value.
    """
    use_centers_of_other, use_edges_of_other = _center_choice_to_flags(
        center_choice
    )
    return _compute_center_distances(
        labels,
        objects_indices,
        center_objects_segmented,
        use_centers_of_other,
        use_edges_of_other,
        wants_scaled,
        maximum_radius,
    )


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


def calculate_zernikes_for_image(
    image_name,
    object_name,
    indices,
    pixels,
    image_mask,
    ijv,
    l,
    z,
    zernike_indexes,
    wants_phase,
):
    lib_measurements = LibraryMeasurements()

    mask = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])

    mask[mask] = image_mask[ijv[mask, 0], ijv[mask, 1]]

    l_ = l[mask]

    z_ = z[mask, :]

    if len(l_) == 0:
        for i, (n, m) in enumerate(zernike_indexes):
            lib_measurements.add_measurement(
                object_name,
                get_zernike_magnitude_name(image_name, n, m),
                numpy.zeros(0),
            )

            if wants_phase:
                lib_measurements.add_measurement(
                    object_name,
                    get_zernike_phase_name(image_name, n, m),
                    numpy.zeros(0),
                )

        return lib_measurements

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

        lib_measurements.add_measurement(
            object_name,
            get_zernike_magnitude_name(image_name, n, m),
            magnitude,
        )

        if wants_phase:
            phase = numpy.arctan2(vr, vi)

            lib_measurements.add_measurement(
                object_name,
                get_zernike_phase_name(image_name, n, m),
                phase,
            )

    return lib_measurements


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
    LibraryMeasurements,
    List[StatsRow],
    Dict[str, NDArray[numpy.float_]],
    Optional[CenterDistanceCache],
]:
    """Compute the radial intensity distribution for one (image, object, bin_count) triple.

    Returns a 4-tuple:
      - lib_measurements: ``LibraryMeasurements`` with per-object measurements
        keyed under ``object_name``.
      - statistics: display rows, one per bin (+ overflow if unscaled).
      - heatmap_arrays: dict keyed by the requested feature templates,
        mapping each to a 2D heatmap array (zeros if no objects).
      - center_distances: the cache tuple to feed back as
        ``cached_center_distances`` on the next call sharing this
        (object, center) pair.
    """
    if heatmap_feature_templates is None:
        heatmap_feature_templates = []

    heatmap_arrays = {
        template: numpy.zeros(object_labels.shape)
        for template in heatmap_feature_templates
    }

    lib_measurements = LibraryMeasurements()

    nobjects = int(numpy.max(object_labels))

    if nobjects == 0:
        stats_row, measurement_pairs = record_empty_object_measurements(
            image_name, object_name, bin_count, wants_scaled
        )
        for feature_name, value in measurement_pairs:
            lib_measurements.add_measurement(object_name, feature_name, value)
        return (
            lib_measurements,
            [stats_row],
            heatmap_arrays,
            cached_center_distances,
        )

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

    for feature_name, value in measurement_pairs:
        lib_measurements.add_measurement(object_name, feature_name, value)

    return lib_measurements, statistics, heatmap_arrays, cached_center_distances
