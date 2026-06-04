from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

import numpy
import scipy.ndimage
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, validate_call

from cellprofiler_library.functions.measurement import (
    compute_center_distances,
    compute_per_bin_distributions,
    compute_radial_cv_for_bin,
    compute_radial_indexes,
    prepare_object_zernike_polynomials, # passthrough import
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
from cellprofiler_library.types import Image2DBinaryMask, Image2DGrayscale, ObjectSegmentation, ObjectIndices, ObjectLabelMask, ObjectSegmentationIJV, ObjectLabel


def get_zernike_magnitude_name(image_name: str, n: int, m: int):
    return "_".join(
        (
            C_RADIAL_DISTRIBUTION,
            Feature.ZERNIKE_MAGNITUDE.value,
            image_name,
            str(n),
            str(m),
        )
    )

def get_zernike_phase_name(image_name: str, n: int, m: int):
    return "_".join(
        (
            C_RADIAL_DISTRIBUTION,
            Feature.ZERNIKE_PHASE.value,
            image_name,
            str(n),
            str(m),
        )
    )

def record_bin_measurements(
    image_name: str,
    object_name: str,
    bin_count: int,
    wants_scaled: bool,
    labels: ObjectSegmentation,
    pixel_data: Image2DGrayscale,
    nobjects: int,
    normalized_distance: NDArray[numpy.float_],
    i_center: NDArray[numpy.float_],
    j_center: NDArray[numpy.float_],
    good_mask: ObjectLabelMask,
    heatmaps: List[TemplateMeasurementFormat],
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

    statistics: MeasureObjectIntensityDistributionStatistics = []
    measurement_pairs: List[Tuple[str, NDArray[numpy.float_]]] = []
    heatmap_arrays: Dict[TemplateMeasurementFormat, NDArray[numpy.float_]] = {
        template: numpy.zeros(labels.shape)
        for template in heatmaps
    }

    for nbin in range(bin_count + (0 if wants_scaled else 1)):
        # per-object radial CV across 8 wedges for this bin.
        bin_mask, bin_labels, radial_cv, empty_object_mask = compute_radial_cv_for_bin(
            nbin, labels, pixel_data, good_mask, bin_indexes, radial_index, nobjects,
        )

        # Feature/measurement consolidation: name each per-bin feature, append
        # to the measurement pairs, and update heatmap arrays in place.
        for measurement, feature, overflow_feature in (
            (
                fraction_at_distance[:, nbin],
                TemplateMeasurementFormat(TemplateMeasurementFormat.RD_FRAC_AT_D),
                TemplateMeasurementFormat.RD_OVERFLOW_FRAC_AT_D,
            ),
            (
                mean_pixel_fraction[:, nbin],
                TemplateMeasurementFormat(TemplateMeasurementFormat.RD_MEAN_FRAC),
                TemplateMeasurementFormat.RD_OVERFLOW_MEAN_FRAC,
            ),
            (
                numpy.array(radial_cv),
                TemplateMeasurementFormat(TemplateMeasurementFormat.RD_RADIAL_CV),
                TemplateMeasurementFormat.RD_OVERFLOW_RADIAL_CV,
            ),
            (
                numpy.full(nobjects, nbin + 1),
                TemplateMeasurementFormat(TemplateMeasurementFormat.RD_BIN_NUM),
                TemplateMeasurementFormat.RD_OVERFLOW_BIN_NUM,
            ),
        ):
            if nbin == bin_count:
                measurement_name = overflow_feature % image_name
            else:
                measurement_name = feature % (image_name, nbin + 1, bin_count)

            measurement_pairs.append((measurement_name, measurement))

            if feature in heatmaps:
                heatmap_arrays[feature][bin_mask] = measurement[bin_labels - 1]

        # Mask empty-object CVs before averaging into the stats table so the
        # mean isn't biased by zero placeholders.
        radial_cv.mask = empty_object_mask

        bin_name = str(nbin + 1) if nbin < bin_count else "Overflow"

        statistics += [
            (
                image_name,
                object_name,
                bin_name,
                str(bin_count),
                numpy.round(numpy.mean(masked_fraction_at_distance[:, nbin]), 4),
                numpy.round(numpy.mean(masked_mean_pixel_fraction[:, nbin]), 4),
                numpy.round(numpy.mean(radial_cv), 4),
            )
        ]

    return statistics, measurement_pairs, heatmap_arrays


def calculate_zernikes_for_image(
    image_name: str,
    object_name: str,
    indices: ObjectIndices,
    pixels: Image2DGrayscale,
    image_mask: Image2DBinaryMask,
    ijv: ObjectSegmentationIJV,
    label_vec: NDArray[ObjectLabel],
    zernike_polynomials: NDArray[numpy.complex128],
    zernike_indexes: NDArray[numpy.int32],
    wants_phase: bool,
):
    lib_measurements = LibraryMeasurements()

    mask = (ijv[:, 0] < pixels.shape[0]) & (ijv[:, 1] < pixels.shape[1])

    mask[mask] = image_mask[ijv[mask, 0], ijv[mask, 1]]

    label_vec_ = label_vec[mask]

    zernike_polynomials_ = zernike_polynomials[mask, :]

    if len(label_vec_) == 0:
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
        numpy.ones(label_vec_.shape, int), labels=label_vec_, index=indices
    )

    for i, (n, m) in enumerate(zernike_indexes):
        vr = scipy.ndimage.sum(
            pixels[ijv[mask, 0], ijv[mask, 1]] * zernike_polynomials_[:, i].real,
            labels=label_vec_,
            index=indices,
        )

        vi = scipy.ndimage.sum(
            pixels[ijv[mask, 0], ijv[mask, 1]] * zernike_polynomials_[:, i].imag,
            labels=label_vec_,
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
    # normalized_distance
    NDArray[numpy.float_],
    # i_center
    NDArray[numpy.float_],
    # j_center
    NDArray[numpy.float_],
    # good_mask
    NDArray[numpy.bool_],
]

StatsRow = Tuple[str, str, str, str, Any, Any, Any]
MeasureObjectIntensityDistributionStatistics = List[StatsRow]


class MeasureObjectIntensityDistributionDisplayData(BaseModel):
    """Display-data bundle produced when ``return_visualization_data=True``.

    Holds the per-bin statistics table (one row per bin, plus an overflow
    row when unscaled) and the populated heatmap arrays keyed by the
    feature template strings that were requested via
    ``heatmap_feature_templates``.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True
    )

    statistics: MeasureObjectIntensityDistributionStatistics
    heatmap_arrays: Dict[TemplateMeasurementFormat, NDArray[numpy.float_]]


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measureobjectintensitydistribution(
    pixel_data: Annotated[
        Image2DGrayscale,
        Field(description="Pixel array for the image, cropped to the object label matrix."),
    ],
    image_name: Annotated[
        str, Field(description="Name of the image being measured.")
    ],
    object_labels: Annotated[
        ObjectSegmentation,
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
        Optional[ObjectSegmentation],
        Field(
            description="2D label matrix of centering objects (for CENTERS_OF_OTHER / EDGES_OF_OTHER). None when using SELF."
        ),
    ] = None,
    heatmap_feature_templates: Annotated[
        Optional[List[TemplateMeasurementFormat]],
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
    return_visualization_data: Annotated[
        bool,
        Field(
            description="If True, also return a MeasureObjectIntensityDistributionDisplayData with the statistics table and heatmap arrays."
        ),
    ] = False,
) -> Union[
    Tuple[LibraryMeasurements, Optional[CenterDistanceCache]],
    Tuple[
        LibraryMeasurements,
        MeasureObjectIntensityDistributionDisplayData,
        Optional[CenterDistanceCache],
    ],
]:
    """Compute the radial intensity distribution for one (image, object, bin_count) triple.

    Returns:
      - When ``return_visualization_data`` is False (default):
        ``(lib_measurements, cached_center_distances)``.
      - When True: ``(lib_measurements, display_data, cached_center_distances)``
        where ``display_data`` holds the per-bin statistics table and the
        populated heatmap arrays.

    The cache tuple is always returned so the caller can feed it back into
    the next call sharing the same (object, center) pair.
    """
    if heatmap_feature_templates is None:
        heatmap_feature_templates = []

    lib_measurements = LibraryMeasurements()

    nobjects = int(numpy.max(object_labels))

    if nobjects == 0:
        stats_row = (image_name, object_name, "no objects", "-", "-", "-", "-")

        measurement_pairs: List[Tuple[str, NDArray[numpy.float_]]] = []
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

        for feature_name, value in measurement_pairs:
            lib_measurements.add_measurement(object_name, feature_name, value)

        if return_visualization_data:
            return (
                lib_measurements,
                MeasureObjectIntensityDistributionDisplayData(
                    statistics=[stats_row],
                    heatmap_arrays={
                        template: numpy.zeros(object_labels.shape)
                        for template in heatmap_feature_templates
                    },
                ),
                cached_center_distances,
            )
        return lib_measurements, cached_center_distances

    if cached_center_distances is None:
        cached_center_distances = compute_center_distances(
            object_labels,
            objects_indices,
            center_object_labels,
            center_choice == CenterChoice.CENTERS_OF_OTHER.value,
            center_choice == CenterChoice.EDGES_OF_OTHER.value,
            wants_scaled,
            maximum_radius,
        )

    normalized_distance, i_center, j_center, good_mask = cached_center_distances

    statistics, measurement_pairs, heatmap_arrays = record_bin_measurements(
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
        heatmap_feature_templates,
    )

    for feature_name, value in measurement_pairs:
        lib_measurements.add_measurement(object_name, feature_name, value)

    if return_visualization_data:
        return (
            lib_measurements,
            MeasureObjectIntensityDistributionDisplayData(
                statistics=statistics, heatmap_arrays=heatmap_arrays
            ),
            cached_center_distances,
        )

    return lib_measurements, cached_center_distances
