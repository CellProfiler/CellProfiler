from cellprofiler_library.opts.measureimageoverlap import DM
from cellprofiler_library.functions.measurement import (
    measure_image_overlap_statistics,
    compute_earth_movers_distance,
)


def measureimageoverlap(
    ground_truth_image,
    test_image,
    mask=None,
    calculate_emd=False,
    max_distance=250,
    penalize_missing=False,
    decimation_method: DM = DM.KMEANS,
    max_points=250,
):

    data = measure_image_overlap_statistics(
        ground_truth_image=ground_truth_image, test_image=test_image, mask=mask
    )

    if calculate_emd:
        emd = compute_earth_movers_distance(
            ground_truth_image=ground_truth_image,
            test_image=test_image,
            max_distance=max_distance,
            penalize_missing=penalize_missing,
            decimation_method=decimation_method,
            max_points=max_points,
        )
        data.update({"EarthMoversDistance": emd})
    return data
