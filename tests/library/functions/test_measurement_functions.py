import numpy as np

from cellprofiler_library.functions.measurement import measure_image_intensities


def test_float32_mean_preserves_double_precision():
    pixels = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    measurements, _ = measure_image_intensities(pixels)
    mean = measurements[1]
    assert isinstance(mean, np.float64)
    assert mean == 0.20000000794728598
