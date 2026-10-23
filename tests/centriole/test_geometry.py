import warnings

import numpy as np

from centriole.cpmorphology import lines_intersect, within_hull


def test_2d_geometry_without_deprecated_cross_products():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for dtype in (np.int32, np.float32, np.float64):
            points = np.array([[0, 0], [4, 4], [0, 4], [4, 0]], dtype=dtype)
            assert lines_intersect(*points)
            hull = np.array([[0, 0], [0, 5], [5, 5], [5, 0]], dtype=dtype)
            assert within_hull(np.array([2, 2], dtype=dtype), hull)
            assert not within_hull(np.array([6, 2], dtype=dtype), hull)
