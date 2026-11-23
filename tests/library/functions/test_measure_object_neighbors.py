import numpy as np

from cellprofiler_library.functions.measurement import measure_object_neighbors
from cellprofiler_library.opts.measureobjectneighbors import DistanceMethod


def _measure(olabels, nlabels):
    has_pixels = np.ones(olabels.max(), bool)
    return measure_object_neighbors(
        olabels,
        olabels,
        nlabels,
        nlabels,
        False,
        olabels.ndim,
        20,
        DistanceMethod.EXPAND,
        has_pixels,
        int(olabels.max()),
    )


def _two_neighbors_3d():
    olabels = np.zeros((3, 10, 20), int)
    olabels[0, 5, 5] = 1
    nlabels = np.zeros((3, 10, 20), int)
    nlabels[0, 5, 15] = 1
    nlabels[2, 8, 5] = 2
    return olabels, nlabels


def test_closest_neighbors_3d_use_all_axes():
    olabels, nlabels = _two_neighbors_3d()
    res = _measure(olabels, nlabels)
    first_number, second_number = res[1], res[2]
    first_distance, second_distance, angle = res[3], res[4], res[5]
    assert first_number[0] == 2
    assert second_number[0] == 1
    np.testing.assert_allclose(first_distance, [np.sqrt(13)])
    np.testing.assert_allclose(second_distance, [10])
    np.testing.assert_allclose(angle, [90])


def test_closest_distance_3d_along_last_axis():
    olabels = np.zeros((3, 10, 20), int)
    olabels[1, 5, 2] = 1
    nlabels = np.zeros((3, 10, 20), int)
    nlabels[1, 5, 9] = 1
    nlabels[1, 5, 0] = 2
    res = _measure(olabels, nlabels)
    np.testing.assert_allclose(res[3], [2])
    np.testing.assert_allclose(res[4], [7])
    np.testing.assert_allclose(res[5], [180])


def test_closest_neighbors_2d_unchanged():
    olabels = np.zeros((20, 10), int)
    olabels[2, 2] = 1
    nlabels = np.zeros((20, 10), int)
    nlabels[5, 2] = 2
    nlabels[2, 6] = 1
    res = _measure(olabels, nlabels)
    assert res[1][0] == 2
    assert res[2][0] == 1
    np.testing.assert_allclose(res[3], [3])
    np.testing.assert_allclose(res[4], [4])
    np.testing.assert_allclose(res[5], [90])


def test_no_objects_3d():
    olabels = np.zeros((3, 10, 20), int)
    nlabels = np.zeros((3, 10, 20), int)
    nlabels[0, 5, 15] = 1
    nlabels[2, 8, 5] = 2
    res = _measure(olabels, nlabels)
    for value in res[1:6]:
        assert len(value) == 0
