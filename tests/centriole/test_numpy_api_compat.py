import numpy as np
import pytest

from centriole.cpmorphology import distance2_to_line
from centriole.princomp import princomp


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_line_distance_preserves_caller_shapes_and_storage(dtype, layout):
    points = []
    for coordinates in ((1, 2), (0, 0), (4, 0)):
        point = np.asarray(coordinates, dtype=dtype)
        if layout == "strided":
            storage = np.full(4, 99, dtype=dtype)
            storage[::2] = point
            point = storage[::2]
        elif layout == "readonly":
            point = np.frombuffer(point.tobytes(), dtype=dtype)
        points.append(point)
    aliases = list(points)
    pointers = [point.__array_interface__["data"][0] for point in points]

    assert distance2_to_line(*points) == 4

    for point, alias, pointer, coordinates in zip(
        points, aliases, pointers, ((1, 2), (0, 0), (4, 0))
    ):
        assert point is alias
        assert point.shape == (1, 2)
        assert point.__array_interface__["data"][0] == pointer
        assert point.flags.writeable == (layout != "readonly")
        np.testing.assert_array_equal(point, np.asarray([coordinates], dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_principal_components_keep_real_dtype_and_order(dtype):
    observations = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 3), (0, 0, -3)),
        dtype=dtype,
    )
    coefficients = princomp(observations)
    assert coefficients.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(coefficients, np.eye(3, dtype=dtype)[:, ::-1])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("phase", [1, 1 + 1j])
def test_principal_components_keep_complex_dtype(dtype, phase):
    observations = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 3), (0, 0, -3)),
        dtype=dtype,
    ) * phase
    coefficients = princomp(observations)
    assert coefficients.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(coefficients, np.eye(3, dtype=dtype)[:, ::-1])
