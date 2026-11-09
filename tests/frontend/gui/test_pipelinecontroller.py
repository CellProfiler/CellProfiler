import numpy

# pipelinecontroller can't be imported before cpframe: they import each other
import cellprofiler.gui.cpframe  # noqa: F401
from cellprofiler.gui.pipelinecontroller import sort_image_numbers


def order(values, ascending=True):
    return sort_image_numbers(sorted(values), values.__getitem__, ascending)


def test_sort_image_numbers_numbers_by_value():
    values = {1: 10, 2: 2.5, 3: 2, 4: 100}

    assert order(values) == [3, 2, 1, 4]
    assert order(values, ascending=False) == [4, 1, 2, 3]


def test_sort_image_numbers_numbers_ahead_of_text():
    values = {1: "b", 2: 7, 3: "a", 4: 1}

    assert order(values) == [4, 2, 3, 1]


def test_sort_image_numbers_ties_by_image_number():
    values = {1: 5, 2: 5, 3: 1}

    assert order(values) == [3, 1, 2]


def test_sort_image_numbers_mixed_missing_numeric_column():
    # NaN compares False with everything, so left in the keys it stops
    # [10, nan, 2] from being ordered at all
    nan = float("nan")
    values = {1: 10.0, 2: nan, 3: 2.0, 4: numpy.float64("nan"), 5: 7.0, 6: None}

    assert order(values) == [3, 5, 1, 2, 4, 6]
    # missing values stay last, in image number order, when descending
    assert order(values, ascending=False) == [1, 5, 3, 2, 4, 6]


def test_sort_image_numbers_all_missing():
    values = {3: float("nan"), 1: float("nan"), 2: float("nan")}

    assert order(values) == [1, 2, 3]
