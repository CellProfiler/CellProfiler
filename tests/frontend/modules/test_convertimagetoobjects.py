from unittest.mock import Mock

import numpy
import pytest
from cellprofiler.modules.convertimagetoobjects import ConvertImageToObjects
from cellprofiler_core.image import Image, ImageSetList
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace


@pytest.fixture
def make_workspace():
    measurements = Measurements()

    def make(data, *, preserve_labels=True, cast_to_bool=False, show_window=True):
        module = ConvertImageToObjects()
        module.x_name.value = "Input"
        module.y_name.value = "Objects"
        module.preserve_labels.value = preserve_labels
        module.cast_to_bool.value = cast_to_bool
        module.show_window = show_window
        image_set_list = ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add("Input", Image(data, dimensions=data.ndim))
        pipeline = Pipeline()
        pipeline.add_module(module)
        return Workspace(
            pipeline, module, image_set, ObjectSet(), measurements, image_set_list
        )

    yield make
    measurements.close()


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("empty", [False, True], ids=["sparse", "empty"])
def test_preserved_label_statistics(make_workspace, dimensions, empty):
    # Label 7 has two disconnected parts; 65535 must not imply 65535 objects.
    data = numpy.array([[7, 0, 65535], [0, 0, 0], [7, 0, 0]], dtype=numpy.uint16)
    if empty:
        data[:] = 0
    if dimensions == 3:
        data = numpy.stack([data, numpy.zeros_like(data)])
    workspace = make_workspace(data)

    workspace.module.run(workspace)

    numpy.testing.assert_array_equal(workspace.get_objects("Objects").segmented, data)
    coverage_label = (
        "Volume covered by objects" if dimensions == 3 else "Area covered by objects"
    )
    coverage = "0.0 %" if empty else ("16.7 %" if dimensions == 3 else "33.3 %")
    assert workspace.display_data.statistics == [
        ("# of objects", "0" if empty else "2"),
        (coverage_label, coverage),
    ]
    # The existing measurements remain separate from the display-only count.
    assert workspace.measurements.get_current_image_measurement("Count_Objects") == (
        0 if empty else 65535
    )


@pytest.mark.parametrize("binary", [False, True], ids=["relabel", "binary"])
def test_statistics_after_conversion(make_workspace, binary):
    data = numpy.array([[7, 0, 9], [0, 0, 0], [7, 0, 0]], dtype=numpy.uint16)
    if binary:
        data = data > 0
    workspace = make_workspace(data, preserve_labels=False, cast_to_bool=binary)

    workspace.module.run(workspace)

    numpy.testing.assert_array_equal(
        workspace.get_objects("Objects").segmented,
        [[1, 0, 2], [0, 0, 0], [3, 0, 0]],
    )
    assert workspace.display_data.statistics == [
        ("# of objects", "3"),
        ("Area covered by objects", "33.3 %"),
    ]


def test_statistics_without_background(make_workspace):
    workspace = make_workspace(numpy.full((3, 3), 7, dtype=numpy.uint16))

    workspace.module.run(workspace)

    assert workspace.display_data.statistics == [
        ("# of objects", "1"),
        ("Area covered by objects", "100.0 %"),
    ]


def test_run_without_display(make_workspace):
    data = numpy.array([[0, 7], [7, 0]], dtype=numpy.uint16)
    workspace = make_workspace(data, show_window=False)

    workspace.module.run(workspace)

    numpy.testing.assert_array_equal(workspace.get_objects("Objects").segmented, data)
    assert vars(workspace.display_data) == {}
    assert workspace.measurements.has_feature("Image", "Count_Objects")


@pytest.mark.parametrize("dimensions", [2, 3])
def test_display_preserves_images_and_shared_axes(make_workspace, dimensions):
    data = numpy.ones((2,) * dimensions, dtype=numpy.uint16)
    workspace = make_workspace(data)
    workspace.module.run(workspace)
    figure = Mock()

    # Redrawing must not append duplicate statistics.
    for _ in range(2):
        figure.reset_mock()
        workspace.module.display(workspace, figure)

        figure.set_subplots.assert_called_once_with(
            dimensions=dimensions, subplots=(2, 2)
        )
        figure.subplot_imshow.assert_called_once_with(
            colormap="gray",
            image=workspace.display_data.x_data,
            title="Input",
            x=0,
            y=0,
        )
        figure.subplot.assert_called_once_with(0, 0)
        figure.subplot_imshow_labels.assert_called_once_with(
            image=workspace.display_data.y_data,
            sharexy=figure.subplot.return_value,
            title="Objects",
            x=1,
            y=0,
        )
        coverage_label = (
            "Volume covered by objects"
            if dimensions == 3
            else "Area covered by objects"
        )
        figure.subplot_table.assert_called_once_with(
            1, 1, [["1"], ["100.0 %"]], row_labels=["# of objects", coverage_label]
        )
