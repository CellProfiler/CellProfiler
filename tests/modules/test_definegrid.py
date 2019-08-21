"""test_definegrid - test the DefineGrid module
"""

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.grid as cpg
import cellprofiler.image as cpi
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.definegrid as D
import pytest

GRID_NAME = "grid"
INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"
OBJECTS_NAME = "objects"


def make_workspace(image, labels):
    module = D.DefineGrid()
    module.set_module_num(1)
    module.grid_image.value = GRID_NAME
    module.manual_image.value = INPUT_IMAGE_NAME
    module.display_image_name.value = INPUT_IMAGE_NAME
    module.object_name.value = OBJECTS_NAME
    module.save_image_name.value = OUTPUT_IMAGE_NAME
    image_set_list = cpi.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(INPUT_IMAGE_NAME, cpi.Image(image))
    object_set = cpo.ObjectSet()
    objects = cpo.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, OBJECTS_NAME)
    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)
        assert not isinstance(event, cpp.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    measurements = cpmeas.Measurements()
    workspace = cpw.Workspace(
        pipeline, module, image_set, object_set, measurements, image_set_list
    )
    return workspace, module


def test_grid_automatic():
    image = np.zeros((50, 100))
    labels = np.zeros((50, 100), int)
    ii, jj = np.mgrid[0:50, 0:100]
    #
    # Make two circles at 10,11 and 40, 92
    #
    first_x, first_y = (11, 10)
    second_x, second_y = (92, 40)
    rows = 4
    columns = 10
    spacing_y = 10
    spacing_x = 9
    for i in range(rows):
        for j in range(columns):
            center_i = first_y + spacing_y * i
            center_j = first_x + spacing_x * j
            labels[(ii - center_i) ** 2 + (jj - center_j) ** 2 <= 9] = (
                i * columns + j + 1
            )
    workspace, module = make_workspace(image, labels)
    assert isinstance(module, D.DefineGrid)
    assert isinstance(workspace, cpw.Workspace)
    module.grid_rows.value = rows
    module.grid_columns.value = columns
    module.ordering.value = D.NUM_BY_COLUMNS
    module.auto_or_manual.value = D.AM_AUTOMATIC
    module.wants_image.value = True
    module.run(workspace)
    gridding = workspace.get_grid(GRID_NAME)
    assert isinstance(gridding, cpg.Grid)
    assert gridding.rows == rows
    assert gridding.columns == columns
    assert gridding.x_spacing == spacing_x
    assert gridding.y_spacing == spacing_y
    assert gridding.x_location_of_lowest_x_spot == first_x
    assert gridding.y_location_of_lowest_y_spot == first_y
    assert np.all(gridding.x_locations == first_x + np.arange(columns) * spacing_x)
    assert np.all(gridding.y_locations == first_y + np.arange(rows) * spacing_y)
    spot_table = np.arange(rows * columns) + 1
    spot_table.shape = (rows, columns)
    assert np.all(gridding.spot_table == spot_table)

    m = workspace.measurements
    assert isinstance(m, cpmeas.Measurements)
    for feature, value in (
        (D.F_COLUMNS, columns),
        (D.F_ROWS, rows),
        (D.F_X_LOCATION_OF_LOWEST_X_SPOT, first_x),
        (D.F_Y_LOCATION_OF_LOWEST_Y_SPOT, first_y),
        (D.F_X_SPACING, spacing_x),
        (D.F_Y_SPACING, spacing_y),
    ):
        measurement = "_".join((D.M_CATEGORY, GRID_NAME, feature))
        assert m.has_feature(cpmeas.IMAGE, measurement)
        assert m.get_current_image_measurement(measurement) == value

    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert image is not None


def test_fail():
    image = np.zeros((50, 100))
    labels = np.zeros((50, 100), int)
    labels[20:40, 51:62] = 1
    workspace, module = make_workspace(image, labels)
    assert isinstance(module, D.DefineGrid)
    assert isinstance(workspace, cpw.Workspace)
    module.ordering.value = D.NUM_BY_COLUMNS
    module.auto_or_manual.value = D.AM_AUTOMATIC
    module.wants_image.value = True
    with pytest.raises(RuntimeError):
        module.run(workspace)


def test_coordinates_plus_savedimagesize():
    image = np.zeros((50, 100))
    labels = np.zeros((50, 100), int)
    first_x, first_y = (11, 10)
    second_x, second_y = (92, 40)
    rows = 4
    columns = 10
    spacing_y = 10
    spacing_x = 9
    workspace, module = make_workspace(image, labels)
    assert isinstance(module, D.DefineGrid)
    assert isinstance(workspace, cpw.Workspace)
    module.grid_rows.value = rows
    module.grid_columns.value = columns
    module.ordering.value = D.NUM_BY_COLUMNS
    module.auto_or_manual.value = D.AM_MANUAL
    module.manual_choice.value = D.MAN_COORDINATES
    module.first_spot_coordinates.value = "%d,%d" % (first_x, first_y)
    module.second_spot_coordinates.value = "%d,%d" % (second_x, second_y)
    module.first_spot_col.value = 1
    module.first_spot_row.value = 1
    module.second_spot_col.value = columns
    module.second_spot_row.value = rows
    module.grid_rows.value = rows
    module.grid_columns.value = columns
    module.wants_image.value = True
    module.run(workspace)
    gridding = workspace.get_grid(GRID_NAME)
    assert isinstance(gridding, cpg.Grid)
    assert gridding.rows == rows
    assert gridding.columns == columns
    assert gridding.x_spacing == spacing_x
    assert gridding.y_spacing == spacing_y
    assert gridding.x_location_of_lowest_x_spot == first_x
    assert gridding.y_location_of_lowest_y_spot == first_y
    assert np.all(gridding.x_locations == first_x + np.arange(columns) * spacing_x)
    assert np.all(gridding.y_locations == first_y + np.arange(rows) * spacing_y)
    spot_table = np.arange(rows * columns) + 1
    spot_table.shape = (rows, columns)
    assert np.all(gridding.spot_table == spot_table)

    m = workspace.measurements
    assert isinstance(m, cpmeas.Measurements)
    for feature, value in (
        (D.F_COLUMNS, columns),
        (D.F_ROWS, rows),
        (D.F_X_LOCATION_OF_LOWEST_X_SPOT, first_x),
        (D.F_Y_LOCATION_OF_LOWEST_Y_SPOT, first_y),
        (D.F_X_SPACING, spacing_x),
        (D.F_Y_SPACING, spacing_y),
    ):
        measurement = "_".join((D.M_CATEGORY, GRID_NAME, feature))
        assert m.has_feature(cpmeas.IMAGE, measurement)
        assert m.get_current_image_measurement(measurement) == value

    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert image is not None
    shape = image.pixel_data.shape
    assert shape[0] == 50
    assert shape[1] == 100
