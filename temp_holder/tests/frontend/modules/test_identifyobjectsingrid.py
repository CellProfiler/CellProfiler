import numpy


import cellprofiler_core.utilities.grid
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.definegrid
import cellprofiler.modules.identifyobjectsingrid
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

OUTPUT_OBJECTS_NAME = "outputobjects"
GRID_NAME = "mygrid"
GUIDING_OBJECTS_NAME = "inputobjects"


def make_workspace(gridding, labels=None):
    module = cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid()
    module.set_module_num(1)
    module.grid_name.value = GRID_NAME
    module.output_objects_name.value = OUTPUT_OBJECTS_NAME
    module.guiding_object_name.value = GUIDING_OBJECTS_NAME
    image_set_list = cellprofiler_core.image.ImageSetList()
    object_set = cellprofiler_core.object.ObjectSet()
    if labels is not None:
        my_objects = cellprofiler_core.object.Objects()
        my_objects.segmented = labels
        object_set.add_objects(my_objects, GUIDING_OBJECTS_NAME)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    workspace.set_grid(GRID_NAME, gridding)
    return workspace, module


def make_rectangular_grid(gridding):
    assert isinstance(gridding, cellprofiler_core.utilities.grid.Grid)
    i0 = gridding.y_location_of_lowest_y_spot
    j0 = gridding.x_location_of_lowest_x_spot
    di = gridding.y_spacing
    dj = gridding.x_spacing
    ni = gridding.rows
    nj = gridding.columns
    i, j = numpy.mgrid[0 : (i0 + di * (ni + 1)), 0 : (j0 + di * (nj + 1))]
    i = numpy.round((i - i0).astype(float) / di).astype(int)
    j = numpy.round((j - j0).astype(float) / dj).astype(int)
    mask = (i >= 0) & (j >= 0) & (i < ni) & (j < nj)
    grid = numpy.zeros((int(gridding.image_height), int(gridding.image_width)), int)
    g = grid[: i.shape[0], : i.shape[1]]
    g[mask[: g.shape[0], : g.shape[1]]] = gridding.spot_table[i[mask], j[mask]]
    return grid


def test_forced_location():
    d = cellprofiler.modules.definegrid.DefineGrid()
    d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
    #
    # Grid with x spacing = 10, y spacing = 20
    #
    diameter = 6
    gridding = d.build_grid_info(15, 25, 1, 1, 25, 45, 2, 2)
    expected = make_rectangular_grid(gridding)
    i, j = numpy.mgrid[0 : expected.shape[0], 0 : expected.shape[1]]
    ispot, jspot = numpy.mgrid[0 : gridding.rows, 0 : gridding.columns]
    y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    y_locations[gridding.spot_table.flatten()] = gridding.y_locations[ispot.flatten()]
    x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    x_locations[gridding.spot_table.flatten()] = gridding.x_locations[jspot.flatten()]
    idist = i - y_locations[expected]
    jdist = j - x_locations[expected]
    expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
    workspace, module = make_workspace(gridding)
    assert isinstance(
        module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid
    )
    module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
    module.diameter.value = diameter
    module.shape_choice.value = (
        cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_FORCED
    )
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
    assert numpy.all(labels == expected[0 : labels.shape[0], 0 : labels.shape[1]])
    #
    # Check measurements
    #
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
    assert numpy.all(xm == x_locations[1:])
    ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
    assert numpy.all(ym == y_locations[1:])
    count = m.get_current_image_measurement("Count_%s" % OUTPUT_OBJECTS_NAME)
    assert count == gridding.rows * gridding.columns

    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(columns) == 4
    count_feature = "Count_%s" % OUTPUT_OBJECTS_NAME
    assert all(
        [
            column[0]
            == ("Image" if column[1] == count_feature else OUTPUT_OBJECTS_NAME)
            for column in columns
        ]
    )
    assert all(
        [
            column[1]
            in (
                "Location_Center_X",
                "Location_Center_Y",
                count_feature,
                "Number_Object_Number",
            )
            for column in columns
        ]
    )
    #
    # Check the measurements
    #
    categories = list(module.get_categories(None, OUTPUT_OBJECTS_NAME))
    assert len(categories) == 2
    categories.sort()
    assert categories[0] == "Location"
    assert categories[1] == "Number"
    categories = module.get_categories(None, "Image")
    assert len(categories) == 1
    assert categories[0] == "Count"
    measurements = module.get_measurements(
        None, "Image", "Count"
    )
    assert len(measurements) == 1
    assert measurements[0] == OUTPUT_OBJECTS_NAME
    measurements = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Location")
    assert len(measurements) == 2
    assert all(m in ("Center_X", "Center_Y") for m in measurements)
    assert "Center_X" in measurements
    assert "Center_Y" in measurements
    measurements = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Number")
    assert len(measurements) == 1
    assert measurements[0] == "Object_Number"


def test_forced_location_auto():
    #
    # Automatic diameter
    #
    d = cellprofiler.modules.definegrid.DefineGrid()
    d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
    diameter = 7
    gridding = d.build_grid_info(15, 25, 1, 1, 25, 45, 2, 2)
    expected = make_rectangular_grid(gridding)
    i, j = numpy.mgrid[0 : expected.shape[0], 0 : expected.shape[1]]
    ispot, jspot = numpy.mgrid[0 : gridding.rows, 0 : gridding.columns]
    y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    y_locations[gridding.spot_table.flatten()] = gridding.y_locations[ispot.flatten()]
    x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    x_locations[gridding.spot_table.flatten()] = gridding.x_locations[jspot.flatten()]
    idist = i - y_locations[expected]
    jdist = j - x_locations[expected]
    expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
    #
    # Make a fuzzy mask to account for the diameter being +/- 1
    #
    mask = numpy.ones(expected.shape, bool)
    mask[
        numpy.abs(numpy.sqrt(idist ** 2 + jdist ** 2) - float(diameter + 1) / 2) <= 1
    ] = False
    #
    # Make a labels matrix that's like the expected one, but
    # is relabeled randomly
    #
    guide_labels = expected.copy()
    numpy.random.seed(0)
    p = numpy.random.permutation(numpy.arange(expected.max() + 1))
    p[p == 0] = p[0]
    p[0] = 0
    guide_labels = p[guide_labels]
    workspace, module = make_workspace(gridding, guide_labels)
    assert isinstance(
        module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid
    )
    module.diameter_choice.value = (
        cellprofiler.modules.identifyobjectsingrid.AM_AUTOMATIC
    )
    module.shape_choice.value = (
        cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_FORCED
    )
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
    assert numpy.all(
        labels[mask] == expected[0 : labels.shape[0], 0 : labels.shape[1]][mask]
    )


def test_natural_circle():
    d = cellprofiler.modules.definegrid.DefineGrid()
    d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
    #
    # Grid with x spacing = 10, y spacing = 20
    #
    diameter = 6
    gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
    expected = make_rectangular_grid(gridding)
    i, j = numpy.mgrid[0 : expected.shape[0], 0 : expected.shape[1]]
    ispot, jspot = numpy.mgrid[0 : gridding.rows, 0 : gridding.columns]
    y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    y_locations[gridding.spot_table.flatten()] = gridding.y_locations[ispot.flatten()]
    x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    x_locations[gridding.spot_table.flatten()] = gridding.x_locations[jspot.flatten()]
    #
    # Perturb the X and Y locations and diameters randomly
    #
    numpy.random.seed(0)
    x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
    y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
    random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
    idist = i - y_locations[expected]
    jdist = j - x_locations[expected]
    guide_labels = expected.copy()
    expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
    guide_labels[
        idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2
    ] = 0
    workspace, module = make_workspace(gridding, guide_labels)
    assert isinstance(
        module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid
    )
    module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
    module.diameter.value = diameter
    module.shape_choice.value = (
        cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_NATURAL
    )
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
    assert numpy.all(labels == expected[0 : labels.shape[0], 0 : labels.shape[1]])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
    assert numpy.all(xm == x_locations[1:])
    ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
    assert numpy.all(ym == y_locations[1:])
    count = m.get_current_image_measurement("Count_%s" % OUTPUT_OBJECTS_NAME)
    assert count == gridding.rows * gridding.columns


def test_natural_circle_edges():
    #
    # Put objects near the edges of the circle and make sure
    # they are filtered out
    #
    d = cellprofiler.modules.definegrid.DefineGrid()
    d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
    #
    # Grid with x spacing = 10, y spacing = 20
    #
    diameter = 6
    gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
    expected = make_rectangular_grid(gridding)
    i, j = numpy.mgrid[0 : expected.shape[0], 0 : expected.shape[1]]
    ispot, jspot = numpy.mgrid[0 : gridding.rows, 0 : gridding.columns]
    y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    y_locations[gridding.spot_table.flatten()] = gridding.y_locations[ispot.flatten()]
    x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    x_locations[gridding.spot_table.flatten()] = gridding.x_locations[jspot.flatten()]
    #
    # save some bad places - at the corners of the grids
    #
    bad_x_locations = (x_locations - gridding.x_spacing / 2).astype(int)
    bad_y_locations = (y_locations - gridding.y_spacing / 2).astype(int)
    #
    # Perturb the X and Y locations and diameters randomly
    #
    numpy.random.seed(0)
    x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
    y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
    random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
    idist = i - y_locations[expected]
    jdist = j - x_locations[expected]
    guide_labels = expected.copy()
    expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
    guide_labels[
        idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2
    ] = 0
    #
    # Add objects in bad places
    #
    for i_off in (-1, 0, 1):
        for j_off in (-1, 0, 1):
            guide_labels[bad_y_locations + i_off, bad_x_locations + j_off] = (
                numpy.arange(len(bad_y_locations))
                + gridding.rows * gridding.columns
                + 1
            )
    #
    # run the module
    #
    workspace, module = make_workspace(gridding, guide_labels)
    assert isinstance(
        module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid
    )
    module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
    module.diameter.value = diameter
    module.shape_choice.value = (
        cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_NATURAL
    )
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
    assert numpy.all(labels == expected[0 : labels.shape[0], 0 : labels.shape[1]])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
    assert numpy.all(xm == x_locations[1:])
    ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
    assert numpy.all(ym == y_locations[1:])
    count = m.get_current_image_measurement("Count_%s" % OUTPUT_OBJECTS_NAME)
    assert count == gridding.rows * gridding.columns


def test_img_891():
    """Regression test of img-891, last spot filtered out"""
    d = cellprofiler.modules.definegrid.DefineGrid()
    d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
    #
    # Grid with x spacing = 10, y spacing = 20
    #
    diameter = 6
    gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
    expected = make_rectangular_grid(gridding)
    i, j = numpy.mgrid[0 : expected.shape[0], 0 : expected.shape[1]]
    ispot, jspot = numpy.mgrid[0 : gridding.rows, 0 : gridding.columns]
    y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    y_locations[gridding.spot_table.flatten()] = gridding.y_locations[ispot.flatten()]
    x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    x_locations[gridding.spot_table.flatten()] = gridding.x_locations[jspot.flatten()]
    #
    # Perturb the X and Y locations and diameters randomly
    #
    numpy.random.seed(0)
    x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
    y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
    random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
    idist = i - y_locations[expected]
    jdist = j - x_locations[expected]
    guide_labels = expected.copy()
    expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
    guide_labels[
        idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2
    ] = 0
    #
    # Erase the last one... this triggered the bug
    #
    expected[expected == numpy.max(guide_labels)] = 0
    guide_labels[guide_labels == numpy.max(guide_labels)] = 0
    workspace, module = make_workspace(gridding, guide_labels)
    assert isinstance(
        module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid
    )
    module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
    module.diameter.value = diameter
    module.shape_choice.value = (
        cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_NATURAL
    )
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
    assert numpy.all(labels == expected[0 : labels.shape[0], 0 : labels.shape[1]])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
    assert len(xm) == 96
    assert numpy.all(xm[:-1] == x_locations[1:-1])
    assert numpy.isnan(xm[-1])
    ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
    assert numpy.all(ym[:-1] == y_locations[1:-1])
    count = m.get_current_image_measurement("Count_%s" % OUTPUT_OBJECTS_NAME)
    assert count == gridding.rows * gridding.columns


def test_natural():
    # Use natural objects.
    #
    # Put objects near the edges of the circle and make sure
    # they are filtered out.
    #
    # Randomly distribute points in objects to keep between
    # two different groups of objects.
    #
    d = cellprofiler.modules.definegrid.DefineGrid()
    d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
    #
    # Grid with x spacing = 10, y spacing = 20
    #
    diameter = 6
    gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
    guide_labels = make_rectangular_grid(gridding)
    i, j = numpy.mgrid[0 : guide_labels.shape[0], 0 : guide_labels.shape[1]]
    ispot, jspot = numpy.mgrid[0 : gridding.rows, 0 : gridding.columns]
    y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    y_locations[gridding.spot_table.flatten()] = gridding.y_locations[ispot.flatten()]
    x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
    x_locations[gridding.spot_table.flatten()] = gridding.x_locations[jspot.flatten()]
    #
    # save some bad places - at the corners of the grids
    #
    bad_x_locations = (x_locations - gridding.x_spacing / 2).astype(int)
    bad_y_locations = (y_locations - gridding.y_spacing / 2).astype(int)
    #
    # Perturb the X and Y locations and diameters randomly
    #
    numpy.random.seed(0)
    x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
    y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
    random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
    idist = i - y_locations[guide_labels]
    jdist = j - x_locations[guide_labels]
    guide_labels[
        idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2
    ] = 0
    expected = guide_labels.copy()
    guide_labels[guide_labels != 0] += (
        gridding.rows
        * gridding.columns
        * (numpy.random.uniform(size=numpy.sum(guide_labels != 0)) > 0.5)
    )
    #
    # Take 1/2 of the points and assign them to a second class of objects.
    # All of the objects should be merged.
    #
    #
    # Add objects in bad places
    #
    for i_off in (-1, 0, 1):
        for j_off in (-1, 0, 1):
            guide_labels[bad_y_locations + i_off, bad_x_locations + j_off] = (
                numpy.arange(len(bad_y_locations))
                + gridding.rows * gridding.columns * 2
                + 1
            )
    #
    # Scramble the label numbers
    #
    p = numpy.random.permutation(numpy.arange(numpy.max(guide_labels) + 1))
    p[p == 0] = p[0]
    p[0] = 0
    guide_labels = p[guide_labels]
    #
    # run the module
    #
    workspace, module = make_workspace(gridding, guide_labels)
    assert isinstance(
        module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid
    )
    module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
    module.diameter.value = diameter
    module.shape_choice.value = cellprofiler.modules.identifyobjectsingrid.SHAPE_NATURAL
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
    assert numpy.all(labels == expected[0 : labels.shape[0], 0 : labels.shape[1]])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
    assert numpy.all(xm == x_locations[1:])
    ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
    assert numpy.all(ym == y_locations[1:])
    count = m.get_current_image_measurement("Count_%s" % OUTPUT_OBJECTS_NAME)
    assert count == gridding.rows * gridding.columns
