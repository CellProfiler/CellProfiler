import centrosome.cpmorphology
import centrosome.outline
import numpy

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.expandorshrinkobjects
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()

INPUT_NAME = "input"
OUTPUT_NAME = "output"
OUTLINES_NAME = "outlines"


def make_workspace(
    labels, operation, iterations=1, wants_outlines=False, wants_fill_holes=False
):
    object_set = cellprofiler.object.ObjectSet()
    objects = cellprofiler.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, INPUT_NAME)
    module = cellprofiler.modules.expandorshrinkobjects.ExpandOrShrink()
    module.object_name.value = INPUT_NAME
    module.output_object_name.value = OUTPUT_NAME
    module.operation.value = operation
    module.iterations.value = iterations
    module.wants_fill_holes.value = wants_fill_holes
    module.set_module_num(1)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(module)
    image_set_list = cellprofiler.image.ImageSetList()
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        cellprofiler.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_expand():
    """Expand an object once"""
    labels = numpy.zeros((10, 10), int)
    labels[4, 4] = 1
    expected = numpy.zeros((10, 10), int)
    expected[numpy.array([4, 3, 4, 5, 4], int), numpy.array([3, 4, 4, 4, 5], int)] = 1
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_EXPAND
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)
    assert OUTLINES_NAME not in workspace.get_outline_names()
    m = workspace.measurements
    assert isinstance(m, cellprofiler.measurement.Measurements)
    count = m.get_current_image_measurement("Count_" + OUTPUT_NAME)
    if not numpy.isscalar(count):
        count = count[0]
    assert count == 1
    location_x = m.get_current_measurement(OUTPUT_NAME, "Location_Center_X")
    assert len(location_x) == 1
    assert location_x[0] == 4
    location_y = m.get_current_measurement(OUTPUT_NAME, "Location_Center_Y")
    assert len(location_y) == 1
    assert location_y[0] == 4


def test_expand_twice():
    '''Expand an object "twice"'''
    labels = numpy.zeros((10, 10), int)
    labels[4, 4] = 1
    i, j = numpy.mgrid[0:10, 0:10] - 4
    expected = (i ** 2 + j ** 2 <= 4).astype(int)
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_EXPAND, 2
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)


def test_expand_two():
    """Expand two objects once"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 3] = 1
    labels[6, 5] = 2
    i, j = numpy.mgrid[0:10, 0:10]
    expected = ((i - 2) ** 2 + (j - 3) ** 2 <= 1).astype(int) + (
        (i - 6) ** 2 + (j - 5) ** 2 <= 1
    ).astype(int) * 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_EXPAND, 1
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)


def test_expand_inf():
    """Expand two objects infinitely"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 3] = 1
    labels[6, 5] = 2
    i, j = numpy.mgrid[0:10, 0:10]
    distance = ((i - 2) ** 2 + (j - 3) ** 2) - ((i - 6) ** 2 + (j - 5) ** 2)
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_EXPAND_INF
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented[distance < 0] == 1)
    assert numpy.all(objects.segmented[distance > 0] == 2)


def test_divide():
    """Divide two touching objects"""
    labels = numpy.ones((10, 10), int)
    labels[5:, :] = 2
    expected = labels.copy()
    expected[4:6, :] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_DIVIDE
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)


def test_dont_divide():
    """Don't divide an object that would disappear"""
    labels = numpy.ones((10, 10), int)
    labels[9, 9] = 2
    expected = labels.copy()
    expected[8, 9] = 0
    expected[8, 8] = 0
    expected[9, 8] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_DIVIDE
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)


def test_shrink():
    """Shrink once"""
    labels = numpy.zeros((10, 10), int)
    labels[1:9, 1:9] = 1
    expected = centrosome.cpmorphology.thin(labels, iterations=1)
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_SHRINK, 1
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)


def test_shrink_inf():
    """Shrink infinitely"""
    labels = numpy.zeros((10, 10), int)
    labels[1:8, 1:8] = 1
    expected = numpy.zeros((10, 10), int)
    expected[4, 4] = 1
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_SHRINK_INF
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)


def test_shrink_inf_fill_holes():
    """Shrink infinitely after filling a hole"""
    labels = numpy.zeros((10, 10), int)
    labels[1:8, 1:8] = 1
    labels[4, 4] = 0
    expected = numpy.zeros((10, 10), int)
    expected[4, 4] = 1
    # Test failure without filling the hole
    workspace, module = make_workspace(
        labels, cellprofiler.modules.expandorshrinkobjects.O_SHRINK_INF
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert not numpy.all(objects.segmented == expected)
    # Test success after filling the hole
    workspace, module = make_workspace(
        labels,
        cellprofiler.modules.expandorshrinkobjects.O_SHRINK_INF,
        wants_fill_holes=True,
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_NAME)
    assert numpy.all(objects.segmented == expected)
