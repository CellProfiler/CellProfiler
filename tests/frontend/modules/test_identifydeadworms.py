import centrosome.cpmorphology
import numpy
import scipy.ndimage
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import C_COUNT, M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y, COLTYPE_INTEGER, \
    COLTYPE_FLOAT, M_NUMBER_OBJECT_NUMBER, FF_COUNT


import cellprofiler.modules.identifydeadworms
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

IMAGE_NAME = "myimage"
OBJECTS_NAME = "myobjects"


def test_load_v1():
    data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10479

IdentifyDeadWorms:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
Input image:BinaryWorms
Objects name:DeadWorms
Worm width:6
Worm length:114
Number of angles:180
"""
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms)
    assert module.image_name == "BinaryWorms"
    assert module.object_name == "DeadWorms"
    assert module.worm_width == 6
    assert module.worm_length == 114
    assert module.angle_count == 180
    assert module.wants_automatic_distance


def test_load_v2():
    data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10479

IdentifyDeadWorms:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Input image:BinaryWorms
Objects name:DeadWorms
Worm width:6
Worm length:114
Number of angles:180
Automatically calculate distance parameters?:No
Spatial distance:6
Angular distance:45
"""
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms)
    assert module.image_name == "BinaryWorms"
    assert module.object_name == "DeadWorms"
    assert module.worm_width == 6
    assert module.worm_length == 114
    assert module.angle_count == 180
    assert not module.wants_automatic_distance
    assert module.space_distance == 6
    assert module.angular_distance == 45


def make_workspace(pixel_data, mask=None):
    image = cellprofiler_core.image.Image(pixel_data, mask)
    image_set_list = cellprofiler_core.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, image)

    module = cellprofiler.modules.identifydeadworms.IdentifyDeadWorms()
    module.set_module_num(1)
    module.image_name.value = IMAGE_NAME
    module.object_name.value = OBJECTS_NAME

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_zeros():
    """Run the module with an image of all zeros"""
    workspace, module = make_workspace(numpy.zeros((20, 10), bool))
    module.run(workspace)
    count = workspace.measurements.get_current_image_measurement(
        "_".join((C_COUNT, OBJECTS_NAME))
    )
    assert count == 0


def test_one_worm():
    """Find a single worm"""
    image = numpy.zeros((20, 20), bool)
    index, count, i, j = centrosome.cpmorphology.get_line_pts(
        numpy.array([1, 6, 19, 14]),
        numpy.array([5, 0, 13, 18]),
        numpy.array([6, 19, 14, 1]),
        numpy.array([0, 13, 18, 5]),
    )
    image[i, j] = True
    image = scipy.ndimage.binary_fill_holes(image)
    workspace, module = make_workspace(image)
    module.worm_length.value = 12
    module.worm_width.value = 5
    module.angle_count.value = 16
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    count = m.get_current_image_measurement(
        "_".join((C_COUNT, OBJECTS_NAME))
    )
    assert count == 1
    x = m.get_current_measurement(
        OBJECTS_NAME, M_LOCATION_CENTER_X
    )
    assert len(x) == 1
    assert round(abs(x[0] - 9.0), 1) == 0
    y = m.get_current_measurement(
        OBJECTS_NAME, M_LOCATION_CENTER_Y
    )
    assert len(y) == 1
    assert round(abs(y[0] - 10.0), 1) == 0
    a = m.get_current_measurement(
        OBJECTS_NAME, cellprofiler.modules.identifydeadworms.M_ANGLE
    )
    assert len(a) == 1
    assert round(abs(a[0] - 135), 0) == 0


def test_crossing_worms():
    """Find two worms that cross"""
    image = numpy.zeros((20, 20), bool)
    index, count, i, j = centrosome.cpmorphology.get_line_pts(
        numpy.array([1, 4, 19, 16]),
        numpy.array([3, 0, 15, 18]),
        numpy.array([4, 19, 16, 1]),
        numpy.array([0, 15, 18, 3]),
    )
    image[i, j] = True
    index, count, i, j = centrosome.cpmorphology.get_line_pts(
        numpy.array([0, 3, 18, 15]),
        numpy.array([16, 19, 4, 1]),
        numpy.array([3, 18, 15, 0]),
        numpy.array([19, 4, 1, 16]),
    )
    image[i, j] = True
    image = scipy.ndimage.binary_fill_holes(image)
    workspace, module = make_workspace(image)
    module.worm_length.value = 17
    module.worm_width.value = 5
    module.angle_count.value = 16
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    count = m.get_current_image_measurement(
        "_".join((C_COUNT, OBJECTS_NAME))
    )
    assert count == 2
    a = m.get_current_measurement(
        OBJECTS_NAME, cellprofiler.modules.identifydeadworms.M_ANGLE
    )
    assert len(a) == 2
    if a[0] > 90:
        order = numpy.array([0, 1])
    else:
        order = numpy.array([1, 0])
    assert round(abs(a[order[0]] - 135), 0) == 0
    assert round(abs(a[order[1]] - 45), 0) == 0
    x = m.get_current_measurement(
        OBJECTS_NAME, M_LOCATION_CENTER_X
    )
    assert len(x) == 2
    assert round(abs(x[order[0]] - 9.0), 0) == 0
    assert round(abs(x[order[1]] - 10.0), 0) == 0
    y = m.get_current_measurement(
        OBJECTS_NAME, M_LOCATION_CENTER_Y
    )
    assert len(y) == 2
    assert round(abs(y[order[0]] - 10.0), 0) == 0
    assert round(abs(y[order[1]] - 9.0), 0) == 0


def test_measurement_columns():
    """Test get_measurement_columns"""
    workspace, module = make_workspace(numpy.zeros((20, 10), bool))
    assert isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms)
    columns = module.get_measurement_columns(workspace.pipeline)
    expected = (
        (
            OBJECTS_NAME,
            M_LOCATION_CENTER_X,
            COLTYPE_INTEGER,
        ),
        (
            OBJECTS_NAME,
            M_LOCATION_CENTER_Y,
            COLTYPE_INTEGER,
        ),
        (
            OBJECTS_NAME,
            cellprofiler.modules.identifydeadworms.M_ANGLE,
            COLTYPE_FLOAT,
        ),
        (
            OBJECTS_NAME,
            M_NUMBER_OBJECT_NUMBER,
            COLTYPE_INTEGER,
        ),
        (
            "Image",
            FF_COUNT % OBJECTS_NAME,
            COLTYPE_INTEGER,
        ),
    )
    assert len(columns) == len(expected)
    for e in expected:
        assert any(
            all([x == y for x, y in zip(c, e)]) for c in columns
        ), "could not find " + repr(e)


def test_find_adjacent_by_distance_empty():
    workspace, module = make_workspace(numpy.zeros((20, 10), bool))
    assert isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms)

    first, second = module.find_adjacent_by_distance(
        numpy.zeros(0), numpy.zeros(0), numpy.zeros(0)
    )
    assert len(first) == 0
    assert len(second) == 0


def test_find_adjacent_by_distance_one():
    workspace, module = make_workspace(numpy.zeros((20, 10), bool))
    assert isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms)

    first, second = module.find_adjacent_by_distance(
        numpy.zeros(1), numpy.zeros(1), numpy.zeros(1)
    )
    assert len(first) == 1
    assert first[0] == 0
    assert len(second) == 1
    assert second[0] == 0


def test_find_adjacent_by_distance_easy():
    #
    # Feed "find_adjacent_by_distance" points whose "i" are all
    # within the space_distance
    #
    workspace, module = make_workspace(numpy.zeros((20, 10), bool))
    assert isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms)
    module.space_distance.value = 10
    # Take find_adjacent_by_distance internals into account: consecutive i
    # will create a single cross-product
    #
    i = numpy.arange(10)
    j = numpy.arange(10)
    # Break into two groups: 0-4 (5x5) and 5-9 (5x5)
    j[5:] += 10
    a = numpy.zeros(10)
    first, second = module.find_adjacent_by_distance(i, j, a)
    order = numpy.lexsort((second, first))
    first = first[order]
    second = second[order]
    assert len(first) == 50
    assert len(second) == 50
    for i in range(50):
        assert first[i] == int(i / 5)
    for i in range(25):
        assert second[i] == i % 5
        assert second[i + 25] == (i % 5) + 5


def test_find_adjacent_by_distance_hard():
    #
    # Feed "find_adjacent_by_distance" points whose "i" are not all
    # within the space_distance
    #
    workspace, module = make_workspace(numpy.zeros((20, 10), bool))
    assert isinstance(module, cellprofiler.modules.identifydeadworms.IdentifyDeadWorms)
    module.space_distance.value = 10
    r = numpy.random.RandomState(44)
    for idx, scramble in enumerate(
        [numpy.arange(13)] + [r.permutation(numpy.arange(13)) for ii in range(10)]
    ):
        # Take find_adjacent_by_distance internals into account: non consecutive i
        # will create two cross-products
        #
        i = numpy.arange(13)
        j = numpy.arange(13)
        # Break into three groups: 0-2 (3x3), 3-6 (4x4) and 7-11 (5x5)
        # with one loner at end
        i[3:] += 10
        i[7:] += 10
        #
        # Make last in last group not match by i+j
        #
        i[-1] += 7
        j[-1] += 8
        a = numpy.zeros(13)
        #
        # Scramble i, j and a
        #
        i = i[scramble]
        j = j[scramble]
        a = a[scramble]
        #
        # a reported value of "n" corresponds to whatever index in scramble
        # that contains "n"
        #
        unscramble = numpy.zeros(13, int)
        unscramble[scramble] = numpy.arange(13)
        first, second = module.find_adjacent_by_distance(i, j, a)
        assert len(first) == 9 + 16 + 25 + 1
        assert len(second) == 9 + 16 + 25 + 1
        for f, s in zip(first, second):
            assert (i[f] - i[s]) ** 2 + (j[f] - j[s]) ** 2 <= 100
