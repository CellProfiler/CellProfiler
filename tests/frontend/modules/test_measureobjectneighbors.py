import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import COLTYPE_INTEGER, COLTYPE_FLOAT, NEIGHBORS, R_FIRST_OBJECT_NUMBER, \
    R_SECOND_OBJECT_NUMBER
from cellprofiler_core.measurement import RelationshipKey

import cellprofiler.modules.measureobjectneighbors
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

OBJECTS_NAME = "objectsname"
NEIGHBORS_NAME = "neighborsname"


def make_workspace(labels, mode, distance=0, neighbors_labels=None):
    """Make a workspace for testing MeasureObjectNeighbors"""
    module = cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors()
    module.set_module_num(1)
    module.object_name.value = OBJECTS_NAME
    module.distance_method.value = mode
    module.distance.value = distance
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)
    object_set = cellprofiler_core.object.ObjectSet()
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    measurements = cellprofiler_core.measurement.Measurements()
    measurements.group_index = 1
    measurements.group_number = 1
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, image_set, object_set, measurements, image_set_list
    )
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, OBJECTS_NAME)
    if neighbors_labels is None:
        module.neighbors_name.value = OBJECTS_NAME
    else:
        module.neighbors_name.value = NEIGHBORS_NAME
        objects = cellprofiler_core.object.Objects()
        objects.segmented = neighbors_labels
        object_set.add_objects(objects, NEIGHBORS_NAME)
    return workspace, module


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("measureobjectneighbors/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors
    )
    assert module.object_name == "glia"
    assert module.neighbors_name == "neurites"
    assert (
        module.distance_method == cellprofiler.modules.measureobjectneighbors.D_EXPAND
    )
    assert module.distance == 2
    assert not module.wants_count_image
    assert module.count_image_name == "countimage"
    assert module.count_colormap == "pink"
    assert not module.wants_percent_touching_image
    assert module.touching_image_name == "touchingimage"
    assert module.touching_colormap == "purple"


def test_empty():
    """Test a labels matrix with no objects"""
    workspace, module = make_workspace(
        numpy.zeros((10, 10), int),
        cellprofiler.modules.measureobjectneighbors.D_EXPAND,
        5,
    )
    module.run(workspace)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Expanded"
    )
    assert len(neighbors) == 0
    features = m.get_feature_names(OBJECTS_NAME)
    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(features) == len(columns)
    for column in columns:
        assert column[0] == OBJECTS_NAME
        assert column[1] in features
        assert column[2] == (
            COLTYPE_INTEGER
            if column[1].find("Number") != -1
            else COLTYPE_FLOAT
        )


def test_one():
    """Test a labels matrix with a single object"""
    labels = numpy.zeros((10, 10), int)
    labels[3:5, 4:6] = 1
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_EXPAND, 5
    )
    module.run(workspace)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Expanded"
    )
    assert len(neighbors) == 1
    assert neighbors[0] == 0
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_Expanded")
    assert len(pct) == 1
    assert pct[0] == 0


def test_two_expand():
    """Test a labels matrix with two objects"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1
    labels[8, 7] = 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_EXPAND, 5
    )
    module.run(workspace)
    assert tuple(module.get_categories(None, OBJECTS_NAME)) == ("Neighbors",)
    assert tuple(module.get_measurements(None, OBJECTS_NAME, "Neighbors")) == tuple(
        cellprofiler.modules.measureobjectneighbors.M_ALL
    )
    assert tuple(
        module.get_measurement_scales(
            None, OBJECTS_NAME, "Neighbors", "NumberOfNeighbors", None
        )
    ) == ("Expanded",)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Expanded"
    )
    assert len(neighbors) == 2
    assert numpy.all(neighbors == 1)
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_Expanded")
    #
    # This is what the patch looks like:
    #  P P P P P P P P P P
    #  P I I I I I I I O O
    #  P I I I I I O O O N
    #  P I I I I O O N N N
    #  P I I I O O N N N N
    #  P I I O O N N N N N
    #  P I O O N N N N N N
    #  O O O N N N N N N N
    #  O N N N N N N N N N
    #  N N N N N N N N N N
    #
    # where P = perimeter, but not overlapping the second object
    #       I = interior, not perimeter
    #       O = dilated 2nd object overlaps perimeter
    #       N = neigbor object, not overlapping
    #
    # There are 33 perimeter pixels (P + O) and 17 perimeter pixels
    # that overlap the dilated neighbor (O).
    #
    assert len(pct) == 2
    assert round(abs(pct[0] - 100.0 * 17.0 / 33.0), 7) == 0
    fo = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_FirstClosestObjectNumber_Expanded"
    )
    assert len(fo) == 2
    assert fo[0] == 2
    assert fo[1] == 1
    x = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_FirstClosestDistance_Expanded"
    )
    assert round(abs(len(x) - 2), 7) == 0
    assert round(abs(x[0] - numpy.sqrt(61)), 7) == 0
    assert round(abs(x[1] - numpy.sqrt(61)), 7) == 0


def test_two_not_adjacent():
    """Test a labels matrix with two objects, not adjacent"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1
    labels[8, 7] = 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5
    )
    module.run(workspace)
    assert tuple(
        module.get_measurement_scales(
            None, OBJECTS_NAME, "Neighbors", "NumberOfNeighbors", None
        )
    ) == ("Adjacent",)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Adjacent"
    )
    assert len(neighbors) == 2
    assert numpy.all(neighbors == 0)
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_Adjacent")
    assert len(pct) == 2
    assert numpy.all(pct == 0)


def test_adjacent():
    """Test a labels matrix with two objects, adjacent"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1
    labels[2, 3] = 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5
    )
    module.run(workspace)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Adjacent"
    )
    assert len(neighbors) == 2
    assert numpy.all(neighbors == 1)
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_Adjacent")
    assert len(pct) == 2
    assert round(abs(pct[0] - 100), 7) == 0
    fo = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_FirstClosestObjectNumber_Adjacent"
    )
    assert len(fo) == 2
    assert fo[0] == 2
    assert fo[1] == 1


def test_manual_not_touching():
    """Test a labels matrix with two objects not touching"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1  # Pythagoras triangle 3-4-5
    labels[5, 6] = 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 4
    )
    module.run(workspace)
    assert tuple(
        module.get_measurement_scales(
            None, OBJECTS_NAME, "Neighbors", "NumberOfNeighbors", None
        )
    ) == ("4",)
    m = workspace.measurements
    neighbors = m.get_current_measurement(OBJECTS_NAME, "Neighbors_NumberOfNeighbors_4")
    assert len(neighbors) == 2
    assert numpy.all(neighbors == 0)
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_4")
    assert len(pct) == 2
    assert round(abs(pct[0] - 0), 7) == 0


def test_manual_touching():
    """Test a labels matrix with two objects touching"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1  # Pythagoras triangle 3-4-5
    labels[5, 6] = 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 5
    )
    module.run(workspace)
    m = workspace.measurements
    neighbors = m.get_current_measurement(OBJECTS_NAME, "Neighbors_NumberOfNeighbors_5")
    assert len(neighbors) == 2
    assert numpy.all(neighbors == 1)
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_5")
    assert len(pct) == 2
    assert round(abs(pct[0] - 100), 7) == 0

    fo = m.get_current_measurement(OBJECTS_NAME, "Neighbors_FirstClosestObjectNumber_5")
    assert len(fo) == 2
    assert fo[0] == 2
    assert fo[1] == 1


def test_three():
    """Test the angles between three objects"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1  # x=3,y=4,5 triangle
    labels[2, 5] = 2
    labels[6, 2] = 3
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 5
    )
    module.run(workspace)
    m = workspace.measurements
    fo = m.get_current_measurement(OBJECTS_NAME, "Neighbors_FirstClosestObjectNumber_5")
    assert len(fo) == 3
    assert fo[0] == 2
    assert fo[1] == 1
    assert fo[2] == 1
    so = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_SecondClosestObjectNumber_5"
    )
    assert len(so) == 3
    assert so[0] == 3
    assert so[1] == 3
    assert so[2] == 2
    d = m.get_current_measurement(OBJECTS_NAME, "Neighbors_SecondClosestDistance_5")
    assert len(d) == 3
    assert round(abs(d[0] - 4), 7) == 0
    assert round(abs(d[1] - 5), 7) == 0
    assert round(abs(d[2] - 5), 7) == 0

    angle = m.get_current_measurement(OBJECTS_NAME, "Neighbors_AngleBetweenNeighbors_5")
    assert len(angle) == 3
    assert round(abs(angle[0] - 90), 7) == 0
    assert round(abs(angle[1] - numpy.arccos(3.0 / 5.0) * 180.0 / numpy.pi), 7) == 0
    assert round(abs(angle[2] - numpy.arccos(4.0 / 5.0) * 180.0 / numpy.pi), 7) == 0


def test_touching_discarded():
    """Make sure that we count edge-touching discarded objects

    Regression test of IMG-1012.
    """
    labels = numpy.zeros((10, 10), int)
    labels[2, 3] = 1
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5
    )
    object_set = workspace.object_set
    assert isinstance(object_set, cellprofiler_core.object.ObjectSet)
    objects = object_set.get_objects(OBJECTS_NAME)
    assert isinstance(objects, cellprofiler_core.object.Objects)

    sm_labels = labels.copy() * 3
    sm_labels[-1, -1] = 1
    sm_labels[0:2, 3] = 2
    objects.small_removed_segmented = sm_labels
    module.run(workspace)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Adjacent"
    )
    assert len(neighbors) == 1
    assert numpy.all(neighbors == 1)
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_Adjacent")
    assert len(pct) == 1
    assert round(abs(pct[0] - 100), 7) == 0
    fo = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_FirstClosestObjectNumber_Adjacent"
    )
    assert len(fo) == 1
    assert fo[0] == 0

    angle = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_AngleBetweenNeighbors_Adjacent"
    )
    assert len(angle) == 1
    assert not numpy.isnan(angle)[0]


def test_all_discarded():
    """Test the case where all objects touch the edge

    Regression test of a follow-on bug to IMG-1012
    """
    labels = numpy.zeros((10, 10), int)
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5
    )
    object_set = workspace.object_set
    assert isinstance(object_set, cellprofiler_core.object.ObjectSet)
    objects = object_set.get_objects(OBJECTS_NAME)
    assert isinstance(objects, cellprofiler_core.object.Objects)

    # Needs 2 objects to trigger the bug
    sm_labels = numpy.zeros((10, 10), int)
    sm_labels[0:2, 3] = 1
    sm_labels[-3:-1, 5] = 2
    objects.small_removed_segmented = sm_labels
    module.run(workspace)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Adjacent"
    )
    assert len(neighbors) == 0
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_Adjacent")
    assert len(pct) == 0
    fo = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_FirstClosestObjectNumber_Adjacent"
    )
    assert len(fo) == 0


def test_NeighborCountImage():
    """Test production of a neighbor-count image"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1  # x=3,y=4,5 triangle
    labels[2, 5] = 2
    labels[6, 2] = 3
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 4
    )
    module.wants_count_image.value = True
    module.count_image_name.value = "my_image"
    module.count_colormap.value = "jet"
    module.run(workspace)
    image = workspace.image_set.get_image("my_image").pixel_data
    assert tuple(image.shape) == (10, 10, 3)
    # Everything off of the images should be black
    assert numpy.all(image[labels[labels == 0], :] == 0)
    # The corners should match 1 neighbor and should get the same color
    assert numpy.all(image[2, 5, :] == image[6, 2, :])
    # The pixel at the right angle should have a different color
    assert not numpy.all(image[2, 2, :] == image[2, 5, :])


def test_PercentTouchingImage():
    """Test production of a percent touching image"""
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1
    labels[2, 5] = 2
    labels[6, 2] = 3
    labels[7, 2] = 3
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 4
    )
    module.wants_percent_touching_image.value = True
    module.touching_image_name.value = "my_image"
    module.touching_colormap.value = "jet"
    module.run(workspace)
    image = workspace.image_set.get_image("my_image").pixel_data
    assert tuple(image.shape) == (10, 10, 3)
    # Everything off of the images should be black
    assert numpy.all(image[labels[labels == 0], :] == 0)
    # 1 and 2 are at 100 %
    assert numpy.all(image[2, 2, :] == image[2, 5, :])
    # 3 is at 50% and should have a different color
    assert not numpy.all(image[2, 2, :] == image[6, 2, :])


def test_get_measurement_columns():
    """Test the get_measurement_columns method"""
    module = cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors()
    module.object_name.value = OBJECTS_NAME
    module.neighbors_name.value = OBJECTS_NAME
    module.distance.value = 5
    for distance_method, scale in (
        (
            cellprofiler.modules.measureobjectneighbors.D_EXPAND,
            cellprofiler.modules.measureobjectneighbors.S_EXPANDED,
        ),
        (
            cellprofiler.modules.measureobjectneighbors.D_ADJACENT,
            cellprofiler.modules.measureobjectneighbors.S_ADJACENT,
        ),
        (cellprofiler.modules.measureobjectneighbors.D_WITHIN, "5"),
    ):
        module.distance_method.value = distance_method
        columns = module.get_measurement_columns(None)
        features = [
            "%s_%s_%s"
            % (cellprofiler.modules.measureobjectneighbors.C_NEIGHBORS, feature, scale)
            for feature in cellprofiler.modules.measureobjectneighbors.M_ALL
        ]
        assert len(columns) == len(features)
        for column in columns:
            assert column[1] in features, "Unexpected column name: %s" % column[1]


def test_get_measurement_columns_neighbors():
    module = cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors()
    module.object_name.value = OBJECTS_NAME
    module.neighbors_name.value = NEIGHBORS_NAME
    module.distance.value = 5
    for distance_method, scale in (
        (
            cellprofiler.modules.measureobjectneighbors.D_EXPAND,
            cellprofiler.modules.measureobjectneighbors.S_EXPANDED,
        ),
        (
            cellprofiler.modules.measureobjectneighbors.D_ADJACENT,
            cellprofiler.modules.measureobjectneighbors.S_ADJACENT,
        ),
        (cellprofiler.modules.measureobjectneighbors.D_WITHIN, "5"),
    ):
        module.distance_method.value = distance_method
        columns = module.get_measurement_columns(None)
        features = [
            "%s_%s_%s_%s"
            % (
                cellprofiler.modules.measureobjectneighbors.C_NEIGHBORS,
                feature,
                NEIGHBORS_NAME,
                scale,
            )
            for feature in cellprofiler.modules.measureobjectneighbors.M_ALL
        ]
        assert len(columns) == len(features)
        for column in columns:
            assert column[1] in features, "Unexpected column name: %s" % column[1]


def test_neighbors_zeros():
    blank_labels = numpy.zeros((20, 10), int)
    one_object = numpy.zeros((20, 10), int)
    one_object[2:-2, 2:-2] = 1

    cases = (
        (blank_labels, blank_labels, 0, 0),
        (blank_labels, one_object, 0, 1),
        (one_object, blank_labels, 1, 0),
    )
    for olabels, nlabels, ocount, ncount in cases:
        for mode in cellprofiler.modules.measureobjectneighbors.D_ALL:
            workspace, module = make_workspace(olabels, mode, neighbors_labels=nlabels)
            assert isinstance(
                module,
                cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors,
            )
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m,cellprofiler_core.measurement.Measurements)
            for feature in module.all_features:
                v = m.get_current_measurement(
                    OBJECTS_NAME, module.get_measurement_name(feature)
                )
                assert len(v) == ocount


def test_one_neighbor():
    olabels = numpy.zeros((20, 10), int)
    olabels[2, 2] = 1
    nlabels = numpy.zeros((20, 10), int)
    nlabels[-2, -2] = 1
    for mode in cellprofiler.modules.measureobjectneighbors.D_ALL:
        workspace, module = make_workspace(
            olabels, mode, distance=20, neighbors_labels=nlabels
        )
        assert isinstance(
            module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors
        )
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m,cellprofiler_core.measurement.Measurements)
        v = m.get_current_measurement(
            OBJECTS_NAME,
            module.get_measurement_name(
                cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_OBJECT_NUMBER
            ),
        )
        assert len(v) == 1
        assert v[0] == 1
        v = m.get_current_measurement(
            OBJECTS_NAME,
            module.get_measurement_name(
                cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_OBJECT_NUMBER
            ),
        )
        assert len(v) == 1
        assert v[0] == 0
        v = m.get_current_measurement(
            OBJECTS_NAME,
            module.get_measurement_name(
                cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_DISTANCE
            ),
        )
        assert len(v) == 1
        assert round(abs(v[0] - numpy.sqrt(16 ** 2 + 6 ** 2)), 7) == 0
        v = m.get_current_measurement(
            OBJECTS_NAME,
            module.get_measurement_name(
                cellprofiler.modules.measureobjectneighbors.M_NUMBER_OF_NEIGHBORS
            ),
        )
        assert len(v) == 1
        assert v[0] == (
            0 if mode == cellprofiler.modules.measureobjectneighbors.D_ADJACENT else 1
        )


def test_two_neighbors():
    olabels = numpy.zeros((20, 10), int)
    olabels[2, 2] = 1
    nlabels = numpy.zeros((20, 10), int)
    nlabels[5, 2] = 2
    nlabels[2, 6] = 1
    workspace, module = make_workspace(
        olabels,
        cellprofiler.modules.measureobjectneighbors.D_EXPAND,
        distance=20,
        neighbors_labels=nlabels,
    )
    assert isinstance(
        module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_OBJECT_NUMBER
        ),
    )
    assert len(v) == 1
    assert v[0] == 2
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_OBJECT_NUMBER
        ),
    )
    assert len(v) == 1
    assert v[0] == 1
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_DISTANCE
        ),
    )
    assert len(v) == 1
    assert round(abs(v[0] - 3), 7) == 0
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_DISTANCE
        ),
    )
    assert len(v) == 1
    assert round(abs(v[0] - 4), 7) == 0
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_ANGLE_BETWEEN_NEIGHBORS
        ),
    )
    assert len(v) == 1
    assert round(abs(v[0] - 90), 7) == 0


def test_different_neighbors_touching():
    olabels = numpy.zeros((20, 10), int)
    olabels[2:5, 2:5] = 1
    nlabels = numpy.zeros((20, 10), int)
    nlabels[3, 5] = 2
    nlabels[5, 2] = 1
    workspace, module = make_workspace(
        olabels,
        cellprofiler.modules.measureobjectneighbors.D_ADJACENT,
        distance=0,
        neighbors_labels=nlabels,
    )
    assert isinstance(
        module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_OBJECT_NUMBER
        ),
    )
    assert len(v) == 1
    assert v[0] == 2
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_OBJECT_NUMBER
        ),
    )
    assert len(v) == 1
    assert v[0] == 1
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_DISTANCE
        ),
    )
    assert len(v) == 1
    assert v[0] == 2
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_SECOND_CLOSEST_DISTANCE
        ),
    )
    assert len(v) == 1
    assert round(v[0], 7) == round(5 ** (0.5), 7)
    v = m.get_current_measurement(
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_PERCENT_TOUCHING
        ),
    )
    assert len(v) == 1
    assert round(v[0], 7) == 62.5


def test_relationships():
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 2
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    k = m.get_relationship_groups()
    assert len(k) == 1
    k = k[0]
    assert isinstance(k, RelationshipKey)
    assert k.module_number == 1
    assert k.object_name1 == OBJECTS_NAME
    assert k.object_name2 == OBJECTS_NAME
    assert k.relationship == NEIGHBORS
    r = m.get_relationships(
        k.module_number, k.relationship, k.object_name1, k.object_name2
    )
    assert len(r) == 8
    ro1 = r[R_FIRST_OBJECT_NUMBER]
    ro2 = r[R_SECOND_OBJECT_NUMBER]
    numpy.testing.assert_array_equal(
        numpy.unique(ro1[ro2 == 3]), numpy.array([1, 2, 4, 5])
    )
    numpy.testing.assert_array_equal(
        numpy.unique(ro2[ro1 == 3]), numpy.array([1, 2, 4, 5])
    )


def test_neighbors():
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 1, 1, 1, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 3, 3, 3, 0, 0, 0],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 4, 4, 4, 0, 0, 0, 5, 5, 5],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    nlabels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_WITHIN, 2, nlabels
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    k = m.get_relationship_groups()
    assert len(k) == 1
    k = k[0]
    assert isinstance(k, RelationshipKey)
    assert k.module_number == 1
    assert k.object_name1 == OBJECTS_NAME
    assert k.object_name2 == NEIGHBORS_NAME
    assert k.relationship == NEIGHBORS
    r = m.get_relationships(
        k.module_number, k.relationship, k.object_name1, k.object_name2
    )
    assert len(r) == 3
    ro1 = r[R_FIRST_OBJECT_NUMBER]
    ro2 = r[R_SECOND_OBJECT_NUMBER]
    assert numpy.all(ro2 == 1)
    numpy.testing.assert_array_equal(numpy.unique(ro1), numpy.array([1, 3, 4]))


def test_missing_object():
    # Regression test of issue 434
    #
    # Catch case of no pixels for an object
    #
    labels = numpy.zeros((10, 10), int)
    labels[2, 2] = 1
    labels[2, 3] = 3
    workspace, module = make_workspace(
        labels, cellprofiler.modules.measureobjectneighbors.D_ADJACENT, 5
    )
    module.run(workspace)
    m = workspace.measurements
    neighbors = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_NumberOfNeighbors_Adjacent"
    )
    numpy.testing.assert_array_equal(neighbors, [1, 0, 1])
    pct = m.get_current_measurement(OBJECTS_NAME, "Neighbors_PercentTouching_Adjacent")
    numpy.testing.assert_array_almost_equal(pct, [100.0, 0, 100.0])
    fo = m.get_current_measurement(
        OBJECTS_NAME, "Neighbors_FirstClosestObjectNumber_Adjacent"
    )
    numpy.testing.assert_array_equal(fo, [3, 0, 1])


def test_small_removed():
    # Regression test of issue #1179
    #
    # neighbor_objects.small_removed_segmented + objects touching border
    # with higher object numbers
    #
    neighbors = numpy.zeros((11, 13), int)
    neighbors[5:7, 4:8] = 1
    neighbors_unedited = numpy.zeros((11, 13), int)
    neighbors_unedited[5:7, 4:8] = 1
    neighbors_unedited[0:4, 4:8] = 2

    objects = numpy.zeros((11, 13), int)
    objects[1:6, 5:7] = 1

    workspace, module = make_workspace(
        objects,
        cellprofiler.modules.measureobjectneighbors.D_WITHIN,
        neighbors_labels=neighbors,
    )
    no = workspace.object_set.get_objects(NEIGHBORS_NAME)
    no.small_removed_segmented = neighbors_unedited
    no.segmented = neighbors
    module.run(workspace)
    m = workspace.measurements
    v = m[
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_NUMBER_OF_NEIGHBORS
        ),
        1,
    ]
    assert len(v) == 1
    assert v[0] == 2


def test_object_is_missing():
    # regression test of #1639
    #
    # Object # 2 should match neighbor # 1, but because of
    # an error in masking distances, neighbor #1 is masked out
    #
    olabels = numpy.zeros((20, 10), int)
    olabels[2, 2] = 2
    nlabels = numpy.zeros((20, 10), int)
    nlabels[2, 3] = 1
    nlabels[5, 2] = 2
    workspace, module = make_workspace(
        olabels,
        cellprofiler.modules.measureobjectneighbors.D_EXPAND,
        distance=20,
        neighbors_labels=nlabels,
    )
    assert isinstance(
        module, cellprofiler.modules.measureobjectneighbors.MeasureObjectNeighbors
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    ftr = module.get_measurement_name(
        cellprofiler.modules.measureobjectneighbors.M_FIRST_CLOSEST_OBJECT_NUMBER
    )
    values = m[OBJECTS_NAME, ftr]
    assert values[1] == 1


def test_small_removed_same():
    # Regression test of issue #1672
    #
    # Objects with small removed failed.
    #
    objects = numpy.zeros((11, 13), int)
    objects[5:7, 1:3] = 1
    objects[6:8, 5:7] = 2
    objects_unedited = objects.copy()
    objects_unedited[0:2, 0:2] = 3

    workspace, module = make_workspace(
        objects, cellprofiler.modules.measureobjectneighbors.D_EXPAND, distance=1
    )
    no = workspace.object_set.get_objects(OBJECTS_NAME)
    no.unedited_segmented = objects_unedited
    no.small_removed_segmented = objects
    module.run(workspace)
    m = workspace.measurements
    v = m[
        OBJECTS_NAME,
        module.get_measurement_name(
            cellprofiler.modules.measureobjectneighbors.M_NUMBER_OF_NEIGHBORS
        ),
        1,
    ]
    assert len(v) == 2
    assert v[0] == 1
