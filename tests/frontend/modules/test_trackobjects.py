import centrosome.filter
import numpy
import six.moves

from cellprofiler_core.constants.measurement import (
    GROUP_NUMBER,
    GROUP_INDEX,
    R_FIRST_IMAGE_NUMBER,
    R_SECOND_IMAGE_NUMBER,
    R_FIRST_OBJECT_NUMBER,
    R_SECOND_OBJECT_NUMBER,
    C_COUNT,
    MCA_AVAILABLE_POST_GROUP,
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y,
)
from cellprofiler_core.image import ImageSetList
import cellprofiler_core.measurement
from cellprofiler_core.object import ObjectSet, Objects

import cellprofiler.modules.trackobjects
import tests.frontend.modules
from cellprofiler_core.pipeline import Pipeline, LoadException, RunException
from cellprofiler_core.workspace import Workspace

OBJECT_NAME = "objects"


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("trackobjects/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = Pipeline()

    def callback(caller, event):
        assert not isinstance(event, LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    assert module.tracking_method == "LAP"
    assert module.object_name.value == "Nuclei"
    assert module.pixel_radius.value == 80
    assert module.display_type.value == "Color and Number"
    assert not module.wants_image
    assert module.measurement == "AreaShape_Area"
    assert module.image_name == "TrackedCells"
    assert module.wants_second_phase
    assert module.split_cost == 41
    assert module.merge_cost == 42
    assert module.max_gap_score == 53
    assert module.max_split_score == 54
    assert module.max_merge_score == 55
    assert module.max_frame_distance == 6


def test_load_v4():
    file = tests.frontend.modules.get_test_resources_directory("trackobjects/v4.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = Pipeline()

    def callback(caller, event):
        assert not isinstance(event, LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 3
    for module, tracking_method, model, save_img, phase2, meas, dop in zip(
        pipeline.modules(),
        ("Measurements", "Overlap", "Distance"),
        (
            cellprofiler.modules.trackobjects.M_BOTH,
            cellprofiler.modules.trackobjects.M_RANDOM,
            cellprofiler.modules.trackobjects.M_VELOCITY,
        ),
        (True, False, True),
        (True, False, True),
        ("Slothfulness", "Prescience", "Trepidation"),
        (
            cellprofiler.modules.trackobjects.DT_COLOR_AND_NUMBER,
            cellprofiler.modules.trackobjects.DT_COLOR_ONLY,
            cellprofiler.modules.trackobjects.DT_COLOR_AND_NUMBER,
        ),
    ):
        assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
        assert module.tracking_method == tracking_method
        assert module.model == model
        assert module.wants_image.value == save_img
        assert module.wants_second_phase.value == phase2
        assert module.measurement == meas
        assert module.pixel_radius == 50
        assert module.display_type == dop
        assert module.image_name == "TrackByLAP"
        assert module.radius_std == 3
        assert module.radius_limit.min == 3.0
        assert module.radius_limit.max == 10.0
        assert module.gap_cost == 40
        assert module.split_cost == 1
        assert module.merge_cost == 1
        assert module.max_gap_score == 51
        assert module.max_split_score == 52
        assert module.max_merge_score == 53
        assert module.max_frame_distance == 4


def test_load_v5():
    file = tests.frontend.modules.get_test_resources_directory("trackobjects/v5.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = Pipeline()

    def callback(caller, event):
        assert not isinstance(event, LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    m = pipeline.modules()[0]
    assert isinstance(m, cellprofiler.modules.trackobjects.TrackObjects)
    assert m.tracking_method == "LAP"
    assert m.object_name == "Turtles"
    assert m.measurement == "Steadiness"
    assert m.pixel_radius == 44
    assert m.display_type == cellprofiler.modules.trackobjects.DT_COLOR_AND_NUMBER
    assert not m.wants_image
    assert m.image_name == "TrackedTurtles"
    assert m.model == cellprofiler.modules.trackobjects.M_BOTH
    assert m.radius_std == 3
    assert m.radius_limit.min == 3
    assert m.radius_limit.max == 11
    assert m.wants_second_phase
    assert m.gap_cost == 39
    assert m.split_cost == 41
    assert m.merge_cost == 42
    assert m.max_frame_distance == 8
    assert m.wants_minimum_lifetime
    assert m.min_lifetime == 2
    assert not m.wants_maximum_lifetime
    assert m.max_lifetime == 1000


def test_load_v6():
    file = tests.frontend.modules.get_test_resources_directory("trackobjects/v6.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = Pipeline()

    def callback(caller, event):
        assert not isinstance(event, LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    m = pipeline.modules()[0]
    assert isinstance(m, cellprofiler.modules.trackobjects.TrackObjects)
    assert m.tracking_method == "LAP"
    assert m.object_name == "Turtles"
    assert m.measurement == "Steadiness"
    assert m.pixel_radius == 44
    assert m.display_type == cellprofiler.modules.trackobjects.DT_COLOR_AND_NUMBER
    assert not m.wants_image
    assert m.image_name == "TrackedTurtles"
    assert m.model == cellprofiler.modules.trackobjects.M_BOTH
    assert m.radius_std == 3
    assert m.radius_limit.min == 3
    assert m.radius_limit.max == 11
    assert m.wants_second_phase
    assert m.gap_cost == 39
    assert m.split_cost == 41
    assert m.merge_cost == 42
    assert m.max_frame_distance == 8
    assert m.wants_minimum_lifetime
    assert m.min_lifetime == 2
    assert not m.wants_maximum_lifetime
    assert m.max_lifetime == 1000
    assert m.mitosis_cost == 79
    assert m.mitosis_max_distance == 41


def runTrackObjects(labels_list, fn=None, measurement=None):
    """Run two cycles of TrackObjects

    labels1 - the labels matrix for the first cycle
    labels2 - the labels matrix for the second cycle
    fn - a callback function called with the module and workspace. It has
         the signature, fn(module, workspace, n) where n is 0 when
         called prior to prepare_run, 1 prior to first iteration
         and 2 prior to second iteration.

    returns the measurements
    """
    module = cellprofiler.modules.trackobjects.TrackObjects()
    module.set_module_num(1)
    module.object_name.value = OBJECT_NAME
    module.pixel_radius.value = 50
    module.measurement.value = "measurement"
    measurements = cellprofiler_core.measurement.Measurements()
    measurements.add_all_measurements(
        "Image", GROUP_NUMBER, [1] * len(labels_list),
    )
    measurements.add_all_measurements(
        "Image", GROUP_INDEX, list(range(1, len(labels_list) + 1)),
    )
    pipeline = Pipeline()
    pipeline.add_module(module)
    image_set_list = ImageSetList()

    if fn:
        fn(module, None, 0)
    module.prepare_run(
        Workspace(pipeline, module, None, None, measurements, image_set_list)
    )

    first = True
    for labels, index in zip(labels_list, list(range(len(labels_list)))):
        object_set = ObjectSet()
        objects = Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECT_NAME)
        image_set = image_set_list.get_image_set(index)
        if first:
            first = False
        else:
            measurements.next_image_set()
        if measurement is not None:
            measurements.add_measurement(
                OBJECT_NAME, "measurement", numpy.array(measurement[index])
            )
        workspace = Workspace(
            pipeline, module, image_set, object_set, measurements, image_set_list
        )
        if fn:
            fn(module, workspace, index + 1)

        module.run(workspace)
    return measurements


def test_track_nothing():
    """Run TrackObjects on an empty labels matrix"""
    columns = []

    def fn(module, workspace, index, columns=columns):
        if workspace is not None and index == 0:
            columns += module.get_measurement_columns(workspace.pipeline)

    measurements = runTrackObjects(
        (numpy.zeros((10, 10), int), numpy.zeros((10, 10), int)), fn
    )

    features = [
        feature
        for feature in measurements.get_feature_names(OBJECT_NAME)
        if feature.startswith(cellprofiler.modules.trackobjects.F_PREFIX)
    ]
    assert all(
        [column[1] in features for column in columns if column[0] == OBJECT_NAME]
    )
    for feature in cellprofiler.modules.trackobjects.F_ALL:
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "50"))
        assert name in features
        value = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(value) == 0

    features = [
        feature
        for feature in measurements.get_feature_names("Image")
        if feature.startswith(cellprofiler.modules.trackobjects.F_PREFIX)
    ]
    assert all([column[1] in features for column in columns if column[0] == "Image"])
    for feature in cellprofiler.modules.trackobjects.F_IMAGE_ALL:
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "50")
        )
        assert name in features
        value = measurements.get_current_image_measurement(name)
        assert value == 0


def test_00_track_one_then_nothing():
    """Run track objects on an object that disappears

    Regression test of IMG-1090
    """
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 2:7] = 1
    measurements = runTrackObjects((labels, numpy.zeros((10, 10), int)))
    feature = "_".join(
        (
            cellprofiler.modules.trackobjects.F_PREFIX,
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT,
            OBJECT_NAME,
            "50",
        )
    )
    value = measurements.get_current_image_measurement(feature)
    assert value == 1


def test_track_one_distance():
    """Track an object that doesn't move using distance"""
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 2:7] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 1
            module.tracking_method.value = "Distance"

    measurements = runTrackObjects((labels, labels), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "1"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert round(abs(m(cellprofiler.modules.trackobjects.F_TRAJECTORY_X) - 0), 7) == 0
    assert round(abs(m(cellprofiler.modules.trackobjects.F_TRAJECTORY_Y) - 0), 7) == 0
    assert (
        round(abs(m(cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED) - 0), 7) == 0
    )
    assert (
        round(abs(m(cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE) - 0), 7)
        == 0
    )
    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 1
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 1
    assert m(cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER) == 1
    assert m(cellprofiler.modules.trackobjects.F_LIFETIME) == 2

    def m(feature):
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "1")
        )
        return measurements.get_current_image_measurement(name)

    assert m(cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_SPLIT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_MERGE_COUNT) == 0
    check_relationships(measurements, [1], [1], [2], [1])


def test_track_one_moving():
    """Track an object that moves"""

    labels_list = []
    distance = 0
    last_i, last_j = (0, 0)
    for i_off, j_off in ((0, 0), (2, 0), (2, 1), (0, 1)):
        distance = i_off - last_i + j_off - last_j
        last_i, last_j = (i_off, j_off)
        labels = numpy.zeros((10, 10), int)
        labels[4 + i_off : 7 + i_off, 4 + j_off : 7 + j_off] = 1
        labels_list.append(labels)

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 3
            module.tracking_method.value = "Distance"

    measurements = runTrackObjects(labels_list, fn)

    def m(feature, expected):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "3"))
        value_set = measurements.get_measurement(OBJECT_NAME, name, measurements.get_image_numbers())
        assert len(expected) == len(value_set)
        for values, x in zip(value_set, expected):
            assert len(values) == 1
            assert round(abs(values[0] - x), 7) == 0

    m(cellprofiler.modules.trackobjects.F_TRAJECTORY_X, [0, 0, 1, 0])
    m(cellprofiler.modules.trackobjects.F_TRAJECTORY_Y, [0, 2, 0, -2])
    m(cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED, [0, 2, 1, 2])
    m(cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE, [0, 2, 3, 5])
    m(cellprofiler.modules.trackobjects.F_LABEL, [1, 1, 1, 1])
    m(cellprofiler.modules.trackobjects.F_LIFETIME, [1, 2, 3, 4])
    m(
        cellprofiler.modules.trackobjects.F_LINEARITY,
        [1, 1, numpy.sqrt(5) / 3, 1.0 / 5.0],
    )

    def m(feature):
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "3")
        )
        return measurements.get_current_image_measurement(name)

    assert m(cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_SPLIT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_MERGE_COUNT) == 0
    image_numbers = numpy.arange(1, len(labels_list) + 1)
    object_numbers = numpy.ones(len(image_numbers))
    check_relationships(
        measurements,
        image_numbers[:-1],
        object_numbers[:-1],
        image_numbers[1:],
        object_numbers[1:],
    )


def test_track_split():
    """Track an object that splits"""
    labels1 = numpy.zeros((11, 9), int)
    labels1[1:10, 1:8] = 1
    labels2 = numpy.zeros((10, 10), int)
    labels2[1:6, 1:8] = 1
    labels2[6:10, 1:8] = 2

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 5
            module.tracking_method.value = "Distance"

    measurements = runTrackObjects((labels1, labels2, labels2), fn)

    def m(feature, idx):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "5"))
        values = measurements.get_measurement(OBJECT_NAME, name, idx + 1)
        assert len(values) == 2
        return values

    labels = m(cellprofiler.modules.trackobjects.F_LABEL, 2)
    assert len(labels) == 2
    assert numpy.all(labels == 1)
    parents = m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER, 1)
    assert numpy.all(parents == 1)
    assert numpy.all(m(cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER, 1) == 1)
    parents = m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER, 2)
    assert numpy.all(parents == numpy.array([1, 2]))
    assert numpy.all(m(cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER, 2) == 2)

    def m(feature):
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "5")
        )
        return measurements.get_measurement("Image", name, measurements.get_image_numbers())[1]

    assert m(cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_SPLIT_COUNT) == 1
    assert m(cellprofiler.modules.trackobjects.F_MERGE_COUNT) == 0
    check_relationships(
        measurements, [1, 1, 2, 2], [1, 1, 1, 2], [2, 2, 3, 3], [1, 2, 1, 2]
    )


def test_track_negative():
    """Track unrelated objects"""
    labels1 = numpy.zeros((10, 10), int)
    labels1[1:5, 1:5] = 1
    labels2 = numpy.zeros((10, 10), int)
    labels2[6:9, 6:9] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 1
            module.tracking_method.value = "Distance"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "1"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 0

    def m(feature):
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "1")
        )
        return measurements.get_current_image_measurement(name)

    assert m(cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT) == 1
    assert m(cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT) == 1
    assert m(cellprofiler.modules.trackobjects.F_SPLIT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_MERGE_COUNT) == 0


def test_track_ambiguous():
    """Track disambiguation from among two possible parents"""
    labels1 = numpy.zeros((20, 20), int)
    labels1[1:4, 1:4] = 1
    labels1[16:19, 16:19] = 2
    labels2 = numpy.zeros((20, 20), int)
    labels2[10:15, 10:15] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 20
            module.tracking_method.value = "Distance"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "20"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 2


def test_overlap_positive():
    """Track overlapping objects"""
    labels1 = numpy.zeros((10, 10), int)
    labels1[3:6, 4:7] = 1
    labels2 = numpy.zeros((10, 10), int)
    labels2[4:7, 5:9] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 2
            module.tracking_method.value = "Overlap"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "2"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 1
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 1


def test_overlap_negative():
    """Track objects that don't overlap"""
    labels1 = numpy.zeros((20, 20), int)
    labels1[3:6, 4:7] = 1
    labels2 = numpy.zeros((20, 20), int)
    labels2[14:17, 15:19] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 2
            module.tracking_method.value = "Overlap"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "2"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 0


def test_overlap_ambiguous():
    """Track an object that overlaps two parents"""
    labels1 = numpy.zeros((20, 20), int)
    labels1[1:5, 1:5] = 1
    labels1[15:19, 15:19] = 2
    labels2 = numpy.zeros((20, 20), int)
    labels2[4:18, 4:18] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 2
            module.tracking_method.value = "Overlap"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "2"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 2


def test_measurement_positive():
    """Test tracking an object by measurement"""
    labels1 = numpy.zeros((10, 10), int)
    labels1[3:6, 4:7] = 1
    labels2 = numpy.zeros((10, 10), int)
    labels2[4:7, 5:9] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 2
            module.tracking_method.value = "Measurements"

    measurements = runTrackObjects((labels1, labels2), fn, [[1], [1]])

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "2"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 1
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 1


def test_measurement_negative():
    """Test tracking with too great a jump between successive images"""
    labels1 = numpy.zeros((20, 20), int)
    labels1[3:6, 4:7] = 1
    labels2 = numpy.zeros((20, 20), int)
    labels2[14:17, 15:19] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 2
            module.tracking_method.value = "Measurements"

    measurements = runTrackObjects((labels1, labels2), fn, [[1], [1]])

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "2"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 0


def test_ambiguous():
    """Test measurement with ambiguous parent choice"""
    labels1 = numpy.zeros((20, 20), int)
    labels1[1:5, 1:5] = 1
    labels1[15:19, 15:19] = 2
    labels2 = numpy.zeros((20, 20), int)
    labels2[6:14, 6:14] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 4
            module.tracking_method.value = "Measurements"

    measurements = runTrackObjects((labels1, labels2), fn, [[1, 10], [9]])

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "4"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 2


def test_cross_numbered_objects():
    """Test labeling when object 1 in one image becomes object 2 in next"""

    i, j = numpy.mgrid[0:10, 0:20]
    labels = (i > 5) + (j > 10) * 2
    pp = numpy.array(list(centrosome.filter.permutations([1, 2, 3, 4])))

    def fn(module, workspace, idx):
        if idx == 0:
            module.tracking_method.value = "LAP"

    measurements = runTrackObjects([numpy.array(p)[labels] for p in pp], fn)

    def m(feature, i):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature))
        values = measurements[OBJECT_NAME, name, i + 1]
        assert len(values) == 4
        return values

    for i, p in enumerate(pp):
        l = m(cellprofiler.modules.trackobjects.F_LABEL, i)
        numpy.testing.assert_array_equal(numpy.arange(1, 5), p[l - 1])
        if i > 0:
            p_prev = pp[i - 1]
            order = numpy.lexsort([p])
            expected_po = p_prev[order]
            po = m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER, i)
            numpy.testing.assert_array_equal(po, expected_po)
            pi = m(cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER, i)
            numpy.testing.assert_array_equal(pi, i)
    image_numbers, _ = numpy.mgrid[1 : (len(pp) + 1), 0:4]
    check_relationships(
        measurements,
        image_numbers[:-1, :].flatten(),
        pp[:-1, :].flatten(),
        image_numbers[1:, :].flatten(),
        pp[1:, :].flatten(),
    )


def test_measurement_columns():
    """Test get_measurement_columns function"""
    module = cellprofiler.modules.trackobjects.TrackObjects()
    module.object_name.value = OBJECT_NAME
    module.tracking_method.value = "Distance"
    module.pixel_radius.value = 10
    columns = module.get_measurement_columns(None)
    assert len(columns) == len(cellprofiler.modules.trackobjects.F_ALL) + len(
        cellprofiler.modules.trackobjects.F_IMAGE_ALL
    )
    for object_name, features in (
        (OBJECT_NAME, cellprofiler.modules.trackobjects.F_ALL),
        ("Image", cellprofiler.modules.trackobjects.F_IMAGE_ALL,),
    ):
        for feature in features:
            if object_name == OBJECT_NAME:
                name = "_".join(
                    (cellprofiler.modules.trackobjects.F_PREFIX, feature, "10")
                )
            else:
                name = "_".join(
                    (
                        cellprofiler.modules.trackobjects.F_PREFIX,
                        feature,
                        OBJECT_NAME,
                        "10",
                    )
                )
            index = [column[1] for column in columns].index(name)
            assert index != -1
            column = columns[index]
            assert column[0] == object_name


def test_measurement_columns_lap():
    """Test get_measurement_columns function for LAP"""
    module = cellprofiler.modules.trackobjects.TrackObjects()
    module.object_name.value = OBJECT_NAME
    module.tracking_method.value = "LAP"
    module.model.value = cellprofiler.modules.trackobjects.M_BOTH
    second_phase = [
        cellprofiler.modules.trackobjects.F_LINKING_DISTANCE,
        cellprofiler.modules.trackobjects.F_MOVEMENT_MODEL,
    ]
    for wants in (True, False):
        module.wants_second_phase.value = wants
        columns = module.get_measurement_columns(None)
        # 2, 2, 4 for the static model
        # 4, 4, 16 for the velocity model
        other_features = [
            cellprofiler.modules.trackobjects.F_AREA,
            cellprofiler.modules.trackobjects.F_LINKING_DISTANCE,
            cellprofiler.modules.trackobjects.F_LINK_TYPE,
            cellprofiler.modules.trackobjects.F_MOVEMENT_MODEL,
            cellprofiler.modules.trackobjects.F_STANDARD_DEVIATION,
        ]
        if wants:
            other_features += [
                cellprofiler.modules.trackobjects.F_GAP_LENGTH,
                cellprofiler.modules.trackobjects.F_GAP_SCORE,
                cellprofiler.modules.trackobjects.F_MERGE_SCORE,
                cellprofiler.modules.trackobjects.F_SPLIT_SCORE,
                cellprofiler.modules.trackobjects.F_MITOSIS_SCORE,
            ]
        assert (
            len(columns)
            == len(cellprofiler.modules.trackobjects.F_ALL)
            + len(cellprofiler.modules.trackobjects.F_IMAGE_ALL)
            + len(other_features)
            + 2
            + 2
            + 4
            + 4
            + 4
            + 16
        )
        kalman_features = [
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_STATE,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_STATE,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_STATE,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_STATE,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_STATE,
                cellprofiler.modules.trackobjects.F_VX,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_STATE,
                cellprofiler.modules.trackobjects.F_VY,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_NOISE,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_NOISE,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_NOISE,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_NOISE,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_NOISE,
                cellprofiler.modules.trackobjects.F_VX,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_NOISE,
                cellprofiler.modules.trackobjects.F_VY,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_X,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_X,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_Y,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_STATIC_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_X,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_X,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_X,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_X,
                cellprofiler.modules.trackobjects.F_VX,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_X,
                cellprofiler.modules.trackobjects.F_VY,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_Y,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_Y,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_Y,
                cellprofiler.modules.trackobjects.F_VX,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_Y,
                cellprofiler.modules.trackobjects.F_VY,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VX,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VX,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VX,
                cellprofiler.modules.trackobjects.F_VX,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VX,
                cellprofiler.modules.trackobjects.F_VY,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VY,
                cellprofiler.modules.trackobjects.F_X,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VY,
                cellprofiler.modules.trackobjects.F_Y,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VY,
                cellprofiler.modules.trackobjects.F_VX,
            ),
            cellprofiler.modules.trackobjects.kalman_feature(
                cellprofiler.modules.trackobjects.F_VELOCITY_MODEL,
                cellprofiler.modules.trackobjects.F_COV,
                cellprofiler.modules.trackobjects.F_VY,
                cellprofiler.modules.trackobjects.F_VY,
            ),
        ]
        for object_name, features in (
            (
                OBJECT_NAME,
                cellprofiler.modules.trackobjects.F_ALL
                + kalman_features
                + other_features,
            ),
            ("Image", cellprofiler.modules.trackobjects.F_IMAGE_ALL,),
        ):
            for feature in features:
                if object_name == OBJECT_NAME:
                    name = "_".join(
                        (cellprofiler.modules.trackobjects.F_PREFIX, feature)
                    )
                else:
                    name = "_".join(
                        (
                            cellprofiler.modules.trackobjects.F_PREFIX,
                            feature,
                            OBJECT_NAME,
                        )
                    )
                index = [column[1] for column in columns].index(name)
                assert index != -1
                column = columns[index]
                assert column[0] == object_name
                if wants or feature in second_phase:
                    assert len(column) == 4
                    assert MCA_AVAILABLE_POST_GROUP in column[3]
                    assert column[3][MCA_AVAILABLE_POST_GROUP]
                else:
                    assert (
                        (len(column) == 3)
                        or (MCA_AVAILABLE_POST_GROUP not in column[3])
                        or (not column[3][MCA_AVAILABLE_POST_GROUP])
                    )


def test_measurements():
    """Test the different measurement pieces"""
    module = cellprofiler.modules.trackobjects.TrackObjects()
    module.object_name.value = OBJECT_NAME
    module.image_name.value = "image"
    module.pixel_radius.value = 10
    categories = module.get_categories(None, "Foo")
    assert len(categories) == 0
    categories = module.get_categories(None, OBJECT_NAME)
    assert len(categories) == 1
    assert categories[0] == cellprofiler.modules.trackobjects.F_PREFIX
    features = module.get_measurements(None, OBJECT_NAME, "Foo")
    assert len(features) == 0
    features = module.get_measurements(
        None, OBJECT_NAME, cellprofiler.modules.trackobjects.F_PREFIX
    )
    assert len(features) == len(cellprofiler.modules.trackobjects.F_ALL)
    assert all(
        [feature in cellprofiler.modules.trackobjects.F_ALL for feature in features]
    )
    scales = module.get_measurement_scales(
        None, OBJECT_NAME, cellprofiler.modules.trackobjects.F_PREFIX, "Foo", "image"
    )
    assert len(scales) == 0
    for feature in cellprofiler.modules.trackobjects.F_ALL:
        scales = module.get_measurement_scales(
            None,
            OBJECT_NAME,
            cellprofiler.modules.trackobjects.F_PREFIX,
            feature,
            "image",
        )
        assert len(scales) == 1
        assert int(scales[0]) == 10


def make_lap2_workspace(objs, nimages, group_numbers=None, group_indexes=None):
    """Make a workspace to test the second half of LAP

    objs - a N x 7 array of "objects" composed of the
           following pieces per object
           objs[0] - image set # for object
           objs[1] - label for object
           objs[2] - parent image #
           objs[3] - parent object #
           objs[4] - x coordinate for object
           objs[5] - y coordinate for object
           objs[6] - area for object
    nimages - # of image sets
    group_numbers - group numbers for each image set, defaults to all 1
    group_indexes - group indexes for each image set, defaults to range
    """
    module = cellprofiler.modules.trackobjects.TrackObjects()
    module.set_module_num(1)
    module.object_name.value = OBJECT_NAME
    module.tracking_method.value = "LAP"
    module.wants_second_phase.value = True
    module.wants_lifetime_filtering.value = False
    module.wants_minimum_lifetime.value = False
    module.min_lifetime.value = 1
    module.wants_maximum_lifetime.value = False
    module.max_lifetime.value = 100

    module.pixel_radius.value = 50

    pipeline = Pipeline()

    def callback(caller, event):
        assert not isinstance(event, RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)

    m = cellprofiler_core.measurement.Measurements()
    if objs.shape[0] > 0:
        nobjects = numpy.bincount(objs[:, 0].astype(int))
    else:
        nobjects = numpy.zeros(nimages, int)
    for i in range(nimages):
        m.next_image_set(i + 1)
        for index, feature, dtype in (
            (
                1,
                module.measurement_name(cellprofiler.modules.trackobjects.F_LABEL),
                int,
            ),
            (
                2,
                module.measurement_name(
                    cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER
                ),
                int,
            ),
            (
                3,
                module.measurement_name(
                    cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER
                ),
                int,
            ),
            (4, M_LOCATION_CENTER_X, float),
            (5, M_LOCATION_CENTER_Y, float),
            (
                6,
                module.measurement_name(cellprofiler.modules.trackobjects.F_AREA),
                float,
            ),
        ):
            values = objs[objs[:, 0] == i, index].astype(dtype)
            m.add_measurement(OBJECT_NAME, feature, values, i + 1)
        m.add_measurement("Image", "ImageNumber", i + 1)
        m.add_measurement(
            "Image",
            GROUP_NUMBER,
            1 if group_numbers is None else group_numbers[i],
            image_set_number=i + 1,
        )
        m.add_measurement(
            "Image",
            GROUP_INDEX,
            i if group_indexes is None else group_indexes[i],
            image_set_number=i + 1,
        )
        #
        # Add blanks of the right sizes for measurements that are recalculated
        #
        m.add_measurement(
            "Image",
            "_".join((C_COUNT, OBJECT_NAME)),
            nobjects[i],
            image_set_number=i + 1,
        )
        for feature in (
            cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED,
            cellprofiler.modules.trackobjects.F_DISPLACEMENT,
            cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE,
            cellprofiler.modules.trackobjects.F_TRAJECTORY_X,
            cellprofiler.modules.trackobjects.F_TRAJECTORY_Y,
            cellprofiler.modules.trackobjects.F_LINEARITY,
            cellprofiler.modules.trackobjects.F_LIFETIME,
            cellprofiler.modules.trackobjects.F_FINAL_AGE,
            cellprofiler.modules.trackobjects.F_LINKING_DISTANCE,
            cellprofiler.modules.trackobjects.F_LINK_TYPE,
            cellprofiler.modules.trackobjects.F_MOVEMENT_MODEL,
            cellprofiler.modules.trackobjects.F_STANDARD_DEVIATION,
        ):
            dtype = (
                int
                if feature
                in (
                    cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER,
                    cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER,
                    cellprofiler.modules.trackobjects.F_LIFETIME,
                    cellprofiler.modules.trackobjects.F_LINK_TYPE,
                    cellprofiler.modules.trackobjects.F_MOVEMENT_MODEL,
                )
                else float
            )
            m.add_measurement(
                OBJECT_NAME,
                module.measurement_name(feature),
                numpy.NaN * numpy.ones(nobjects[i], dtype)
                if feature == cellprofiler.modules.trackobjects.F_FINAL_AGE
                else numpy.zeros(nobjects[i], dtype),
                image_set_number=i + 1,
            )
        for feature in (
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT,
            cellprofiler.modules.trackobjects.F_MERGE_COUNT,
        ):
            m.add_measurement(
                "Image",
                module.image_measurement_name(feature),
                0,
                image_set_number=i + 1,
            )
    #
    # Figure out how many new and lost objects per image set
    #
    label_sets = [set() for i in range(nimages)]
    for row in objs:
        label_sets[row[0]].add(row[1])
    if group_numbers is None:
        group_numbers = numpy.ones(nimages, int)
    if group_indexes is None:
        group_indexes = numpy.arange(nimages) + 1
    #
    # New objects are ones without matching labels in the previous set
    #
    for i in range(0, nimages):
        if group_indexes[i] == 1:
            new_objects = len(label_sets[i])
            lost_objects = 0
        else:
            new_objects = sum(
                [1 for label in label_sets[i] if label not in label_sets[i - 1]]
            )
            lost_objects = sum(
                [1 for label in label_sets[i - 1] if label not in label_sets[i]]
            )
        m.add_measurement(
            "Image",
            module.image_measurement_name(
                cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT
            ),
            new_objects,
            image_set_number=i + 1,
        )
        m.add_measurement(
            "Image",
            module.image_measurement_name(
                cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT
            ),
            lost_objects,
            image_set_number=i + 1,
        )
    m.image_set_number = nimages

    image_set_list = ImageSetList()
    for i in range(nimages):
        image_set = image_set_list.get_image_set(i)
    workspace = Workspace(pipeline, module, image_set, ObjectSet(), m, image_set_list,)
    return workspace, module


def check_measurements(workspace, d):
    """Check measurements against expected values

    workspace - workspace that was run
    d - dictionary of feature name and list of expected measurement values
    """
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    module = workspace.module
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    for feature, expected in list(d.items()):
        if numpy.isscalar(expected[0]):
            mname = module.image_measurement_name(feature)
            values = m.get_measurement("Image", mname, m.get_image_numbers())
            assert len(expected) == len(values), (
                "Expected # image sets (%d) != actual (%d) for %s"
                % (len(expected), len(values), feature)
            )
            assert all([v == e for v, e in zip(values, expected)]), (
                "Values don't match for " + feature
            )
        else:
            mname = module.measurement_name(feature)
            values = m.get_measurement(OBJECT_NAME, mname, m.get_image_numbers())
            assert len(expected) == len(values), (
                "Expected # image sets (%d) != actual (%d) for %s"
                % (len(expected), len(values), feature)
            )
            for i, (e, v) in enumerate(zip(expected, values)):
                assert len(e) == len(v), (
                    "Expected # of objects (%d) != actual (%d) for %s:%d"
                    % (len(e), len(v), feature, i)
                )
                numpy.testing.assert_almost_equal(v, e)


def check_relationships(
    m,
    expected_parent_image_numbers,
    expected_parent_object_numbers,
    expected_child_image_numbers,
    expected_child_object_numbers,
):
    """Check the relationship measurements against expected"""
    expected_parent_image_numbers = numpy.atleast_1d(expected_parent_image_numbers)
    expected_child_image_numbers = numpy.atleast_1d(expected_child_image_numbers)
    expected_parent_object_numbers = numpy.atleast_1d(expected_parent_object_numbers)
    expected_child_object_numbers = numpy.atleast_1d(expected_child_object_numbers)
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    r = m.get_relationships(
        1, cellprofiler.modules.trackobjects.R_PARENT, OBJECT_NAME, OBJECT_NAME
    )
    actual_parent_image_numbers = r[R_FIRST_IMAGE_NUMBER]
    actual_parent_object_numbers = r[R_FIRST_OBJECT_NUMBER]
    actual_child_image_numbers = r[R_SECOND_IMAGE_NUMBER]
    actual_child_object_numbers = r[R_SECOND_OBJECT_NUMBER]
    assert len(actual_parent_image_numbers) == len(expected_parent_image_numbers)
    #
    # Sort similarly
    #
    for i1, o1, i2, o2 in (
        (
            expected_parent_image_numbers,
            expected_parent_object_numbers,
            expected_child_image_numbers,
            expected_child_object_numbers,
        ),
        (
            actual_parent_image_numbers,
            actual_parent_object_numbers,
            actual_child_image_numbers,
            actual_child_object_numbers,
        ),
    ):
        order = numpy.lexsort((i1, o1, i2, o2))
        for x in (i1, o1, i2, o2):
            x[:] = x[order]
    for expected, actual in zip(
        (
            expected_parent_image_numbers,
            expected_parent_object_numbers,
            expected_child_image_numbers,
            expected_child_object_numbers,
        ),
        (
            actual_parent_image_numbers,
            actual_parent_object_numbers,
            actual_child_image_numbers,
            actual_child_object_numbers,
        ),
    ):
        numpy.testing.assert_array_equal(expected, actual)


def test_lap_none():
    """Run the second part of LAP on one image of nothing"""
    with MonkeyPatchedDelete():
        workspace, module = make_lap2_workspace(numpy.zeros((0, 7)), 1)
        assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
        module.run_as_data_tool(workspace)
        check_measurements(
            workspace,
            {
                cellprofiler.modules.trackobjects.F_LABEL: [numpy.zeros(0, int)],
                cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED: [numpy.zeros(0)],
                cellprofiler.modules.trackobjects.F_DISPLACEMENT: [numpy.zeros(0)],
                cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE: [
                    numpy.zeros(0)
                ],
                cellprofiler.modules.trackobjects.F_TRAJECTORY_X: [numpy.zeros(0)],
                cellprofiler.modules.trackobjects.F_TRAJECTORY_Y: [numpy.zeros(0)],
                cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [0],
                cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0],
                cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0],
                cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0],
            },
        )


def test_lap_one():
    """Run the second part of LAP on one image of one object"""
    with MonkeyPatchedDelete():
        workspace, module = make_lap2_workspace(
            numpy.array([[0, 1, 0, 0, 100, 100, 25]]), 1
        )
        assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
        module.run_as_data_tool(workspace)
        check_measurements(
            workspace,
            {
                cellprofiler.modules.trackobjects.F_LABEL: [numpy.array([1])],
                cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                    numpy.array([0])
                ],
                cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                    numpy.array([0])
                ],
                cellprofiler.modules.trackobjects.F_DISPLACEMENT: [numpy.zeros(1)],
                cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE: [
                    numpy.zeros(1)
                ],
                cellprofiler.modules.trackobjects.F_TRAJECTORY_X: [numpy.zeros(1)],
                cellprofiler.modules.trackobjects.F_TRAJECTORY_Y: [numpy.zeros(1)],
                cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1],
                cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0],
                cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0],
                cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0],
            },
        )


def test_bridge_gap():
    """Bridge a gap of zero frames between two objects"""
    with MonkeyPatchedDelete():
        workspace, module = make_lap2_workspace(
            numpy.array([[0, 1, 0, 0, 1, 2, 25], [2, 2, 0, 0, 101, 102, 25]]), 3
        )
        assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
        #
        # The cost of bridging the gap should be 141. We set the alternative
        # score to 142 so that bridging wins.
        #
        module.gap_cost.value = 142
        module.max_gap_score.value = 142
        module.run_as_data_tool(workspace)
        distance = numpy.array([numpy.sqrt(2 * 100 * 100)])
        check_measurements(
            workspace,
            {
                cellprofiler.modules.trackobjects.F_LABEL: [
                    numpy.array([1]),
                    numpy.zeros(0),
                    numpy.array([1]),
                ],
                cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                    numpy.array([0]),
                    numpy.zeros(0, int),
                    numpy.array([1]),
                ],
                cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                    numpy.array([0]),
                    numpy.zeros(0, int),
                    numpy.array([1]),
                ],
                cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED: [
                    numpy.zeros(1),
                    numpy.zeros(0),
                    distance,
                ],
                cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE: [
                    numpy.zeros(1),
                    numpy.zeros(0),
                    distance,
                ],
                cellprofiler.modules.trackobjects.F_TRAJECTORY_X: [
                    numpy.zeros(1),
                    numpy.zeros(0),
                    numpy.array([100]),
                ],
                cellprofiler.modules.trackobjects.F_TRAJECTORY_Y: [
                    numpy.zeros(1),
                    numpy.zeros(0),
                    numpy.array([100]),
                ],
                cellprofiler.modules.trackobjects.F_LINEARITY: [
                    numpy.array([numpy.nan]),
                    numpy.zeros(0),
                    numpy.array([1]),
                ],
                cellprofiler.modules.trackobjects.F_LIFETIME: [
                    numpy.ones(1),
                    numpy.zeros(0),
                    numpy.array([2]),
                ],
                cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                    numpy.array([numpy.nan]),
                    numpy.zeros(0),
                    numpy.array([2]),
                ],
                cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 0, 0],
                cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 0, 0],
                cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
                cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
            },
        )
        check_relationships(workspace.measurements, [1], [1], [3], [1])


def test_maintain_gap():
    """Maintain object identity across a large gap"""
    with MonkeyPatchedDelete():
        workspace, module = make_lap2_workspace(
            numpy.array([[0, 1, 0, 0, 1, 2, 25], [2, 2, 0, 0, 101, 102, 25]]), 3
        )
        assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
        #
        # The cost of creating the gap should be 140 and the cost of
        # bridging the gap should be 141.
        #
        module.gap_cost.value = 140
        module.max_gap_score.value = 142
        module.run_as_data_tool(workspace)
        check_measurements(
            workspace,
            {
                cellprofiler.modules.trackobjects.F_LABEL: [
                    numpy.array([1]),
                    numpy.zeros(0),
                    numpy.array([2]),
                ],
                cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                    numpy.array([0]),
                    numpy.zeros(0),
                    numpy.array([0]),
                ],
                cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                    numpy.array([0]),
                    numpy.zeros(0),
                    numpy.array([0]),
                ],
                cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 0, 1],
                cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 1, 0],
                cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
                cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
            },
        )


def test_filter_gap():
    """Filter a gap due to an unreasonable score"""
    with MonkeyPatchedDelete():
        workspace, module = make_lap2_workspace(
            numpy.array([[0, 1, 0, 0, 1, 2, 25], [2, 2, 0, 0, 101, 102, 25]]), 3
        )
        assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
        #
        # The cost of creating the gap should be 142 and the cost of
        # bridging the gap should be 141. However, the gap should be filtered
        # by the max score
        #
        module.gap_cost.value = 142
        module.max_gap_score.value = 140
        module.run_as_data_tool(workspace)
        check_measurements(
            workspace,
            {
                cellprofiler.modules.trackobjects.F_LABEL: [
                    numpy.array([1]),
                    numpy.zeros(0),
                    numpy.array([2]),
                ],
                cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                    numpy.array([0]),
                    numpy.zeros(0),
                    numpy.array([0]),
                ],
                cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                    numpy.array([0]),
                    numpy.zeros(0),
                    numpy.array([0]),
                ],
            },
        )


def test_split():
    """Track an object splitting"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 100, 100, 50],
                [1, 1, 1, 1, 110, 110, 25],
                [1, 2, 0, 0, 90, 90, 25],
                [2, 1, 2, 1, 113, 114, 25],
                [2, 2, 2, 2, 86, 87, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The split score should be 20*sqrt(2) more than the null so a split
    # alternative cost of 15 is too much and 14 too little. Values
    # doulbed to mat
    #
    module.split_cost.value = 30
    module.max_split_score.value = 30
    module.run_as_data_tool(workspace)
    d200 = numpy.sqrt(200)
    tot = numpy.sqrt(13 ** 2 + 14 ** 2)
    lin = tot / (d200 + 5)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([1, 1]),
                numpy.array([1, 1]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 1]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 1]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED: [
                numpy.zeros(1),
                numpy.ones(2) * d200,
                numpy.array([5, 5]),
            ],
            cellprofiler.modules.trackobjects.F_DISPLACEMENT: [
                numpy.zeros(1),
                numpy.ones(2) * d200,
                numpy.array([tot, tot]),
            ],
            cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE: [
                numpy.zeros(1),
                numpy.ones(2) * d200,
                numpy.ones(2) * d200 + 5,
            ],
            cellprofiler.modules.trackobjects.F_TRAJECTORY_X: [
                numpy.zeros(1),
                numpy.array([10, -10]),
                numpy.array([3, -4]),
            ],
            cellprofiler.modules.trackobjects.F_TRAJECTORY_Y: [
                numpy.zeros(1),
                numpy.array([10, -10]),
                numpy.array([4, -3]),
            ],
            cellprofiler.modules.trackobjects.F_LINEARITY: [
                numpy.array([numpy.nan]),
                numpy.array([1, 1]),
                numpy.array([lin, lin]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.ones(1),
                numpy.array([2, 2]),
                numpy.array([3, 3]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([3, 3]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 0, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 1, 0],
        },
    )


def test_dont_split():
    """Track an object splitting"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 100, 100, 50],
                [1, 1, 1, 1, 110, 110, 25],
                [1, 2, 0, 0, 90, 90, 25],
                [2, 1, 2, 1, 110, 110, 25],
                [2, 2, 2, 2, 90, 90, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    module.split_cost.value = 28
    module.max_split_score.value = 30
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([1, 2]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 0]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 0]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.ones(1),
                numpy.array([2, 1]),
                numpy.array([3, 2]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([3, 2]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 1, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
        },
    )


def test_split_filter():
    """Prevent a split by setting the filter too low"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 100, 100, 50],
                [1, 1, 1, 1, 110, 110, 25],
                [1, 2, 0, 0, 90, 90, 25],
                [2, 1, 2, 1, 110, 110, 25],
                [2, 2, 2, 2, 90, 90, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    module.split_cost.value = 30
    module.max_split_score.value = 28
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([1, 2]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 0]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 0]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.array([1]),
                numpy.array([2, 1]),
                numpy.array([3, 2]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([3, 2]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 1, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
        },
    )


def test_merge():
    """Merge two objects into one"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 110, 110, 25],
                [0, 2, 0, 0, 90, 90, 25],
                [1, 1, 1, 1, 110, 110, 25],
                [1, 2, 1, 2, 90, 90, 25],
                [2, 1, 2, 1, 100, 100, 50],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    module.merge_cost.value = 30
    module.max_merge_score.value = 30
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1, 1]),
                numpy.array([1, 1]),
                numpy.array([1]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0, 0]),
                numpy.array([1, 1]),
                numpy.array([2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0, 0]),
                numpy.array([1, 2]),
                numpy.array([1]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.array([1, 1]),
                numpy.array([2, 2]),
                numpy.array([3]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([3]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [2, 0, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 1],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
        },
    )


def test_dont_merge():
    """Don't merge because of low alternative merge cost"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 110, 110, 25],
                [0, 2, 0, 0, 90, 90, 25],
                [1, 1, 1, 1, 110, 110, 25],
                [1, 2, 1, 2, 90, 90, 25],
                [2, 1, 2, 1, 100, 100, 50],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
    #
    module.merge_cost.value = 28
    module.max_merge_score.value = 30
    module.run_as_data_tool(workspace)
    labels = workspace.measurements.get_measurement(
        OBJECT_NAME, module.measurement_name(cellprofiler.modules.trackobjects.F_LABEL), workspace.measurements.get_image_numbers()
    )
    assert len(labels) == 3
    assert len(labels[0]) == 2
    assert labels[0][0] == 1
    assert labels[0][1] == 2
    assert len(labels[1]) == 2
    assert labels[1][0] == 1
    assert labels[1][1] == 2
    assert len(labels[2]) == 1
    assert labels[2][0] == 1


def test_filter_merge():
    """Don't merge because of low alternative merge cost"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 110, 110, 25],
                [0, 2, 0, 0, 90, 90, 25],
                [1, 1, 1, 1, 110, 110, 25],
                [1, 2, 1, 2, 90, 90, 25],
                [2, 1, 2, 1, 100, 100, 50],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
    #
    module.merge_cost.value = 30
    module.max_merge_score.value = 28
    module.run_as_data_tool(workspace)
    labels = workspace.measurements.get_measurement(
        OBJECT_NAME, module.measurement_name(cellprofiler.modules.trackobjects.F_LABEL), workspace.measurements.get_image_numbers()
    )
    assert len(labels) == 3
    assert len(labels[0]) == 2
    assert labels[0][0] == 1
    assert labels[0][1] == 2
    assert len(labels[1]) == 2
    assert labels[1][0] == 1
    assert labels[1][1] == 2
    assert len(labels[2]) == 1
    assert labels[2][0] == 1


def test_img_1111():
    """Regression test of img-1111"""
    data = numpy.array(
        [
            [9, 1, 0, 0, 225, 20, 50],
            [9, 2, 0, 0, 116, 223, 31],
            [25, 3, 0, 0, 43, 291, 26],
            [28, 4, 0, 0, 410, 436, 24],
            [29, 5, 0, 0, 293, 166, 23],
            [29, 4, 29, 1, 409, 436, 24],
            [30, 5, 30, 1, 293, 167, 30],
            [32, 6, 0, 0, 293, 164, 69],
            [33, 6, 33, 1, 292, 166, 37],
            [35, 7, 0, 0, 290, 165, 63],
            [36, 7, 36, 1, 290, 166, 38],
            [39, 8, 0, 0, 287, 163, 28],
            [40, 8, 40, 1, 287, 163, 21],
            [44, 9, 0, 0, 54, 288, 20],
            [77, 10, 0, 0, 514, 211, 49],
            [78, 10, 78, 1, 514, 210, 42],
            [79, 10, 79, 1, 514, 209, 73],
            [80, 10, 80, 1, 514, 208, 49],
            [81, 10, 81, 1, 515, 209, 38],
            [98, 11, 0, 0, 650, 54, 24],
            [102, 12, 0, 0, 586, 213, 46],
            [104, 13, 0, 0, 586, 213, 27],
            [106, 14, 0, 0, 587, 212, 54],
            [107, 14, 107, 1, 587, 212, 40],
            [113, 15, 0, 0, 17, 145, 51],
            [116, 16, 0, 0, 45, 153, 21],
            [117, 17, 0, 0, 53, 148, 44],
            [117, 18, 0, 0, 90, 278, 87],
            [119, 19, 0, 0, 295, 184, 75],
            [120, 19, 120, 1, 295, 184, 79],
            [121, 19, 121, 1, 295, 182, 75],
            [123, 20, 0, 0, 636, 7, 20],
            [124, 20, 124, 1, 635, 7, 45],
            [124, 21, 0, 0, 133, 171, 22],
            [124, 22, 0, 0, 417, 365, 65],
            [126, 23, 0, 0, 125, 182, 77],
            [126, 24, 0, 0, 358, 306, 48],
            [126, 25, 0, 0, 413, 366, 60],
            [127, 26, 0, 0, 141, 173, 71],
            [127, 25, 127, 3, 413, 366, 35],
            [128, 27, 0, 0, 131, 192, 76],
            [129, 28, 0, 0, 156, 182, 74],
            [130, 29, 0, 0, 147, 194, 56],
            [131, 30, 0, 0, 152, 185, 56],
            [132, 30, 132, 1, 154, 188, 78],
            [133, 31, 0, 0, 142, 186, 64],
            [133, 32, 0, 0, 91, 283, 23],
            [134, 33, 0, 0, 150, 195, 80],
        ]
    )
    data = data[:8, :]
    workspace, module = make_lap2_workspace(data, numpy.max(data[:, 0]) + 1)
    module.run_as_data_tool(workspace)


def test_multi_group():
    """Run several tests in different groups"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 1, 2, 25],
                [2, 2, 0, 0, 101, 102, 25],
                [3, 1, 0, 0, 100, 100, 50],
                [4, 1, 4, 1, 110, 110, 25],
                [4, 2, 0, 0, 90, 90, 25],
                [5, 1, 5, 1, 113, 114, 25],
                [5, 2, 5, 2, 86, 87, 25],
                [6, 1, 0, 0, 110, 110, 25],
                [6, 2, 0, 0, 90, 90, 25],
                [7, 1, 7, 1, 110, 110, 25],
                [7, 2, 7, 2, 90, 90, 25],
                [8, 1, 8, 1, 104, 102, 50],
            ]
        ),
        9,
        group_numbers=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        group_indexes=[1, 2, 3, 1, 2, 3, 1, 2, 3],
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The cost of bridging the gap should be 141. We set the alternative
    # score to 142 so that bridging wins.
    #
    module.gap_cost.value = 142
    module.max_gap_score.value = 142
    module.split_cost.value = 30
    module.max_split_score.value = 30
    module.merge_cost.value = 30
    module.max_merge_score.value = 30
    module.run_as_data_tool(workspace)
    distance = numpy.array([numpy.sqrt(2 * 100 * 100)])
    d200 = numpy.sqrt(200)
    tot = numpy.sqrt(13 ** 2 + 14 ** 2)
    lin = tot / (d200 + 5)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.zeros(0),
                numpy.array([1]),
                numpy.array([1]),
                numpy.array([1, 1]),
                numpy.array([1, 1]),
                numpy.array([1, 1]),
                numpy.array([1, 1]),
                numpy.array([1]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.zeros(0),
                numpy.array([1]),
                numpy.array([0]),
                numpy.array([4, 4]),
                numpy.array([5, 5]),
                numpy.array([0, 0]),
                numpy.array([7, 7]),
                numpy.array([8]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.zeros(0),
                numpy.array([1]),
                numpy.array([0]),
                numpy.array([1, 1]),
                numpy.array([1, 2]),
                numpy.array([0, 0]),
                numpy.array([1, 2]),
                numpy.array([1]),
            ],
            cellprofiler.modules.trackobjects.F_DISPLACEMENT: [
                numpy.zeros(1),
                numpy.zeros(0),
                distance,
                numpy.zeros(1),
                numpy.ones(2) * d200,
                numpy.array([tot, tot]),
                numpy.zeros(2),
                numpy.zeros(2),
                numpy.array([10]),
            ],
            cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE: [
                numpy.zeros(1),
                numpy.zeros(0),
                distance,
                numpy.zeros(1),
                numpy.ones(2) * d200,
                numpy.ones(2) * d200 + 5,
                numpy.zeros(2),
                numpy.zeros(2),
                numpy.array([10]),
            ],
            cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED: [
                numpy.zeros(1),
                numpy.zeros(0),
                distance,
                numpy.zeros(1),
                numpy.ones(2) * d200,
                numpy.array([5, 5]),
                numpy.zeros(2),
                numpy.zeros(2),
                numpy.array([10]),
            ],
            cellprofiler.modules.trackobjects.F_TRAJECTORY_X: [
                numpy.zeros(1),
                numpy.zeros(0),
                numpy.array([100]),
                numpy.zeros(1),
                numpy.array([10, -10]),
                numpy.array([3, -4]),
                numpy.zeros(2),
                numpy.zeros(2),
                numpy.array([-6]),
            ],
            cellprofiler.modules.trackobjects.F_TRAJECTORY_Y: [
                numpy.zeros(1),
                numpy.zeros(0),
                numpy.array([100]),
                numpy.zeros(1),
                numpy.array([10, -10]),
                numpy.array([4, -3]),
                numpy.zeros(2),
                numpy.zeros(2),
                numpy.array([-8]),
            ],
            cellprofiler.modules.trackobjects.F_LINEARITY: [
                numpy.array([numpy.nan]),
                numpy.zeros(0),
                numpy.array([1]),
                numpy.array([numpy.nan]),
                numpy.array([1, 1]),
                numpy.array([lin, lin]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.ones(1),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.ones(1),
                numpy.zeros(0),
                numpy.array([2]),
                numpy.ones(1),
                numpy.array([2, 2]),
                numpy.array([3, 3]),
                numpy.ones(2),
                numpy.array([2, 2]),
                numpy.array([3]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([numpy.nan]),
                numpy.zeros(0),
                numpy.array([2]),
                numpy.array([numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([3, 3]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([3]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [
                1,
                0,
                0,
                1,
                0,
                0,
                2,
                0,
                0,
            ],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
        },
    )


def test_filter_by_final_age():
    """Filter an object by the final age"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 100, 100, 50],
                [1, 1, 1, 1, 110, 110, 50],
                [1, 2, 0, 0, 90, 90, 25],
                [2, 1, 2, 1, 100, 100, 50],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The split score should be between 14 and 15.  Set the split
    # alternative cost to 28 so that the split is inhibited.
    #
    module.split_cost.value = 28
    module.max_split_score.value = 30
    #
    # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
    #
    module.merge_cost.value = 28
    module.max_merge_score.value = 30
    module.wants_lifetime_filtering.value = True
    module.wants_minimum_lifetime.value = True
    module.min_lifetime.value = 1
    module.wants_maximum_lifetime.value = False
    module.max_lifetime.value = 100
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([1, numpy.NaN]),
                numpy.array([1]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 0]),
                numpy.array([2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 0]),
                numpy.array([1]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.array([1]),
                numpy.array([2, 1]),
                numpy.array([3]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([numpy.nan]),
                numpy.array([numpy.nan, 1]),
                numpy.array([3]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 1, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 0, 1],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
        },
    )


def test_mitosis():
    """Track a mitosis"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 103, 104, 50],
                [1, 2, 0, 0, 110, 110, 25],
                [1, 3, 0, 0, 90, 90, 25],
                [2, 2, 2, 1, 113, 114, 25],
                [2, 3, 2, 2, 86, 87, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The parent is off by np.sqrt(3*3+4*4) = 5, so an alternative of
    # 4 loses and 6 wins
    #
    module.merge_cost.value = 1
    module.gap_cost.value = 1
    module.mitosis_cost.value = 6
    module.mitosis_max_distance.value = 20
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([1, 1]),
                numpy.array([1, 1]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 1]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 1]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.ones(1),
                numpy.array([2, 2]),
                numpy.array([3, 3]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([numpy.nan]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([3, 3]),
            ],
            cellprofiler.modules.trackobjects.F_LINK_TYPE: [
                numpy.array([cellprofiler.modules.trackobjects.LT_NONE]),
                numpy.array(
                    [
                        cellprofiler.modules.trackobjects.LT_MITOSIS,
                        cellprofiler.modules.trackobjects.LT_MITOSIS,
                    ]
                ),
                numpy.array(
                    [
                        cellprofiler.modules.trackobjects.LT_NONE,
                        cellprofiler.modules.trackobjects.LT_NONE,
                    ]
                ),
            ],
            cellprofiler.modules.trackobjects.F_MITOSIS_SCORE: [
                numpy.array([numpy.nan]),
                numpy.array([5, 5]),
                numpy.array([numpy.nan, numpy.nan]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 0, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 1, 0],
        },
    )


def test_no_mitosis():
    """Don't track a mitosis"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 103, 104, 50],
                [1, 2, 0, 0, 110, 110, 25],
                [1, 3, 0, 0, 90, 90, 25],
                [2, 2, 2, 1, 113, 114, 25],
                [2, 3, 2, 2, 86, 87, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The parent is off by np.sqrt(3*3+4*4) = 5, so an alternative of
    # 4 loses and 6 wins
    #
    module.merge_cost.value = 1
    module.mitosis_cost.value = 4
    module.mitosis_max_distance.value = 20
    module.gap_cost.value = 1
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([2, 3]),
                numpy.array([2, 3]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([0, 0]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([0, 0]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.ones(1),
                numpy.array([1, 1]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([1]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 2, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 1, 0],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
        },
    )


def test_mitosis_distance_filter():
    """Don't track a mitosis"""
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 103, 104, 50],
                [1, 2, 0, 0, 110, 110, 25],
                [1, 3, 0, 0, 90, 90, 25],
                [2, 2, 2, 1, 113, 114, 25],
                [2, 3, 2, 2, 86, 87, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    #
    # The parent is off by np.sqrt(3*3+4*4) = 5, so an alternative of
    # 4 loses and 6 wins
    #
    module.merge_cost.value = 1
    module.mitosis_cost.value = 6
    module.mitosis_max_distance.value = 15
    module.gap_cost.value = 1
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([2, 3]),
                numpy.array([2, 3]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([0, 0]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([0, 0]),
                numpy.array([1, 2]),
            ],
            cellprofiler.modules.trackobjects.F_LIFETIME: [
                numpy.ones(1),
                numpy.array([1, 1]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_FINAL_AGE: [
                numpy.array([1]),
                numpy.array([numpy.nan, numpy.nan]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT: [1, 2, 0],
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT: [0, 1, 0],
            cellprofiler.modules.trackobjects.F_MERGE_COUNT: [0, 0, 0],
            cellprofiler.modules.trackobjects.F_SPLIT_COUNT: [0, 0, 0],
        },
    )


def test_alternate_child_mitoses():
    # Test that LAP can pick the best of two possible child alternates
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 103, 104, 50],
                [1, 2, 0, 0, 110, 110, 25],
                [1, 3, 0, 0, 91, 91, 25],
                [1, 4, 0, 0, 90, 90, 25],
                [2, 2, 2, 1, 113, 114, 25],
                [2, 3, 2, 2, 86, 87, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    module.merge_cost.value = 1
    module.gap_cost.value = 1
    module.mitosis_cost.value = 6
    module.mitosis_max_distance.value = 20
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1]),
                numpy.array([1, 1, 2]),
                numpy.array([1, 1]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 1, 0]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0]),
                numpy.array([1, 1, 0]),
                numpy.array([1, 2]),
            ],
        },
    )


def test_alternate_parent_mitoses():
    # Test that LAP can pick the best of two possible parent alternates
    workspace, module = make_lap2_workspace(
        numpy.array(
            [
                [0, 1, 0, 0, 100, 100, 50],
                [0, 2, 0, 0, 103, 104, 50],
                [1, 3, 0, 0, 110, 110, 25],
                [1, 4, 0, 0, 90, 90, 25],
                [2, 3, 2, 1, 113, 114, 25],
                [2, 4, 2, 2, 86, 87, 25],
            ]
        ),
        3,
    )
    assert isinstance(module, cellprofiler.modules.trackobjects.TrackObjects)
    module.merge_cost.value = 1
    module.gap_cost.value = 1
    module.mitosis_cost.value = 6
    module.mitosis_max_distance.value = 20
    module.run_as_data_tool(workspace)
    check_measurements(
        workspace,
        {
            cellprofiler.modules.trackobjects.F_LABEL: [
                numpy.array([1, 2]),
                numpy.array([1, 1]),
                numpy.array([1, 1]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER: [
                numpy.array([0, 0]),
                numpy.array([1, 1]),
                numpy.array([2, 2]),
            ],
            cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER: [
                numpy.array([0, 0]),
                numpy.array([1, 1]),
                numpy.array([1, 2]),
            ],
        },
    )


class MonkeyPatchedDelete(object):
    """Monkey patch np.delete inside of a scope

    For regression test of issue #1571 - negative
    indices in calls to numpy.delete

    Usage:
        with MonkeyPatchedDelete():
            ... do test ...
    """

    def __init__(self, test=None):
        __test = test

    def __enter__(self):
        self.old_delete = numpy.delete
        numpy.delete = self.monkey_patched_delete

    def __exit__(self, type, value, traceback):
        numpy.delete = self.old_delete

    def monkey_patched_delete(self, array, indices, axis):
        # __test.assertTrue(numpy.all(indices >= 0))
        return self.old_delete(array, indices, axis)


def test_save_image():
    module = cellprofiler.modules.trackobjects.TrackObjects()
    module.set_module_num(1)
    module.object_name.value = OBJECT_NAME
    module.pixel_radius.value = 50
    module.wants_image.value = True
    module.image_name.value = "outimage"
    measurements = cellprofiler_core.measurement.Measurements()
    measurements.add_image_measurement(GROUP_NUMBER, 1)
    measurements.add_image_measurement(GROUP_INDEX, 1)
    pipeline = Pipeline()
    pipeline.add_module(module)
    image_set_list = ImageSetList()

    module.prepare_run(
        Workspace(pipeline, module, None, None, measurements, image_set_list)
    )

    first = True
    object_set = ObjectSet()
    objects = Objects()
    objects.segmented = numpy.zeros((640, 480), int)
    object_set.add_objects(objects, OBJECT_NAME)
    image_set = image_set_list.get_image_set(0)
    workspace = Workspace(
        pipeline, module, image_set, object_set, measurements, image_set_list
    )
    module.run(workspace)
    image = workspace.image_set.get_image(module.image_name.value)
    shape = image.pixel_data.shape
    assert shape[0] == 640
    assert shape[1] == 480


def test_get_no_gap_pair_scores():
    for F, L, max_gap in (
        (numpy.zeros((0, 3)), numpy.zeros((0, 3)), 1),
        (numpy.ones((1, 3)), numpy.ones((1, 3)), 1),
        (numpy.ones((2, 3)), numpy.ones((2, 3)), 1),
    ):
        t = cellprofiler.modules.trackobjects.TrackObjects()
        a, d = t.get_gap_pair_scores(F, L, max_gap)
        assert tuple(a.shape) == (0, 2)
        assert len(d) == 0


def test_get_gap_pair_scores():
    L = numpy.array(
        [
            [0.0, 0.0, 1, 0, 0, 0, 1],
            [1.0, 1.0, 5, 0, 0, 0, 1],
            [3.0, 3.0, 8, 0, 0, 0, 1],
            [2.0, 2.0, 9, 0, 0, 0, 1],
            [0.0, 0.0, 9, 0, 0, 0, 1],
            [0.0, 0.0, 9, 0, 0, 0, 1],
        ]
    )
    F = numpy.array(
        [
            [0.0, 0.0, 0, 0, 0, 0, 1],
            [1.0, 0.0, 4, 0, 0, 0, 1],
            [3.0, 0.0, 6, 0, 0, 0, 1],
            [4.0, 0.0, 7, 0, 0, 0, 1],
            [1.0, 0.0, 2, 0, 0, 0, 2],
            [1.0, 0.0, 2, 0, 0, 0, 0.5],
        ]
    )
    expected = numpy.array([[0, 1], [0, 4], [0, 5], [1, 2], [1, 3]])
    expected_d = numpy.sqrt(
        numpy.sum((L[expected[:, 0], :2] - F[expected[:, 1], :2]) ** 2, 1)
    )
    expected_rho = numpy.array([1, 2, 2, 1, 1])
    t = cellprofiler.modules.trackobjects.TrackObjects()
    a, d = t.get_gap_pair_scores(F, L, 4)
    order = numpy.lexsort((a[:, 1], a[:, 0]))
    a, d = a[order], d[order]
    numpy.testing.assert_array_equal(a, expected)

    numpy.testing.assert_array_almost_equal(d, expected_d * expected_rho)


def test_neighbour_track_nothing():
    """Run TrackObjects on an empty labels matrix"""
    columns = []

    def fn(module, workspace, index, columns=columns):
        if workspace is not None and index == 0:
            columns += module.get_measurement_columns(workspace.pipeline)
            module.tracking_method.value = "Follow Neighbors"

    measurements = runTrackObjects(
        (numpy.zeros((10, 10), int), numpy.zeros((10, 10), int)), fn
    )

    features = [
        feature
        for feature in measurements.get_feature_names(OBJECT_NAME)
        if feature.startswith(cellprofiler.modules.trackobjects.F_PREFIX)
    ]
    assert all(
        [column[1] in features for column in columns if column[0] == OBJECT_NAME]
    )
    for feature in cellprofiler.modules.trackobjects.F_ALL:
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "50"))
        assert name in features
        value = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(value) == 0

    features = [
        feature
        for feature in measurements.get_feature_names("Image")
        if feature.startswith(cellprofiler.modules.trackobjects.F_PREFIX)
    ]
    assert all([column[1] in features for column in columns if column[0] == "Image"])
    for feature in cellprofiler.modules.trackobjects.F_IMAGE_ALL:
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "50")
        )
        assert name in features
        value = measurements.get_current_image_measurement(name)
        assert value == 0


def test_00_neighbour_track_one_then_nothing():
    """Run track objects on an object that disappears

    Regression test of IMG-1090
    """
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 2:7] = 1

    def fn(module, workspace, index):
        if workspace is not None and index == 0:
            module.tracking_method.value = "Follow Neighbors"

    measurements = runTrackObjects((labels, numpy.zeros((10, 10), int)), fn)
    feature = "_".join(
        (
            cellprofiler.modules.trackobjects.F_PREFIX,
            cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT,
            OBJECT_NAME,
            "50",
        )
    )
    value = measurements.get_current_image_measurement(feature)
    assert value == 1


def test_neighbour_track_one_by_distance():
    """Track an object that doesn't move."""
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 2:7] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 1
            module.tracking_method.value = "Follow Neighbors"

    measurements = runTrackObjects((labels, labels), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "1"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert round(abs(m(cellprofiler.modules.trackobjects.F_TRAJECTORY_X) - 0), 7) == 0
    assert round(abs(m(cellprofiler.modules.trackobjects.F_TRAJECTORY_Y) - 0), 7) == 0
    assert (
        round(abs(m(cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED) - 0), 7) == 0
    )
    assert (
        round(abs(m(cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE) - 0), 7)
        == 0
    )
    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 1
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 1
    assert m(cellprofiler.modules.trackobjects.F_PARENT_IMAGE_NUMBER) == 1
    assert m(cellprofiler.modules.trackobjects.F_LIFETIME) == 2

    def m(feature):
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "1")
        )
        return measurements.get_current_image_measurement(name)

    assert m(cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_SPLIT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_MERGE_COUNT) == 0
    check_relationships(measurements, [1], [1], [2], [1])


def test_neighbour_track_one_moving():
    """Track an object that moves"""

    labels_list = []
    distance = 0
    last_i, last_j = (0, 0)
    for i_off, j_off in ((0, 0), (2, 0), (2, 1), (0, 1)):
        distance = i_off - last_i + j_off - last_j
        last_i, last_j = (i_off, j_off)
        labels = numpy.zeros((10, 10), int)
        labels[4 + i_off : 7 + i_off, 4 + j_off : 7 + j_off] = 1
        labels_list.append(labels)

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 3
            module.tracking_method.value = "Follow Neighbors"

    measurements = runTrackObjects(labels_list, fn)

    def m(feature, expected):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "3"))
        value_set = measurements.get_measurement(OBJECT_NAME, name, measurements.get_image_numbers())
        assert len(expected) == len(value_set)
        for values, x in zip(value_set, expected):
            assert len(values) == 1
            assert round(abs(values[0] - x), 7) == 0

    m(cellprofiler.modules.trackobjects.F_TRAJECTORY_X, [0, 0, 1, 0])
    m(cellprofiler.modules.trackobjects.F_TRAJECTORY_Y, [0, 2, 0, -2])
    m(cellprofiler.modules.trackobjects.F_DISTANCE_TRAVELED, [0, 2, 1, 2])
    m(cellprofiler.modules.trackobjects.F_INTEGRATED_DISTANCE, [0, 2, 3, 5])
    m(cellprofiler.modules.trackobjects.F_LABEL, [1, 1, 1, 1])
    m(cellprofiler.modules.trackobjects.F_LIFETIME, [1, 2, 3, 4])
    m(
        cellprofiler.modules.trackobjects.F_LINEARITY,
        [1, 1, numpy.sqrt(5) / 3, 1.0 / 5.0],
    )

    def m(feature):
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "3")
        )
        return measurements.get_current_image_measurement(name)

    assert m(cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_SPLIT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_MERGE_COUNT) == 0
    image_numbers = numpy.arange(1, len(labels_list) + 1)
    object_numbers = numpy.ones(len(image_numbers))
    check_relationships(
        measurements,
        image_numbers[:-1],
        object_numbers[:-1],
        image_numbers[1:],
        object_numbers[1:],
    )


def test_neighbour_track_negative():
    """Track unrelated objects"""
    labels1 = numpy.zeros((10, 10), int)
    labels1[1:5, 1:5] = 1
    labels2 = numpy.zeros((10, 10), int)
    labels2[6:9, 6:9] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 1
            module.tracking_method.value = "Follow Neighbors"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "1"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 0

    def m(feature):
        name = "_".join(
            (cellprofiler.modules.trackobjects.F_PREFIX, feature, OBJECT_NAME, "1")
        )
        return measurements.get_current_image_measurement(name)

    assert m(cellprofiler.modules.trackobjects.F_NEW_OBJECT_COUNT) == 1
    assert m(cellprofiler.modules.trackobjects.F_LOST_OBJECT_COUNT) == 1
    assert m(cellprofiler.modules.trackobjects.F_SPLIT_COUNT) == 0
    assert m(cellprofiler.modules.trackobjects.F_MERGE_COUNT) == 0


def test_neighbour_track_ambiguous():
    """Track disambiguation from among two possible parents"""
    labels1 = numpy.zeros((20, 20), int)
    labels1[1:4, 1:4] = 1
    labels1[16:19, 16:19] = 2
    labels2 = numpy.zeros((20, 20), int)
    labels2[10:15, 10:15] = 1

    def fn(module, workspace, idx):
        if idx == 0:
            module.pixel_radius.value = 20
            module.tracking_method.value = "Follow Neighbors"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "20"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    assert m(cellprofiler.modules.trackobjects.F_LABEL) == 2
    assert m(cellprofiler.modules.trackobjects.F_PARENT_OBJECT_NUMBER) == 2


def test_neighbour_track_group_with_drop():
    """Track groups with one lost"""
    labels1 = numpy.zeros((20, 20), int)
    labels1[2, 2] = 1
    labels1[4, 2] = 2
    labels1[2, 4] = 3
    labels1[4, 4] = 4

    labels2 = numpy.zeros((20, 20), int)
    labels2[16, 16] = 1
    labels2[18, 16] = 2
    # labels2[16,18] = 3 is no longer present
    labels2[18, 18] = 4

    def fn(module, workspace, idx):
        if idx == 0:
            module.drop_cost.value = 100  # make it always try to match
            module.pixel_radius.value = 200
            module.average_cell_diameter.value = 5
            module.tracking_method.value = "Follow Neighbors"

    measurements = runTrackObjects((labels1, labels2), fn)

    def m(feature):
        name = "_".join((cellprofiler.modules.trackobjects.F_PREFIX, feature, "20"))
        values = measurements.get_current_measurement(OBJECT_NAME, name)
        assert len(values) == 1
        return values[0]

    check_relationships(measurements, [1, 1, 1], [1, 2, 4], [2, 2, 2], [1, 2, 4])
