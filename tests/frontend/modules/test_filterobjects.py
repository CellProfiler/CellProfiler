import contextlib
import io
import os
import pickle
import tempfile

import numpy
import pytest

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.modules
from cellprofiler_core.constants.measurement import FF_PARENT, FF_COUNT, FF_CHILDREN_COUNT, M_LOCATION_CENTER_X, \
    M_LOCATION_CENTER_Y, M_LOCATION_CENTER_Z, M_NUMBER_OBJECT_NUMBER


import cellprofiler.modules.filterobjects
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_IMAGE = "input_image"
INPUT_OBJECTS = "input_objects"
ENCLOSING_OBJECTS = "my_enclosing_objects"
OUTPUT_OBJECTS = "output_objects"
TEST_FTR = "my_measurement"
MISSPELLED_FTR = "m_measurement"


def make_workspace(object_dict={}, image_dict={}):
    """Make a workspace for testing FilterByObjectMeasurement"""
    module = cellprofiler.modules.filterobjects.FilterByObjectMeasurement()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    object_set = cellprofiler_core.object.ObjectSet()
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    for key in list(image_dict.keys()):
        image_set.add(key, cellprofiler_core.image.Image(image_dict[key]))
    for key in list(object_dict.keys()):
        o = cellprofiler_core.object.Objects()
        o.segmented = object_dict[key]
        object_set.add_objects(o, key)
    return workspace, module


@contextlib.contextmanager
def make_classifier(
    module,
    answers,
    classes=None,
    class_names=None,
    rules_class=None,
    name="Classifier",
    feature_names=[f"{INPUT_OBJECTS}_{TEST_FTR}"],
):
    """Returns the filename of the classifier pickle"""
    assert isinstance(module, cellprofiler.modules.filterobjects.FilterObjects)
    if classes is None:
        classes = numpy.arange(1, numpy.max(answers) + 1)
    if class_names is None:
        class_names = ["Class%d" for _ in classes]
    if rules_class is None:
        rules_class = class_names[0]
    s = make_classifier_pickle(answers, classes, class_names, name, feature_names)
    fd, filename = tempfile.mkstemp(".model")
    os.write(fd, s)
    os.close(fd)

    module.mode.value = cellprofiler.modules.filterobjects.MODE_CLASSIFIERS
    module.rules_class.value = rules_class
    module.rules_directory.set_custom_path(os.path.dirname(filename))
    module.rules_file_name.value = os.path.split(filename)[1]
    yield
    try:
        os.remove(filename)
    except:
        pass


def test_zeros_single():
    """Test keep single object on an empty labels matrix"""
    workspace, module = make_workspace({INPUT_OBJECTS: numpy.zeros((10, 10), int)})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == 0)


def test_zeros_per_object():
    """Test keep per object filtering on an empty labels matrix"""
    workspace, module = make_workspace(
        {
            INPUT_OBJECTS: numpy.zeros((10, 10), int),
            ENCLOSING_OBJECTS: numpy.zeros((10, 10), int),
        }
    )
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
    )
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == 0)


def test_zeros_filter():
    """Test object filtering on an empty labels matrix"""
    workspace, module = make_workspace({INPUT_OBJECTS: numpy.zeros((10, 10), int)})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
    module.measurements[0].min_limit.value = 0
    module.measurements[0].max_limit.value = 1000
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == 0)


def test_keep_single_min():
    """Keep a single object (min) from among two"""
    labels = numpy.zeros((10, 10), int)
    labels[2:4, 3:5] = 1
    labels[6:9, 5:8] = 2
    expected = labels.copy()
    expected[labels == 1] = 0
    expected[labels == 2] = 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([2, 1]))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)
    parents = m.get_current_measurement(
        OUTPUT_OBJECTS, FF_PARENT % INPUT_OBJECTS
    )
    assert len(parents) == 1
    assert parents[0] == 2
    assert (
        m.get_current_image_measurement(
            FF_COUNT % OUTPUT_OBJECTS
        )
        == 1
    )
    feature = FF_CHILDREN_COUNT % OUTPUT_OBJECTS
    child_count = m.get_current_measurement(INPUT_OBJECTS, feature)
    assert len(child_count) == 2
    assert child_count[0] == 0
    assert child_count[1] == 1


def test_keep_single_max():
    """Keep a single object (max) from among two"""
    labels = numpy.zeros((10, 10), int)
    labels[2:4, 3:5] = 1
    labels[6:9, 5:8] = 2
    expected = labels.copy()
    expected[labels == 1] = 0
    expected[labels == 2] = 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 2]))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_keep_one_min():
    """Keep two sub-objects (min) from among four enclosed by two"""
    sub_labels = numpy.zeros((20, 20), int)
    expected = numpy.zeros((20, 20), int)
    for i, j, k, e in ((0, 0, 1, 0), (10, 0, 2, 1), (0, 10, 3, 2), (10, 10, 4, 0)):
        sub_labels[i + 2 : i + 5, j + 3 : j + 7] = k
        expected[i + 2 : i + 5, j + 3 : j + 7] = e
    labels = numpy.zeros((20, 20), int)
    labels[:, :10] = 1
    labels[:, 10:] = 2
    workspace, module = make_workspace(
        {INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels}
    )
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MINIMAL_PER_OBJECT
    )
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([2, 1, 3, 4]))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_keep_one_max():
    """Keep two sub-objects (max) from among four enclosed by two"""
    sub_labels = numpy.zeros((20, 20), int)
    expected = numpy.zeros((20, 20), int)
    for i, j, k, e in ((0, 0, 1, 0), (10, 0, 2, 1), (0, 10, 3, 2), (10, 10, 4, 0)):
        sub_labels[i + 2 : i + 5, j + 3 : j + 7] = k
        expected[i + 2 : i + 5, j + 3 : j + 7] = e
    labels = numpy.zeros((20, 20), int)
    labels[:, :10] = 1
    labels[:, 10:] = 2
    workspace, module = make_workspace(
        {INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels}
    )
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
    )
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 2, 4, 3]))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_keep_maximal_most_overlap():
    labels = numpy.zeros((10, 20), int)
    labels[:, :10] = 1
    labels[:, 10:] = 2
    sub_labels = numpy.zeros((10, 20), int)
    sub_labels[2, 4] = 1
    sub_labels[4:6, 8:15] = 2
    sub_labels[8, 15] = 3
    expected = sub_labels * (sub_labels != 3)
    workspace, module = make_workspace(
        {INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels}
    )
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
    )
    module.per_object_assignment.value = (
        cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP
    )
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_keep_minimal_most_overlap():
    labels = numpy.zeros((10, 20), int)
    labels[:, :10] = 1
    labels[:, 10:] = 2
    sub_labels = numpy.zeros((10, 20), int)
    sub_labels[2, 4] = 1
    sub_labels[4:6, 8:15] = 2
    sub_labels[8, 15] = 3
    expected = sub_labels * (sub_labels != 3)
    workspace, module = make_workspace(
        {INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels}
    )
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MINIMAL_PER_OBJECT
    )
    module.per_object_assignment.value = (
        cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP
    )
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([4, 2, 3]))
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_filter():
    """Filter objects by limits"""
    n = 40
    labels = numpy.zeros((10, n * 10), int)
    for i in range(40):
        labels[2:5, i * 10 + 3 : i * 10 + 7] = i + 1
    numpy.random.seed(0)
    values = numpy.random.uniform(size=n)
    idx = 1
    my_min = 0.3
    my_max = 0.7
    expected = numpy.zeros(labels.shape, int)
    for i, value in zip(list(range(n)), values):
        if my_min <= value <= my_max:
            expected[labels == i + 1] = idx
            idx += 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
    module.measurements[0].wants_minimum.value = True
    module.measurements[0].min_limit.value = my_min
    module.measurements[0].wants_maximum.value = True
    module.measurements[0].max_limit.value = my_max
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_filter():
    """Filter objects by min limits"""
    n = 40
    labels = numpy.zeros((10, n * 10), int)
    for i in range(40):
        labels[2:5, i * 10 + 3 : i * 10 + 7] = i + 1
    numpy.random.seed(0)
    values = numpy.random.uniform(size=n)
    idx = 1
    my_min = 0.3
    expected = numpy.zeros(labels.shape, int)
    for i, value in zip(list(range(n)), values):
        if value >= my_min:
            expected[labels == i + 1] = idx
            idx += 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
    module.measurements[0].min_limit.value = my_min
    module.measurements[0].max_limit.value = 0.7
    module.measurements[0].wants_maximum.value = False
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_filter():
    """Filter objects by maximum limits"""
    n = 40
    labels = numpy.zeros((10, n * 10), int)
    for i in range(40):
        labels[2:5, i * 10 + 3 : i * 10 + 7] = i + 1
    numpy.random.seed(0)
    values = numpy.random.uniform(size=n)
    idx = 1
    my_max = 0.7
    expected = numpy.zeros(labels.shape, int)
    for i, value in zip(list(range(n)), values):
        if value <= my_max:
            expected[labels == i + 1] = idx
            idx += 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
    module.measurements[0].min_limit.value = 0.3
    module.measurements[0].wants_minimum.value = False
    module.measurements[0].max_limit.value = my_max
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_filter_two():
    """Filter objects by two measurements"""
    n = 40
    labels = numpy.zeros((10, n * 10), int)
    for i in range(40):
        labels[2:5, i * 10 + 3 : i * 10 + 7] = i + 1
    numpy.random.seed(0)
    values = numpy.zeros((n, 2))
    values = numpy.random.uniform(size=(n, 2))
    idx = 1
    my_max = numpy.array([0.7, 0.5])
    expected = numpy.zeros(labels.shape, int)
    for i, v1, v2 in zip(list(range(n)), values[:, 0], values[:, 1]):
        if v1 <= my_max[0] and v2 <= my_max[1]:
            expected[labels == i + 1] = idx
            idx += 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.add_measurement()
    m = workspace.measurements
    for i in range(2):
        measurement_name = "measurement%d" % (i + 1)
        module.measurements[i].measurement.value = measurement_name
        module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
        module.measurements[i].min_limit.value = 0.3
        module.measurements[i].wants_minimum.value = False
        module.measurements[i].max_limit.value = my_max[i]
        m.add_measurement(INPUT_OBJECTS, measurement_name, values[:, i])
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(labels.segmented == expected)


def test_renumber_other():
    """Renumber an associated object"""
    n = 40
    labels = numpy.zeros((10, n * 10), int)
    alternates = numpy.zeros((10, n * 10), int)
    for i in range(40):
        labels[2:5, i * 10 + 3 : i * 10 + 7] = i + 1
        alternates[3:7, i * 10 + 2 : i * 10 + 5] = i + 1
    numpy.random.seed(0)
    values = numpy.random.uniform(size=n)
    idx = 1
    my_min = 0.3
    my_max = 0.7
    expected = numpy.zeros(labels.shape, int)
    expected_alternates = numpy.zeros(alternates.shape, int)
    for i, value in zip(list(range(n)), values):
        if my_min <= value <= my_max:
            expected[labels == i + 1] = idx
            expected_alternates[alternates == i + 1] = idx
            idx += 1
    workspace, module = make_workspace(
        {INPUT_OBJECTS: labels, "my_alternates": alternates}
    )
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_LIMITS
    module.measurements[0].min_limit.value = my_min
    module.measurements[0].max_limit.value = my_max
    module.add_additional_object()
    module.additional_objects[0].object_name.value = "my_alternates"
    module.additional_objects[0].target_name.value = "my_additional_result"
    m = workspace.measurements
    m.add_measurement(INPUT_OBJECTS, TEST_FTR, values)
    module.run(workspace)
    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    alternates = workspace.object_set.get_objects("my_additional_result")
    assert numpy.all(labels.segmented == expected)
    assert numpy.all(alternates.segmented == expected_alternates)


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("filterobjects/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.filterobjects.FilterObjects)
    assert module.x_name == "Things"
    assert module.y_name == "FilteredThings"
    assert module.mode == cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
    assert (
        module.rules_directory.dir_choice
        == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
    )
    assert module.rules_file_name == "myrules.txt"
    assert module.measurements[0].measurement == "Intensity_MeanIntensity_DNA"
    assert module.filter_choice == cellprofiler.modules.filterobjects.FI_MINIMAL


def test_load_v4():
    file = tests.frontend.modules.get_test_resources_directory("filterobjects/v4.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.filterobjects.FilterObjects)
    assert module.y_name == "MyFilteredObjects"
    assert module.x_name == "MyObjects"
    assert module.mode == cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
    assert module.filter_choice == cellprofiler.modules.filterobjects.FI_LIMITS
    assert (
        module.rules_directory.dir_choice
        == cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME
    )
    assert module.rules_directory.custom_path == "./rules"
    assert module.rules_file_name == "myrules.txt"
    assert module.measurement_count.value == 2
    assert module.additional_object_count.value == 2
    assert module.measurements[0].measurement == "Intensity_LowerQuartileIntensity_DNA"
    assert module.measurements[0].wants_minimum
    assert not module.measurements[0].wants_maximum
    assert round(abs(module.measurements[0].min_limit.value - 0.2), 7) == 0
    assert round(abs(module.measurements[0].max_limit.value - 1.5), 7) == 0
    assert module.measurements[1].measurement == "Intensity_UpperQuartileIntensity_DNA"
    assert not module.measurements[1].wants_minimum
    assert module.measurements[1].wants_maximum
    assert round(abs(module.measurements[1].min_limit.value - 0.9), 7) == 0
    assert round(abs(module.measurements[1].max_limit.value - 1.8), 7) == 0
    for group, name in zip(module.additional_objects, ("Cells", "Cytoplasm")):
        assert group.object_name == name
        assert group.target_name == "Filtered%s" % name


def test_load_v5():
    file = tests.frontend.modules.get_test_resources_directory("filterobjects/v5.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.filterobjects.FilterObjects)
    assert module.y_name == "MyFilteredObjects"
    assert module.x_name == "MyObjects"
    assert module.mode == cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
    assert module.filter_choice == cellprofiler.modules.filterobjects.FI_LIMITS
    assert (
        module.rules_directory.dir_choice
        == cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME
    )
    assert module.rules_directory.custom_path == "./rules"
    assert module.rules_file_name == "myrules.txt"
    assert module.rules_class == "1"
    assert module.measurement_count.value == 2
    assert module.additional_object_count.value == 2
    assert module.measurements[0].measurement == "Intensity_LowerQuartileIntensity_DNA"
    assert module.measurements[0].wants_minimum
    assert not module.measurements[0].wants_maximum
    assert round(abs(module.measurements[0].min_limit.value - 0.2), 7) == 0
    assert round(abs(module.measurements[0].max_limit.value - 1.5), 7) == 0
    assert module.measurements[1].measurement == "Intensity_UpperQuartileIntensity_DNA"
    assert not module.measurements[1].wants_minimum
    assert module.measurements[1].wants_maximum
    assert round(abs(module.measurements[1].min_limit.value - 0.9), 7) == 0
    assert round(abs(module.measurements[1].max_limit.value - 1.8), 7) == 0
    for group, name in zip(module.additional_objects, ("Cells", "Cytoplasm")):
        assert group.object_name == name
        assert group.target_name == "Filtered%s" % name


def test_load_v6():
    file = tests.frontend.modules.get_test_resources_directory("filterobjects/v6.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.filterobjects.FilterObjects)
    assert module.y_name == "MyFilteredObjects"
    assert module.x_name == "MyObjects"
    assert module.mode == cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
    assert module.filter_choice == cellprofiler.modules.filterobjects.FI_LIMITS
    assert module.per_object_assignment == cellprofiler.modules.filterobjects.PO_BOTH
    assert (
        module.rules_directory.dir_choice
        == cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME
    )
    assert module.rules_directory.custom_path == "./rules"
    assert module.rules_file_name == "myrules.txt"
    assert module.rules_class == "1"
    assert module.measurement_count.value == 2
    assert module.additional_object_count.value == 2
    assert module.measurements[0].measurement == "Intensity_LowerQuartileIntensity_DNA"
    assert module.measurements[0].wants_minimum
    assert not module.measurements[0].wants_maximum
    assert round(abs(module.measurements[0].min_limit.value - 0.2), 7) == 0
    assert round(abs(module.measurements[0].max_limit.value - 1.5), 7) == 0
    assert module.measurements[1].measurement == "Intensity_UpperQuartileIntensity_DNA"
    assert not module.measurements[1].wants_minimum
    assert module.measurements[1].wants_maximum
    assert round(abs(module.measurements[1].min_limit.value - 0.9), 7) == 0
    assert round(abs(module.measurements[1].max_limit.value - 1.8), 7) == 0
    for group, name in zip(module.additional_objects, ("Cells", "Cytoplasm")):
        assert group.object_name == name
        assert group.target_name == "Filtered%s" % name


def test_load_v7():
    file = tests.frontend.modules.get_test_resources_directory("filterobjects/v7.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.filterobjects.FilterObjects)
    assert module.y_name == "MyFilteredObjects"
    assert module.x_name == "MyObjects"
    assert module.mode == cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
    assert module.filter_choice == cellprofiler.modules.filterobjects.FI_LIMITS
    assert (
        module.per_object_assignment
        == cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP
    )
    assert (
        module.rules_directory.dir_choice
        == cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME
    )
    assert module.rules_directory.custom_path == "./rules"
    assert module.rules_file_name == "myrules.txt"
    assert module.rules_class == "1"
    assert module.measurement_count.value == 2
    assert module.additional_object_count.value == 2
    assert module.measurements[0].measurement == "Intensity_LowerQuartileIntensity_DNA"
    assert module.measurements[0].wants_minimum
    assert not module.measurements[0].wants_maximum
    assert round(abs(module.measurements[0].min_limit.value - 0.2), 7) == 0
    assert round(abs(module.measurements[0].max_limit.value - 1.5), 7) == 0
    assert module.measurements[1].measurement == "Intensity_UpperQuartileIntensity_DNA"
    assert not module.measurements[1].wants_minimum
    assert module.measurements[1].wants_maximum
    assert round(abs(module.measurements[1].min_limit.value - 0.9), 7) == 0
    assert round(abs(module.measurements[1].max_limit.value - 1.8), 7) == 0
    for group, name in zip(module.additional_objects, ("Cells", "Cytoplasm")):
        assert group.object_name == name
        assert group.target_name == "Filtered%s" % name


def test_filter_by_rule():
    labels = numpy.zeros((10, 20), int)
    labels[3:5, 4:9] = 1
    labels[7:9, 6:12] = 2
    labels[4:9, 14:18] = 3
    workspace, module = make_workspace({"MyObjects": labels})
    assert isinstance(
        module, cellprofiler.modules.filterobjects.FilterByObjectMeasurement
    )
    m = workspace.measurements
    m.add_measurement("MyObjects", "MyMeasurement", numpy.array([1.5, 2.3, 1.8]))
    rules_file_contents = "IF (MyObjects_MyMeasurement > 2.0, [1.0,-1.0], [-1.0,1.0])\n"
    rules_path = tempfile.mktemp()
    fd = open(rules_path, "wt")
    try:
        fd.write(rules_file_contents)
        fd.close()
        rules_dir, rules_file = os.path.split(rules_path)
        module.x_name.value = "MyObjects"
        module.mode.value = cellprofiler.modules.filterobjects.MODE_RULES
        module.rules_file_name.value = rules_file
        module.rules_directory.dir_choice = (
            cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        )
        module.rules_directory.custom_path = rules_dir
        module.y_name.value = "MyTargetObjects"
        module.run(workspace)
        target_objects = workspace.object_set.get_objects("MyTargetObjects")
        target_labels = target_objects.segmented
        assert numpy.all(target_labels[labels == 2] > 0)
        assert numpy.all(target_labels[labels != 2] == 0)
    finally:
        os.remove(rules_path)


def test_filter_by_3_class_rule():
    rules_file_contents = (
        "IF (MyObjects_MyMeasurement > 2.0, [1.0,-1.0,-1.0], [-0.5,0.5,0.5])\n"
        "IF (MyObjects_MyMeasurement > 1.6, [0.5,0.5,-0.5], [-1.0,-1.0,1.0])\n"
    )
    expected_class = [None, "3", "1", "2"]
    rules_path = tempfile.mktemp()
    with open(rules_path, "wt") as fd:
        fd.write(rules_file_contents)
    try:
        for rules_class in ("1", "2", "3"):
            labels = numpy.zeros((10, 20), int)
            labels[3:5, 4:9] = 1
            labels[7:9, 6:12] = 2
            labels[4:9, 14:18] = 3
            workspace, module = make_workspace({"MyObjects": labels})
            assert isinstance(
                module, cellprofiler.modules.filterobjects.FilterByObjectMeasurement
            )
            m = workspace.measurements
            m.add_measurement(
                "MyObjects", "MyMeasurement", numpy.array([1.5, 2.3, 1.8])
            )
            rules_dir, rules_file = os.path.split(rules_path)
            module.x_name.value = "MyObjects"
            module.mode.value = cellprofiler.modules.filterobjects.MODE_RULES
            module.rules_file_name.value = rules_file
            module.rules_directory.dir_choice = (
                cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
            )
            module.rules_directory.custom_path = rules_dir
            module.rules_class.value = rules_class
            module.y_name.value = "MyTargetObjects"
            module.run(workspace)
            target_objects = workspace.object_set.get_objects("MyTargetObjects")
            target_labels = target_objects.segmented
            kept = expected_class.index(rules_class)
            assert numpy.all(target_labels[labels == kept] > 0)
            assert numpy.all(target_labels[labels != kept] == 0)
    finally:
        os.remove(rules_path)

def test_filter_by_fuzzy_rule():
    labels = numpy.zeros((10, 20), int)
    labels[3:5, 4:9] = 1
    labels[7:9, 6:12] = 2
    labels[4:9, 14:18] = 3
    workspace, module = make_workspace({"MyObjects": labels})
    assert isinstance(
        module, cellprofiler.modules.filterobjects.FilterByObjectMeasurement
    )
    m = workspace.measurements
    m.add_measurement("MyObjects", "MyMeasurement", numpy.array([1.5, 2.3, 1.8]))
    rules_file_contents = "IF (MyObjects_MMeasurement > 2.0, [1.0,-1.0], [-1.0,1.0])\n"
    rules_path = tempfile.mktemp()
    fd = open(rules_path, "wt")
    try:
        fd.write(rules_file_contents)
        fd.close()
        rules_dir, rules_file = os.path.split(rules_path)
        module.x_name.value = "MyObjects"
        module.mode.value = cellprofiler.modules.filterobjects.MODE_RULES
        module.rules_file_name.value = rules_file
        module.rules_directory.dir_choice = (
            cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        )
        module.rules_directory.custom_path = rules_dir
        module.y_name.value = "MyTargetObjects"
        module.allow_fuzzy.value=True
        module.run(workspace)
        target_objects = workspace.object_set.get_objects("MyTargetObjects")
        target_labels = target_objects.segmented
        assert numpy.all(target_labels[labels == 2] > 0)
        assert numpy.all(target_labels[labels != 2] == 0)
        module.allow_fuzzy.value=False
        with pytest.raises(AssertionError):
            module.run(workspace)
    finally:
        os.remove(rules_path)

def test_discard_border_objects():
    """Test the mode to discard border objects"""
    labels = numpy.zeros((10, 10), int)
    labels[1:4, 0:3] = 1
    labels[4:8, 1:5] = 2
    labels[:, 9] = 3

    expected = numpy.zeros((10, 10), int)
    expected[4:8, 1:5] = 1

    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.mode.value = cellprofiler.modules.filterobjects.MODE_BORDER
    module.run(workspace)
    output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(expected == output_objects.segmented)


def test_discard_mask_objects():
    """Test discarding objects that touch the mask of objects parent img"""
    mask = numpy.ones((10, 10), bool)
    mask[5, 5] = False
    labels = numpy.zeros((10, 10), int)
    labels[1:4, 1:4] = 1
    labels[5:8, 5:8] = 2
    expected = labels.copy()
    expected[expected == 2] = 0

    workspace, module = make_workspace({})
    parent_image = cellprofiler_core.image.Image(numpy.zeros((10, 10)), mask=mask)
    workspace.image_set.add(INPUT_IMAGE, parent_image)

    input_objects = cellprofiler_core.object.Objects()
    input_objects.segmented = labels
    input_objects.parent_image = parent_image

    workspace.object_set.add_objects(input_objects, INPUT_OBJECTS)

    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.mode.value = cellprofiler.modules.filterobjects.MODE_BORDER
    module.run(workspace)
    output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(expected == output_objects.segmented)


def test_unedited_segmented():
    # Test transferral of unedited segmented segmentation
    # from source to target

    unedited = numpy.zeros((10, 10), dtype=int)
    unedited[0:4, 0:4] = 1
    unedited[6:8, 0:4] = 2
    unedited[6:8, 6:8] = 3
    segmented = numpy.zeros((10, 10), dtype=int)
    segmented[6:8, 6:8] = 1
    segmented[6:8, 0:4] = 2

    workspace, module = make_workspace({})
    input_objects = cellprofiler_core.object.Objects()
    workspace.object_set.add_objects(input_objects, INPUT_OBJECTS)
    input_objects.segmented = segmented
    input_objects.unedited_segmented = unedited
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.mode.value = cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL
    m = workspace.measurements
    m[INPUT_OBJECTS, TEST_FTR] = numpy.array([2, 1])
    module.run(workspace)
    output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    numpy.testing.assert_equal(output_objects.unedited_segmented, unedited)


def test_small_removed_segmented():
    # Test output objects' small_removed_segmented
    #
    # It should be the small_removed_segmented of the
    # source minus the filtered

    unedited = numpy.zeros((10, 10), dtype=int)
    unedited[0:4, 0:4] = 1
    unedited[6:8, 0:4] = 2
    unedited[6:8, 6:8] = 3
    segmented = numpy.zeros((10, 10), dtype=int)
    segmented[6:8, 6:8] = 1
    segmented[6:8, 0:4] = 2

    workspace, module = make_workspace({})
    input_objects = cellprofiler_core.object.Objects()
    input_objects.segmented = segmented
    input_objects.unedited_segmented = unedited
    workspace.object_set.add_objects(input_objects, INPUT_OBJECTS)
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.mode.value = cellprofiler.modules.filterobjects.MODE_MEASUREMENTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MINIMAL
    m = workspace.measurements
    m[INPUT_OBJECTS, TEST_FTR] = numpy.array([2, 1])
    module.run(workspace)
    output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    small_removed = output_objects.small_removed_segmented
    mask = (unedited != 3) & (unedited != 0)
    assert numpy.all(small_removed[mask] != 0)
    assert numpy.all(small_removed[~mask] == 0)


def test_classify_none():
    workspace, module = make_workspace({INPUT_OBJECTS: numpy.zeros((10, 10), int)})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    with make_classifier(module, numpy.zeros(0, int), classes=[1]):
        workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(0)
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        assert output_objects.count == 0


def test_classify_true():
    labels = numpy.zeros((10, 10), int)
    labels[4:7, 4:7] = 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    with make_classifier(module, numpy.ones(1, int), classes=[1, 2]):
        workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(1)
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        assert output_objects.count == 1


def test_classify_false():
    labels = numpy.zeros((10, 10), int)
    labels[4:7, 4:7] = 1
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    with make_classifier(module, numpy.ones(1, int) * 2, classes=[1, 2]):
        workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(1)
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        assert output_objects.count == 0


def test_classify_many():
    labels = numpy.zeros((10, 10), int)
    labels[1:4, 1:4] = 1
    labels[5:7, 5:7] = 2
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    with make_classifier(module, numpy.array([1, 2]), classes=[1, 2]):
        workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(2)
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        assert output_objects.count == 1
        labels_out = output_objects.get_labels()[0][0]
        numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 1)
        numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 0)


def test_classify_keep_removed():
    labels = numpy.zeros((10, 10), int)
    labels[1:4, 1:4] = 1
    labels[5:7, 5:7] = 2
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.keep_removed_objects.value = True
    module.removed_objects_name.value = "RemovedObjects"
    with make_classifier(module, numpy.array([1, 2]), classes=[1, 2]):
        workspace.measurements[INPUT_OBJECTS, TEST_FTR] = numpy.zeros(2)
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        removed_objects = workspace.object_set.get_objects("RemovedObjects")
        assert output_objects.count == 1
        assert removed_objects.count == 1
        labels_out = output_objects.get_labels()[0][0]
        numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 1)
        numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 0)
        removed_labels_out = removed_objects.get_labels()[0][0]
        numpy.testing.assert_array_equal(removed_labels_out[1:4, 1:4], 0)
        numpy.testing.assert_array_equal(removed_labels_out[5:7, 5:7], 1)

def test_classify_many_fuzzy():
    labels = numpy.zeros((10, 10), int)
    labels[1:4, 1:4] = 1
    labels[5:7, 5:7] = 2
    workspace, module = make_workspace({INPUT_OBJECTS: labels})
    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    with make_classifier(module, numpy.array([1, 2]), classes=[1, 2]):
        workspace.measurements[INPUT_OBJECTS, MISSPELLED_FTR] = numpy.zeros(2)
        module.allow_fuzzy.value = True
        module.run(workspace)
        output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        assert output_objects.count == 1
        labels_out = output_objects.get_labels()[0][0]
        numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 1)
        numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 0)
        module.allow_fuzzy.value = False
        with pytest.raises(AssertionError):
            module.run(workspace)


def test_measurements():
    workspace, module = make_workspace(
        object_dict={
            INPUT_OBJECTS: numpy.zeros((10, 10), dtype=numpy.uint8),
            "additional_objects": numpy.zeros((10, 10), dtype=numpy.uint8),
        }
    )

    module.x_name.value = INPUT_OBJECTS

    module.y_name.value = OUTPUT_OBJECTS

    module.add_additional_object()

    module.additional_objects[0].object_name.value = "additional_objects"

    module.additional_objects[0].target_name.value = "additional_result"

    module.measurements[0].measurement.value = TEST_FTR

    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL

    measurements = workspace.measurements

    measurements.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.zeros((0,)))

    module.run(workspace)

    object_names = [
        (INPUT_OBJECTS, OUTPUT_OBJECTS),
        ("additional_objects", "additional_result"),
    ]

    for input_object_name, output_object_name in object_names:
        assert measurements.has_current_measurements(
            "Image",
            FF_COUNT % output_object_name,
        )

        assert measurements.has_current_measurements(
            input_object_name,
            FF_CHILDREN_COUNT % output_object_name,
        )

        output_object_features = [
            FF_PARENT % input_object_name,
            M_LOCATION_CENTER_X,
            M_LOCATION_CENTER_Y,
            M_LOCATION_CENTER_Z,
            M_NUMBER_OBJECT_NUMBER,
        ]

        for feature in output_object_features:
            assert measurements.has_current_measurements(output_object_name, feature)


def test_discard_border_objects_volume():
    labels = numpy.zeros((9, 16, 16), dtype=numpy.uint8)
    labels[:3, 1:5, 1:5] = 1  # touches bottom z-slice
    labels[-3:, -5:-1, -5:-1] = 2  # touches top z-slice
    labels[2:7, 6:10, 6:10] = 3
    labels[2:5, :5, -5:-1] = 4  # touches top edge
    labels[6:9, -5:-1, :5] = 5  # touches left edge

    expected = numpy.zeros_like(labels)
    expected[2:7, 6:10, 6:10] = 1

    src_objects = cellprofiler_core.object.Objects()
    src_objects.segmented = labels

    workspace, module = make_workspace({INPUT_OBJECTS: labels})

    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.mode.value = cellprofiler.modules.filterobjects.MODE_BORDER

    module.run(workspace)

    output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)

    numpy.testing.assert_array_equal(output_objects.segmented, expected)


def test_keep_maximal_per_object_most_overlap_volume():
    labels = numpy.zeros((1, 10, 20), int)
    labels[:, :, :10] = 1
    labels[:, :, 10:] = 2

    sub_labels = numpy.zeros((1, 10, 20), int)
    sub_labels[:, 2, 4] = 1
    sub_labels[:, 4:6, 8:15] = 2
    sub_labels[:, 8, 15] = 3

    expected = sub_labels * (sub_labels != 3)

    workspace, module = make_workspace(
        {INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels}
    )

    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
    )
    module.per_object_assignment.value = (
        cellprofiler.modules.filterobjects.PO_PARENT_WITH_MOST_OVERLAP
    )

    m = workspace.measurements

    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

    module.run(workspace)

    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

    assert numpy.all(labels.segmented == expected)


def test_keep_minimal_per_object_both_parents_image():
    labels = numpy.zeros((10, 20), int)
    labels[:, :10] = 1
    labels[:, 10:] = 2

    sub_labels = numpy.zeros((10, 20), int)
    sub_labels[2, 4] = 1
    sub_labels[4:6, 8:15] = 2
    sub_labels[8, 15] = 3

    expected = numpy.zeros_like(sub_labels)
    expected[2, 4] = 1
    expected[8, 15] = 2

    workspace, module = make_workspace(
        {INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels}
    )

    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MINIMAL_PER_OBJECT
    )
    module.per_object_assignment.value = cellprofiler.modules.filterobjects.PO_BOTH

    m = workspace.measurements

    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

    module.run(workspace)

    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

    assert numpy.all(labels.segmented == expected)


def test_keep_maximal_both_parents_volume():
    labels = numpy.zeros((2, 10, 20), int)
    labels[:, :, :10] = 1
    labels[:, :, 10:] = 2

    sub_labels = numpy.zeros((2, 10, 20), int)
    sub_labels[:, 2, 4] = 1
    sub_labels[:, 4:6, 8:15] = 2
    sub_labels[:, 8, 15] = 3

    expected = numpy.zeros_like(sub_labels)
    expected[sub_labels == 2] = 1

    workspace, module = make_workspace(
        {INPUT_OBJECTS: sub_labels, ENCLOSING_OBJECTS: labels}
    )

    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.enclosing_object_name.value = ENCLOSING_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = (
        cellprofiler.modules.filterobjects.FI_MAXIMAL_PER_OBJECT
    )
    module.per_object_assignment.value = cellprofiler.modules.filterobjects.PO_BOTH

    m = workspace.measurements

    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

    module.run(workspace)

    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

    assert numpy.all(labels.segmented == expected)


def test_keep_maximal_volume():
    labels = numpy.zeros((2, 10, 20), int)
    labels[:, 2, 4] = 1
    labels[:, 4:6, 8:15] = 2
    labels[:, 8, 15] = 3

    expected = numpy.zeros_like(labels)
    expected[labels == 2] = 1

    workspace, module = make_workspace({INPUT_OBJECTS: labels})

    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL

    m = workspace.measurements

    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

    module.run(workspace)

    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

    assert numpy.all(labels.segmented == expected)


def test_keep_removed_volume():
    labels = numpy.zeros((2, 10, 20), int)
    labels[:, 2, 4] = 1
    labels[:, 4:6, 8:15] = 2
    labels[:, 8, 15] = 3

    expected = numpy.zeros_like(labels)
    expected[labels == 2] = 1

    expected_removed = numpy.zeros_like(labels)
    expected_removed[labels == 1] = 1
    expected_removed[labels == 3] = 2


    workspace, module = make_workspace({INPUT_OBJECTS: labels})

    module.x_name.value = INPUT_OBJECTS
    module.y_name.value = OUTPUT_OBJECTS
    module.keep_removed_objects.value = True
    module.removed_objects_name.value = "RemovedObjects"
    module.measurements[0].measurement.value = TEST_FTR
    module.filter_choice.value = cellprofiler.modules.filterobjects.FI_MAXIMAL

    m = workspace.measurements

    m.add_measurement(INPUT_OBJECTS, TEST_FTR, numpy.array([1, 4, 2]))

    module.run(workspace)

    labels = workspace.object_set.get_objects(OUTPUT_OBJECTS)

    removed_labels = workspace.object_set.get_objects("RemovedObjects")

    assert numpy.all(labels.segmented == expected)

    assert numpy.all(removed_labels.segmented == expected_removed)


class FakeClassifier(object):
    def __init__(self, answers, classes):
        """initializer

        answers - a vector of answers to be returned by "predict"

        classes - a vector of class numbers to be used to populate classes_
        """
        self.answers_ = answers
        self.classes_ = classes

    def predict(self, *args, **kwargs):
        return self.answers_


def make_classifier_pickle(answers, classes, class_names, name, feature_names):
    """Make a pickle of a fake classifier

    answers - the answers you want to get back after calling classifier.predict
    classes - the class #s for the answers.
    class_names - one name per class in the order they appear in classes
    name - the name of the classifier
    feature_names - the names of the features fed into the classifier
    """
    classifier = FakeClassifier(answers, classes)
    return pickle.dumps([classifier, class_names, name, feature_names])
