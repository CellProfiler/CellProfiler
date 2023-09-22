import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import (
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y,
    FF_COUNT,
    FF_PARENT,
    COLTYPE_FLOAT,
    M_NUMBER_OBJECT_NUMBER,
    COLTYPE_INTEGER,
    FF_CHILDREN_COUNT,
)

import cellprofiler.modules.splitormergeobjects
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_OBJECTS_NAME = "inputobjects"
OUTPUT_OBJECTS_NAME = "outputobjects"
IMAGE_NAME = "image"
OUTLINE_NAME = "outlines"


def test_load_v5():
    file = tests.frontend.modules.get_test_resources_directory("splitormergeobjects/v5.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.loadtxt(six.moves.StringIO(data))
    module = pipeline.modules()[0]

    assert module.objects_name.value == "IdentifyPrimaryObjects"
    assert module.output_objects_name.value == "SplitOrMergeObjects"
    assert module.relabel_option.value == "Merge"
    assert module.distance_threshold.value == 0
    assert not module.wants_image.value
    assert module.image_name.value == "None"
    assert module.minimum_intensity_fraction.value == 0.9
    assert module.where_algorithm.value == "Closest point"
    assert module.merge_option.value == "Distance"
    assert module.parent_object.value == "None"
    assert module.merging_method.value == "Disconnected"


def test_load_v4():
    file = tests.frontend.modules.get_test_resources_directory("splitormergeobjects/v4.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.loadtxt(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.splitormergeobjects.SplitOrMergeObjects
    )
    assert module.objects_name == "blobs"
    assert module.output_objects_name == "RelabeledBlobs"
    assert (
        module.relabel_option == cellprofiler.modules.splitormergeobjects.OPTION_MERGE
    )
    assert module.distance_threshold == 2
    assert not module.wants_image
    assert module.image_name == "Guide"
    assert module.minimum_intensity_fraction == 0.8
    assert (
        module.where_algorithm
        == cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT
    )
    assert module.merge_option == cellprofiler.modules.splitormergeobjects.UNIFY_PARENT
    assert module.parent_object == "Nuclei"
    assert (
        module.merging_method == cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL
    )

    module = pipeline.modules()[1]
    assert (
        module.relabel_option == cellprofiler.modules.splitormergeobjects.OPTION_SPLIT
    )
    assert module.wants_image
    assert (
        module.where_algorithm == cellprofiler.modules.splitormergeobjects.CA_CENTROIDS
    )
    assert (
        module.merge_option == cellprofiler.modules.splitormergeobjects.UNIFY_DISTANCE
    )
    assert (
        module.merging_method
        == cellprofiler.modules.splitormergeobjects.UM_DISCONNECTED
    )


def rruunn(
    input_labels,
    relabel_option,
    merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_DISTANCE,
    unify_method=cellprofiler.modules.splitormergeobjects.UM_DISCONNECTED,
    distance_threshold=5,
    minimum_intensity_fraction=0.9,
    where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT,
    image=None,
    parent_object="Parent_object",
    parents_of=None,
):
    """Run the SplitOrMergeObjects module

    returns the labels matrix and the workspace.
    """
    module = cellprofiler.modules.splitormergeobjects.SplitOrMergeObjects()
    module.set_module_num(1)
    module.objects_name.value = INPUT_OBJECTS_NAME
    module.output_objects_name.value = OUTPUT_OBJECTS_NAME
    module.relabel_option.value = relabel_option
    module.merge_option.value = merge_option
    module.merging_method.value = unify_method
    module.parent_object.value = parent_object
    module.distance_threshold.value = distance_threshold
    module.minimum_intensity_fraction.value = minimum_intensity_fraction
    module.wants_image.value = image is not None
    module.where_algorithm.value = where_algorithm

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    if image is not None:
        img = cellprofiler_core.image.Image(image)
        image_set.add(IMAGE_NAME, img)
        module.image_name.value = IMAGE_NAME

    object_set = cellprofiler_core.object.ObjectSet()
    o = cellprofiler_core.object.Objects()
    o.segmented = input_labels
    object_set.add_objects(o, INPUT_OBJECTS_NAME)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    if parents_of is not None:
        m = workspace.measurements
        ftr = FF_PARENT % parent_object
        m[INPUT_OBJECTS_NAME, ftr] = parents_of
    module.run(workspace)
    output_objects = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
    return output_objects.segmented, workspace


def test_split_zero():
    labels, workspace = rruunn(
        numpy.zeros((10, 20), int),
        cellprofiler.modules.splitormergeobjects.OPTION_SPLIT,
    )
    assert numpy.all(labels == 0)
    assert labels.shape[0] == 10
    assert labels.shape[1] == 20

    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    count = m.get_current_image_measurement(FF_COUNT % OUTPUT_OBJECTS_NAME)
    assert count == 0
    for feature_name in (
        M_LOCATION_CENTER_X,
        M_LOCATION_CENTER_Y,
    ):
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, feature_name)
        assert len(values) == 0

    module = workspace.module
    assert isinstance(
        module, cellprofiler.modules.splitormergeobjects.SplitOrMergeObjects
    )
    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(columns) == 6
    for object_name, feature_name, coltype in (
        (OUTPUT_OBJECTS_NAME, M_LOCATION_CENTER_X, COLTYPE_FLOAT,),
        (OUTPUT_OBJECTS_NAME, M_LOCATION_CENTER_Y, COLTYPE_FLOAT,),
        (OUTPUT_OBJECTS_NAME, M_NUMBER_OBJECT_NUMBER, COLTYPE_INTEGER,),
        (INPUT_OBJECTS_NAME, FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME, COLTYPE_INTEGER,),
        (OUTPUT_OBJECTS_NAME, FF_PARENT % INPUT_OBJECTS_NAME, COLTYPE_INTEGER,),
        ("Image", FF_COUNT % OUTPUT_OBJECTS_NAME, COLTYPE_INTEGER,),
    ):
        assert any(
            [
                object_name == c[0] and feature_name == c[1] and coltype == c[2]
                for c in columns
            ]
        )
    categories = module.get_categories(workspace.pipeline, "Image")
    assert len(categories) == 1
    assert categories[0] == "Count"
    categories = module.get_categories(workspace.pipeline, OUTPUT_OBJECTS_NAME)
    assert len(categories) == 3
    assert any(["Location" in categories])
    assert any(["Parent" in categories])
    assert any(["Number" in categories])
    categories = module.get_categories(workspace.pipeline, INPUT_OBJECTS_NAME)
    assert len(categories) == 1
    assert categories[0] == "Children"
    f = module.get_measurements(workspace.pipeline, "Image", "Count")
    assert len(f) == 1
    assert f[0] == OUTPUT_OBJECTS_NAME
    f = module.get_measurements(workspace.pipeline, OUTPUT_OBJECTS_NAME, "Location")
    assert len(f) == 2
    assert all([any([x == y for y in f]) for x in ("Center_X", "Center_Y")])
    f = module.get_measurements(workspace.pipeline, OUTPUT_OBJECTS_NAME, "Parent")
    assert len(f) == 1
    assert f[0] == INPUT_OBJECTS_NAME

    f = module.get_measurements(workspace.pipeline, OUTPUT_OBJECTS_NAME, "Number")
    assert len(f) == 1
    assert f[0] == "Object_Number"

    f = module.get_measurements(workspace.pipeline, INPUT_OBJECTS_NAME, "Children")
    assert len(f) == 1
    assert f[0] == "%s_Count" % OUTPUT_OBJECTS_NAME


def test_split_one():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels_out, workspace = rruunn(
        labels, cellprofiler.modules.splitormergeobjects.OPTION_SPLIT
    )
    assert numpy.all(labels == labels_out)

    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    count = m.get_current_image_measurement(FF_COUNT % OUTPUT_OBJECTS_NAME)
    assert count == 1
    for feature_name, value in (
        (M_LOCATION_CENTER_X, 5),
        (M_LOCATION_CENTER_Y, 3),
        (FF_PARENT % INPUT_OBJECTS_NAME, 1),
    ):
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, feature_name)
        assert len(values) == 1
        assert round(abs(values[0] - value), 7) == 0

    values = m.get_current_measurement(
        INPUT_OBJECTS_NAME, FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME,
    )
    assert len(values) == 1
    assert values[0] == 1


def test_split_one_into_two():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 1
    labels_out, workspace = rruunn(
        labels, cellprofiler.modules.splitormergeobjects.OPTION_SPLIT
    )
    index = numpy.array([labels_out[3, 5], labels_out[3, 15]])
    assert index[0] != index[1]
    assert all([x in index for x in (1, 2)])
    expected = numpy.zeros((10, 20), int)
    expected[2:5, 3:8] = index[0]
    expected[2:5, 13:18] = index[1]
    assert numpy.all(labels_out == expected)
    m = workspace.measurements
    values = m.get_current_measurement(
        OUTPUT_OBJECTS_NAME, FF_PARENT % INPUT_OBJECTS_NAME,
    )
    assert len(values) == 2
    assert numpy.all(values == 1)
    values = m.get_current_measurement(
        INPUT_OBJECTS_NAME, FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME,
    )
    assert len(values) == 1
    assert values[0] == 2


def test_unify_zero():
    labels, workspace = rruunn(
        numpy.zeros((10, 20), int),
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
    )
    assert numpy.all(labels == 0)
    assert labels.shape[0] == 10
    assert labels.shape[1] == 20


def test_unify_one():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels_out, workspace = rruunn(
        labels, cellprofiler.modules.splitormergeobjects.OPTION_MERGE
    )
    assert numpy.all(labels == labels_out)


def test_unify_two_to_one():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        distance_threshold=6,
    )
    assert numpy.all(labels_out[labels != 0] == 1)
    assert numpy.all(labels_out[labels == 0] == 0)


def test_unify_two_stays_two():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        distance_threshold=4,
    )
    assert numpy.all(labels_out == labels)


def test_unify_image_centroids():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    image = numpy.ones((10, 20)) * (labels > 0) * 0.5
    image[3, 8:13] = 0.41
    image[3, 5] = 0.6
    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        distance_threshold=6,
        image=image,
        minimum_intensity_fraction=0.8,
        where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CENTROIDS,
    )
    assert numpy.all(labels_out[labels != 0] == 1)
    assert numpy.all(labels_out[labels == 0] == 0)


def test_dont_unify_image_centroids():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    image = numpy.ones((10, 20)) * labels * 0.5
    image[3, 8:12] = 0.41
    image[3, 5] = 0.6
    image[3, 15] = 0.6
    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        distance_threshold=6,
        image=image,
        minimum_intensity_fraction=0.8,
        where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CENTROIDS,
    )
    assert numpy.all(labels_out == labels)


def test_unify_image_closest_point():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    image = numpy.ones((10, 20)) * (labels > 0) * 0.6
    image[2, 8:13] = 0.41
    image[2, 7] = 0.5
    image[2, 13] = 0.5
    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        distance_threshold=6,
        image=image,
        minimum_intensity_fraction=0.8,
        where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT,
    )
    assert numpy.all(labels_out[labels != 0] == 1)
    assert numpy.all(labels_out[labels == 0] == 0)


def test_dont_unify_image_closest_point():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    image = numpy.ones((10, 20)) * labels * 0.6
    image[3, 8:12] = 0.41
    image[2, 7] = 0.5
    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        distance_threshold=6,
        image=image,
        minimum_intensity_fraction=0.8,
        where_algorithm=cellprofiler.modules.splitormergeobjects.CA_CLOSEST_POINT,
    )
    assert numpy.all(labels_out == labels)


def test_unify_per_parent():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2

    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_PARENT,
        parent_object="Parent_object",
        parents_of=numpy.array([1, 1]),
    )
    assert numpy.all(labels_out[labels != 0] == 1)


def test_unify_convex_hull():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    expected = numpy.zeros(labels.shape, int)
    expected[2:5, 3:18] = 1

    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_PARENT,
        unify_method=cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL,
        parent_object="Parent_object",
        parents_of=numpy.array([1, 1]),
    )
    assert numpy.all(labels_out == expected)


def test_unify_nothing():
    labels = numpy.zeros((10, 20), int)
    for um in (
        cellprofiler.modules.splitormergeobjects.UM_DISCONNECTED,
        cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL,
    ):
        labels_out, workspace = rruunn(
            labels,
            cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
            merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_PARENT,
            unify_method=cellprofiler.modules.splitormergeobjects.UM_CONVEX_HULL,
            parent_object="Parent_object",
            parents_of=numpy.zeros(0, int),
        )
        assert numpy.all(labels_out == 0)

def test_unify_per_parent_non_consecutive():
    labels = numpy.zeros((10, 20), int)
    labels[2:5, 3:8] = 1
    labels[2:5, 13:18] = 2
    labels[7:9, 13:18] = 3

    labels_out, workspace = rruunn(
        labels,
        cellprofiler.modules.splitormergeobjects.OPTION_MERGE,
        merge_option=cellprofiler.modules.splitormergeobjects.UNIFY_PARENT,
        parent_object="Parent_object",
        parents_of=numpy.array([1, 1, 3]),
    )
    assert numpy.max(labels_out == 2)
