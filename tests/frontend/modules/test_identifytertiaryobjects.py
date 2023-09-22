import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.measurement
import cellprofiler_core.modules
from cellprofiler_core.constants.measurement import FF_COUNT, COLTYPE_INTEGER, M_LOCATION_CENTER_X, COLTYPE_FLOAT, \
    M_LOCATION_CENTER_Y, M_NUMBER_OBJECT_NUMBER, FF_CHILDREN_COUNT, FF_PARENT, R_FIRST_IMAGE_NUMBER, \
    R_SECOND_IMAGE_NUMBER, R_FIRST_OBJECT_NUMBER, R_SECOND_OBJECT_NUMBER


import cellprofiler.modules.identifytertiaryobjects
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

PRIMARY = "primary"
SECONDARY = "secondary"
TERTIARY = "tertiary"
OUTLINES = "Outlines"


def on_pipeline_event(caller, event):
    assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)


def make_workspace(primary_labels, secondary_labels):
    """Make a workspace that has objects for the input labels

    returns a workspace with the following
        object_set - has object with name "primary" containing
                        the primary labels
                        has object with name "secondary" containing
                        the secondary labels
    """
    isl = cellprofiler_core.image.ImageSetList()
    module = cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion()
    module.set_module_num(1)
    module.primary_objects_name.value = PRIMARY
    module.secondary_objects_name.value = SECONDARY
    module.subregion_objects_name.value = TERTIARY
    workspace = cellprofiler_core.workspace.Workspace(
        cellprofiler_core.pipeline.Pipeline(),
        module,
        isl.get_image_set(0),
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        isl,
    )
    workspace.pipeline.add_module(module)

    for labels, name in ((primary_labels, PRIMARY), (secondary_labels, SECONDARY)):
        objects = cellprofiler_core.object.Objects()
        objects.segmented = labels
        workspace.object_set.add_objects(objects, name)
    return workspace


def test_zeros():
    """Test IdentifyTertiarySubregion on an empty image"""
    primary_labels = numpy.zeros((10, 10), int)
    secondary_labels = numpy.zeros((10, 10), int)
    workspace = make_workspace(primary_labels, secondary_labels)
    module = workspace.module
    module.run(workspace)
    measurements = workspace.measurements
    assert "Image" in measurements.get_object_names()
    count_feature = "Count_%s" % TERTIARY
    assert count_feature in measurements.get_feature_names("Image")
    value = measurements.get_current_measurement("Image", count_feature)
    assert numpy.product(value.shape) == 1
    assert value == 0
    assert TERTIARY in workspace.object_set.get_object_names()
    output_objects = workspace.object_set.get_objects(TERTIARY)
    assert numpy.all(output_objects.segmented == primary_labels)
    columns = module.get_measurement_columns(workspace.pipeline)
    for object_name in (
        "Image",
        PRIMARY,
        SECONDARY,
        TERTIARY,
    ):
        ocolumns = [x for x in columns if x[0] == object_name]
        features = measurements.get_feature_names(object_name)
        assert len(ocolumns) == len(features)
        assert all([column[1] in features for column in ocolumns])


def test_one_object():
    """Test creation of a single tertiary object"""
    primary_labels = numpy.zeros((10, 10), int)
    secondary_labels = numpy.zeros((10, 10), int)
    primary_labels[3:6, 4:7] = 1
    secondary_labels[2:7, 3:8] = 1
    expected_labels = numpy.zeros((10, 10), int)
    expected_labels[2:7, 3:8] = 1
    expected_labels[4, 5] = 0
    workspace = make_workspace(primary_labels, secondary_labels)
    module = workspace.module
    module.run(workspace)
    measurements = workspace.measurements
    assert "Image" in measurements.get_object_names()
    count_feature = "Count_%s" % TERTIARY
    assert count_feature in measurements.get_feature_names("Image")
    value = measurements.get_current_measurement("Image", count_feature)
    assert numpy.product(value.shape) == 1
    assert value == 1

    assert TERTIARY in measurements.get_object_names()
    child_count_feature = "Children_%s_Count" % TERTIARY
    for parent_name in (PRIMARY, SECONDARY):
        parents_of_feature = "Parent_%s" % parent_name
        assert parents_of_feature in measurements.get_feature_names(TERTIARY)
        value = measurements.get_current_measurement(TERTIARY, parents_of_feature)
        assert numpy.product(value.shape), 1
        assert value[0], 1
        assert child_count_feature in measurements.get_feature_names(parent_name)
        value = measurements.get_current_measurement(parent_name, child_count_feature)
        assert numpy.product(value.shape), 1
        assert value[0], 1

    for axis, expected in (("X", 5), ("Y", 4)):
        feature = "Location_Center_%s" % axis
        assert feature in measurements.get_feature_names(TERTIARY)
        value = measurements.get_current_measurement(TERTIARY, feature)
        assert numpy.product(value.shape), 1
        assert value[0] == expected

    assert TERTIARY in workspace.object_set.get_object_names()
    output_objects = workspace.object_set.get_objects(TERTIARY)
    assert numpy.all(output_objects.segmented == expected_labels)


def test_two_objects():
    """Test creation of two tertiary objects"""
    primary_labels = numpy.zeros((10, 20), int)
    secondary_labels = numpy.zeros((10, 20), int)
    expected_primary_parents = numpy.zeros((10, 20), int)
    expected_secondary_parents = numpy.zeros((10, 20), int)
    centers = ((4, 5, 1, 2), (4, 15, 2, 1))
    for x, y, primary_label, secondary_label in centers:
        primary_labels[x - 1 : x + 2, y - 1 : y + 2] = primary_label
        secondary_labels[x - 2 : x + 3, y - 2 : y + 3] = secondary_label
        expected_primary_parents[x - 2 : x + 3, y - 2 : y + 3] = primary_label
        expected_primary_parents[x, y] = 0
        expected_secondary_parents[x - 2 : x + 3, y - 2 : y + 3] = secondary_label
        expected_secondary_parents[x, y] = 0

    workspace = make_workspace(primary_labels, secondary_labels)
    module = workspace.module
    module.run(workspace)
    measurements = workspace.measurements
    count_feature = "Count_%s" % TERTIARY
    value = measurements.get_current_measurement("Image", count_feature)
    assert value == 2

    child_count_feature = "Children_%s_Count" % TERTIARY
    output_labels = workspace.object_set.get_objects(TERTIARY).segmented
    for parent_name, idx, parent_labels in (
        (PRIMARY, 2, expected_primary_parents),
        (SECONDARY, 3, expected_secondary_parents),
    ):
        parents_of_feature = "Parent_%s" % parent_name
        cvalue = measurements.get_current_measurement(parent_name, child_count_feature)
        assert numpy.all(cvalue == 1)
        pvalue = measurements.get_current_measurement(TERTIARY, parents_of_feature)
        for value in (pvalue, cvalue):
            assert numpy.product(value.shape), 2
        #
        # Make an array that maps the parent label index to the
        # corresponding child label index
        #
        label_map = numpy.zeros((len(centers) + 1,), int)
        for center in centers:
            label = center[idx]
            label_map[label] = pvalue[center[idx] - 1]
        expected_labels = label_map[parent_labels]
        assert numpy.all(expected_labels == output_labels)


def test_overlapping_secondary():
    """Make sure that an overlapping tertiary is assigned to the larger parent"""
    expected_primary_parents = numpy.zeros((10, 20), int)
    expected_secondary_parents = numpy.zeros((10, 20), int)
    primary_labels = numpy.zeros((10, 20), int)
    secondary_labels = numpy.zeros((10, 20), int)
    primary_labels[3:6, 3:10] = 2
    primary_labels[3:6, 10:17] = 1
    secondary_labels[2:7, 2:12] = 1
    expected_primary_parents[2:7, 2:12] = 2
    expected_primary_parents[4, 4:12] = 0  # the middle of the primary
    expected_primary_parents[4, 9] = 2  # the outline of primary # 2
    expected_primary_parents[4, 10] = 2  # the outline of primary # 1
    expected_secondary_parents[expected_primary_parents > 0] = 1
    workspace = make_workspace(primary_labels, secondary_labels)
    module = workspace.module
    assert isinstance(
        module, cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion
    )
    module.run(workspace)
    measurements = workspace.measurements
    output_labels = workspace.object_set.get_objects(TERTIARY).segmented
    for parent_name, parent_labels in (
        (PRIMARY, expected_primary_parents),
        (SECONDARY, expected_secondary_parents),
    ):
        parents_of_feature = "Parent_%s" % parent_name
        pvalue = measurements.get_current_measurement(TERTIARY, parents_of_feature)
        label_map = numpy.zeros((numpy.product(pvalue.shape) + 1,), int)
        label_map[1:] = pvalue.flatten()
        mapped_labels = label_map[output_labels]
        assert numpy.all(parent_labels == mapped_labels)


def test_wrong_size():
    """Regression test of img-961, what if objects have different sizes?

    Slightly bizarre use case: maybe if user wants to measure background
    outside of cells in a plate of wells???
    """
    expected_primary_parents = numpy.zeros((20, 20), int)
    expected_secondary_parents = numpy.zeros((20, 20), int)
    primary_labels = numpy.zeros((10, 30), int)
    secondary_labels = numpy.zeros((20, 20), int)
    primary_labels[3:6, 3:10] = 2
    primary_labels[3:6, 10:17] = 1
    secondary_labels[2:7, 2:12] = 1
    expected_primary_parents[2:7, 2:12] = 2
    expected_primary_parents[4, 4:12] = 0  # the middle of the primary
    expected_primary_parents[4, 9] = 2  # the outline of primary # 2
    expected_primary_parents[4, 10] = 2  # the outline of primary # 1
    expected_secondary_parents[expected_primary_parents > 0] = 1
    workspace = make_workspace(primary_labels, secondary_labels)
    module = workspace.module
    assert isinstance(
        module, cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion
    )
    module.run(workspace)


def test_get_measurement_columns():
    """Test the get_measurement_columns method"""
    module = cellprofiler.modules.identifytertiaryobjects.IdentifyTertiarySubregion()
    module.primary_objects_name.value = PRIMARY
    module.secondary_objects_name.value = SECONDARY
    module.subregion_objects_name.value = TERTIARY
    columns = module.get_measurement_columns(None)
    expected = (
        (
            "Image",
            FF_COUNT % TERTIARY,
            COLTYPE_INTEGER,
        ),
        (
            TERTIARY,
            M_LOCATION_CENTER_X,
            COLTYPE_FLOAT,
        ),
        (
            TERTIARY,
            M_LOCATION_CENTER_Y,
            COLTYPE_FLOAT,
        ),
        (
            TERTIARY,
            M_NUMBER_OBJECT_NUMBER,
            COLTYPE_INTEGER,
        ),
        (
            PRIMARY,
            FF_CHILDREN_COUNT % TERTIARY,
            COLTYPE_INTEGER,
        ),
        (
            SECONDARY,
            FF_CHILDREN_COUNT % TERTIARY,
            COLTYPE_INTEGER,
        ),
        (
            TERTIARY,
            FF_PARENT % PRIMARY,
            COLTYPE_INTEGER,
        ),
        (
            TERTIARY,
            FF_PARENT % SECONDARY,
            COLTYPE_INTEGER,
        ),
    )
    assert len(columns) == len(expected)
    for column in columns:
        assert any([all([cv == ev for cv, ev in zip(column, ec)]) for ec in expected])


def test_do_not_shrink():
    """Test the option to not shrink the smaller objects"""
    primary_labels = numpy.zeros((10, 10), int)
    secondary_labels = numpy.zeros((10, 10), int)
    primary_labels[3:6, 4:7] = 1
    secondary_labels[2:7, 3:8] = 1
    expected_labels = numpy.zeros((10, 10), int)
    expected_labels[2:7, 3:8] = 1
    expected_labels[3:6, 4:7] = 0

    workspace = make_workspace(primary_labels, secondary_labels)
    module = workspace.module
    module.shrink_primary.value = False
    module.run(workspace)
    measurements = workspace.measurements

    output_objects = workspace.object_set.get_objects(TERTIARY)
    assert numpy.all(output_objects.segmented == expected_labels)


def test_do_not_shrink_identical():
    """Test a case where the primary and secondary objects are identical"""
    primary_labels = numpy.zeros((20, 20), int)
    secondary_labels = numpy.zeros((20, 20), int)
    expected_labels = numpy.zeros((20, 20), int)

    # first and third objects have different sizes
    primary_labels[3:6, 4:7] = 1
    secondary_labels[2:7, 3:8] = 1
    expected_labels[2:7, 3:8] = 1
    expected_labels[3:6, 4:7] = 0

    primary_labels[13:16, 4:7] = 3
    secondary_labels[12:17, 3:8] = 3
    expected_labels[12:17, 3:8] = 3
    expected_labels[13:16, 4:7] = 0

    # second object and fourth have same size

    primary_labels[3:6, 14:17] = 2
    secondary_labels[3:6, 14:17] = 2
    primary_labels[13:16, 14:17] = 4
    secondary_labels[13:16, 14:17] = 4
    workspace = make_workspace(primary_labels, secondary_labels)

    module = workspace.module
    module.shrink_primary.value = False
    module.run(workspace)
    output_objects = workspace.object_set.get_objects(TERTIARY)
    assert numpy.all(output_objects.segmented == expected_labels)

    measurements = workspace.measurements
    count_feature = "Count_%s" % TERTIARY
    value = measurements.get_current_measurement("Image", count_feature)
    assert value == 3

    child_count_feature = "Children_%s_Count" % TERTIARY
    for parent_name in PRIMARY, SECONDARY:
        parent_of_feature = "Parent_%s" % parent_name
        parent_of = measurements.get_current_measurement(TERTIARY, parent_of_feature)
        child_count = measurements.get_current_measurement(
            parent_name, child_count_feature
        )
        for parent, expected_child_count in ((1, 1), (2, 0), (3, 1), (4, 0)):
            assert child_count[parent - 1] == expected_child_count
        for child in (1, 3):
            assert parent_of[child - 1] == child

    for location_feature in (
        M_LOCATION_CENTER_X,
        M_LOCATION_CENTER_Y,
    ):
        values = measurements.get_current_measurement(TERTIARY, location_feature)
        assert numpy.all(numpy.isnan(values) == [False, True, False])


def test_do_not_shrink_missing():
    # Regression test of 705

    for missing in range(1, 3):
        for missing_primary in False, True:
            primary_labels = numpy.zeros((20, 20), int)
            secondary_labels = numpy.zeros((20, 20), int)
            expected_labels = numpy.zeros((20, 20), int)
            centers = ((5, 5), (15, 5), (5, 15))
            pidx = 1
            sidx = 1
            for idx, (i, j) in enumerate(centers):
                if (idx + 1 != missing) or not missing_primary:
                    primary_labels[(i - 1) : (i + 2), (j - 1) : (j + 2)] = pidx
                    pidx += 1
                if (idx + 1 != missing) or missing_primary:
                    secondary_labels[(i - 2) : (i + 3), (j - 2) : (j + 3)] = sidx
                    sidx += 1
            expected_labels = secondary_labels * (primary_labels == 0)
            workspace = make_workspace(primary_labels, secondary_labels)

            module = workspace.module
            module.shrink_primary.value = False
            module.run(workspace)
            output_objects = workspace.object_set.get_objects(TERTIARY)
            assert numpy.all(output_objects.segmented == expected_labels)

            m = workspace.measurements

            child_name = module.subregion_objects_name.value
            primary_name = module.primary_objects_name.value
            ftr = FF_PARENT % primary_name
            pparents = m[child_name, ftr]
            assert len(pparents) == (3 if missing_primary else 2)
            if missing_primary:
                assert pparents[missing - 1] == 0

            secondary_name = module.secondary_objects_name.value
            ftr = FF_PARENT % secondary_name
            pparents = m[child_name, ftr]
            assert len(pparents) == (3 if missing_primary else 2)
            if not missing_primary:
                assert all([x in pparents for x in range(1, 3)])

            ftr = FF_CHILDREN_COUNT % child_name
            children = m[primary_name, ftr]
            assert len(children) == (2 if missing_primary else 3)
            if not missing_primary:
                assert children[missing - 1] == 0
                assert numpy.all(numpy.delete(children, missing - 1) == 1)
            else:
                assert numpy.all(children == 1)

            children = m[secondary_name, ftr]
            assert len(children) == (3 if missing_primary else 2)
            assert numpy.all(children == 1)


def test_no_relationships():
    workspace = make_workspace(numpy.zeros((10, 10), int), numpy.zeros((10, 10), int))
    workspace.module.run(workspace)
    m = workspace.measurements
    for parent, relationship in (
        (PRIMARY, cellprofiler.modules.identifytertiaryobjects.R_REMOVED),
        (SECONDARY, cellprofiler.modules.identifytertiaryobjects.R_PARENT),
    ):
        result = m.get_relationships(
            workspace.module.module_num, relationship, parent, TERTIARY
        )
        assert len(result) == 0


def test_relationships():
    primary = numpy.zeros((10, 30), int)
    secondary = numpy.zeros((10, 30), int)
    for i in range(3):
        center_j = 5 + i * 10
        primary[3:6, (center_j - 1) : (center_j + 2)] = i + 1
        secondary[2:7, (center_j - 2) : (center_j + 3)] = i + 1
    workspace = make_workspace(primary, secondary)
    workspace.module.run(workspace)
    m = workspace.measurements
    for parent, relationship in (
        (PRIMARY, cellprofiler.modules.identifytertiaryobjects.R_REMOVED),
        (SECONDARY, cellprofiler.modules.identifytertiaryobjects.R_PARENT),
    ):
        result = m.get_relationships(
            workspace.module.module_num, relationship, parent, TERTIARY
        )
        assert len(result) == 3
        for i in range(3):
            assert result[R_FIRST_IMAGE_NUMBER][i] == 1
            assert result[R_SECOND_IMAGE_NUMBER][i] == 1
            assert (
                result[R_FIRST_OBJECT_NUMBER][i] == i + 1
            )
            assert (
                result[R_SECOND_OBJECT_NUMBER][i] == i + 1
            )


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("identifytertiaryobjects/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.loadtxt(six.moves.StringIO(data))
    module = pipeline.modules()[0]

    assert module.secondary_objects_name.value == "IdentifySecondaryObjects"
    assert module.primary_objects_name.value == "IdentifyPrimaryObjects"
    assert module.subregion_objects_name.value == "IdentifyTertiaryObjects"
    assert module.shrink_primary.value
