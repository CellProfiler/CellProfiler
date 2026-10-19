import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.measurement
import cellprofiler_core.modules
from cellprofiler_core.constants.measurement import FF_COUNT, COLTYPE_INTEGER, M_LOCATION_CENTER_X, COLTYPE_FLOAT, \
    M_LOCATION_CENTER_Y, FF_PARENT, M_NUMBER_OBJECT_NUMBER, FF_CHILDREN_COUNT, C_COUNT, C_LOCATION, C_NUMBER, C_PARENT, \
    C_CHILDREN, FTR_CENTER_X, FTR_CENTER_Y, FTR_OBJECT_NUMBER


import cellprofiler.modules.maskobjects
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_OBJECTS = "inputobjects"
OUTPUT_OBJECTS = "outputobjects"
MASKING_OBJECTS = "maskingobjects"
MASKING_IMAGE = "maskingobjects"
OUTPUT_OUTLINES = "outputoutlines"


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("maskobjects/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 4

    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.maskobjects.MaskObjects)
    assert module.object_name.value == "Nuclei"
    assert module.mask_choice.value == cellprofiler.modules.maskobjects.MC_OBJECTS
    assert module.masking_objects.value == "Wells"
    assert module.remaining_objects.value == "MaskedNuclei"
    assert (
        module.retain_or_renumber.value == cellprofiler.modules.maskobjects.R_RENUMBER
    )
    assert module.overlap_choice.value == cellprofiler.modules.maskobjects.P_MASK
    assert not module.wants_inverted_mask

    module = pipeline.modules()[1]
    assert isinstance(module, cellprofiler.modules.maskobjects.MaskObjects)
    assert module.object_name.value == "Cells"
    assert module.mask_choice.value == cellprofiler.modules.maskobjects.MC_IMAGE
    assert module.masking_image.value == "WellBoundary"
    assert module.remaining_objects.value == "MaskedCells"
    assert module.retain_or_renumber.value == cellprofiler.modules.maskobjects.R_RETAIN
    assert module.overlap_choice.value == cellprofiler.modules.maskobjects.P_KEEP
    assert not module.wants_inverted_mask

    module = pipeline.modules()[2]
    assert isinstance(module, cellprofiler.modules.maskobjects.MaskObjects)
    assert module.object_name.value == "Cytoplasm"
    assert module.mask_choice.value == cellprofiler.modules.maskobjects.MC_OBJECTS
    assert module.masking_objects.value == "Cells"
    assert module.remaining_objects.value == "MaskedCytoplasm"
    assert (
        module.retain_or_renumber.value == cellprofiler.modules.maskobjects.R_RENUMBER
    )
    assert module.overlap_choice.value == cellprofiler.modules.maskobjects.P_REMOVE
    assert not module.wants_inverted_mask

    module = pipeline.modules()[3]
    assert isinstance(module, cellprofiler.modules.maskobjects.MaskObjects)
    assert module.object_name.value == "Speckles"
    assert module.mask_choice.value == cellprofiler.modules.maskobjects.MC_OBJECTS
    assert module.masking_objects.value == "Cells"
    assert module.remaining_objects.value == "MaskedSpeckles"
    assert (
        module.retain_or_renumber.value == cellprofiler.modules.maskobjects.R_RENUMBER
    )
    assert (
        module.overlap_choice.value
        == cellprofiler.modules.maskobjects.P_REMOVE_PERCENTAGE
    )
    assert round(abs(module.overlap_fraction.value - 0.3), 7) == 0
    assert not module.wants_inverted_mask


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("maskobjects/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1

    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.maskobjects.MaskObjects)
    assert module.object_name.value == "Nuclei"
    assert module.mask_choice.value == cellprofiler.modules.maskobjects.MC_OBJECTS
    assert module.masking_objects.value == "Wells"
    assert module.remaining_objects.value == "MaskedNuclei"
    assert (
        module.retain_or_renumber.value == cellprofiler.modules.maskobjects.R_RENUMBER
    )
    assert module.overlap_choice.value == cellprofiler.modules.maskobjects.P_MASK
    assert module.wants_inverted_mask


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("maskobjects/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.loadtxt(six.moves.StringIO(data))
    module = pipeline.modules()[0]

    assert module.object_name.value == "IdentifyPrimaryObjects"
    assert module.remaining_objects.value == "MaskObjects"
    assert module.mask_choice.value == "Objects"
    assert module.masking_objects.value == "FilterObjects"
    assert module.masking_image.value == "None"
    assert module.overlap_choice.value == "Keep overlapping region"
    assert module.overlap_fraction.value == 0.5
    assert module.retain_or_renumber.value == "Renumber"
    assert not module.wants_inverted_mask.value


def make_workspace(
    labels, overlap_choice, masking_objects=None, masking_image=None, renumber=True
):
    module = cellprofiler.modules.maskobjects.MaskObjects()
    module.set_module_num(1)
    module.object_name.value = INPUT_OBJECTS
    module.remaining_objects.value = OUTPUT_OBJECTS
    module.mask_choice.value = (
        cellprofiler.modules.maskobjects.MC_OBJECTS
        if masking_objects is not None
        else cellprofiler.modules.maskobjects.MC_IMAGE
    )
    module.masking_image.value = MASKING_IMAGE
    module.masking_objects.value = MASKING_OBJECTS
    module.retain_or_renumber.value = (
        cellprofiler.modules.maskobjects.R_RENUMBER
        if renumber
        else cellprofiler.modules.maskobjects.R_RETAIN
    )
    module.overlap_choice.value = overlap_choice

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)

    object_set = cellprofiler_core.object.ObjectSet()
    io = cellprofiler_core.object.Objects()
    io.segmented = labels
    object_set.add_objects(io, INPUT_OBJECTS)

    if masking_objects is not None:
        oo = cellprofiler_core.object.Objects()
        oo.segmented = masking_objects
        object_set.add_objects(oo, MASKING_OBJECTS)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    if masking_image is not None:
        mi = cellprofiler_core.image.Image(masking_image)
        image_set.add(MASKING_IMAGE, mi)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_measurement_columns():
    """Test get_measurement_columns"""
    workspace, module = make_workspace(
        numpy.zeros((20, 10), int),
        cellprofiler.modules.maskobjects.P_MASK,
        numpy.zeros((20, 10), int),
    )
    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(columns) == 6
    for expected in (
        (
            "Image",
            FF_COUNT % OUTPUT_OBJECTS,
            COLTYPE_INTEGER,
        ),
        (
            OUTPUT_OBJECTS,
            M_LOCATION_CENTER_X,
            COLTYPE_FLOAT,
        ),
        (
            OUTPUT_OBJECTS,
            M_LOCATION_CENTER_Y,
            COLTYPE_FLOAT,
        ),
        (
            OUTPUT_OBJECTS,
            FF_PARENT % INPUT_OBJECTS,
            COLTYPE_INTEGER,
        ),
        (
            OUTPUT_OBJECTS,
            M_NUMBER_OBJECT_NUMBER,
            COLTYPE_INTEGER,
        ),
        (
            INPUT_OBJECTS,
            FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            COLTYPE_INTEGER,
        ),
    ):
        assert any(
            [all([c in e for c, e in zip(column, expected)]) for column in columns]
        )


def test_measurement_categories():
    workspace, module = make_workspace(
        numpy.zeros((20, 10), int),
        cellprofiler.modules.maskobjects.MC_OBJECTS,
        numpy.zeros((20, 10), int),
    )
    categories = module.get_categories(workspace.pipeline, "Foo")
    assert len(categories) == 0

    categories = module.get_categories(
        workspace.pipeline, "Image"
    )
    assert len(categories) == 1
    assert categories[0] == C_COUNT

    categories = module.get_categories(workspace.pipeline, OUTPUT_OBJECTS)
    assert len(categories) == 3
    for category, expected in zip(
        sorted(categories),
        (
            C_LOCATION,
            C_NUMBER,
            C_PARENT,
        ),
    ):
        assert category == expected

    categories = module.get_categories(workspace.pipeline, INPUT_OBJECTS)
    assert len(categories) == 1
    assert categories[0] == C_CHILDREN


def test_measurements():
    workspace, module = make_workspace(
        numpy.zeros((20, 10), int),
        cellprofiler.modules.maskobjects.P_MASK,
        numpy.zeros((20, 10), int),
    )
    ftr_count = (
        FF_CHILDREN_COUNT % OUTPUT_OBJECTS
    ).split("_", 1)[1]
    d = {
        "Foo": {},
        "Image": {
            "Foo": [],
            C_COUNT: [OUTPUT_OBJECTS],
        },
        OUTPUT_OBJECTS: {
            "Foo": [],
            C_LOCATION: [
                FTR_CENTER_X,
                FTR_CENTER_Y,
            ],
            C_PARENT: [INPUT_OBJECTS],
            C_NUMBER: [
                FTR_OBJECT_NUMBER
            ],
        },
        INPUT_OBJECTS: {
            "Foo": [],
            C_CHILDREN: [ftr_count],
        },
    }
    for object_name in list(d.keys()):
        od = d[object_name]
        for category in list(od.keys()):
            features = module.get_measurements(
                workspace.pipeline, object_name, category
            )
            expected = od[category]
            assert len(features) == len(expected)
            for feature, e in zip(sorted(features), sorted(expected)):
                assert feature == e


def test_mask_nothing():
    workspace, module = make_workspace(
        numpy.zeros((20, 10), int),
        cellprofiler.modules.maskobjects.MC_OBJECTS,
        numpy.zeros((20, 10), int),
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    value = m.get_current_image_measurement(
        FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 0
    for object_name, feature in (
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_X),
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_Y),
        (OUTPUT_OBJECTS, M_NUMBER_OBJECT_NUMBER),
        (OUTPUT_OBJECTS, FF_PARENT % INPUT_OBJECTS),
        (
            INPUT_OBJECTS,
            FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == 0
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == 0)


def test_mask_with_objects():
    labels = numpy.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = numpy.zeros((20, 10), int)
    mask[3:17, 2:6] = 1
    expected = labels.copy()
    expected[mask == 0] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_MASK, mask
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)

    expected_x = numpy.array([4, 4])
    expected_y = numpy.array([5, 14])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    value = m.get_current_image_measurement(
        FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 2

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            M_NUMBER_OBJECT_NUMBER,
            numpy.array([1, 2]),
        ),
        (
            OUTPUT_OBJECTS,
            FF_PARENT % INPUT_OBJECTS,
            numpy.array([1, 2]),
        ),
        (
            INPUT_OBJECTS,
            FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            numpy.array([1, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            assert value == e


def test_mask_with_image():
    labels = numpy.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = numpy.zeros((20, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[~mask] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_MASK, masking_image=mask
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)

    expected_x = numpy.array([4, 4])
    expected_y = numpy.array([5, 14])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    value = m.get_current_image_measurement(
        FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 2

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            M_NUMBER_OBJECT_NUMBER,
            numpy.array([1, 2]),
        ),
        (
            OUTPUT_OBJECTS,
            FF_PARENT % INPUT_OBJECTS,
            numpy.array([1, 2]),
        ),
        (
            INPUT_OBJECTS,
            FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            numpy.array([1, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            assert value == e


def test_mask_renumber():
    labels = numpy.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 3
    labels[22:28, 3:7] = 2
    mask = numpy.zeros((30, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[~mask] = 0
    expected[expected == 3] = 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_MASK, masking_image=mask
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)

    expected_x = numpy.array([4, 4])
    expected_y = numpy.array([5, 14])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    value = m.get_current_image_measurement(
        FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 2

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            M_NUMBER_OBJECT_NUMBER,
            numpy.array([1, 2]),
        ),
        (
            OUTPUT_OBJECTS,
            FF_PARENT % INPUT_OBJECTS,
            numpy.array([1, 3]),
        ),
        (
            INPUT_OBJECTS,
            FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            numpy.array([1, 0, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            assert value == e


def test_mask_retain():
    labels = numpy.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 3
    labels[22:28, 3:7] = 2
    mask = numpy.zeros((30, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[~mask] = 0
    workspace, module = make_workspace(
        labels,
        cellprofiler.modules.maskobjects.P_MASK,
        masking_image=mask,
        renumber=False,
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)

    expected_x = numpy.array([4, None, 4])
    expected_y = numpy.array([5, None, 14])
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    value = m.get_current_image_measurement(
        FF_COUNT % OUTPUT_OBJECTS
    )
    assert value == 3

    for object_name, feature, expected in (
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_X, expected_x),
        (OUTPUT_OBJECTS, M_LOCATION_CENTER_Y, expected_y),
        (
            OUTPUT_OBJECTS,
            M_NUMBER_OBJECT_NUMBER,
            numpy.array([1, 2, 3]),
        ),
        (
            OUTPUT_OBJECTS,
            FF_PARENT % INPUT_OBJECTS,
            numpy.array([1, 2, 3]),
        ),
        (
            INPUT_OBJECTS,
            FF_CHILDREN_COUNT % OUTPUT_OBJECTS,
            numpy.array([1, 0, 1]),
        ),
    ):
        data = m.get_current_measurement(object_name, feature)
        assert len(data) == len(expected)
        for value, e in zip(data, expected):
            if e is None:
                assert numpy.isnan(value)
            else:
                assert value == e


def test_mask_invert():
    labels = numpy.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    #
    # Make a mask that covers only object # 1 and that is missing
    # one pixel of that object. Invert it in anticipation of invert op
    #
    mask = labels == 1
    mask[2, 3] = False
    mask = ~mask
    expected = labels
    expected[labels != 1] = 0
    expected[2, 3] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_MASK, masking_image=mask
    )
    assert isinstance(module, cellprofiler.modules.maskobjects.MaskObjects)
    module.wants_inverted_mask.value = True
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)


def test_keep():
    labels = numpy.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 3
    labels[22:28, 3:7] = 2
    mask = numpy.zeros((30, 10), bool)
    mask[3:17, 2:6] = True
    expected = labels.copy()
    expected[expected == 2] = 0
    expected[expected == 3] = 2
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_KEEP, masking_image=mask
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)


def test_remove():
    labels = numpy.zeros((20, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = numpy.zeros((20, 10), bool)
    mask[2:17, 2:7] = True
    expected = labels.copy()
    expected[labels == 2] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_REMOVE, masking_image=mask
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)


def test_remove_percent():
    labels = numpy.zeros((20, 10), int)
    labels[2:8, 3:6] = 1
    labels[12:18, 3:7] = 2
    mask = numpy.zeros((20, 10), bool)
    mask[3:17, 2:6] = True
    # loses 3 of 18 from object 1 = .1666
    # loses 9 of 24 from object 2 = .375
    expected = labels.copy()
    expected[labels == 2] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_REMOVE_PERCENTAGE, masking_image=mask
    )
    module.overlap_fraction.value = 0.75
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)


def test_different_object_sizes():
    labels = numpy.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = numpy.zeros((20, 20), int)
    mask[3:17, 2:6] = 1
    expected = labels.copy()
    expected[:20, :][mask[:, :10] == 0] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_MASK, mask
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)


def test_different_image_sizes():
    labels = numpy.zeros((30, 10), int)
    labels[2:8, 3:7] = 1
    labels[12:18, 3:7] = 2
    mask = numpy.zeros((20, 20), bool)
    mask[3:17, 2:6] = 1
    expected = labels.copy()
    expected[:20, :][mask[:, :10] == 0] = 0
    workspace, module = make_workspace(
        labels, cellprofiler.modules.maskobjects.P_MASK, masking_image=mask
    )
    module.run(workspace)
    objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
    assert numpy.all(objects.segmented == expected)
