import six.moves

import cellprofiler.measurement
import cellprofiler.modules.editobjectsmanually
import cellprofiler.pipeline

INPUT_OBJECTS_NAME = "inputobjects"
OUTPUT_OBJECTS_NAME = "outputobjects"


def test_load_v1():
    with open("./tests/resources/modules/editobjectsmanually/v1.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.editobjectsmanually.EditObjectsManually
    )
    assert module.object_name == "Nuclei"
    assert module.filtered_objects == "EditedNuclei"
    assert module.renumber_choice == cellprofiler.modules.editobjectsmanually.R_RENUMBER
    assert not module.wants_image_display


def test_load_v2():
    with open("./tests/resources/modules/editobjectsmanually/v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.editobjectsmanually.EditObjectsManually
    )
    assert module.object_name == "Nuclei"
    assert module.filtered_objects == "EditedNuclei"
    assert module.renumber_choice == cellprofiler.modules.editobjectsmanually.R_RETAIN
    assert module.wants_image_display
    assert module.image_name == "DNA"
    assert not module.allow_overlap


def test_load_v3():
    with open("./tests/resources/modules/editobjectsmanually/v3.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.editobjectsmanually.EditObjectsManually
    )
    assert module.object_name == "Nuclei"
    assert module.filtered_objects == "EditedNuclei"
    assert module.renumber_choice == cellprofiler.modules.editobjectsmanually.R_RETAIN
    assert module.wants_image_display
    assert module.image_name == "DNA"
    assert module.allow_overlap


def test_load_v4():
    with open("./tests/resources/modules/editobjectsmanually/v4.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.loadtxt(six.moves.StringIO(data))
    module = pipeline.modules()[0]

    assert module.object_name.value == "IdentifyPrimaryObjects"
    assert module.filtered_objects.value == "EditedObjects"
    assert module.renumber_choice.value == "Renumber"
    assert module.wants_image_display.value
    assert module.image_name.value == "DNA"
    assert not module.allow_overlap.value


def test_measurements():
    module = cellprofiler.modules.editobjectsmanually.EditObjectsManually()
    module.object_name.value = INPUT_OBJECTS_NAME
    module.filtered_objects.value = OUTPUT_OBJECTS_NAME

    columns = module.get_measurement_columns(None)
    expected_columns = [
        (
            cellprofiler.measurement.IMAGE,
            cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS_NAME,
            cellprofiler.measurement.COLTYPE_INTEGER,
        ),
        (
            OUTPUT_OBJECTS_NAME,
            cellprofiler.measurement.M_NUMBER_OBJECT_NUMBER,
            cellprofiler.measurement.COLTYPE_INTEGER,
        ),
        (
            OUTPUT_OBJECTS_NAME,
            cellprofiler.measurement.M_LOCATION_CENTER_X,
            cellprofiler.measurement.COLTYPE_FLOAT,
        ),
        (
            OUTPUT_OBJECTS_NAME,
            cellprofiler.measurement.M_LOCATION_CENTER_Y,
            cellprofiler.measurement.COLTYPE_FLOAT,
        ),
        (
            OUTPUT_OBJECTS_NAME,
            cellprofiler.measurement.FF_PARENT % INPUT_OBJECTS_NAME,
            cellprofiler.measurement.COLTYPE_INTEGER,
        ),
        (
            INPUT_OBJECTS_NAME,
            cellprofiler.measurement.FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME,
            cellprofiler.measurement.COLTYPE_INTEGER,
        ),
    ]

    for column in columns:
        assert any(
            [
                all([column[i] == expected[i] for i in range(3)])
                for expected in expected_columns
            ]
        ), ("Unexpected column: %s, %s, %s" % column)
        # Make sure no duplicates
        assert (
            len(["x" for c in columns if all([column[i] == c[i] for i in range(3)])])
            == 1
        )
    for expected in expected_columns:
        assert any(
            [all([column[i] == expected[i] for i in range(3)]) for column in columns]
        ), ("Missing column: %s, %s, %s" % expected)

    #
    # Check the measurement features
    #
    d = {
        cellprofiler.measurement.IMAGE: {
            cellprofiler.measurement.C_COUNT: [OUTPUT_OBJECTS_NAME],
            "Foo": [],
        },
        INPUT_OBJECTS_NAME: {
            cellprofiler.measurement.C_CHILDREN: ["%s_Count" % OUTPUT_OBJECTS_NAME],
            "Foo": [],
        },
        OUTPUT_OBJECTS_NAME: {
            cellprofiler.measurement.C_LOCATION: [
                cellprofiler.measurement.FTR_CENTER_X,
                cellprofiler.measurement.FTR_CENTER_Y,
            ],
            cellprofiler.measurement.C_PARENT: [INPUT_OBJECTS_NAME],
            cellprofiler.measurement.C_NUMBER: [
                cellprofiler.measurement.FTR_OBJECT_NUMBER
            ],
            "Foo": [],
        },
        "Foo": {},
    }

    for object_name, category_d in list(d.items()):
        #
        # Check get_categories for the object
        #
        categories = module.get_categories(None, object_name)
        assert len(categories) == len(
            [k for k in list(category_d.keys()) if k != "Foo"]
        )
        for category in categories:
            assert category in category_d
        for category in list(category_d.keys()):
            if category != "Foo":
                assert category in categories

        for category, expected_features in list(category_d.items()):
            #
            # check get_measurements for each category
            #
            features = module.get_measurements(None, object_name, category)
            assert len(features) == len(expected_features)
            for feature in features:
                assert feature in expected_features, "Unexpected feature: %s" % feature
            for feature in expected_features:
                assert feature in features, "Missing feature: %s" % feature
