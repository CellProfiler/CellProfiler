import contextlib
import os
import pickle
import tempfile

import numpy
import pytest

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT, COLTYPE_INTEGER


import cellprofiler.modules.classifyobjects
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace

OBJECTS_NAME = "myobjects"
MEASUREMENT_NAME_1 = "Measurement1"
MEASUREMENT_NAME_2 = "Measurement2"
IMAGE_NAME = "image"
TEST_FTR = "my_measurement"
MISSPELLED_FTR = "m_measurement"


def make_workspace(labels, contrast_choice, measurement1=None, measurement2=None):
    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, OBJECTS_NAME)

    measurements = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.classifyobjects.ClassifyObjects()
    m_names = []
    if measurement1 is not None:
        measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_1, measurement1)
        m_names.append(MEASUREMENT_NAME_1)
    if measurement2 is not None:
        measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_2, measurement2)
        module.add_single_measurement()
        m_names.append(MEASUREMENT_NAME_2)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)

    module.contrast_choice.value = contrast_choice
    if (
        module.contrast_choice
        == cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT
    ):
        for i, m in enumerate(m_names):
            group = module.single_measurements[i]
            group.object_name.value = OBJECTS_NAME
            group.measurement.value = m
            group.image_name.value = IMAGE_NAME
    else:
        module.object_name.value = OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.first_measurement.value = MEASUREMENT_NAME_1
        module.second_measurement.value = MEASUREMENT_NAME_2
    module.set_module_num(1)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, image_set, object_set, measurements, image_set_list
    )
    return workspace, module

def make_workspace_classifier(object_dict={}, image_dict={},grouplist = []):
    """Make a workspace for testing FilterByObjectMeasurement"""
    module = cellprofiler.modules.classifyobjects.ClassifyObjects()
    module.contrast_choice.value = cellprofiler.modules.classifyobjects.BY_MODEL
    module.object_name.value = OBJECTS_NAME
    module.create_class_sets.value = True
    for group in range(len(grouplist)):
        if group >= len(module.desired_classes):
            settings_group = cellprofiler_core.setting.SettingsGroup()
            settings_group.can_delete = False
            settings_group.append("class_name",cellprofiler_core.setting.choice.Choice('text',grouplist))
            settings_group.append("class_objects_name", cellprofiler_core.setting.text.LabelName('text','dummy'))
            module.desired_classes.append(settings_group)
        module.desired_classes[group].class_name.value = str(grouplist[group])
        module.desired_classes[group].class_objects_name.value = str(grouplist[group])
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
    feature_names=[f"{OBJECTS_NAME}_{TEST_FTR}"],
):
    """Returns the filename of the classifier pickle"""
    assert isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects)
    if classes is None:
        if class_names != None:
            classes = numpy.arange(1, len(class_names) + 1)
        else:
            classes = numpy.arange(1, numpy.max(answers) + 1)
    if class_names is None:
        class_names = [str(_) for _ in classes]
    if rules_class is None:
        rules_class = class_names[0]
    s = make_classifier_pickle(answers, classes, class_names, name, feature_names)
    fd, filename = tempfile.mkstemp(".model")
    os.write(fd, s)
    os.close(fd)

    module.contrast_choice.value = cellprofiler.modules.classifyobjects.BY_MODEL
    module.model_directory.set_custom_path(os.path.dirname(filename))
    module.model_file_name.value = os.path.split(filename)[1]
    yield
    try:
        os.remove(filename)
    except:
        pass

def test_classify_single_none():
    """Make sure the single measurement mode can handle no objects"""
    workspace, module = make_workspace(
        numpy.zeros((10, 10), int),
        cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT,
        numpy.zeros((0,), float),
    )
    module.run(workspace)
    for m_name in (
        "Classify_Measurement1_Bin_1",
        "Classify_Measurement1_Bin_2",
        "Classify_Measurement1_Bin_3",
    ):
        m = workspace.measurements.get_current_measurement(OBJECTS_NAME, m_name)
        assert len(m) == 0


def test_classify_single_even():
    m = numpy.array((0.5, 0, 1, 0.1))
    labels = numpy.zeros((20, 10), int)
    labels[2:5, 3:7] = 1
    labels[12:15, 1:4] = 2
    labels[6:11, 5:9] = 3
    labels[16:19, 5:9] = 4
    workspace, module = make_workspace(
        labels, cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT, m
    )
    module.single_measurements[
        0
    ].bin_choice.value = cellprofiler.modules.classifyobjects.BC_EVEN
    module.single_measurements[0].low_threshold.value = 0.2
    module.single_measurements[0].high_threshold.value = 0.7
    module.single_measurements[0].bin_count.value = 1
    module.single_measurements[0].wants_low_bin.value = True
    module.single_measurements[0].wants_high_bin.value = True
    module.single_measurements[0].wants_images.value = True

    expected_obj = dict(
        Classify_Measurement1_Bin_1=(0, 1, 0, 1),
        Classify_Measurement1_Bin_2=(1, 0, 0, 0),
        Classify_Measurement1_Bin_3=(0, 0, 1, 0),
    )
    expected_img = dict(
        Classify_Measurement1_Bin_1_NumObjectsPerBin=2,
        Classify_Measurement1_Bin_2_NumObjectsPerBin=1,
        Classify_Measurement1_Bin_3_NumObjectsPerBin=1,
        Classify_Measurement1_Bin_1_PctObjectsPerBin=50.0,
        Classify_Measurement1_Bin_2_PctObjectsPerBin=25.0,
        Classify_Measurement1_Bin_3_PctObjectsPerBin=25.0,
    )
    module.run(workspace)
    for measurement, expected_values in list(expected_obj.items()):
        values = workspace.measurements.get_current_measurement(
            OBJECTS_NAME, measurement
        )
        assert len(values) == 4
        assert numpy.all(values == numpy.array(expected_values))
    for measurement, expected_values in list(expected_img.items()):
        values = workspace.measurements.get_current_measurement(
            "Image", measurement
        )
        assert values == expected_values

    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert numpy.all(pixel_data[labels == 0, :] == 0)
    colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
    for i, color in enumerate(colors + [colors[1]]):
        assert numpy.all(pixel_data[labels == i + 1, :] == color)

    columns = module.get_measurement_columns(None)
    assert len(columns) == 9
    assert len(set([column[1] for column in columns])) == 9  # no duplicates
    for column in columns:
        if column[0] != OBJECTS_NAME:  # Must be image
            assert column[0] == "Image"
            assert column[1] in list(expected_img.keys())
            assert (
                column[2] == COLTYPE_INTEGER
                if column[1].endswith(
                    cellprofiler.modules.classifyobjects.F_NUM_PER_BIN
                )
                else COLTYPE_FLOAT
            )
        else:
            assert column[0] == OBJECTS_NAME
            assert column[1] in list(expected_obj.keys())
            assert column[2] == COLTYPE_INTEGER

    categories = module.get_categories(None, "Image")
    assert len(categories) == 1
    assert categories[0] == cellprofiler.modules.classifyobjects.M_CATEGORY
    names = module.get_measurements(None, "Image", "foo")
    assert len(names) == 0
    categories = module.get_categories(None, OBJECTS_NAME)
    assert len(categories) == 1
    assert categories[0] == cellprofiler.modules.classifyobjects.M_CATEGORY
    names = module.get_measurements(None, OBJECTS_NAME, "foo")
    assert len(names) == 0
    names = module.get_measurements(
        None, "foo", cellprofiler.modules.classifyobjects.M_CATEGORY
    )
    assert len(names) == 0
    names = module.get_measurements(
        None, OBJECTS_NAME, cellprofiler.modules.classifyobjects.M_CATEGORY
    )
    assert len(names) == 3
    assert len(set(names)) == 3
    assert all(
        [
            "_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
            in list(expected_obj.keys())
            for name in names
        ]
    )
    names = module.get_measurements(
        None,
        "Image",
        cellprofiler.modules.classifyobjects.M_CATEGORY,
    )
    assert len(names) == 6
    assert len(set(names)) == 6
    assert all(
        [
            "_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
            in list(expected_img.keys())
            for name in names
        ]
    )


def test_classify_single_custom():
    m = numpy.array((0.5, 0, 1, 0.1))
    labels = numpy.zeros((20, 10), int)
    labels[2:5, 3:7] = 1
    labels[12:15, 1:4] = 2
    labels[6:11, 5:9] = 3
    labels[16:19, 5:9] = 4
    workspace, module = make_workspace(
        labels, cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT, m
    )
    module.single_measurements[
        0
    ].bin_choice.value = cellprofiler.modules.classifyobjects.BC_CUSTOM
    module.single_measurements[0].custom_thresholds.value = ".2,.7"
    module.single_measurements[0].bin_count.value = 14  # should ignore
    module.single_measurements[0].wants_custom_names.value = True
    module.single_measurements[0].wants_low_bin.value = True
    module.single_measurements[0].wants_high_bin.value = True
    module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
    module.single_measurements[0].wants_images.value = True

    expected_img = dict(
        Classify_Three_NumObjectsPerBin=2,
        Classify_Three_PctObjectsPerBin=50.0,
        Classify_Blind_NumObjectsPerBin=1,
        Classify_Blind_PctObjectsPerBin=25.0,
        Classify_Mice_NumObjectsPerBin=1,
        Classify_Mice_PctObjectsPerBin=25.0,
    )
    expected_obj = dict(
        Classify_Three=(0, 1, 0, 1),
        Classify_Blind=(1, 0, 0, 0),
        Classify_Mice=(0, 0, 1, 0),
    )
    module.run(workspace)
    for measurement, expected_values in list(expected_obj.items()):
        values = workspace.measurements.get_current_measurement(
            OBJECTS_NAME, measurement
        )
        assert len(values) == 4
        assert numpy.all(values == numpy.array(expected_values))
    for measurement, expected_values in list(expected_img.items()):
        values = workspace.measurements.get_current_measurement(
            "Image", measurement
        )
        assert values == expected_values
    image = workspace.image_set.get_image(IMAGE_NAME)
    pixel_data = image.pixel_data
    assert numpy.all(pixel_data[labels == 0, :] == 0)
    colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
    for i, color in enumerate(colors + [colors[1]]):
        assert numpy.all(pixel_data[labels == i + 1, :] == color)

    columns = module.get_measurement_columns(None)
    assert len(columns) == 9
    assert len(set([column[1] for column in columns])) == 9  # no duplicates
    for column in columns:
        if column[0] != OBJECTS_NAME:  # Must be image
            assert column[0] == "Image"
            assert column[1] in list(expected_img.keys())
            assert (
                column[2] == COLTYPE_INTEGER
                if column[1].endswith(
                    cellprofiler.modules.classifyobjects.F_NUM_PER_BIN
                )
                else COLTYPE_FLOAT
            )
        else:
            assert column[0] == OBJECTS_NAME
            assert column[1] in list(expected_obj.keys())
            assert column[2] == COLTYPE_INTEGER

    categories = module.get_categories(None, "Image")
    assert len(categories) == 1
    categories = module.get_categories(None, OBJECTS_NAME)
    assert len(categories) == 1
    assert categories[0] == cellprofiler.modules.classifyobjects.M_CATEGORY
    names = module.get_measurements(None, OBJECTS_NAME, "foo")
    assert len(names) == 0
    names = module.get_measurements(
        None, "foo", cellprofiler.modules.classifyobjects.M_CATEGORY
    )
    assert len(names) == 0
    names = module.get_measurements(
        None, OBJECTS_NAME, cellprofiler.modules.classifyobjects.M_CATEGORY
    )
    assert len(names) == 3
    assert len(set(names)) == 3
    assert all(
        [
            "_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
            in list(expected_obj.keys())
            for name in names
        ]
    )
    names = module.get_measurements(
        None,
        "Image",
        cellprofiler.modules.classifyobjects.M_CATEGORY,
    )
    assert len(names) == 6
    assert len(set(names)) == 6
    assert all(
        [
            "_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
            in list(expected_img.keys())
            for name in names
        ]
    )


def test_last_is_nan():
    # regression test for issue #1553
    #
    # Test that classify objects classifies an object whose measurement
    # is NaN as none of the categories. Test for no exception thrown
    # if showing the figure and last object has a measurement of NaN
    #
    for leave_last_out in (False, True):
        m = numpy.array((0.5, 0, 1, numpy.NaN))
        if leave_last_out:
            m = m[:-1]
        labels = numpy.zeros((20, 10), int)
        labels[2:5, 3:7] = 1
        labels[12:15, 1:4] = 2
        labels[6:11, 5:9] = 3
        labels[16:19, 5:9] = 4
        workspace, module = make_workspace(
            labels, cellprofiler.modules.classifyobjects.BY_SINGLE_MEASUREMENT, m
        )
        module.single_measurements[
            0
        ].bin_choice.value = cellprofiler.modules.classifyobjects.BC_CUSTOM
        module.single_measurements[0].custom_thresholds.value = ".2,.7"
        module.single_measurements[0].bin_count.value = 14  # should ignore
        module.single_measurements[0].wants_custom_names.value = True
        module.single_measurements[0].wants_low_bin.value = True
        module.single_measurements[0].wants_high_bin.value = True
        module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
        module.single_measurements[0].wants_images.value = True

        expected_img = dict(
            Classify_Three_NumObjectsPerBin=1,
            Classify_Three_PctObjectsPerBin=25.0,
            Classify_Blind_NumObjectsPerBin=1,
            Classify_Blind_PctObjectsPerBin=25.0,
            Classify_Mice_NumObjectsPerBin=1,
            Classify_Mice_PctObjectsPerBin=25.0,
        )
        expected_obj = dict(
            Classify_Three=(0, 1, 0, 0),
            Classify_Blind=(1, 0, 0, 0),
            Classify_Mice=(0, 0, 1, 0),
        )
        module.run(workspace)
        for measurement, expected_values in list(expected_obj.items()):
            values = workspace.measurements.get_current_measurement(
                OBJECTS_NAME, measurement
            )
            assert len(values) == 4
            assert numpy.all(values == numpy.array(expected_values))
        for measurement, expected_values in list(expected_img.items()):
            values = workspace.measurements.get_current_measurement(
                "Image", measurement
            )
            assert values == expected_values
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        assert numpy.all(pixel_data[labels == 0, :] == 0)
        colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5), (16, 5))]
        for i, color in enumerate(colors + [colors[1]]):
            assert numpy.all(pixel_data[labels == i + 1, :] == color)


def test_two_none():
    workspace, module = make_workspace(
        numpy.zeros((10, 10), int),
        cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS,
        numpy.zeros((0,), float),
        numpy.zeros((0,), float),
    )
    module.run(workspace)
    for lh1 in ("low", "high"):
        for lh2 in ("low", "high"):
            m_name = "Classify_Measurement1_%s_Measurement2_%s" % (lh1, lh2)
            m = workspace.measurements.get_current_measurement(OBJECTS_NAME, m_name)
            assert len(m) == 0


def test_two():
    numpy.random.seed(0)
    labels = numpy.zeros((10, 20), int)
    index = 1
    for i_min, i_max in ((1, 4), (6, 9)):
        for j_min, j_max in ((2, 6), (8, 11), (13, 18)):
            labels[i_min:i_max, j_min:j_max] = index
            index += 1
    num_labels = index - 1
    exps = numpy.exp(numpy.arange(numpy.max(labels)))
    m1 = numpy.random.permutation(exps)
    m2 = numpy.random.permutation(exps)
    for wants_custom_names in (False, True):
        for tm1 in (
            cellprofiler.modules.classifyobjects.TM_MEAN,
            cellprofiler.modules.classifyobjects.TM_MEDIAN,
            cellprofiler.modules.classifyobjects.TM_CUSTOM,
        ):
            for tm2 in (
                cellprofiler.modules.classifyobjects.TM_MEAN,
                cellprofiler.modules.classifyobjects.TM_MEDIAN,
                cellprofiler.modules.classifyobjects.TM_CUSTOM,
            ):
                workspace, module = make_workspace(
                    labels,
                    cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS,
                    m1,
                    m2,
                )
                assert isinstance(
                    module, cellprofiler.modules.classifyobjects.ClassifyObjects
                )
                module.first_threshold_method.value = tm1
                module.first_threshold.value = 8
                module.second_threshold_method.value = tm2
                module.second_threshold.value = 70
                module.wants_image.value = True

                def cutoff(method, custom_cutoff):
                    if method == cellprofiler.modules.classifyobjects.TM_MEAN:
                        return numpy.mean(exps)
                    elif method == cellprofiler.modules.classifyobjects.TM_MEDIAN:
                        return numpy.median(exps)
                    else:
                        return custom_cutoff

                c1 = cutoff(tm1, module.first_threshold.value)
                c2 = cutoff(tm2, module.second_threshold.value)
                m1_over = m1 >= c1
                m2_over = m2 >= c2
                if wants_custom_names:
                    f_names = ("TL", "TR", "BL", "BR")
                    module.wants_custom_names.value = True
                    module.low_low_custom_name.value = f_names[0]
                    module.low_high_custom_name.value = f_names[1]
                    module.high_low_custom_name.value = f_names[2]
                    module.high_high_custom_name.value = f_names[3]
                else:
                    f_names = (
                        "Measurement1_low_Measurement2_low",
                        "Measurement1_low_Measurement2_high",
                        "Measurement1_high_Measurement2_low",
                        "Measurement1_high_Measurement2_high",
                    )
                m_names = [
                    "_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
                    for name in f_names
                ]

                module.run(workspace)
                columns = module.get_measurement_columns(None)
                for column in columns:
                    if column[0] != OBJECTS_NAME:  # Must be image
                        assert column[0] == "Image"
                        assert (
                            column[2] == COLTYPE_INTEGER
                            if column[1].endswith(
                                cellprofiler.modules.classifyobjects.F_NUM_PER_BIN
                            )
                            else COLTYPE_FLOAT
                        )
                    else:
                        assert column[0] == OBJECTS_NAME
                        assert (
                            column[2] == COLTYPE_INTEGER
                        )

                assert len(columns) == 12
                assert (
                    len(set([column[1] for column in columns])) == 12
                )  # no duplicates

                categories = module.get_categories(
                    None, "Image"
                )
                assert len(categories) == 1
                categories = module.get_categories(None, OBJECTS_NAME)
                assert len(categories) == 1
                assert categories[0] == cellprofiler.modules.classifyobjects.M_CATEGORY
                names = module.get_measurements(None, OBJECTS_NAME, "foo")
                assert len(names) == 0
                names = module.get_measurements(
                    None, "foo", cellprofiler.modules.classifyobjects.M_CATEGORY
                )
                assert len(names) == 0
                names = module.get_measurements(
                    None, OBJECTS_NAME, cellprofiler.modules.classifyobjects.M_CATEGORY
                )
                assert len(names) == 4

                for m_name, expected in zip(
                    m_names,
                    (
                        (~m1_over) & (~m2_over),
                        (~m1_over) & m2_over,
                        m1_over & ~m2_over,
                        m1_over & m2_over,
                    ),
                ):
                    m = workspace.measurements.get_current_measurement(
                        "Image",
                        "_".join(
                            (m_name, cellprofiler.modules.classifyobjects.F_NUM_PER_BIN)
                        ),
                    )
                    assert m == expected.astype(int).sum()
                    m = workspace.measurements.get_current_measurement(
                        "Image",
                        "_".join(
                            (m_name, cellprofiler.modules.classifyobjects.F_PCT_PER_BIN)
                        ),
                    )
                    assert m == 100.0 * float(expected.astype(int).sum()) / num_labels
                    m = workspace.measurements.get_current_measurement(
                        OBJECTS_NAME, m_name
                    )
                    assert numpy.all(m == expected.astype(int))
                    assert m_name in [column[1] for column in columns]
                    assert m_name in [
                        "_".join(
                            (cellprofiler.modules.classifyobjects.M_CATEGORY, name)
                        )
                        for name in names
                    ]
                image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
                assert numpy.all(image[labels == 0, :] == 0)
                colors = image[(labels > 0) & (m[labels - 1] == 1), :]
                if colors.shape[0] > 0:
                    assert all(
                        [numpy.all(colors[:, i] == colors[0, i]) for i in range(3)]
                    )


def test_nans():
    # Test for NaN values in two measurements.
    #
    labels = numpy.zeros((10, 15), int)
    labels[3:5, 3:5] = 1
    labels[6:8, 3:5] = 3
    labels[3:5, 6:8] = 4
    labels[6:8, 6:8] = 5
    labels[3:5, 10:12] = 2

    m1 = numpy.array((1, 2, numpy.NaN, 1, numpy.NaN))
    m2 = numpy.array((1, 2, 1, numpy.NaN, numpy.NaN))
    for leave_last_out in (False, True):
        end = numpy.max(labels) - 1 if leave_last_out else numpy.max(labels)
        workspace, module = make_workspace(
            labels,
            cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS,
            m1[:end],
            m2[:end],
        )
        assert isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects)
        module.first_threshold_method.value = (
            cellprofiler.modules.classifyobjects.TM_MEAN
        )
        module.first_threshold.value = 2
        module.second_threshold_method.value = (
            cellprofiler.modules.classifyobjects.TM_MEAN
        )
        module.second_threshold.value = 2
        module.wants_image.value = True
        module.wants_custom_names.value = False
        module.run(workspace)
        f_names = (
            "Measurement1_low_Measurement2_low",
            "Measurement1_low_Measurement2_high",
            "Measurement1_high_Measurement2_low",
            "Measurement1_high_Measurement2_high",
        )
        m_names = [
            "_".join((cellprofiler.modules.classifyobjects.M_CATEGORY, name))
            for name in f_names
        ]
        m = workspace.measurements
        for m_name, expected in zip(
            m_names,
            [
                numpy.array((1, 0, 0, 0, 0)),
                numpy.array((0, 0, 0, 0, 0)),
                numpy.array((0, 0, 0, 0, 0)),
                numpy.array((0, 1, 0, 0, 0)),
            ],
        ):
            values = m[OBJECTS_NAME, m_name]
            numpy.testing.assert_array_equal(values, expected)


def test_nan_offset_by_1():
    # Regression test of 1636
    labels = numpy.zeros((10, 15), int)
    labels[3:5, 3:5] = 1
    labels[6:8, 3:5] = 2

    m1 = numpy.array((4, numpy.NaN))
    m2 = numpy.array((4, 4))
    workspace, module = make_workspace(
        labels, cellprofiler.modules.classifyobjects.BY_TWO_MEASUREMENTS, m1, m2
    )
    assert isinstance(module, cellprofiler.modules.classifyobjects.ClassifyObjects)
    module.first_threshold_method.value = cellprofiler.modules.classifyobjects.TM_MEAN
    module.first_threshold.value = 2
    module.second_threshold_method.value = cellprofiler.modules.classifyobjects.TM_MEAN
    module.second_threshold.value = 2
    module.wants_image.value = True
    module.wants_custom_names.value = False
    module.run(workspace)
    image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
    colors = module.get_colors(4)
    reverse = numpy.zeros(image.shape[:2], int)
    for idx, color in enumerate(colors):
        reverse[numpy.all(image == color[numpy.newaxis, numpy.newaxis, :3], 2)] = idx
    assert numpy.all(reverse[labels == 1] == 4)

def test_classify_none():
    classes=['a']
    workspace, module = make_workspace_classifier(
        {OBJECTS_NAME: numpy.zeros((10, 10), int)},
        grouplist = classes)
    with make_classifier(module, numpy.zeros(0, int), class_names=classes):
        workspace.measurements[OBJECTS_NAME, TEST_FTR] = numpy.zeros(0)
        module.run(workspace)
        for eachclass in classes:
            output_objects = workspace.object_set.get_objects(eachclass)
            assert output_objects.count == 0


def test_classify_one():
    classes=['a','b']
    labels = numpy.zeros((10, 10), int)
    labels[4:7, 4:7] = 1
    workspace, module = make_workspace_classifier(
        {OBJECTS_NAME: labels},
        grouplist = classes)
    with make_classifier(module, [1], class_names=classes):
        workspace.measurements[OBJECTS_NAME, TEST_FTR] = numpy.zeros(1)
        module.run(workspace)
        for eachclass in classes:
            output_objects = workspace.object_set.get_objects(eachclass)
            output_object_count = output_objects.count
            if eachclass == 'a':
                assert output_object_count == 1
            else:
                assert output_object_count == 0

def test_classify_many():
    classes = ['a','b']
    labels = numpy.zeros((10, 10), int)
    labels[1:4, 1:4] = 1
    labels[5:7, 5:7] = 2
    workspace, module = make_workspace_classifier(
        {OBJECTS_NAME: labels},
        grouplist = classes)
    with make_classifier(module, numpy.array([1, 2]), class_names=classes):
        workspace.measurements[OBJECTS_NAME, TEST_FTR] = numpy.zeros(2)
        module.run(workspace)
        for eachclass in classes:
            output_objects = workspace.object_set.get_objects(eachclass)
            output_objects = workspace.object_set.get_objects(eachclass)
            assert output_objects.count == 1
            labels_out = output_objects.get_labels()[0][0]
            if eachclass == 'a':
                numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 1)
                numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 0)
            else:
                numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 0)
                numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 1)     

def test_classify_many_fuzzy():
    classes = ['a','b']
    labels = numpy.zeros((10, 10), int)
    labels[1:4, 1:4] = 1
    labels[5:7, 5:7] = 2
    workspace, module = make_workspace_classifier(
        {OBJECTS_NAME: labels},
        grouplist = classes)
    with make_classifier(module, numpy.array([1, 2]), class_names=classes):
        workspace.measurements[OBJECTS_NAME, MISSPELLED_FTR] = numpy.zeros(2)
        module.allow_fuzzy.value = True
        module.run(workspace)
        for eachclass in classes:
            output_objects = workspace.object_set.get_objects(eachclass)
            output_objects = workspace.object_set.get_objects(eachclass)
            assert output_objects.count == 1
            labels_out = output_objects.get_labels()[0][0]
            if eachclass == 'a':
                numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 1)
                numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 0)
            else:
                numpy.testing.assert_array_equal(labels_out[1:4, 1:4], 0)
                numpy.testing.assert_array_equal(labels_out[5:7, 5:7], 1)
        module.allow_fuzzy.value = False 
        with pytest.raises(AssertionError):
            module.run(workspace)    
                

class FakeClassifier(object):
    def __init__(self, answers, classes, proba):
        """initializer

        answers - a vector of answers to be returned by "predict"

        classes - a vector of class numbers to be used to populate classes_
        """
        self.answers_ = answers
        self.classes_ = classes
        self.proba_ = proba

    def predict(self, *args, **kwargs):
        return self.answers_
    def predict_proba(self, *args):
        return(self.proba_)


def make_classifier_pickle(answers, classes, class_names, name, feature_names):
    """Make a pickle of a fake classifier

    answers - the answers you want to get back after calling classifier.predict
    classes - the class #s for the answers.
    class_names - one name per class in the order they appear in classes
    name - the name of the classifier
    feature_names - the names of the features fed into the classifier
    """
    proba = numpy.zeros([len(answers),len(classes)])
    classifier = FakeClassifier(answers, classes, proba)
    return pickle.dumps([classifier, class_names, name, feature_names])