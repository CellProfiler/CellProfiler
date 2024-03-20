import io

import centrosome.threshold
import numpy
from cellprofiler_core.constants.measurement import FF_COUNT, COLTYPE_FLOAT, COLTYPE_INTEGER, R_FIRST_IMAGE_NUMBER, \
    R_SECOND_IMAGE_NUMBER, R_FIRST_OBJECT_NUMBER, R_SECOND_OBJECT_NUMBER
from cellprofiler_core.constants.module._identify import TS_GLOBAL, O_TWO_CLASS, O_FOREGROUND


import tests.frontend.modules
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.identifysecondaryobjects
import cellprofiler.modules.threshold
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

INPUT_OBJECTS_NAME = "input_objects"
OUTPUT_OBJECTS_NAME = "output_objects"
NEW_OBJECTS_NAME = "new_objects"
IMAGE_NAME = "image"
THRESHOLD_IMAGE_NAME = "threshold"


def test_load_v9():
    file = tests.frontend.modules.get_test_resources_directory(
        "identifysecondaryobjects/v9.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    module = pipeline.modules()[-1]
    assert isinstance(
        module, cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects
    )
    assert module.x_name == "ChocolateChips"
    assert module.y_name == "Cookies"
    assert module.image_name == "BakingSheet"
    assert module.method == cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    assert module.distance_to_dilate == 11
    assert module.regularization_factor == 0.125
    assert module.wants_discard_edge
    assert not module.wants_discard_primary
    assert module.new_primary_objects_name == "FilteredChocolateChips"
    assert module.fill_holes
    assert (
        module.threshold.threshold_scope == TS_GLOBAL
    )
    assert (
        module.threshold.global_operation.value == cellprofiler.modules.threshold.TM_LI
    )
    assert module.threshold.threshold_smoothing_scale.value == 1.3488
    assert module.threshold.threshold_correction_factor == 1
    assert module.threshold.threshold_range.min == 0.0
    assert module.threshold.threshold_range.max == 1.0
    assert module.threshold.manual_threshold == 0.3
    assert module.threshold.thresholding_measurement == "Count_Cookies"
    assert (
        module.threshold.two_class_otsu
        == O_TWO_CLASS
    )
    assert (
        module.threshold.assign_middle_to_foreground
        == O_FOREGROUND
    )
    assert module.threshold.adaptive_window_size == 9


def test_load_v10():
    file = tests.frontend.modules.get_test_resources_directory(
        "identifysecondaryobjects/v10.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    module = pipeline.modules()[0]

    assert module.x_name.value == "IdentifyPrimaryObjects"
    assert module.y_name.value == "IdentifySecondaryObjects"
    assert module.method.value == "Propagation"
    assert module.image_name.value == "DNA"
    assert module.distance_to_dilate.value == 10
    assert module.regularization_factor.value == 0.05
    assert not module.wants_discard_edge.value
    assert not module.wants_discard_primary.value
    assert module.new_primary_objects_name.value == "FilteredNuclei"
    assert module.fill_holes.value


def make_workspace(
    image, segmented, unedited_segmented=None, small_removed_segmented=None
):
    p = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    p.add_listener(callback)
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(image)
    objects = cellprofiler_core.object.Objects()
    if unedited_segmented is not None:
        objects.unedited_segmented = unedited_segmented
    if small_removed_segmented is not None:
        objects.small_removed_segmented = small_removed_segmented
    objects.segmented = segmented
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    return workspace, module


def test_zeros_propagation():
    workspace, module = make_workspace(
        numpy.zeros((10, 10)), numpy.zeros((10, 10), int)
    )
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.run(workspace)
    m = workspace.measurements
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 0
    columns = module.get_measurement_columns(workspace.pipeline)
    for object_name in (
        "Image",
        OUTPUT_OBJECTS_NAME,
        INPUT_OBJECTS_NAME,
    ):
        ocolumns = [x for x in columns if x[0] == object_name]
        features = m.get_feature_names(object_name)
        assert len(ocolumns) == len(features)
        assert all([column[1] in features for column in ocolumns])
    assert "my_outlines" not in workspace.get_outline_names()


def test_one_object_propagation():
    img = numpy.zeros((10, 10))
    img[2:7, 2:7] = 0.5
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 3:6] = 1
    workspace, module = make_workspace(img, labels)
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.25
    module.run(workspace)
    m = workspace.measurements
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 1
    expected = numpy.zeros((10, 10), int)
    expected[2:7, 2:7] = 1
    assert numpy.all(objects_out.segmented == expected)
    child_counts = m.get_current_measurement(
        INPUT_OBJECTS_NAME, "Children_%s_Count" % OUTPUT_OBJECTS_NAME
    )
    assert len(child_counts) == 1
    assert child_counts[0] == 1
    parents = m.get_current_measurement(
        OUTPUT_OBJECTS_NAME, "Parent_%s" % INPUT_OBJECTS_NAME
    )
    assert len(parents) == 1
    assert parents[0] == 1


def test_two_objects_propagation_image():
    img = numpy.zeros((10, 20))
    img[2:7, 2:7] = 0.3
    img[2:7, 7:17] = 0.5
    labels = numpy.zeros((10, 20), int)
    labels[3:6, 3:6] = 1
    labels[3:6, 13:16] = 2
    workspace, module = make_workspace(img, labels)
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.regularization_factor.value = 0  # propagate by image
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.2
    module.run(workspace)
    m = workspace.measurements
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 2
    objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 10), int)
    expected[2:7, 2:7] = 1
    expected[2:7, 7:17] = 2
    mask = numpy.ones((10, 10), bool)
    mask[:, 7:9] = False
    assert numpy.all(objects_out.segmented[:10, :10][mask] == expected[mask])


def test_two_objects_propagation_distance():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    img = numpy.zeros((10, 20))
    img[2:7, 2:7] = 0.3
    img[2:7, 7:17] = 0.5
    image = cellprofiler_core.image.Image(img)
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros((10, 20), int)
    labels[3:6, 3:6] = 1
    labels[3:6, 13:16] = 2
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.regularization_factor.value = 1000  # propagate by distance
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.2
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 2
    objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 20), int)
    expected[2:7, 2:10] = 1
    expected[2:7, 10:17] = 2
    mask = numpy.ones((10, 20), bool)
    mask[:, 9:11] = False
    assert numpy.all(objects_out.segmented[mask] == expected[mask])


def test_zeros_watershed_gradient():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(numpy.zeros((10, 10)))
    objects = cellprofiler_core.object.Objects()
    objects.unedited_segmented = numpy.zeros((10, 10), int)
    objects.small_removed_segmented = numpy.zeros((10, 10), int)
    objects.segmented = numpy.zeros((10, 10), int)
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 0


def test_one_object_watershed_gradient():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    img = numpy.zeros((10, 10))
    img[2:7, 2:7] = 0.5
    image = cellprofiler_core.image.Image(img)
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 3:6] = 1
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G
    module.threshold.threshold_scope.value = (
        TS_GLOBAL
    )
    module.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 1
    objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 10), int)
    expected[2:7, 2:7] = 1
    assert numpy.all(objects_out.segmented == expected)
    assert "Location_Center_X" in m.get_feature_names(OUTPUT_OBJECTS_NAME)
    values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
    assert numpy.product(values.shape) == 1
    assert values[0] == 4
    assert "Location_Center_Y" in m.get_feature_names(OUTPUT_OBJECTS_NAME)
    values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
    assert numpy.product(values.shape) == 1
    assert values[0] == 4


def test_two_objects_watershed_gradient():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    img = numpy.zeros((10, 20))
    # There should be a gradient at :,7 which should act
    # as the watershed barrier
    img[2:7, 2:7] = 0.3
    img[2:7, 7:17] = 0.5
    image = cellprofiler_core.image.Image(img)
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros((10, 20), int)
    labels[3:6, 3:6] = 1
    labels[3:6, 13:16] = 2
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.2
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 2
    objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 20), int)
    expected[2:7, 2:7] = 1
    expected[2:7, 7:17] = 2
    mask = numpy.ones((10, 20), bool)
    mask[:, 7:9] = False
    assert numpy.all(objects_out.segmented[mask] == expected[mask])


def test_zeros_watershed_image():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(numpy.zeros((10, 10)))
    objects = cellprofiler_core.object.Objects()
    objects.unedited_segmented = numpy.zeros((10, 10), int)
    objects.small_removed_segmented = numpy.zeros((10, 10), int)
    objects.segmented = numpy.zeros((10, 10), int)
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 0


def test_one_object_watershed_image():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    img = numpy.zeros((10, 10))
    img[2:7, 2:7] = 0.5
    image = cellprofiler_core.image.Image(img)
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 3:6] = 1
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.threshold.threshold_scope.value = (
        TS_GLOBAL
    )
    module.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    module.set_module_num(1)
    p.add_module(module)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 1
    objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 10), int)
    expected[2:7, 2:7] = 1
    assert numpy.all(objects_out.segmented == expected)


def test_two_objects_watershed_image():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    img = numpy.zeros((10, 20))
    # There should be a saddle at 7 which should serve
    # as the watershed barrier
    x, y = numpy.mgrid[0:10, 0:20]
    img[2:7, 2:7] = 0.05 * (7 - y[2:7, 2:7])
    img[2:7, 7:17] = 0.05 * (y[2:7, 7:17] - 6)
    image = cellprofiler_core.image.Image(img)
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros((10, 20), int)
    labels[3:6, 3:6] = 1
    labels[3:6, 13:16] = 2
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.01
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 2
    objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 20), int)
    expected[2:7, 2:7] = 1
    expected[2:7, 7:17] = 2
    mask = numpy.ones((10, 20), bool)
    mask[:, 7] = False
    assert numpy.all(objects_out.segmented[mask] == expected[mask])


def test_zeros_distance_n():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(numpy.zeros((10, 10)))
    objects = cellprofiler_core.object.Objects()
    objects.unedited_segmented = numpy.zeros((10, 10), int)
    objects.small_removed_segmented = numpy.zeros((10, 10), int)
    objects.segmented = numpy.zeros((10, 10), int)
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_N
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 0


def test_one_object_distance_n():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    img = numpy.zeros((10, 10))
    image = cellprofiler_core.image.Image(img)
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 3:6] = 1
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_N
    module.distance_to_dilate.value = 1
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 1
    objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 10), int)
    expected[2:7, 2:7] = 1
    for x in (2, 6):
        for y in (2, 6):
            expected[x, y] = 0
    assert numpy.all(objects_out.segmented == expected)


def test_two_objects_distance_n():
    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    img = numpy.zeros((10, 20))
    image = cellprofiler_core.image.Image(img)
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros((10, 20), int)
    labels[3:6, 3:6] = 1
    labels[3:6, 13:16] = 2
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_N
    module.distance_to_dilate.value = 100
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    assert OUTPUT_OBJECTS_NAME in m.get_object_names()
    assert "Image" in m.get_object_names()
    assert "Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image")
    counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
    assert numpy.product(counts.shape) == 1
    assert counts == 2
    objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
    expected = numpy.zeros((10, 20), int)
    expected[:, :10] = 1
    expected[:, 10:] = 2
    assert numpy.all(objects_out.segmented == expected)


def test_measurements_no_new_primary():
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    for discard_edge in (True, False):
        module.wants_discard_edge.value = discard_edge
        module.wants_discard_primary.value = False
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME

        categories = module.get_categories(None, "Image")
        assert len(categories) == 2
        assert all([any([x == y for x in categories]) for y in ("Count", "Threshold")])
        categories = module.get_categories(None, OUTPUT_OBJECTS_NAME)
        assert len(categories) == 3
        assert all(
            [
                any([x == y for x in categories])
                for y in ("Location", "Parent", "Number")
            ]
        )
        categories = module.get_categories(None, INPUT_OBJECTS_NAME)
        assert len(categories) == 1
        assert categories[0] == "Children"

        categories = module.get_categories(None, NEW_OBJECTS_NAME)
        assert len(categories) == 0

        features = module.get_measurements(
            None, "Image", "Count"
        )
        assert len(features) == 1
        assert features[0] == OUTPUT_OBJECTS_NAME

        features = module.get_measurements(
            None, "Image", "Threshold"
        )
        threshold_features = (
            "OrigThreshold",
            "FinalThreshold",
            "WeightedVariance",
            "SumOfEntropies",
        )
        assert len(features) == 4
        assert all([any([x == y for x in features]) for y in threshold_features])
        for threshold_feature in threshold_features:
            objects = module.get_measurement_objects(
                None,
                "Image",
                "Threshold",
                threshold_feature,
            )
            assert len(objects) == 1
            assert objects[0] == OUTPUT_OBJECTS_NAME

        features = module.get_measurements(None, INPUT_OBJECTS_NAME, "Children")
        assert len(features) == 1
        assert (
            features[0] == "%s_Count" % OUTPUT_OBJECTS_NAME
        )

        features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Parent")
        assert len(features) == 1
        assert features[0] == INPUT_OBJECTS_NAME

        features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Location")
        assert len(features) == 3
        assert all(
            [
                any([x == y for x in features])
                for y in ("Center_X", "Center_Y", "Center_Z")
            ]
        )
        features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Number")
        assert len(features) == 1
        assert features[0] == "Object_Number"

        columns = module.get_measurement_columns(None)
        expected_columns = [
            (
                "Image",
                "Threshold_%s_%s" % (f, OUTPUT_OBJECTS_NAME),
                COLTYPE_FLOAT,
            )
            for f in threshold_features
        ]
        expected_columns += [
            (
                "Image",
                "Count_%s" % OUTPUT_OBJECTS_NAME,
                COLTYPE_INTEGER,
            ),
            (
                INPUT_OBJECTS_NAME,
                "Children_%s_Count" % OUTPUT_OBJECTS_NAME,
                COLTYPE_INTEGER,
            ),
            (
                OUTPUT_OBJECTS_NAME,
                "Location_Center_X",
                COLTYPE_FLOAT,
            ),
            (
                OUTPUT_OBJECTS_NAME,
                "Location_Center_Y",
                COLTYPE_FLOAT,
            ),
            (
                OUTPUT_OBJECTS_NAME,
                "Location_Center_Z",
                COLTYPE_FLOAT,
            ),
            (
                OUTPUT_OBJECTS_NAME,
                "Number_Object_Number",
                COLTYPE_INTEGER,
            ),
            (
                OUTPUT_OBJECTS_NAME,
                "Parent_%s" % INPUT_OBJECTS_NAME,
                COLTYPE_INTEGER,
            ),
        ]
        assert len(columns) == len(expected_columns)
        for column in expected_columns:
            assert any(
                [
                    all([fa == fb for fa, fb in zip(column, expected_column)])
                    for expected_column in expected_columns
                ]
            )


def test_measurements_new_primary():
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.wants_discard_edge.value = True
    module.wants_discard_primary.value = True
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.new_primary_objects_name.value = NEW_OBJECTS_NAME

    categories = module.get_categories(None, "Image")
    assert len(categories) == 2
    assert all([any([x == y for x in categories]) for y in ("Count", "Threshold")])
    categories = module.get_categories(None, OUTPUT_OBJECTS_NAME)
    assert len(categories) == 3
    assert all(
        [any([x == y for x in categories]) for y in ("Location", "Parent", "Number")]
    )
    categories = module.get_categories(None, INPUT_OBJECTS_NAME)
    assert len(categories) == 1
    assert categories[0] == "Children"

    categories = module.get_categories(None, NEW_OBJECTS_NAME)
    assert len(categories) == 4
    assert all(
        [
            any([x == y for x in categories])
            for y in ("Location", "Parent", "Children", "Number")
        ]
    )

    features = module.get_measurements(
        None, "Image", "Count"
    )
    assert len(features) == 2
    assert OUTPUT_OBJECTS_NAME in features
    assert NEW_OBJECTS_NAME in features

    features = module.get_measurements(
        None, "Image", "Threshold"
    )
    threshold_features = (
        "OrigThreshold",
        "FinalThreshold",
        "WeightedVariance",
        "SumOfEntropies",
    )
    assert len(features) == 4
    assert all([any([x == y for x in features]) for y in threshold_features])
    for threshold_feature in threshold_features:
        objects = module.get_measurement_objects(
            None, "Image", "Threshold", threshold_feature
        )
        assert len(objects) == 1
        assert objects[0] == OUTPUT_OBJECTS_NAME

    features = module.get_measurements(None, INPUT_OBJECTS_NAME, "Children")
    assert len(features) == 2
    assert all(
        [
            any([x == y for x in features])
            for y in (
                "%s_Count" % OUTPUT_OBJECTS_NAME,
                "%s_Count" % NEW_OBJECTS_NAME,
            )
        ]
    )

    features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Parent")
    assert len(features) == 2
    assert all(
        [
            any([x == y for x in features])
            for y in (INPUT_OBJECTS_NAME, NEW_OBJECTS_NAME)
        ]
    )

    for oname in (OUTPUT_OBJECTS_NAME, NEW_OBJECTS_NAME):
        features = module.get_measurements(None, oname, "Location")
        assert len(features) == 3
        assert all(
            [
                any([x == y for x in features])
                for y in ("Center_X", "Center_Y", "Center_Z")
            ]
        )

    columns = module.get_measurement_columns(None)
    expected_columns = [
        (
            "Image",
            "Threshold_%s_%s" % (f, OUTPUT_OBJECTS_NAME),
            COLTYPE_FLOAT,
        )
        for f in threshold_features
    ]
    for oname in (NEW_OBJECTS_NAME, OUTPUT_OBJECTS_NAME):
        expected_columns += [
            (
                "Image",
                FF_COUNT % oname,
                COLTYPE_INTEGER,
            ),
            (
                INPUT_OBJECTS_NAME,
                "Children_%s_Count" % oname,
                COLTYPE_INTEGER,
            ),
            (oname, "Location_Center_X", COLTYPE_FLOAT),
            (oname, "Location_Center_Y", COLTYPE_FLOAT),
            (oname, "Location_Center_Z", COLTYPE_FLOAT),
            (
                oname,
                "Number_Object_Number",
                COLTYPE_INTEGER,
            ),
            (oname, "Parent_Primary", COLTYPE_INTEGER),
        ]
    expected_columns += [
        (
            NEW_OBJECTS_NAME,
            "Children_%s_Count" % OUTPUT_OBJECTS_NAME,
            COLTYPE_INTEGER,
        ),
        (
            OUTPUT_OBJECTS_NAME,
            "Parent_%s" % NEW_OBJECTS_NAME,
            COLTYPE_INTEGER,
        ),
    ]
    assert len(columns) == len(expected_columns)
    for column in expected_columns:
        assert any(
            [
                all([fa == fb for fa, fb in zip(column, expected_column)])
                for expected_column in expected_columns
            ]
        )


def test_filter_edge():
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    image = numpy.array(
        [
            [0, 0, 0.5, 0, 0],
            [0, 0.5, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0.5, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected_unedited = numpy.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(image)
    objects = cellprofiler_core.object.Objects()
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.wants_discard_edge.value = True
    module.wants_discard_primary.value = True
    module.new_primary_objects_name.value = NEW_OBJECTS_NAME
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.threshold.threshold_scope.value = (
        TS_GLOBAL
    )
    module.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    module.run(workspace)
    object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
    assert numpy.all(object_out.segmented == 0)
    assert numpy.all(object_out.unedited_segmented == expected_unedited)

    object_out = workspace.object_set.get_objects(NEW_OBJECTS_NAME)
    assert numpy.all(object_out.segmented == 0)
    assert numpy.all(object_out.unedited_segmented == labels)


def test_filter_unedited():
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    labels_unedited = numpy.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    image = numpy.array(
        [
            [0, 0, 0.5, 0, 0],
            [0, 0.5, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0.5, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected = numpy.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    expected_unedited = numpy.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(image)
    objects = cellprofiler_core.object.Objects()
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.unedited_segmented = labels_unedited
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.wants_discard_edge.value = True
    module.wants_discard_primary.value = True
    module.new_primary_objects_name.value = NEW_OBJECTS_NAME
    module.threshold.threshold_scope.value = (
        TS_GLOBAL
    )
    module.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
    assert numpy.all(object_out.segmented == expected)
    assert numpy.all(object_out.unedited_segmented == expected_unedited)

    object_out = workspace.object_set.get_objects(NEW_OBJECTS_NAME)
    assert numpy.all(object_out.segmented == labels)
    assert numpy.all(object_out.unedited_segmented == labels_unedited)


def test_small():
    """Regression test of IMG-791

    A small object in the seed mask should not attract any of the
    secondary object.
    """
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    labels_unedited = numpy.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    image = numpy.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        float,
    )
    expected = image.astype(int)

    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(image)
    objects = cellprofiler_core.object.Objects()
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.unedited_segmented = labels_unedited
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.5
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
    assert numpy.all(object_out.segmented == expected)


def test_small_touching():
    """Test of logic added for IMG-791

    A small object in the seed mask touching the edge should attract
    some of the secondary object
    """
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    labels_unedited = numpy.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
        ]
    )

    image = numpy.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        float,
    )

    p = cellprofiler_core.pipeline.Pipeline()
    o_s = cellprofiler_core.object.ObjectSet()
    i_l = cellprofiler_core.image.ImageSetList()
    image = cellprofiler_core.image.Image(image)
    objects = cellprofiler_core.object.Objects()
    objects.unedited_segmented = labels
    objects.small_removed_segmented = labels
    objects.unedited_segmented = labels_unedited
    objects.segmented = labels
    o_s.add_objects(objects, INPUT_OBJECTS_NAME)
    i_s = i_l.get_image_set(0)
    i_s.add(IMAGE_NAME, image)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
    module.x_name.value = INPUT_OBJECTS_NAME
    module.y_name.value = OUTPUT_OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.5
    module.set_module_num(1)
    p.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(p, module, i_s, o_s, m, i_l)
    module.run(workspace)
    object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
    i, j = numpy.argwhere(labels_unedited == 2)[0]
    assert numpy.all(object_out.segmented[i - 1 :, j - 1 : j + 2] == 0)
    assert len(numpy.unique(object_out.unedited_segmented)) == 3
    assert len(numpy.unique(object_out.unedited_segmented[i - 1 :, j - 1 : j + 2])) == 1


def test_holes_no_holes():
    for wants_fill_holes in (True, False):
        for method in (
            cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_B,
            cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION,
            cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G,
            cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I,
        ):
            labels = numpy.zeros((20, 10), int)
            labels[5, 5] = 1
            labels[15, 5] = 2
            threshold = numpy.zeros((20, 10), bool)
            threshold[1:7, 4:7] = True
            threshold[2, 5] = False
            threshold[14:17, 4:7] = True
            expected = numpy.zeros((20, 10), int)
            expected[1:7, 4:7] = 1
            expected[14:17, 4:7] = 2
            if not wants_fill_holes:
                expected[2, 5] = 0
            workspace, module = make_workspace(threshold * 0.5, labels)
            assert isinstance(
                module,
                cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects,
            )
            module.threshold.threshold_scope.value = (
                cellprofiler.modules.threshold.TS_GLOBAL
            )
            module.threshold.global_operation.value = (
                cellprofiler.modules.threshold.TM_MANUAL
            )
            module.threshold.manual_threshold.value = 0.5
            module.method.value = method
            module.fill_holes.value = wants_fill_holes
            module.distance_to_dilate.value = 10000
            image_set = workspace.image_set
            assert isinstance(image_set, cellprofiler_core.image.ImageSet)

            module.run(workspace)
            object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
            labels_out = object_out.segmented
            indexes = workspace.measurements.get_current_measurement(
                OUTPUT_OBJECTS_NAME, "Parent_" + INPUT_OBJECTS_NAME
            )
            assert len(indexes) == 2
            indexes = numpy.hstack(([0], indexes))
            assert numpy.all(indexes[labels_out] == expected)


def test_relationships_zero():
    workspace, module = make_workspace(
        numpy.zeros((10, 10)), numpy.zeros((10, 10), int)
    )
    assert isinstance(
        module, cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    result = m.get_relationships(
        module.module_num,
        cellprofiler.modules.identifysecondaryobjects.R_PARENT,
        module.x_name.value,
        module.y_name.value,
    )
    assert len(result) == 0


def test_relationships_one():
    img = numpy.zeros((10, 10))
    img[2:7, 2:7] = 0.5
    labels = numpy.zeros((10, 10), int)
    labels[3:6, 3:6] = 1
    workspace, module = make_workspace(img, labels)
    module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
    module.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.threshold.manual_threshold.value = 0.25
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    result = m.get_relationships(
        module.module_num,
        cellprofiler.modules.identifysecondaryobjects.R_PARENT,
        module.x_name.value,
        module.y_name.value,
    )
    assert len(result) == 1
    assert result[R_FIRST_IMAGE_NUMBER][0] == 1
    assert result[R_SECOND_IMAGE_NUMBER][0] == 1
    assert result[R_FIRST_OBJECT_NUMBER][0] == 1
    assert result[R_SECOND_OBJECT_NUMBER][0] == 1


def test_relationships_missing():
    for missing in range(1, 4):
        img = numpy.zeros((10, 30))
        labels = numpy.zeros((10, 30), int)
        for i in range(3):
            object_number = i + 1
            center_j = i * 10 + 4
            labels[3:6, (center_j - 1) : (center_j + 2)] = object_number
            if object_number != missing:
                img[2:7, (center_j - 2) : (center_j + 3)] = 0.5
            else:
                img[0:7, (center_j - 2) : (center_j + 3)] = 0.5
        workspace, module = make_workspace(img, labels)
        assert isinstance(
            module,
            cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects,
        )
        module.method.value = (
            cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        )
        module.threshold.threshold_scope.value = (
            cellprofiler.modules.threshold.TS_GLOBAL
        )
        module.threshold.global_operation.value = (
            cellprofiler.modules.threshold.TM_MANUAL
        )
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = False
        module.threshold.manual_threshold.value = 0.25
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m,cellprofiler_core.measurement.Measurements)
        result = m.get_relationships(
            module.module_num,
            cellprofiler.modules.identifysecondaryobjects.R_PARENT,
            module.x_name.value,
            module.y_name.value,
        )
        assert len(result) == 2
        for i in range(2):
            first_object_number = second_object_number = i + 1
            if first_object_number >= missing:
                first_object_number += 1
            assert result[R_FIRST_IMAGE_NUMBER][i] == 1
            assert result[R_SECOND_IMAGE_NUMBER][i] == 1
            assert (
                result[R_FIRST_OBJECT_NUMBER][i]
                == first_object_number
            )
            assert (
                result[R_SECOND_OBJECT_NUMBER][i]
                == second_object_number
            )
