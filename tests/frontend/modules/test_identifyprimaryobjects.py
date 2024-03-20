import io
import os

import centrosome.threshold
import numpy
import pytest
import scipy.ndimage

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import FF_COUNT, COLTYPE_INTEGER, FF_FINAL_THRESHOLD, COLTYPE_FLOAT, \
    FF_ORIG_THRESHOLD, M_NUMBER_OBJECT_NUMBER, M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y, M_LOCATION_CENTER_Z, \
    FF_SUM_OF_ENTROPIES, FF_WEIGHTED_VARIANCE
from cellprofiler_core.constants.module._identify import TS_GLOBAL, TS_ADAPTIVE, O_TWO_CLASS, O_FOREGROUND, \
    O_THREE_CLASS, O_BACKGROUND, RB_MODE, RB_SD, RB_MEAN, RB_MEDIAN, RB_MAD, TS_MANUAL, TS_MEASUREMENT
from cellprofiler_core.image import VanillaImage
import cellprofiler.modules.identifyprimaryobjects
import cellprofiler.modules.threshold
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace

import tests.frontend
import tests.frontend.modules

IMAGE_NAME = "my_image"
OBJECTS_NAME = "my_objects"
MASKING_OBJECTS_NAME = "masking_objects"
MEASUREMENT_NAME = "my_measurement"


def load_error_handler(caller, event):
    if isinstance(event, cellprofiler_core.pipeline.event.LoadException):
        pytest.fail(event.error.message)


def make_workspace(image, mask=None, labels=None):
    """Make a workspace and IdentifyPrimaryObjects module

    image - the intensity image for thresholding

    mask - if present, the "don't analyze" mask of the intensity image

    labels - if thresholding per-object, the labels matrix needed
    """
    module = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    module.set_module_num(1)
    module.x_name.value = IMAGE_NAME
    module.y_name.value = OBJECTS_NAME

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)
    m = cellprofiler_core.measurement.Measurements()
    cpimage = cellprofiler_core.image.Image(image, mask=mask)
    m.add(IMAGE_NAME, cpimage)
    object_set = cellprofiler_core.object.ObjectSet()
    if labels is not None:
        o = cellprofiler_core.object.Objects()
        o.segmented = labels
        object_set.add_objects(o, MASKING_OBJECTS_NAME)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, m, object_set, m, None
    )
    return workspace, module


def test_init():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()


def test_test_zero_objects():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.use_advanced.value = True
    x.threshold.threshold_range.min = 0.1
    x.threshold.threshold_range.max = 1
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    img = numpy.zeros((25, 25))
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented == 0)
    assert "Image" in measurements.get_object_names()
    assert "my_object" in measurements.get_object_names()
    assert "Threshold_FinalThreshold_my_object" in measurements.get_feature_names(
        "Image"
    )
    assert "Count_my_object" in measurements.get_feature_names("Image")
    count = measurements.get_current_measurement("Image", "Count_my_object")
    assert count == 0
    assert "Location_Center_X" in measurements.get_feature_names("my_object")
    location_center_x = measurements.get_current_measurement(
        "my_object", "Location_Center_X"
    )
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 0
    assert "Location_Center_Y" in measurements.get_feature_names("my_object")
    location_center_y = measurements.get_current_measurement(
        "my_object", "Location_Center_Y"
    )
    assert isinstance(location_center_y, numpy.ndarray)
    assert numpy.product(location_center_y.shape) == 0


def test_test_zero_objects_wa_in_lo_in():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.use_advanced.value = True
    x.threshold.threshold_range.min = 0.1
    x.threshold.threshold_range.max = 1
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    img = numpy.zeros((25, 25))
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented == 0)


def test_test_zero_objects_wa_di_lo_in():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.use_advanced.value = True
    x.threshold.threshold_range.min = 0.1
    x.threshold.threshold_range.max = 1
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    img = numpy.zeros((25, 25))
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented == 0)


def test_test_zero_objects_wa_in_lo_sh():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.use_advanced.value = True
    x.threshold.threshold_range.min = 0.1
    x.threshold.threshold_range.max = 1
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
    img = numpy.zeros((25, 25))
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented == 0)


def test_test_zero_objects_wa_di_lo_sh():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.use_advanced.value = True
    x.threshold.threshold_range.min = 0.1
    x.threshold.threshold_range.max = 1
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
    img = numpy.zeros((25, 25))
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented == 0)


def test_test_one_object():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = False
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = one_cell_image()
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented[img > 0] == 1)
    assert numpy.all(img[segmented == 1] > 0)
    assert "Image" in measurements.get_object_names()
    assert "my_object" in measurements.get_object_names()
    assert "Threshold_FinalThreshold_my_object" in measurements.get_feature_names(
        "Image"
    )
    threshold = measurements.get_current_measurement(
        "Image", "Threshold_FinalThreshold_my_object"
    )
    assert threshold < 0.5
    assert "Count_my_object" in measurements.get_feature_names("Image")
    count = measurements.get_current_measurement("Image", "Count_my_object")
    assert count == 1
    assert "Location_Center_Y" in measurements.get_feature_names("my_object")
    location_center_y = measurements.get_current_measurement(
        "my_object", "Location_Center_Y"
    )
    assert isinstance(location_center_y, numpy.ndarray)
    assert numpy.product(location_center_y.shape) == 1
    assert location_center_y[0] > 8
    assert location_center_y[0] < 12
    assert "Location_Center_X" in measurements.get_feature_names("my_object")
    location_center_x = measurements.get_current_measurement(
        "my_object", "Location_Center_X"
    )
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 1
    assert location_center_x[0] > 13
    assert location_center_x[0] < 16
    columns = x.get_measurement_columns(pipeline)
    for object_name in ("Image", "my_object"):
        ocolumns = [x for x in columns if x[0] == object_name]
        features = measurements.get_feature_names(object_name)
        assert len(ocolumns) == len(features)
        assert all([column[1] in features for column in ocolumns])


def test_test_two_objects():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = False
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = two_cell_image()
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    assert "Image" in measurements.get_object_names()
    assert "my_object" in measurements.get_object_names()
    assert "Threshold_FinalThreshold_my_object" in measurements.get_feature_names(
        "Image"
    )
    threshold = measurements.get_current_measurement(
        "Image", "Threshold_FinalThreshold_my_object"
    )
    assert threshold < 0.6
    assert "Count_my_object" in measurements.get_feature_names("Image")
    count = measurements.get_current_measurement("Image", "Count_my_object")
    assert count == 2
    assert "Location_Center_Y" in measurements.get_feature_names("my_object")
    location_center_y = measurements.get_current_measurement(
        "my_object", "Location_Center_Y"
    )
    assert isinstance(location_center_y, numpy.ndarray)
    assert numpy.product(location_center_y.shape) == 2
    assert location_center_y[0] > 8
    assert location_center_y[0] < 12
    assert location_center_y[1] > 28
    assert location_center_y[1] < 32
    assert "Location_Center_Y" in measurements.get_feature_names("my_object")
    location_center_x = measurements.get_current_measurement(
        "my_object", "Location_Center_X"
    )
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 2
    assert location_center_x[0] > 33
    assert location_center_x[0] < 37
    assert location_center_x[1] > 13
    assert location_center_x[1] < 16


def test_test_threshold_range():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.threshold.threshold_range.min = 0.7
    x.threshold.threshold_range.max = 1
    x.threshold.threshold_correction_factor.value = 0.95
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_LI
    x.exclude_size.value = False
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    img = two_cell_image()
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    assert "Image" in measurements.get_object_names()
    assert "my_object" in measurements.get_object_names()
    assert "Threshold_FinalThreshold_my_object" in measurements.get_feature_names(
        "Image"
    )
    threshold = measurements.get_current_measurement(
        "Image", "Threshold_FinalThreshold_my_object"
    )
    assert threshold < 0.8
    assert threshold > 0.6
    assert "Count_my_object" in measurements.get_feature_names("Image")
    count = measurements.get_current_measurement("Image", "Count_my_object")
    assert count == 1
    assert "Location_Center_Y" in measurements.get_feature_names("my_object")
    location_center_y = measurements.get_current_measurement(
        "my_object", "Location_Center_Y"
    )
    assert isinstance(location_center_y, numpy.ndarray)
    assert numpy.product(location_center_y.shape) == 1
    assert location_center_y[0] > 8
    assert location_center_y[0] < 12
    assert "Location_Center_X" in measurements.get_feature_names("my_object")
    location_center_x = measurements.get_current_measurement(
        "my_object", "Location_Center_X"
    )
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 1
    assert location_center_x[0] > 33
    assert location_center_x[0] < 36


def test_fill_holes():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = False
    x.fill_holes.value = True
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.zeros((40, 40))
    draw_circle(img, (10, 10), 7, 0.5)
    draw_circle(img, (30, 30), 7, 0.5)
    img[10, 10] = 0
    img[30, 30] = 0
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert objects.segmented[10, 10] > 0
    assert objects.segmented[30, 30] > 0


def test_dont_fill_holes():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.threshold.threshold_range.min = 0.7
    x.threshold.threshold_range.max = 1
    x.exclude_size.value = False
    x.fill_holes.value = False
    x.smoothing_filter_size.value = 0
    x.automatic_smoothing.value = False
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    img = numpy.zeros((40, 40))
    draw_circle(img, (10, 10), 7, 0.5)
    draw_circle(img, (30, 30), 7, 0.5)
    img[10, 10] = 0
    img[30, 30] = 0
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert objects.segmented[10, 10] == 0
    assert objects.segmented[30, 30] == 0


def test_01_fill_holes_within_holes():
    "Regression test of img-1431"
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.size_range.min = 1
    x.size_range.max = 2
    x.exclude_size.value = False
    x.fill_holes.value = cellprofiler.modules.identifyprimaryobjects.FH_DECLUMP
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.zeros((40, 40))
    draw_circle(img, (20, 20), 10, 0.5)
    draw_circle(img, (20, 20), 4, 0)
    img[20, 20] = 1
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert objects.segmented[20, 20] == 1
    assert objects.segmented[22, 20] == 1
    assert objects.segmented[26, 20] == 1


def test_test_watershed_shape_shape():
    """Identify by local_maxima:shape & intensity:shape

    Create an object whose intensity is high near the middle
    but has an hourglass shape, then segment it using shape/shape
    """
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.x_name.value = "my_image"
    x.y_name.value = "my_object"
    x.exclude_size.value = False
    x.size_range.value = (2, 10)
    x.fill_holes.value = False
    x.maxima_suppression_size.value = 3
    x.automatic_suppression.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0, 0, 0],
            [0, 0, 0, 0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.8, 0.9, 1, 1, 0.9, 0.8, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0, 0, 0, 0],
            [0, 0, 0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert numpy.max(objects.segmented) == 2


def test_test_watershed_shape_intensity():
    """Identify by local_maxima:shape & watershed:intensity

    Create an object with an hourglass shape to get two maxima, but
    set the intensities so that one maximum gets the middle portion
    """
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.x_name.value = "my_image"
    x.y_name.value = "my_object"
    x.exclude_size.value = False
    x.size_range.value = (2, 10)
    x.fill_holes.value = False
    x.maxima_suppression_size.value = 3
    x.automatic_suppression.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert numpy.max(objects.segmented) == 2
    assert objects.segmented[7, 11] == objects.segmented[7, 4]


def test_test_watershed_intensity_distance_single():
    """Identify by local_maxima:intensity & watershed:shape - one object

    Create an object with an hourglass shape and a peak in the middle.
    It should be segmented into a single object.
    """
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.x_name.value = "my_image"
    x.y_name.value = "my_object"
    x.exclude_size.value = False
    x.size_range.value = (4, 10)
    x.fill_holes.value = False
    x.maxima_suppression_size.value = 3.6
    x.automatic_suppression.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # We do a little blur here so that there's some monotonic decrease
    # from the central peak
    img = scipy.ndimage.gaussian_filter(img, 0.25, mode="constant")
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert numpy.max(objects.segmented) == 1


def test_test_watershed_intensity_distance_triple():
    """Identify by local_maxima:intensity & watershed:shape - 3 objects w/o filter

    Create an object with an hourglass shape and a peak in the middle.
    It should be segmented into a single object.
    """
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.x_name.value = "my_image"
    x.y_name.value = "my_object"
    x.exclude_size.value = False
    x.size_range.value = (2, 10)
    x.fill_holes.value = False
    x.smoothing_filter_size.value = 0
    x.automatic_smoothing.value = False
    x.maxima_suppression_size.value = 7.1
    x.automatic_suppression.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert numpy.max(objects.segmented) == 3


def test_test_watershed_intensity_distance_filter():
    """Identify by local_maxima:intensity & watershed:shape - filtered

    Create an object with an hourglass shape and a peak in the middle.
    It should be segmented into a single object.
    """
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.x_name.value = "my_image"
    x.y_name.value = "my_object"
    x.exclude_size.value = False
    x.size_range.value = (2, 10)
    x.fill_holes.value = False
    x.smoothing_filter_size.value = 1
    x.automatic_smoothing.value = 1
    x.maxima_suppression_size.value = 3.6
    x.automatic_suppression.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert numpy.max(objects.segmented) == 1


def test_test_watershed_intensity_distance_double():
    """Identify by local_maxima:intensity & watershed:shape - two objects

    Create an object with an hourglass shape and peaks in the top and
    bottom, but with a distribution of values that's skewed so that,
    by intensity, one of the images occupies the middle. The middle
    should be shared because the watershed is done using the distance
    transform.
    """
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.x_name.value = "my_image"
    x.y_name.value = "my_object"
    x.exclude_size.value = False
    x.size_range.value = (2, 10)
    x.fill_holes.value = False
    x.smoothing_filter_size.value = 0
    x.automatic_smoothing.value = 0
    x.maxima_suppression_size.value = 3.6
    x.automatic_suppression.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # We do a little blur here so that there's some monotonic decrease
    # from the central peak
    img = scipy.ndimage.gaussian_filter(img, 0.5, mode="constant")
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert numpy.max(objects.segmented) == 2
    assert objects.segmented[12, 7] != objects.segmented[4, 7]


def test_propagate():
    """Test the propagate unclump method"""
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.x_name.value = "my_image"
    x.y_name.value = "my_object"
    x.exclude_size.value = False
    x.size_range.value = (2, 10)
    x.fill_holes.value = False
    x.smoothing_filter_size.value = 0
    x.automatic_smoothing.value = 0
    x.maxima_suppression_size.value = 7
    x.automatic_suppression.value = False
    x.threshold.manual_threshold.value = 0.3
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_PROPAGATE
    x.threshold.threshold_scope.value = TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.threshold_smoothing_scale.value = 0
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # We do a little blur here so that there's some monotonic decrease
    # from the central peak
    img = scipy.ndimage.gaussian_filter(img, 0.5, mode="constant")
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert numpy.max(objects.segmented) == 2
    # This point has a closer "crow-fly" distance to the upper object
    # but should be in the lower one because of the serpentine path
    assert objects.segmented[14, 9] == objects.segmented[9, 9]


def test_fly():
    """Run identify on the fly image"""
    file = tests.frontend.modules.get_test_resources_directory("identifyprimaryobjects/fly.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(pipeline, event):
        assert not isinstance(
            event,
            (
                cellprofiler_core.pipeline.event.RunException,
                cellprofiler_core.pipeline.event.LoadException,
            ),
        )

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    x = pipeline.modules()[0]
    assert isinstance(
        x, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    x.use_advanced.value = True
    img = fly_image()
    image = cellprofiler_core.image.Image(img)
    #
    # Make sure it runs both regular and with reduced image
    #
    for min_size in (9, 15):
        #
        # Exercise clumping / declumping options
        #
        x.size_range.min = min_size
        for unclump_method in (
                cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY,
                cellprofiler.modules.identifyprimaryobjects.UN_SHAPE,
        ):
            x.unclump_method.value = unclump_method
            for watershed_method in (
                    cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY,
                    cellprofiler.modules.identifyprimaryobjects.WA_SHAPE,
                    cellprofiler.modules.identifyprimaryobjects.WA_PROPAGATE,
            ):
                x.watershed_method.value = watershed_method
                image_set_list = cellprofiler_core.image.ImageSetList()
                image_set = image_set_list.get_image_set(0)
                image_set.add(x.x_name.value, image)
                object_set = cellprofiler_core.object.ObjectSet()
                measurements = cellprofiler_core.measurement.Measurements()
                x.run(
                    cellprofiler_core.workspace.Workspace(
                        pipeline, x, image_set, object_set, measurements, None
                    )
                )


def test_maxima_suppression_zero():
    # Regression test for issue #877
    # if maxima_suppression_size = 1 or 0, use a 4-connected structuring
    # element.
    #
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0, 0],
            [0, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 2, 0, 0, 3, 0, 0],
            [0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    for distance in (0, 1):
        x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
        x.use_advanced.value = True
        x.x_name.value = "my_image"
        x.y_name.value = "my_object"
        x.exclude_size.value = False
        x.size_range.value = (2, 10)
        x.fill_holes.value = False
        x.smoothing_filter_size.value = 0
        x.automatic_smoothing.value = 0
        x.maxima_suppression_size.value = distance
        x.automatic_suppression.value = False
        x.threshold.manual_threshold.value = 0.05
        x.unclump_method.value = (
            cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
        )
        x.watershed_method.value = (
            cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
        )
        x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
        x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
        x.threshold.threshold_smoothing_scale.value = 0
        pipeline = cellprofiler_core.pipeline.Pipeline()
        x.set_module_num(1)
        pipeline.add_module(x)
        object_set = cellprofiler_core.object.ObjectSet()
        measurements = cellprofiler_core.measurement.Measurements()
        measurements.add(x.x_name.value, cellprofiler_core.image.Image(img))
        x.run(
            cellprofiler_core.workspace.Workspace(
                pipeline, x, measurements, object_set, measurements, None
            )
        )
        output = object_set.get_objects(x.y_name.value)
        assert output.count == 4
        assert numpy.all(output.segmented[expected == 0] == 0)
        assert len(numpy.unique(output.segmented[expected == 1])) == 1


def test_load_v10():
    # Sorry about this overly-long pipeline, it seemed like we need to
    # revisit many of the choices.
    file = tests.frontend.modules.get_test_resources_directory("identifyprimaryobjects/v10.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    module = pipeline.modules()[4]
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    assert module.x_name == "Channel0"
    assert module.y_name == "Cells"
    assert module.size_range.min == 15
    assert module.size_range.max == 45
    assert module.exclude_size
    assert module.exclude_border_objects
    assert (
            module.unclump_method
            == cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    )
    assert (
            module.watershed_method
            == cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
    )
    assert module.automatic_smoothing
    assert module.smoothing_filter_size == 11
    assert module.automatic_suppression
    assert module.maxima_suppression_size == 9
    assert module.low_res_maxima
    assert (
            module.fill_holes == cellprofiler.modules.identifyprimaryobjects.FH_THRESHOLDING
    )
    assert module.limit_choice == cellprofiler.modules.identifyprimaryobjects.LIMIT_NONE
    assert module.maximum_object_count == 499
    #
    assert (
            module.threshold.threshold_scope
            == TS_ADAPTIVE
    )
    assert module.threshold.local_operation.value == centrosome.threshold.TM_OTSU
    assert module.threshold.threshold_smoothing_scale == 1.3488
    assert round(abs(module.threshold.threshold_correction_factor.value - 0.80), 7) == 0
    assert round(abs(module.threshold.threshold_range.min - 0.01), 7) == 0
    assert round(abs(module.threshold.threshold_range.max - 0.90), 7) == 0
    assert round(abs(module.threshold.manual_threshold.value - 0.03), 7) == 0
    assert module.threshold.thresholding_measurement == "Metadata_Threshold"
    assert (
            module.threshold.two_class_otsu
            == O_TWO_CLASS
    )
    assert (
            module.threshold.assign_middle_to_foreground
            == O_FOREGROUND
    )
    assert module.threshold.adaptive_window_size == 12
    #
    # Test alternate settings using subsequent instances of IDPrimary
    #
    module = pipeline.modules()[5]
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    assert not module.exclude_size
    assert not module.exclude_border_objects
    assert (
            module.unclump_method
            == cellprofiler.modules.identifyprimaryobjects.UN_INTENSITY
    )
    assert (
            module.watershed_method == cellprofiler.modules.identifyprimaryobjects.WA_NONE
    )
    assert not module.automatic_smoothing
    assert not module.automatic_suppression
    assert not module.low_res_maxima
    assert module.fill_holes == cellprofiler.modules.identifyprimaryobjects.FH_NEVER
    assert (
            module.limit_choice == cellprofiler.modules.identifyprimaryobjects.LIMIT_ERASE
    )
    assert (
            module.threshold.threshold_scope == TS_GLOBAL
    )
    assert (
            module.threshold.global_operation.value == cellprofiler.modules.threshold.TM_LI
    )
    assert (
            module.threshold.two_class_otsu
            == O_THREE_CLASS
    )
    assert (
            module.threshold.assign_middle_to_foreground
            == O_BACKGROUND
    )
    assert module.use_advanced.value

    module = pipeline.modules()[6]
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    assert module.unclump_method == cellprofiler.modules.identifyprimaryobjects.UN_NONE
    assert (
            module.watershed_method
            == cellprofiler.modules.identifyprimaryobjects.WA_PROPAGATE
    )
    assert module.limit_choice == "None"
    assert module.threshold.global_operation.value == "None"
    assert module.threshold.threshold_smoothing_scale == 0
    assert module.use_advanced.value

    module = pipeline.modules()[7]
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    assert module.unclump_method == cellprofiler.modules.identifyprimaryobjects.UN_SHAPE
    assert (
            module.watershed_method == cellprofiler.modules.identifyprimaryobjects.WA_SHAPE
    )
    assert (
            module.threshold.threshold_scope == TS_GLOBAL
    )
    assert (
            module.threshold.global_operation.value
            == cellprofiler.modules.threshold.TM_ROBUST_BACKGROUND
    )
    assert module.threshold.lower_outlier_fraction.value == 0.02
    assert module.threshold.upper_outlier_fraction.value == 0.02
    assert (
            module.threshold.averaging_method.value == RB_MODE
    )
    assert (
            module.threshold.variance_method.value == RB_SD
    )
    assert module.threshold.number_of_deviations.value == 0
    assert module.threshold.threshold_correction_factor.value == 1.6
    assert module.use_advanced.value

    module = pipeline.modules()[8]
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    assert module.threshold.threshold_scope == cellprofiler.modules.threshold.TS_GLOBAL
    assert (
            module.threshold.global_operation.value
            == cellprofiler.modules.threshold.TM_MANUAL
    )
    assert module.use_advanced.value

    module = pipeline.modules()[9]
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    assert module.threshold.threshold_scope == cellprofiler.modules.threshold.TS_GLOBAL
    assert (
            module.threshold.global_operation.value
            == cellprofiler.modules.threshold.TM_MEASUREMENT
    )
    assert module.use_advanced.value

    module = pipeline.modules()[10]
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    assert module.threshold.threshold_scope == "None"
    assert (
            module.threshold.global_operation.value
            == cellprofiler.modules.threshold.TM_ROBUST_BACKGROUND
    )
    assert module.threshold.lower_outlier_fraction == 0.05
    assert module.threshold.upper_outlier_fraction == 0.05
    assert (
            module.threshold.averaging_method == RB_MEAN
    )
    assert module.threshold.variance_method == RB_SD
    assert module.threshold.number_of_deviations == 2
    assert module.use_advanced.value


def test_01_load_new_robust_background():
    #
    # Test custom robust background parameters.
    #
    file = tests.frontend.modules.get_test_resources_directory(
        "identifyprimaryobjects/robust_background.pipeline"
    )
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    for module, averaging_method, variance_method in zip(
            pipeline.modules(),
            (
                    RB_MEAN,
                    RB_MEDIAN,
                    RB_MODE,
            ),
            (
                    RB_SD,
                    RB_MAD,
                    RB_MAD,
            ),
    ):
        assert isinstance(
            module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
        )
        assert module.threshold.lower_outlier_fraction == 0.1
        assert module.threshold.upper_outlier_fraction == 0.2
        assert module.threshold.number_of_deviations == 2.5
        assert module.threshold.averaging_method == averaging_method
        assert module.threshold.variance_method == variance_method
        assert module.use_advanced.value


def test_discard_large():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = True
    x.size_range.min = 10
    x.size_range.max = 40
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.manual_threshold.value = 0.3
    img = numpy.zeros((200, 200))
    draw_circle(img, (100, 100), 25, 0.5)
    draw_circle(img, (25, 25), 10, 0.5)
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert objects.segmented[25, 25] == 1, "The small object was not there"
    assert objects.segmented[100, 100] == 0, "The large object was not filtered out"
    assert (
            objects.small_removed_segmented[25, 25] > 0
    ), "The small object was not in the small_removed label set"
    assert (
            objects.small_removed_segmented[100, 100] > 0
    ), "The large object was not in the small-removed label set"
    assert objects.unedited_segmented[
        25, 25
    ], "The small object was not in the unedited set"
    assert objects.unedited_segmented[
        100, 100
    ], "The large object was not in the unedited set"
    location_center_x = measurements.get_current_measurement(
        "my_object", "Location_Center_X"
    )
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 1


def test_keep_large():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = False
    x.size_range.min = 10
    x.size_range.max = 40
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.manual_threshold.value = 0.3
    img = numpy.zeros((200, 200))
    draw_circle(img, (100, 100), 25, 0.5)
    draw_circle(img, (25, 25), 10, 0.5)
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert objects.segmented[25, 25], "The small object was not there"
    assert objects.segmented[100, 100], "The large object was filtered out"
    assert objects.unedited_segmented[
        25, 25
    ], "The small object was not in the unedited set"
    assert objects.unedited_segmented[
        100, 100
    ], "The large object was not in the unedited set"
    location_center_x = measurements.get_current_measurement(
        "my_object", "Location_Center_X"
    )
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 2


def test_discard_small():
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = True
    x.size_range.min = 40
    x.size_range.max = 60
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.manual_threshold.value = 0.3
    img = numpy.zeros((200, 200))
    draw_circle(img, (100, 100), 25, 0.5)
    draw_circle(img, (25, 25), 10, 0.5)
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    objects = object_set.get_objects("my_object")
    assert objects.segmented[25, 25] == 0, "The small object was not filtered out"
    assert objects.segmented[100, 100] == 1, "The large object was not present"
    assert (
            objects.small_removed_segmented[25, 25] == 0
    ), "The small object was in the small_removed label set"
    assert (
            objects.small_removed_segmented[100, 100] > 0
    ), "The large object was not in the small-removed label set"
    assert objects.unedited_segmented[
        25, 25
    ], "The small object was not in the unedited set"
    assert objects.unedited_segmented[
        100, 100
    ], "The large object was not in the unedited set"
    location_center_x = measurements.get_current_measurement(
        "my_object", "Location_Center_X"
    )
    assert isinstance(location_center_x, numpy.ndarray)
    assert numpy.product(location_center_x.shape) == 1


def test_regression_diagonal():
    """Regression test - was using one-connected instead of 3-connected structuring element"""
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = False
    x.smoothing_filter_size.value = 0
    x.automatic_smoothing.value = False
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.threshold_smoothing_scale.value = 0
    x.threshold.manual_threshold.value = 0.5
    img = numpy.zeros((10, 10))
    img[4, 4] = 1
    img[5, 5] = 1
    image = cellprofiler_core.image.Image(img)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented[img > 0] == 1)
    assert numpy.all(img[segmented == 1] > 0)


def test_regression_adaptive_mask():
    """Regression test - mask all but one pixel / adaptive"""
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.use_advanced.value = True
    x.exclude_size.value = False
    x.threshold.threshold_scope.value = centrosome.threshold.TM_ADAPTIVE
    x.threshold.global_operation.value = centrosome.threshold.TM_OTSU
    numpy.random.seed(62)
    img = numpy.random.uniform(size=(100, 100))
    mask = numpy.zeros(img.shape, bool)
    mask[-1, -1] = True
    image = cellprofiler_core.image.Image(img, mask)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add_provider(
        VanillaImage("my_image", image)
    )
    object_set = cellprofiler_core.object.ObjectSet()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    x.run(
        cellprofiler_core.workspace.Workspace(
            pipeline, x, image_set, object_set, measurements, None
        )
    )
    assert len(object_set.object_names) == 1
    assert "my_object" in object_set.object_names
    objects = object_set.get_objects("my_object")
    segmented = objects.segmented
    assert numpy.all(segmented == 0)


# def test_test_robust_background_fly():
#     image = fly_image()
#     workspace, x = make_workspace(image)
#     x.threshold.threshold_scope.value = I.TS_GLOBAL
#     x.threshold_method.value = T.TM_ROBUST_BACKGROUND
#     local_threshold,threshold = x.get_threshold(
#         cpi.Image(image), np.ones(image.shape,bool), workspace)
#     assertTrue(threshold > 0.09)
#     assertTrue(threshold < 0.095)


def test_get_measurement_columns():
    """Test the get_measurement_columns method"""
    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    oname = "my_object"
    x.y_name.value = oname
    x.x_name.value = "my_image"
    x.use_advanced.value = True
    columns = x.get_measurement_columns(None)
    expected_columns = [
        ("Image", format % oname, coltype)
        for format, coltype in (
            (
                FF_COUNT,
                COLTYPE_INTEGER,
            ),
            (
                FF_FINAL_THRESHOLD,
                COLTYPE_FLOAT,
            ),
            (
                FF_ORIG_THRESHOLD,
                COLTYPE_FLOAT,
            ),
            (
                FF_WEIGHTED_VARIANCE,
                COLTYPE_FLOAT,
            ),
            (
                FF_SUM_OF_ENTROPIES,
                COLTYPE_FLOAT,
            ),
        )
    ]
    expected_columns += [
        (oname, feature, COLTYPE_FLOAT)
        for feature in (
            M_LOCATION_CENTER_X,
            M_LOCATION_CENTER_Y,
            M_LOCATION_CENTER_Z,
        )
    ]
    expected_columns += [
        (
            oname,
            M_NUMBER_OBJECT_NUMBER,
            COLTYPE_INTEGER,
        )
    ]
    assert len(columns) == len(expected_columns)
    for column in columns:
        assert any(
            all([colval == exval for colval, exval in zip(column, expected)])
            for expected in expected_columns
        )


def test_regression_holes():
    """Regression test - fill holes caused by filtered object

    This was created as a regression test for the bug, IMG-191, but
    didn't exercise the bug. It's a good test of watershed and filling
    labeled holes in an odd case, so I'm leaving it in.
    """
    #
    # This array has two intensity peaks separated by a border.
    # You should get two objects, one within the other.
    #
    pixels = (
            numpy.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                    [0, 0, 2, 1, 2, 2, 2, 2, 1, 2, 0, 0],
                    [0, 0, 2, 1, 2, 9, 2, 2, 1, 2, 0, 0],
                    [0, 0, 2, 1, 2, 2, 2, 2, 1, 2, 0, 0],
                    [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 9, 9, 0, 0],
                    [0, 0, 2, 2, 2, 2, 2, 2, 9, 9, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                float,
            )
            / 10.0
    )
    expected = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
            [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
            [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
            [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
            [0, 0, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    mask = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        bool,
    )
    workspace, x = make_workspace(pixels)
    x.use_advanced.value = True
    x.exclude_size.value = True
    x.size_range.min = 6
    x.size_range.max = 50
    x.maxima_suppression_size.value = 3
    x.automatic_suppression.value = False
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_INTENSITY
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.threshold_smoothing_scale.value = 0
    x.threshold.manual_threshold.value = 0.05
    x.threshold.threshold_correction_factor.value = 1
    measurements = workspace.measurements
    x.run(workspace)
    my_objects = workspace.object_set.get_objects(OBJECTS_NAME)
    assert my_objects.segmented[3, 3] != 0
    if my_objects.unedited_segmented[3, 3] == 2:
        unedited_segmented = my_objects.unedited_segmented
    else:
        unedited_segmented = numpy.array([0, 2, 1])[my_objects.unedited_segmented]
    assert numpy.all(unedited_segmented[mask] == expected[mask])


def test_regression_holes():
    """Regression test - fill holes caused by filtered object

    This is the real regression test for IMG-191. The smaller object
    is surrounded by pixels below threshold. This prevents filling in
    the unedited case.
    """
    # An update to fill_labeled_holes will remove both the filtered object
    # and the hole
    #
    if True:
        return
    pixels = (
        numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 3, 0, 0, 9, 2, 0, 0, 3, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 2, 2, 2, 9, 2, 0, 0],
                [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            float,
        )
        / 10.0
    )
    expected = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    mask = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        bool,
    )
    image = cellprofiler_core.image.Image(pixels)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add("my_image", image)
    object_set = cellprofiler_core.object.ObjectSet()

    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = True
    x.size_range.min = 4
    x.size_range.max = 50
    x.maxima_suppression_size.value = 3
    x.automatic_suppression.value = False
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.local_operation.value = centrosome.threshold.TM_MANUAL
    x.threshold.manual_threshold.value = 0.1
    x.threshold.threshold_correction_factor.value = 1
    x.set_module_num(1)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(x)
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, x, image_set, object_set, measurements, image_set_list
    )
    x.run(workspace)
    my_objects = object_set.get_objects("my_object")
    assert my_objects.segmented[3, 3] != 0
    assert numpy.all(my_objects.segmented[mask] == expected[mask])


def test_erase_objects():
    """Set up a limit on the # of objects and exceed it - erasing objects"""
    maximum_object_count = 3
    pixels = numpy.zeros((20, 21))
    pixels[2:8, 2:8] = 0.5
    pixels[12:18, 2:8] = 0.5
    pixels[2:8, 12:18] = 0.5
    pixels[12:18, 12:18] = 0.5
    image = cellprofiler_core.image.Image(pixels)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add("my_image", image)
    object_set = cellprofiler_core.object.ObjectSet()

    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.manual_threshold.value = 0.25
    x.threshold.threshold_correction_factor.value = 1
    x.limit_choice.value = cellprofiler.modules.identifyprimaryobjects.LIMIT_ERASE
    x.maximum_object_count.value = maximum_object_count
    x.set_module_num(1)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(x)
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, x, image_set, object_set, measurements, image_set_list
    )
    x.run(workspace)
    assert measurements.get_current_image_measurement("Count_my_object") == 0
    my_objects = object_set.get_objects("my_object")
    assert numpy.all(my_objects.segmented == 0)
    assert numpy.max(my_objects.unedited_segmented) == 4


def test_dont_erase_objects():
    """Ask to erase objects, but don't"""
    maximum_object_count = 5
    pixels = numpy.zeros((20, 21))
    pixels[2:8, 2:8] = 0.5
    pixels[12:18, 2:8] = 0.5
    pixels[2:8, 12:18] = 0.5
    pixels[12:18, 12:18] = 0.5
    image = cellprofiler_core.image.Image(pixels)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add("my_image", image)
    object_set = cellprofiler_core.object.ObjectSet()

    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "my_object"
    x.x_name.value = "my_image"
    x.exclude_size.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.threshold.manual_threshold.value = 0.25
    x.threshold.threshold_correction_factor.value = 1
    x.limit_choice.value = cellprofiler.modules.identifyprimaryobjects.LIMIT_ERASE
    x.maximum_object_count.value = maximum_object_count
    x.set_module_num(1)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(x)
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, x, image_set, object_set, measurements, image_set_list
    )
    x.run(workspace)
    assert measurements.get_current_image_measurement("Count_my_object") == 4
    my_objects = object_set.get_objects("my_object")
    assert numpy.max(my_objects.segmented) == 4


def test_threshold_by_measurement():
    """Set threshold based on mean image intensity"""
    pixels = numpy.zeros((10, 10))
    pixels[2:6, 2:6] = 0.5

    image = cellprofiler_core.image.Image(pixels)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add("MyImage", image)
    object_set = cellprofiler_core.object.ObjectSet()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    measurements = cellprofiler_core.measurement.Measurements()
    measurements.add_image_measurement("MeanIntensity_MyImage", numpy.mean(pixels))

    x = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    x.use_advanced.value = True
    x.y_name.value = "MyObject"
    x.x_name.value = "MyImage"
    x.exclude_size.value = False
    x.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
    x.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    x.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MEASUREMENT
    x.threshold.threshold_smoothing_scale.value = 0
    x.threshold.thresholding_measurement.value = "MeanIntensity_MyImage"
    x.threshold.threshold_correction_factor.value = 1
    x.set_module_num(1)
    pipeline.add_module(x)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, x, image_set, object_set, measurements, image_set_list
    )
    x.run(workspace)
    assert measurements.get_current_image_measurement("Count_MyObject") == 1
    assert measurements.get_current_image_measurement(
        "Threshold_FinalThreshold_MyObject"
    ) == numpy.mean(pixels)


def test_threshold_smoothing_automatic():
    image = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.5, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    workspace, module = make_workspace(image)
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    module.use_advanced.value = True
    module.exclude_size.value = False
    module.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
    module.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    # MCT on this image is zero, so set the threshold at .225
    # with the threshold minimum (manual = no smoothing)
    module.threshold.threshold_scope.value = (
        TS_GLOBAL
    )
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_LI
    module.threshold.threshold_range.min = 0.225
    module.run(workspace)
    labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
    numpy.testing.assert_array_equal(expected, labels)


def test_threshold_smoothing_manual():
    image = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.5, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    workspace, module = make_workspace(image)
    assert isinstance(
        module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
    )
    module.use_advanced.value = True
    module.exclude_size.value = False
    module.unclump_method.value = cellprofiler.modules.identifyprimaryobjects.UN_NONE
    module.watershed_method.value = cellprofiler.modules.identifyprimaryobjects.WA_NONE
    module.threshold.threshold_scope.value = (
        TS_GLOBAL
    )
    module.threshold.global_operation.value = cellprofiler.modules.threshold.TM_LI
    module.threshold.threshold_range.min = 0.125
    module.threshold.threshold_smoothing_scale.value = 3
    module.run(workspace)
    labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
    numpy.testing.assert_array_equal(expected, labels)


def test_threshold_no_smoothing():
    image = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0.4, 0.5, 0.4, 0, 0],
            [0, 0, 0.4, 0.4, 0.4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    expected = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    for tm in (
            TS_MANUAL,
            TS_MEASUREMENT,
    ):
        workspace, module = make_workspace(image)
        assert isinstance(
            module, cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects
        )
        module.use_advanced.value = True
        module.exclude_size.value = False
        module.unclump_method.value = (
            cellprofiler.modules.identifyprimaryobjects.UN_NONE
        )
        module.watershed_method.value = (
            cellprofiler.modules.identifyprimaryobjects.WA_NONE
        )
        module.threshold.threshold_scope.value = (
            cellprofiler.modules.threshold.TS_GLOBAL
        )
        module.threshold.global_operation.value = tm
        module.threshold.manual_threshold.value = 0.125
        module.threshold.thresholding_measurement.value = MEASUREMENT_NAME
        workspace.measurements[
            "Image", MEASUREMENT_NAME
        ] = 0.125
        module.threshold.threshold_smoothing_scale.value = 0
        module.run(workspace)
        labels = workspace.object_set.get_objects(OBJECTS_NAME).segmented
        numpy.testing.assert_array_equal(expected, labels)


def add_noise(img, fraction):
    """Add a fractional amount of noise to an image to make it look real"""
    numpy.random.seed(0)
    noise = numpy.random.uniform(
        low=1 - fraction / 2, high=1 + fraction / 2, size=img.shape
    )
    return img * noise


def one_cell_image():
    img = numpy.zeros((25, 25))
    draw_circle(img, (10, 15), 5, 0.5)
    return add_noise(img, 0.01)


def two_cell_image():
    img = numpy.zeros((50, 50))
    draw_circle(img, (10, 35), 5, 0.8)
    draw_circle(img, (30, 15), 5, 0.6)
    return add_noise(img, 0.01)


def fly_image():
    from cellprofiler_core.reader import get_image_reader
    from cellprofiler_core.pipeline import ImageFile

    path = os.path.join(os.path.dirname(tests.frontend.__file__), "resources/01_POS002_D.TIF")
    rdr = get_image_reader(ImageFile("file://"+path))
    return rdr.read()


def draw_circle(img, center, radius, value):
    x, y = numpy.mgrid[0: img.shape[0], 0: img.shape[1]]
    distance = numpy.sqrt(
        (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])
    )
    img[distance <= radius] = value


def test_01_masked_wv():
    output = centrosome.threshold.weighted_variance(
        numpy.zeros((3, 3)), numpy.zeros((3, 3), bool), 1
    )
    assert output == 0


def test_02_zero_wv():
    output = centrosome.threshold.weighted_variance(
        numpy.zeros((3, 3)), numpy.ones((3, 3), bool), numpy.ones((3, 3), bool)
    )
    assert output == 0


def test_03_fg_0_bg_0():
    """Test all foreground pixels same, all background same, wv = 0"""
    img = numpy.zeros((4, 4))
    img[:, 2:4] = 1
    binary_image = img > 0.5
    output = centrosome.threshold.weighted_variance(
        img, numpy.ones(img.shape, bool), binary_image
    )
    assert output == 0


def test_04_values():
    """Test with two foreground and two background values"""
    #
    # The log of this array is [-4,-3],[-2,-1] and
    # the variance should be (.25 *2 + .25 *2)/4 = .25
    img = numpy.array([[1.0 / 16.0, 1.0 / 8.0], [1.0 / 4.0, 1.0 / 2.0]])
    binary_image = numpy.array([[False, False], [True, True]])
    output = centrosome.threshold.weighted_variance(
        img, numpy.ones((2, 2), bool), binary_image
    )
    assert round(abs(output - 0.25), 7) == 0


def test_05_mask():
    """Test, masking out one of the background values"""
    #
    # The log of this array is [-4,-3],[-2,-1] and
    # the variance should be (.25*2 + .25 *2)/4 = .25
    img = numpy.array(
        [[1.0 / 16.0, 1.0 / 16.0, 1.0 / 8.0], [1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0]]
    )
    mask = numpy.array([[False, True, True], [False, True, True]])
    binary_image = numpy.array([[False, False, False], [True, True, True]])
    output = centrosome.threshold.weighted_variance(img, mask, binary_image)
    assert round(abs(output - 0.25), 7) == 0


def test_01_all_masked():
    output = centrosome.threshold.sum_of_entropies(
        numpy.zeros((3, 3)), numpy.zeros((3, 3), bool), 1
    )
    assert output == 0


def test_020_all_zero():
    """Can't take the log of zero, so all zero matrix = 0"""
    output = centrosome.threshold.sum_of_entropies(
        numpy.zeros((4, 2)), numpy.ones((4, 2), bool), numpy.ones((4, 2), bool)
    )
    assert round(abs(output - 0), 7) == 0
