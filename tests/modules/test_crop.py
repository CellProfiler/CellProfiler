import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import GROUP_INDEX, GROUP_NUMBER, COLTYPE_INTEGER


import cellprofiler.modules.crop
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

INPUT_IMAGE = "input_image"
CROP_IMAGE = "crop_image"
CROP_OBJECTS = "crop_objects"
CROPPING = "cropping"
OUTPUT_IMAGE = "output_image"


def make_workspace(input_pixels, crop_image=None, cropping=None, crop_objects=None):
    """Return a workspace with the given images installed and the crop module"""
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    module = cellprofiler.modules.crop.Crop()
    module.set_module_num(1)
    image_set.add(INPUT_IMAGE, cellprofiler_core.image.Image(input_pixels))
    module.image_name.value = INPUT_IMAGE
    module.cropped_image_name.value = OUTPUT_IMAGE
    if crop_image is not None:
        image_set.add(CROP_IMAGE, cellprofiler_core.image.Image(crop_image))
        module.image_mask_source.value = CROP_IMAGE
    if cropping is not None:
        image_set.add(
            CROPPING,
            cellprofiler_core.image.Image(
                numpy.zeros(cropping.shape), crop_mask=cropping
            ),
        )
        module.cropping_mask_source.value = CROPPING
    object_set = cellprofiler_core.object.ObjectSet()
    if crop_objects is not None:
        objects = cellprofiler_core.object.Objects()
        objects.segmented = crop_objects
        object_set.add_objects(objects, CROP_OBJECTS)

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    m = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, image_set, object_set, m, image_set_list
    )
    m.add_measurement(
        "Image",
        GROUP_INDEX,
        0,
        image_set_number=1,
    )
    m.add_measurement(
        "Image",
        GROUP_NUMBER,
        1,
        image_set_number=1,
    )
    return workspace, module


def test_zeros():
    """Test cropping an image with a mask of all zeros"""
    workspace, module = make_workspace(
        numpy.zeros((10, 10)), crop_image=numpy.zeros((10, 10), bool)
    )
    module.shape.value = cellprofiler.modules.crop.SH_IMAGE
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_NO
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == 0)
    assert numpy.all(output_image.mask == output_image.pixel_data)
    assert numpy.all(output_image.crop_mask == output_image.pixel_data)
    m = workspace.measurements
    assert "Image" in m.get_object_names()
    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(columns) == 2
    assert all([x[0] == "Image" for x in columns])
    assert all([x[2] == COLTYPE_INTEGER for x in columns])
    feature = "Crop_OriginalImageArea_%s" % OUTPUT_IMAGE
    assert feature in [x[1] for x in columns]
    assert feature in m.get_feature_names("Image")
    values = m.get_current_measurement("Image", feature)
    assert round(abs(values - 10 * 10), 7) == 0
    feature = "Crop_AreaRetainedAfterCropping_%s" % OUTPUT_IMAGE
    assert feature in [x[1] for x in columns]
    assert feature in m.get_feature_names("Image")
    values = m.get_current_measurement("Image", feature)
    assert values == 0


def test_zeros_and_remove_all():
    """Test cropping and removing rows and columns on a blank image"""
    workspace, module = make_workspace(
        numpy.zeros((10, 10)), crop_image=numpy.zeros((10, 10), bool)
    )
    module.shape.value = cellprofiler.modules.crop.SH_IMAGE
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_ALL
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.product(output_image.pixel_data.shape) == 0


def test_crop_edges_with_image():
    """Test cropping and removing rows and columns with an image"""
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = x / 100.0 + y / 10.0
    crop_image = numpy.zeros((10, 10), bool)
    crop_image[2, 3] = True
    crop_image[7, 5] = True
    expected_image = numpy.zeros((6, 3), numpy.float32)
    expected_image[0, 0] = input_image[2, 3]
    expected_image[5, 2] = input_image[7, 5]
    workspace, module = make_workspace(input_image, crop_image=crop_image)
    module.shape.value = cellprofiler.modules.crop.SH_IMAGE
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_all_with_image():
    """Test cropping and removing rows and columns with an image"""
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    crop_image = numpy.zeros((10, 10), bool)
    crop_image[2, 3] = True
    crop_image[7, 5] = True
    expected_image = input_image[(2, 7), :][:, (3, 5)]
    expected_image[1, 0] = 0
    expected_image[0, 1] = 0
    workspace, module = make_workspace(input_image, crop_image=crop_image)
    module.shape.value = cellprofiler.modules.crop.SH_IMAGE
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_ALL
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_edges_with_cropping():
    """Test cropping and removing rows and columns with an image cropping"""
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    crop_image = numpy.zeros((10, 10), bool)
    crop_image[2, 3] = True
    crop_image[7, 5] = True
    expected_image = numpy.zeros((6, 3))
    expected_image[0, 0] = input_image[2, 3]
    expected_image[5, 2] = input_image[7, 5]
    workspace, module = make_workspace(input_image, cropping=crop_image)
    module.shape.value = cellprofiler.modules.crop.SH_CROPPING
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_ellipse_x_major():
    """Crop with an ellipse that has its major axis in the X direction"""
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_ELLIPSE
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.ellipse_center.set_value((4, 5))
    module.ellipse_x_radius.value = 3
    module.ellipse_y_radius.value = 2
    expected_image = input_image[3:8, 1:8]
    for i, j in ((0, 0), (1, 0), (0, 1), (0, 2)):
        expected_image[i, j] = 0
        expected_image[-i - 1, j] = 0
        expected_image[i, -j - 1] = 0
        expected_image[-i - 1, -j - 1] = 0
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_ellipse_y_major():
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_ELLIPSE
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.ellipse_center.set_value((5, 4))
    module.ellipse_x_radius.value = 2
    module.ellipse_y_radius.value = 3
    expected_image = input_image[1:8, 3:8]
    for i, j in ((0, 0), (1, 0), (0, 1), (2, 0)):
        expected_image[i, j] = 0
        expected_image[-i - 1, j] = 0
        expected_image[i, -j - 1] = 0
        expected_image[-i - 1, -j - 1] = 0
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle():
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    expected_image = input_image[2:8, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_xmin():
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    expected_image = input_image[2:8, :9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_RECTANGLE
    module.horizontal_limits.set_value((0, 9))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_xmax():
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    expected_image = input_image[2:8, 1:]
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_RECTANGLE
    module.horizontal_limits.set_value((1, "end"))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_ymin():
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    expected_image = input_image[:8, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((0, 8))
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_ymax():
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    expected_image = input_image[2:, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((2, "end"))
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_color_with_rectangle():
    """Regression test: make sure cropping works with a color image"""
    i, j, k = numpy.mgrid[0:10, 0:10, 0:3]
    input_image = (i / 1000.0 + j / 100.0 + k).astype(numpy.float32)
    expected_image = input_image[2:8, 1:9, :]
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_float_bounds():
    x, y = numpy.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(numpy.float32)
    expected_image = input_image[2:8, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cellprofiler.modules.crop.SH_RECTANGLE
    module.horizontal_limits.set_value((1.2, 9.0000003))
    module.vertical_limits.set_value((2.5, 8.999999))
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert numpy.all(output_image.pixel_data == expected_image)


def test_mask_with_objects():
    numpy.random.seed()
    input_image = numpy.random.uniform(size=(20, 10))
    input_objects = numpy.zeros((20, 10), dtype=int)
    input_objects[2:7, 3:8] = 1
    input_objects[12:17, 3:8] = 2
    workspace, module = make_workspace(input_image, crop_objects=input_objects)
    module.shape.value = cellprofiler.modules.crop.SH_OBJECTS
    module.objects_source.value = CROP_OBJECTS
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_NO
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert output_image.has_masking_objects
    assert numpy.all(input_objects == output_image.labels)
    assert numpy.all(output_image.mask == (input_objects > 0))


def test_crop_with_objects():
    numpy.random.seed()
    input_image = numpy.random.uniform(size=(20, 10))
    input_objects = numpy.zeros((20, 10), dtype=int)
    input_objects[2:7, 3:8] = 1
    input_objects[12:17, 3:8] = 2
    workspace, module = make_workspace(input_image, crop_objects=input_objects)
    module.shape.value = cellprofiler.modules.crop.SH_OBJECTS
    module.objects_source.value = CROP_OBJECTS
    module.remove_rows_and_columns.value = cellprofiler.modules.crop.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert output_image.has_masking_objects
    assert numpy.all(input_objects[2:17, 3:8] == output_image.labels)
    assert numpy.all(output_image.mask == (input_objects[2:17, 3:8] > 0))
    assert numpy.all(output_image.crop_mask == (input_objects > 0))
