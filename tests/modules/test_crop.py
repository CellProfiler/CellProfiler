import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.modules.crop as cpmc
import cellprofiler.measurement as cpm
import cellprofiler.object as cpo

INPUT_IMAGE = "input_image"
CROP_IMAGE = "crop_image"
CROP_OBJECTS = "crop_objects"
CROPPING = "cropping"
OUTPUT_IMAGE = "output_image"


def make_workspace(input_pixels, crop_image=None, cropping=None, crop_objects=None):
    """Return a workspace with the given images installed and the crop module"""
    image_set_list = cpi.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    module = cpmc.Crop()
    module.set_module_num(1)
    image_set.add(INPUT_IMAGE, cpi.Image(input_pixels))
    module.image_name.value = INPUT_IMAGE
    module.cropped_image_name.value = OUTPUT_IMAGE
    if crop_image is not None:
        image_set.add(CROP_IMAGE, cpi.Image(crop_image))
        module.image_mask_source.value = CROP_IMAGE
    if cropping is not None:
        image_set.add(CROPPING, cpi.Image(np.zeros(cropping.shape), crop_mask=cropping))
        module.cropping_mask_source.value = CROPPING
    object_set = cpo.ObjectSet()
    if crop_objects is not None:
        objects = cpo.Objects()
        objects.segmented = crop_objects
        object_set.add_objects(objects, CROP_OBJECTS)

    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    m = cpm.Measurements()
    workspace = cpw.Workspace(
        pipeline, module, image_set, object_set, m, image_set_list
    )
    m.add_measurement(cpm.IMAGE, cpm.GROUP_INDEX, 0, image_set_number=1)
    m.add_measurement(cpm.IMAGE, cpm.GROUP_NUMBER, 1, image_set_number=1)
    return workspace, module


def test_zeros():
    """Test cropping an image with a mask of all zeros"""
    workspace, module = make_workspace(
        np.zeros((10, 10)), crop_image=np.zeros((10, 10), bool)
    )
    module.shape.value = cpmc.SH_IMAGE
    module.remove_rows_and_columns.value = cpmc.RM_NO
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == 0)
    assert np.all(output_image.mask == output_image.pixel_data)
    assert np.all(output_image.crop_mask == output_image.pixel_data)
    m = workspace.measurements
    assert "Image" in m.get_object_names()
    columns = module.get_measurement_columns(workspace.pipeline)
    assert len(columns) == 2
    assert all([x[0] == cpm.IMAGE for x in columns])
    assert all([x[2] == cpm.COLTYPE_INTEGER for x in columns])
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
        np.zeros((10, 10)), crop_image=np.zeros((10, 10), bool)
    )
    module.shape.value = cpmc.SH_IMAGE
    module.remove_rows_and_columns.value = cpmc.RM_ALL
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.product(output_image.pixel_data.shape) == 0


def test_crop_edges_with_image():
    """Test cropping and removing rows and columns with an image"""
    x, y = np.mgrid[0:10, 0:10]
    input_image = x / 100.0 + y / 10.0
    crop_image = np.zeros((10, 10), bool)
    crop_image[2, 3] = True
    crop_image[7, 5] = True
    expected_image = np.zeros((6, 3), np.float32)
    expected_image[0, 0] = input_image[2, 3]
    expected_image[5, 2] = input_image[7, 5]
    workspace, module = make_workspace(input_image, crop_image=crop_image)
    module.shape.value = cpmc.SH_IMAGE
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_all_with_image():
    """Test cropping and removing rows and columns with an image"""
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    crop_image = np.zeros((10, 10), bool)
    crop_image[2, 3] = True
    crop_image[7, 5] = True
    expected_image = input_image[(2, 7), :][:, (3, 5)]
    expected_image[1, 0] = 0
    expected_image[0, 1] = 0
    workspace, module = make_workspace(input_image, crop_image=crop_image)
    module.shape.value = cpmc.SH_IMAGE
    module.remove_rows_and_columns.value = cpmc.RM_ALL
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_edges_with_cropping():
    """Test cropping and removing rows and columns with an image cropping"""
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    crop_image = np.zeros((10, 10), bool)
    crop_image[2, 3] = True
    crop_image[7, 5] = True
    expected_image = np.zeros((6, 3))
    expected_image[0, 0] = input_image[2, 3]
    expected_image[5, 2] = input_image[7, 5]
    workspace, module = make_workspace(input_image, cropping=crop_image)
    module.shape.value = cpmc.SH_CROPPING
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_ellipse_x_major():
    """Crop with an ellipse that has its major axis in the X direction"""
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_ELLIPSE
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
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
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_ellipse_y_major():
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_ELLIPSE
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
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
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle():
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    expected_image = input_image[2:8, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_xmin():
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    expected_image = input_image[2:8, :9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_RECTANGLE
    module.horizontal_limits.set_value((0, 9))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_xmax():
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    expected_image = input_image[2:8, 1:]
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_RECTANGLE
    module.horizontal_limits.set_value((1, "end"))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_ymin():
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    expected_image = input_image[:8, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((0, 8))
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_unbounded_ymax():
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    expected_image = input_image[2:, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((2, "end"))
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_color_with_rectangle():
    """Regression test: make sure cropping works with a color image"""
    i, j, k = np.mgrid[0:10, 0:10, 0:3]
    input_image = (i / 1000.0 + j / 100.0 + k).astype(np.float32)
    expected_image = input_image[2:8, 1:9, :]
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_RECTANGLE
    module.horizontal_limits.set_value((1, 9))
    module.vertical_limits.set_value((2, 8))
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_crop_with_rectangle_float_bounds():
    x, y = np.mgrid[0:10, 0:10]
    input_image = (x / 100.0 + y / 10.0).astype(np.float32)
    expected_image = input_image[2:8, 1:9]
    workspace, module = make_workspace(input_image)
    module.shape.value = cpmc.SH_RECTANGLE
    module.horizontal_limits.set_value((1.2, 9.0000003))
    module.vertical_limits.set_value((2.5, 8.999999))
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert np.all(output_image.pixel_data == expected_image)


def test_mask_with_objects():
    np.random.seed()
    input_image = np.random.uniform(size=(20, 10))
    input_objects = np.zeros((20, 10), dtype=int)
    input_objects[2:7, 3:8] = 1
    input_objects[12:17, 3:8] = 2
    workspace, module = make_workspace(input_image, crop_objects=input_objects)
    module.shape.value = cpmc.SH_OBJECTS
    module.objects_source.value = CROP_OBJECTS
    module.remove_rows_and_columns.value = cpmc.RM_NO
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert output_image.has_masking_objects
    assert np.all(input_objects == output_image.labels)
    assert np.all(output_image.mask == (input_objects > 0))


def test_crop_with_objects():
    np.random.seed()
    input_image = np.random.uniform(size=(20, 10))
    input_objects = np.zeros((20, 10), dtype=int)
    input_objects[2:7, 3:8] = 1
    input_objects[12:17, 3:8] = 2
    workspace, module = make_workspace(input_image, crop_objects=input_objects)
    module.shape.value = cpmc.SH_OBJECTS
    module.objects_source.value = CROP_OBJECTS
    module.remove_rows_and_columns.value = cpmc.RM_EDGES
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE)
    assert output_image.has_masking_objects
    assert np.all(input_objects[2:17, 3:8] == output_image.labels)
    assert np.all(output_image.mask == (input_objects[2:17, 3:8] > 0))
    assert np.all(output_image.crop_mask == (input_objects > 0))
