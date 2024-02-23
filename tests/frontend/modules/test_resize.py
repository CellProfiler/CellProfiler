import io

import numpy
import numpy.testing
import skimage.exposure
import skimage.transform

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.resize
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_IMAGE_NAME = "input"
OUTPUT_IMAGE_NAME = "output"


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("resize/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.resize.Resize)
    assert module.x_name == "DNA"
    assert module.y_name == "ResizedDNA"
    assert module.size_method == cellprofiler.modules.resize.R_TO_SIZE
    assert round(abs(module.resizing_factor_x.value - 0.25), 7) == 0
    assert round(abs(module.resizing_factor_y.value - 0.25), 7) == 0
    assert round(abs(module.resizing_factor_z.value - 1), 7) == 0
    assert module.specific_width == 141
    assert module.specific_height == 169
    assert module.interpolation == cellprofiler.modules.resize.I_BILINEAR
    assert module.additional_image_count.value == 1
    additional_image = module.additional_images[0]
    assert additional_image.input_image_name == "Actin"
    assert additional_image.output_image_name == "ResizedActin"

    module = pipeline.modules()[1]
    assert isinstance(module, cellprofiler.modules.resize.Resize)
    assert module.interpolation == cellprofiler.modules.resize.I_BICUBIC


def make_workspace(
    image, size_method, interpolation, mask=None, cropping=None, dimensions=2
):
    module = cellprofiler.modules.resize.Resize()
    module.x_name.value = INPUT_IMAGE_NAME
    module.y_name.value = OUTPUT_IMAGE_NAME
    module.size_method.value = size_method
    module.interpolation.value = interpolation
    module.set_module_num(1)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image = cellprofiler_core.image.Image(image, mask, cropping, dimensions=dimensions)
    image_set.add(INPUT_IMAGE_NAME, image)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_rescale_triple_color():
    i, j = numpy.mgrid[0:10, 0:10]
    image = numpy.zeros((10, 10, 3))
    image[:, :, 0] = i
    image[:, :, 1] = j
    image = skimage.exposure.rescale_intensity(1.0 * image)
    i, j = (numpy.mgrid[0:30, 0:30].astype(float) * 1.0 / 3.0).astype(int)
    expected = numpy.zeros((30, 30, 3))
    expected[:, :, 0] = i
    expected[:, :, 1] = j
    expected = skimage.exposure.rescale_intensity(1.0 * expected)
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 3.0
    module.run(workspace)
    result_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    result = result_image.pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)
    assert result_image.parent_image is workspace.image_set.get_image(INPUT_IMAGE_NAME)


def test_rescale_triple_bw():
    i, j = numpy.mgrid[0:10, 0:10].astype(float)
    image = skimage.exposure.rescale_intensity(1.0 * i)
    i, j = (numpy.mgrid[0:30, 0:30].astype(float) * 1.0 / 3.0).astype(int)
    expected = skimage.exposure.rescale_intensity(1.0 * i)
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 3.0
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)


def test_third():
    i, j = numpy.mgrid[0:30, 0:30]
    image = skimage.exposure.rescale_intensity(1.0 * i)
    expected = skimage.transform.resize(image, (10, 10), order=0, mode="symmetric")
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 1.0 / 3.0
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)


def test_bilinear():
    i, j = numpy.mgrid[0:10, 0:10]
    image = skimage.exposure.rescale_intensity(1.0 * i)
    expected = skimage.transform.resize(image, (30, 30), order=1, mode="symmetric")
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 3.0
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)


def test_bicubic():
    i, j = numpy.mgrid[0:10, 0:10]
    image = skimage.exposure.rescale_intensity(1.0 * i)
    expected = skimage.transform.resize(image, (30, 30), order=3, mode="symmetric")
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_BICUBIC,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 3.0
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)


def test_reshape_double():
    """Make an image twice as large by changing the shape"""
    i, j = numpy.mgrid[0:10, 0:10].astype(float)
    image = skimage.exposure.rescale_intensity(i + j * 10.0)
    expected = skimage.transform.resize(image, (19, 19), order=1, mode="symmetric")
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.specific_width.value = 19
    module.specific_height.value = 19
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)


def test_reshape_half():
    """Make an image half as large by changing the shape"""
    i, j = numpy.mgrid[0:19, 0:19].astype(float) / 2.0
    image = skimage.exposure.rescale_intensity(i + j * 10)
    expected = skimage.transform.resize(image, (10, 10), order=1, mode="symmetric")
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.specific_width.value = 10
    module.specific_height.value = 10
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)


def test_reshape_half_and_double():
    """Make an image twice as large in one dimension and half in other"""
    i, j = numpy.mgrid[0:10, 0:19].astype(float)
    image = skimage.exposure.rescale_intensity(i + j * 5.0)
    expected = skimage.transform.resize(image, (19, 10), order=1, mode="symmetric")
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.specific_width.value = 10
    module.specific_height.value = 19
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_array_almost_equal(result, expected)


def test_reshape_using_another_images_dimensions():
    """'Resize to another image's dimensions"""
    i, j = numpy.mgrid[0:10, 0:19].astype(float)
    image = skimage.exposure.rescale_intensity(1.0 * i + j)
    expected = skimage.transform.resize(image, (19, 10), order=1, mode="symmetric")
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE
    module.specific_image.value = "AnotherImage"
    workspace.image_set.add(
        module.specific_image.value, cellprofiler_core.image.Image(expected)
    )
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    assert expected.shape == result.shape


def test_resize_with_cropping():
    # This is a regression test for issue # 967
    r = numpy.random.RandomState()
    r.seed(501)
    i, j = numpy.mgrid[0:10, 0:20]
    image = i + j
    mask = r.uniform(size=image.shape) > 0.5
    imask = mask.astype(int)
    cropping = numpy.zeros((30, 40), bool)
    cropping[10:20, 10:30] = True
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_BILINEAR,
        mask,
        cropping,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 0.5
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert tuple(result.mask.shape) == (5, 10)
    assert tuple(result.crop_mask.shape) == (5, 10)
    x = result.crop_image_similarly(numpy.zeros(result.crop_mask.shape))
    assert tuple(x.shape) == (5, 10)


def test_resize_with_cropping_bigger():
    # This is a regression test for issue # 967
    r = numpy.random.RandomState()
    r.seed(501)
    i, j = numpy.mgrid[0:10, 0:20]
    image = i + j
    mask = r.uniform(size=image.shape) > 0.5
    imask = mask.astype(int)
    cropping = numpy.zeros((30, 40), bool)
    cropping[10:20, 10:30] = True
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_BILINEAR,
        mask,
        cropping,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 2
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert tuple(result.mask.shape) == (20, 40)
    assert tuple(result.crop_mask.shape) == (20, 40)
    x = result.crop_image_similarly(numpy.zeros(result.crop_mask.shape))
    assert tuple(x.shape) == (20, 40)


def test_resize_color():
    # Regression test of issue #1416
    image = numpy.zeros((20, 22, 3))
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.resizing_factor_x.value = module.resizing_factor_y.value = 0.5
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert tuple(result.pixel_data.shape) == (10, 11, 3)


def test_resize_color_bw():
    # Regression test of issue #1416
    image = numpy.zeros((20, 22, 3))
    tgt_image = numpy.zeros((5, 11))
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE
    module.specific_image.value = "AnotherImage"
    workspace.image_set.add(
        module.specific_image.value, cellprofiler_core.image.Image(tgt_image)
    )
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert tuple(result.pixel_data.shape) == (5, 11, 3)


def test_resize_color_color():
    # Regression test of issue #1416
    image = numpy.zeros((20, 22, 3))
    tgt_image = numpy.zeros((10, 11, 3))
    workspace, module = make_workspace(
        image,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_BILINEAR,
    )
    module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE
    module.specific_image.value = "AnotherImage"
    workspace.image_set.add(
        module.specific_image.value, cellprofiler_core.image.Image(tgt_image)
    )
    module.run(workspace)
    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert tuple(result.pixel_data.shape) == (10, 11, 3)


def test_resize_volume_factor_grayscale():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    mask = data > 0.5

    crop_mask = numpy.zeros_like(data, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.resizing_factor_x.value = module.resizing_factor_y.value = 0.5

    module.resizing_factor_z.value = 1

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected_data = skimage.transform.resize(
        data, (10, 5, 5), order=0, mode="symmetric"
    )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 5, 5), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 5, 5), order=0, mode="constant")
    )

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_factor_color():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10, 3)

    mask = data[:, :, :, 0] > 0.5

    crop_mask = numpy.zeros_like(mask, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.resizing_factor_x.value = module.resizing_factor_y.value = 2.5

    module.resizing_factor_z.value = 1

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected_data = numpy.zeros((10, 25, 25, 3), dtype=data.dtype)

    for idx in range(3):
        expected_data[:, :, :, idx] = skimage.transform.resize(
            data[:, :, :, idx], (10, 25, 25), order=0, mode="symmetric"
        )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 25, 25), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 25, 25), order=0, mode="constant")
    )

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_factor_grayscale_resize_z_even():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    mask = data > 0.5

    crop_mask = numpy.zeros_like(data, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.resizing_factor_x.value = module.resizing_factor_y.value = 0.5

    module.resizing_factor_z.value = 0.5

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected_data = skimage.transform.resize(
        data, (5, 5, 5), order=0, mode="symmetric"
    )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (5, 5, 5), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (5, 5, 5), order=0, mode="constant")
    )

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_factor_grayscale_resize_z_odd():
    numpy.random.seed(73)

    data = numpy.random.rand(9, 10, 10)

    mask = data > 0.5

    crop_mask = numpy.zeros_like(data, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.resizing_factor_x.value = module.resizing_factor_y.value = 0.5

    module.resizing_factor_z.value = 0.5

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected_data = skimage.transform.resize(
        data, (4, 5, 5), order=0, mode="symmetric"
    )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (4, 5, 5), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (4, 5, 5), order=0, mode="constant")
    )

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)

def test_resize_volume_factor_color_resize_z():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10, 3)

    mask = data[:, :, :, 0] > 0.5

    crop_mask = numpy.zeros_like(mask, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.resizing_factor_x.value = module.resizing_factor_y.value = 2.5

    module.resizing_factor_z.value = 2.5

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected_data = numpy.zeros((25, 25, 25, 3), dtype=data.dtype)

    for idx in range(3):
        expected_data[:, :, :, idx] = skimage.transform.resize(
            data[:, :, :, idx], (25, 25, 25), order=0, mode="symmetric"
        )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (25, 25, 25), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (25, 25, 25), order=0, mode="constant")
    )

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_manual_grayscale():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    mask = data > 0.5

    crop_mask = numpy.zeros_like(data, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.use_manual_or_image.value = cellprofiler.modules.resize.C_MANUAL

    module.specific_width.value = 25

    module.specific_height.value = 30

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected_data = skimage.transform.resize(
        data, (10, 30, 25), order=0, mode="symmetric"
    )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 30, 25), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 30, 25), order=0, mode="constant")
    )

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_manual_color():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10, 3)

    mask = data[:, :, :, 0] > 0.5

    crop_mask = numpy.zeros_like(mask, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.use_manual_or_image.value = cellprofiler.modules.resize.C_MANUAL

    module.specific_width.value = 5

    module.specific_height.value = 8

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected_data = numpy.zeros((10, 8, 5, 3), dtype=data.dtype)

    for idx in range(3):
        expected_data[:, :, :, idx] = skimage.transform.resize(
            data[:, :, :, idx], (10, 8, 5), order=0, mode="symmetric"
        )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 8, 5), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 8, 5), order=0, mode="constant")
    )

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_grayscale_other_volume_grayscale():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    mask = data > 0.5

    crop_mask = numpy.zeros_like(data, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

    module.specific_image.value = "Other Image"

    expected_data = skimage.transform.resize(
        data, (10, 30, 25), order=0, mode="symmetric"
    )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 30, 25), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 30, 25), order=0, mode="constant")
    )

    other_image = cellprofiler_core.image.Image(
        expected_data, mask=expected_mask, crop_mask=expected_crop_mask, dimensions=3
    )

    workspace.image_set.add(module.specific_image.value, other_image)

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_grayscale_other_volume_color():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    mask = data > 0.5

    crop_mask = numpy.zeros_like(data, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

    module.specific_image.value = "Other Image"

    expected_data = skimage.transform.resize(
        data, (10, 30, 25), order=0, mode="symmetric"
    )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 30, 25), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 30, 25), order=0, mode="constant")
    )

    other_image = cellprofiler_core.image.Image(
        numpy.random.rand(10, 30, 25, 3),
        mask=expected_mask,
        crop_mask=expected_crop_mask,
        dimensions=3,
    )

    workspace.image_set.add(module.specific_image.value, other_image)

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_color_other_volume_grayscale():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10, 3)

    mask = data[:, :, :, 0] > 0.5

    crop_mask = numpy.zeros_like(mask, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

    module.specific_image.value = "Other Image"

    expected_data = numpy.zeros((10, 30, 25, 3), dtype=data.dtype)

    for idx in range(3):
        expected_data[:, :, :, idx] = skimage.transform.resize(
            data[:, :, :, idx], (10, 30, 25), order=0, mode="symmetric"
        )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 30, 25), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 30, 25), order=0, mode="constant")
    )

    other_image = cellprofiler_core.image.Image(
        expected_data, mask=expected_mask, crop_mask=expected_crop_mask, dimensions=3
    )

    workspace.image_set.add(module.specific_image.value, other_image)

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


def test_resize_volume_color_other_volume_color():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10, 3)

    mask = data[:, :, :, 0] > 0.5

    crop_mask = numpy.zeros_like(mask, dtype=bool)

    crop_mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_TO_SIZE,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
        mask=mask,
        cropping=crop_mask,
        dimensions=3,
    )

    module.use_manual_or_image.value = cellprofiler.modules.resize.C_IMAGE

    module.specific_image.value = "Other Image"

    expected_data = numpy.zeros((10, 30, 25, 3), dtype=data.dtype)

    for idx in range(3):
        expected_data[:, :, :, idx] = skimage.transform.resize(
            data[:, :, :, idx], (10, 30, 25), order=0, mode="symmetric"
        )

    expected_mask = skimage.img_as_bool(
        skimage.transform.resize(mask, (10, 30, 25), order=0, mode="constant")
    )

    expected_crop_mask = skimage.img_as_bool(
        skimage.transform.resize(crop_mask, (10, 30, 25), order=0, mode="constant")
    )

    other_image = cellprofiler_core.image.Image(
        numpy.random.rand(10, 30, 25, 3),
        mask=expected_mask,
        crop_mask=expected_crop_mask,
        dimensions=3,
    )

    workspace.image_set.add(module.specific_image.value, other_image)

    module.run(workspace)

    actual = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    assert actual.volumetric

    numpy.testing.assert_array_almost_equal(actual.pixel_data, expected_data)

    numpy.testing.assert_array_almost_equal(actual.mask, expected_mask)

    numpy.testing.assert_array_almost_equal(actual.crop_mask, expected_crop_mask)


# https://github.com/CellProfiler/CellProfiler/issues/3080
def test_resize_factor_rounding():
    data = numpy.zeros((99, 99))

    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
    )

    module.resizing_factor_x.value = module.resizing_factor_y.value = 0.25

    module.run(workspace)

    assert workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data.shape == (25, 25)


# https://github.com/CellProfiler/CellProfiler/issues/3531
def test_resize_float():
    data = numpy.ones((10, 10), dtype=numpy.float32) * 2
    expected = numpy.ones((5, 5), dtype=numpy.float32) * 2
    workspace, module = make_workspace(
        data,
        cellprofiler.modules.resize.R_BY_FACTOR,
        cellprofiler.modules.resize.I_NEAREST_NEIGHBOR,
    )

    module.resizing_factor_x.value = module.resizing_factor_y.value = 0.5

    module.run(workspace)

    result = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    numpy.testing.assert_allclose(result, expected)
