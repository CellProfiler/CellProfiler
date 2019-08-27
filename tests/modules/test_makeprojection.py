import numpy as np
from six.moves import StringIO

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.makeprojection as M
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw

IMAGE_NAME = "image"
PROJECTED_IMAGE_NAME = "projectedimage"


def test_load_v2():
    with open("./tests/resources/modules/makeprojection/v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cpp.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cpp.LoadExceptionEvent)

    pipeline.load(StringIO(data))
    methods = (
        M.P_AVERAGE,
        M.P_MAXIMUM,
        M.P_MINIMUM,
        M.P_SUM,
        M.P_VARIANCE,
        M.P_POWER,
        M.P_BRIGHTFIELD,
    )
    assert len(pipeline.modules()) == len(methods)
    for method, module in zip(methods, pipeline.modules()):
        assert isinstance(module, M.MakeProjection)
        assert module.image_name == "ch02"
        assert module.projection_type == method
        assert module.projection_image_name == "ProjectionCh00Scale6"
        assert module.frequency == 6


def run_image_set(projection_type, images_and_masks, frequency=9, run_last=True):
    image_set_list = cpi.ImageSetList()
    image_count = len(images_and_masks)
    for i in range(image_count):
        pixel_data, mask = images_and_masks[i]
        if mask is None:
            image = cpi.Image(pixel_data)
        else:
            image = cpi.Image(pixel_data, mask)
        image_set_list.get_image_set(i).add(IMAGE_NAME, image)
    #
    # Add bogus image at end for 2nd group
    #
    bogus_image = cpi.Image(np.zeros((10, 20)))
    image_set_list.get_image_set(image_count).add(IMAGE_NAME, bogus_image)

    pipeline = cpp.Pipeline()
    module = M.MakeProjection()
    module.set_module_num(1)
    module.image_name.value = IMAGE_NAME
    module.projection_image_name.value = PROJECTED_IMAGE_NAME
    module.projection_type.value = projection_type
    module.frequency.value = frequency
    pipeline.add_module(module)
    m = cpmeas.Measurements()
    workspace = cpw.Workspace(pipeline, module, None, None, m, image_set_list)
    module.prepare_run(workspace)
    module.prepare_group(workspace, {}, list(range(1, len(images_and_masks) + 1)))
    for i in range(image_count):
        if i > 0:
            image_set_list.purge_image_set(i - 1)
        w = cpw.Workspace(
            pipeline,
            module,
            image_set_list.get_image_set(i),
            cpo.ObjectSet(),
            m,
            image_set_list,
        )
        if i < image_count - 1 or run_last:
            module.run(w)
    module.post_group(w, {})
    image = w.image_set.get_image(PROJECTED_IMAGE_NAME)
    #
    # Make sure that the image provider is reset after prepare_group
    #
    module.prepare_group(workspace, {}, [image_count + 1])
    image_set = image_set_list.get_image_set(image_count)
    w = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(), m, image_set_list)
    module.run(w)
    image_provider = image_set.get_image_provider(PROJECTED_IMAGE_NAME)
    assert np.max(image_provider.count) == 1

    return image


def test_average():
    np.random.seed(0)
    images_and_masks = [
        (np.random.uniform(size=(10, 10)).astype(np.float32), None) for i in range(3)
    ]
    expected = np.zeros((10, 10), np.float32)
    for image, mask in images_and_masks:
        expected += image
    expected = expected / len(images_and_masks)
    image = run_image_set(M.P_AVERAGE, images_and_masks)
    assert not image.has_mask
    assert np.all(np.abs(image.pixel_data - expected) < np.finfo(float).eps)


def test_average_mask():
    np.random.seed(0)
    images_and_masks = [
        (
            np.random.uniform(size=(100, 100)).astype(np.float32),
            np.random.uniform(size=(100, 100)) > 0.3,
        )
        for i in range(3)
    ]
    expected = np.zeros((100, 100), np.float32)
    expected_count = np.zeros((100, 100), np.float32)
    expected_mask = np.zeros((100, 100), bool)
    for image, mask in images_and_masks:
        expected[mask] += image[mask]
        expected_count[mask] += 1
        expected_mask = mask | expected_mask
    expected = expected / expected_count
    image = run_image_set(M.P_AVERAGE, images_and_masks)
    assert image.has_mask
    assert np.all(expected_mask == image.mask)
    np.testing.assert_almost_equal(
        image.pixel_data[image.mask], expected[expected_mask]
    )


def test_average_color():
    np.random.seed(0)
    images_and_masks = [
        (np.random.uniform(size=(10, 10, 3)).astype(np.float32), None) for i in range(3)
    ]
    expected = np.zeros((10, 10, 3), np.float32)
    for image, mask in images_and_masks:
        expected += image
    expected = expected / len(images_and_masks)
    image = run_image_set(M.P_AVERAGE, images_and_masks)
    assert not image.has_mask
    assert np.all(np.abs(image.pixel_data - expected) < np.finfo(float).eps)


def test_average_masked_color():
    np.random.seed(0)
    images_and_masks = [
        (
            np.random.uniform(size=(10, 10, 3)).astype(np.float32),
            np.random.uniform(size=(10, 10)) > 0.3,
        )
        for i in range(3)
    ]
    expected = np.zeros((10, 10, 3))
    expected_count = np.zeros((10, 10), np.float32)
    expected_mask = np.zeros((10, 10), bool)
    for image, mask in images_and_masks:
        expected[mask, :] += image[mask, :]
        expected_count[mask] += 1
        expected_mask = mask | expected_mask
    expected = expected / expected_count[:, :, np.newaxis]
    image = run_image_set(M.P_AVERAGE, images_and_masks)
    assert image.has_mask
    np.testing.assert_equal(image.mask, expected_mask)
    np.testing.assert_almost_equal(
        image.pixel_data[expected_mask], expected[expected_mask]
    )


def test_maximum():
    np.random.seed(0)
    images_and_masks = [
        (np.random.uniform(size=(10, 10)).astype(np.float32), None) for i in range(3)
    ]
    expected = np.zeros((10, 10), np.float32)
    for image, mask in images_and_masks:
        expected = np.maximum(expected, image)
    image = run_image_set(M.P_MAXIMUM, images_and_masks)
    assert not image.has_mask
    assert np.all(np.abs(image.pixel_data - expected) < np.finfo(float).eps)


def test_maximum_mask():
    np.random.seed(0)
    images_and_masks = [
        (
            np.random.uniform(size=(100, 100)).astype(np.float32),
            np.random.uniform(size=(100, 100)) > 0.3,
        )
        for i in range(3)
    ]
    expected = np.zeros((100, 100), np.float32)
    expected_mask = np.zeros((100, 100), bool)
    for image, mask in images_and_masks:
        expected[mask] = np.maximum(expected[mask], image[mask])
        expected_mask = mask | expected_mask
    image = run_image_set(M.P_MAXIMUM, images_and_masks)
    assert image.has_mask
    assert np.all(expected_mask == image.mask)
    assert np.all(
        np.abs(image.pixel_data[image.mask] - expected[expected_mask])
        < np.finfo(float).eps
    )


def test_maximum_color():
    np.random.seed(0)
    images_and_masks = [
        (np.random.uniform(size=(10, 10, 3)).astype(np.float32), None) for i in range(3)
    ]
    expected = np.zeros((10, 10, 3), np.float32)
    for image, mask in images_and_masks:
        expected = np.maximum(expected, image)
    image = run_image_set(M.P_MAXIMUM, images_and_masks)
    assert not image.has_mask
    assert np.all(np.abs(image.pixel_data - expected) < np.finfo(float).eps)


def test_variance():
    np.random.seed(41)
    images_and_masks = [
        (np.random.uniform(size=(20, 10)).astype(np.float32), None) for i in range(10)
    ]
    image = run_image_set(M.P_VARIANCE, images_and_masks)
    images = np.array([x[0] for x in images_and_masks])
    x = np.sum(images, 0)
    x2 = np.sum(images ** 2, 0)
    expected = x2 / 10.0 - x ** 2 / 100.0
    np.testing.assert_almost_equal(image.pixel_data, expected, 4)


def test_power():
    image = np.ones((20, 10))
    images_and_masks = [(image.copy(), None) for i in range(9)]
    for i, (img, _) in enumerate(images_and_masks):
        img[5, 5] *= np.sin(2 * np.pi * float(i) / 9.0)
    image_out = run_image_set(M.P_POWER, images_and_masks, frequency=9)
    i, j = np.mgrid[: image.shape[0], : image.shape[1]]
    np.testing.assert_almost_equal(image_out.pixel_data[(i != 5) & (j != 5)], 0)
    assert image_out.pixel_data[5, 5] > 1


def test_brightfield():
    image = np.ones((20, 10))
    images_and_masks = [(image.copy(), None) for i in range(9)]
    for i, (img, _) in enumerate(images_and_masks):
        if i < 5:
            img[:5, :5] = 0
        else:
            img[:5, 5:] = 0
    image_out = run_image_set(M.P_BRIGHTFIELD, images_and_masks)
    i, j = np.mgrid[: image.shape[0], : image.shape[1]]
    np.testing.assert_almost_equal(image_out.pixel_data[(i > 5) | (j < 5)], 0)
    np.testing.assert_almost_equal(image_out.pixel_data[(i < 5) & (j >= 5)], 1)


def test_minimum():
    np.random.seed(0)
    images_and_masks = [
        (np.random.uniform(size=(10, 10)).astype(np.float32), None) for i in range(3)
    ]
    expected = np.ones((10, 10), np.float32)
    for image, mask in images_and_masks:
        expected = np.minimum(expected, image)
    image = run_image_set(M.P_MINIMUM, images_and_masks)
    assert not image.has_mask
    assert np.all(np.abs(image.pixel_data - expected) < np.finfo(float).eps)


def test_minimum_mask():
    np.random.seed(72)
    images_and_masks = [
        (
            np.random.uniform(size=(100, 100)).astype(np.float32),
            np.random.uniform(size=(100, 100)) > 0.3,
        )
        for i in range(3)
    ]
    expected = np.ones((100, 100), np.float32)
    expected_mask = np.zeros((100, 100), bool)
    for image, mask in images_and_masks:
        expected[mask] = np.minimum(expected[mask], image[mask])
        expected_mask = mask | expected_mask
    image = run_image_set(M.P_MINIMUM, images_and_masks)
    assert image.has_mask
    assert np.any(image.mask == False)
    assert np.all(expected_mask == image.mask)
    assert np.all(
        np.abs(image.pixel_data[image.mask] - expected[expected_mask])
        < np.finfo(float).eps
    )
    assert np.all(image.pixel_data[~image.mask] == 0)


def test_minimum_color():
    np.random.seed(0)
    images_and_masks = [
        (np.random.uniform(size=(10, 10, 3)).astype(np.float32), None) for i in range(3)
    ]
    expected = np.ones((10, 10, 3), np.float32)
    for image, mask in images_and_masks:
        expected = np.minimum(expected, image)
    image = run_image_set(M.P_MINIMUM, images_and_masks)
    assert not image.has_mask
    assert np.all(np.abs(image.pixel_data - expected) < np.finfo(float).eps)


def test_mask_unmasked():
    np.random.seed(81)
    images_and_masks = [(np.random.uniform(size=(10, 10)), None) for i in range(3)]
    image = run_image_set(M.P_MASK, images_and_masks)
    assert tuple(image.pixel_data.shape) == (10, 10)
    assert np.all(image.pixel_data == True)
    assert not image.has_mask


def test_mask():
    np.random.seed(81)
    images_and_masks = [
        (np.random.uniform(size=(10, 10)), np.random.uniform(size=(10, 10)) > 0.3)
        for i in range(3)
    ]
    expected = np.ones((10, 10), bool)
    for _, mask in images_and_masks:
        expected = expected & mask
    image = run_image_set(M.P_MASK, images_and_masks)
    assert np.all(image.pixel_data == expected)


def test_filtered():
    """Make sure the image shows up in the image set even if filtered

    This is similar to issue # 310 - the last image may be filtered before
    the projection is done and the aggregate image is then missing
    from the image set.
    """
    np.random.seed(81)
    images_and_masks = [(np.random.uniform(size=(10, 10)), None) for i in range(3)]
    image = run_image_set(M.P_AVERAGE, images_and_masks, run_last=False)
    np.testing.assert_array_almost_equal(
        image.pixel_data, (images_and_masks[0][0] + images_and_masks[1][0]) / 2
    )
