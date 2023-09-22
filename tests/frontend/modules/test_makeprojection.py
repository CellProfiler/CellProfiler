import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement


import cellprofiler.modules.makeprojection
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

IMAGE_NAME = "image"
PROJECTED_IMAGE_NAME = "projectedimage"


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("makeprojection/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.load(six.moves.StringIO(data))
    methods = (
        cellprofiler.modules.makeprojection.P_AVERAGE,
        cellprofiler.modules.makeprojection.P_MAXIMUM,
        cellprofiler.modules.makeprojection.P_MINIMUM,
        cellprofiler.modules.makeprojection.P_SUM,
        cellprofiler.modules.makeprojection.P_VARIANCE,
        cellprofiler.modules.makeprojection.P_POWER,
        cellprofiler.modules.makeprojection.P_BRIGHTFIELD,
    )
    assert len(pipeline.modules()) == len(methods)
    for method, module in zip(methods, pipeline.modules()):
        assert isinstance(module, cellprofiler.modules.makeprojection.MakeProjection)
        assert module.image_name == "ch02"
        assert module.projection_type == method
        assert module.projection_image_name == "ProjectionCh00Scale6"
        assert module.frequency == 6


def run_image_set(projection_type, images_and_masks, frequency=9, run_last=True):
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_count = len(images_and_masks)
    for i in range(image_count):
        pixel_data, mask = images_and_masks[i]
        if mask is None:
            image = cellprofiler_core.image.Image(pixel_data)
        else:
            image = cellprofiler_core.image.Image(pixel_data, mask)
        image_set_list.get_image_set(i).add(IMAGE_NAME, image)
    #
    # Add bogus image at end for 2nd group
    #
    bogus_image = cellprofiler_core.image.Image(numpy.zeros((10, 20)))
    image_set_list.get_image_set(image_count).add(IMAGE_NAME, bogus_image)

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.makeprojection.MakeProjection()
    module.set_module_num(1)
    module.image_name.value = IMAGE_NAME
    module.projection_image_name.value = PROJECTED_IMAGE_NAME
    module.projection_type.value = projection_type
    module.frequency.value = frequency
    pipeline.add_module(module)
    m = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, None, None, m, image_set_list
    )
    module.prepare_run(workspace)
    module.prepare_group(workspace, {}, list(range(1, len(images_and_masks) + 1)))
    for i in range(image_count):
        if i > 0:
            image_set_list.purge_image_set(i - 1)
        w = cellprofiler_core.workspace.Workspace(
            pipeline,
            module,
            image_set_list.get_image_set(i),
            cellprofiler_core.object.ObjectSet(),
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
    w = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        m,
        image_set_list,
    )
    module.run(w)
    image_provider = image_set.get_image_provider(PROJECTED_IMAGE_NAME)
    assert numpy.max(image_provider.count) == 1

    return image


def test_average():
    numpy.random.seed(0)
    images_and_masks = [
        (numpy.random.uniform(size=(10, 10)).astype(numpy.float32), None)
        for i in range(3)
    ]
    expected = numpy.zeros((10, 10), numpy.float32)
    for image, mask in images_and_masks:
        expected += image
    expected = expected / len(images_and_masks)
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_AVERAGE, images_and_masks
    )
    assert not image.has_mask
    assert numpy.all(numpy.abs(image.pixel_data - expected) < numpy.finfo(float).eps)


def test_average_mask():
    numpy.random.seed(0)
    images_and_masks = [
        (
            numpy.random.uniform(size=(100, 100)).astype(numpy.float32),
            numpy.random.uniform(size=(100, 100)) > 0.3,
        )
        for i in range(3)
    ]
    expected = numpy.zeros((100, 100), numpy.float32)
    expected_count = numpy.zeros((100, 100), numpy.float32)
    expected_mask = numpy.zeros((100, 100), bool)
    for image, mask in images_and_masks:
        expected[mask] += image[mask]
        expected_count[mask] += 1
        expected_mask = mask | expected_mask
    expected = expected / expected_count
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_AVERAGE, images_and_masks
    )
    assert image.has_mask
    assert numpy.all(expected_mask == image.mask)
    numpy.testing.assert_almost_equal(
        image.pixel_data[image.mask], expected[expected_mask]
    )


def test_average_color():
    numpy.random.seed(0)
    images_and_masks = [
        (numpy.random.uniform(size=(10, 10, 3)).astype(numpy.float32), None)
        for i in range(3)
    ]
    expected = numpy.zeros((10, 10, 3), numpy.float32)
    for image, mask in images_and_masks:
        expected += image
    expected = expected / len(images_and_masks)
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_AVERAGE, images_and_masks
    )
    assert not image.has_mask
    assert numpy.all(numpy.abs(image.pixel_data - expected) < numpy.finfo(float).eps)


def test_average_masked_color():
    numpy.random.seed(0)
    images_and_masks = [
        (
            numpy.random.uniform(size=(10, 10, 3)).astype(numpy.float32),
            numpy.random.uniform(size=(10, 10)) > 0.3,
        )
        for i in range(3)
    ]
    expected = numpy.zeros((10, 10, 3))
    expected_count = numpy.zeros((10, 10), numpy.float32)
    expected_mask = numpy.zeros((10, 10), bool)
    for image, mask in images_and_masks:
        expected[mask, :] += image[mask, :]
        expected_count[mask] += 1
        expected_mask = mask | expected_mask
    expected = expected / expected_count[:, :, numpy.newaxis]
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_AVERAGE, images_and_masks
    )
    assert image.has_mask
    numpy.testing.assert_equal(image.mask, expected_mask)
    numpy.testing.assert_almost_equal(
        image.pixel_data[expected_mask], expected[expected_mask]
    )


def test_maximum():
    numpy.random.seed(0)
    images_and_masks = [
        (numpy.random.uniform(size=(10, 10)).astype(numpy.float32), None)
        for i in range(3)
    ]
    expected = numpy.zeros((10, 10), numpy.float32)
    for image, mask in images_and_masks:
        expected = numpy.maximum(expected, image)
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_MAXIMUM, images_and_masks
    )
    assert not image.has_mask
    assert numpy.all(numpy.abs(image.pixel_data - expected) < numpy.finfo(float).eps)


def test_maximum_mask():
    numpy.random.seed(0)
    images_and_masks = [
        (
            numpy.random.uniform(size=(100, 100)).astype(numpy.float32),
            numpy.random.uniform(size=(100, 100)) > 0.3,
        )
        for i in range(3)
    ]
    expected = numpy.zeros((100, 100), numpy.float32)
    expected_mask = numpy.zeros((100, 100), bool)
    for image, mask in images_and_masks:
        expected[mask] = numpy.maximum(expected[mask], image[mask])
        expected_mask = mask | expected_mask
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_MAXIMUM, images_and_masks
    )
    assert image.has_mask
    assert numpy.all(expected_mask == image.mask)
    assert numpy.all(
        numpy.abs(image.pixel_data[image.mask] - expected[expected_mask])
        < numpy.finfo(float).eps
    )


def test_maximum_color():
    numpy.random.seed(0)
    images_and_masks = [
        (numpy.random.uniform(size=(10, 10, 3)).astype(numpy.float32), None)
        for i in range(3)
    ]
    expected = numpy.zeros((10, 10, 3), numpy.float32)
    for image, mask in images_and_masks:
        expected = numpy.maximum(expected, image)
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_MAXIMUM, images_and_masks
    )
    assert not image.has_mask
    assert numpy.all(numpy.abs(image.pixel_data - expected) < numpy.finfo(float).eps)


def test_variance():
    numpy.random.seed(41)
    images_and_masks = [
        (numpy.random.uniform(size=(20, 10)).astype(numpy.float32), None)
        for i in range(10)
    ]
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_VARIANCE, images_and_masks
    )
    images = numpy.array([x[0] for x in images_and_masks])
    x = numpy.sum(images, 0)
    x2 = numpy.sum(images ** 2, 0)
    expected = x2 / 10.0 - x ** 2 / 100.0
    numpy.testing.assert_almost_equal(image.pixel_data, expected, 4)


def test_power():
    image = numpy.ones((20, 10))
    images_and_masks = [(image.copy(), None) for i in range(9)]
    for i, (img, _) in enumerate(images_and_masks):
        img[5, 5] *= numpy.sin(2 * numpy.pi * float(i) / 9.0)
    image_out = run_image_set(
        cellprofiler.modules.makeprojection.P_POWER, images_and_masks, frequency=9
    )
    i, j = numpy.mgrid[: image.shape[0], : image.shape[1]]
    numpy.testing.assert_almost_equal(image_out.pixel_data[(i != 5) & (j != 5)], 0)
    assert image_out.pixel_data[5, 5] > 1


def test_brightfield():
    image = numpy.ones((20, 10))
    images_and_masks = [(image.copy(), None) for i in range(9)]
    for i, (img, _) in enumerate(images_and_masks):
        if i < 5:
            img[:5, :5] = 0
        else:
            img[:5, 5:] = 0
    image_out = run_image_set(
        cellprofiler.modules.makeprojection.P_BRIGHTFIELD, images_and_masks
    )
    i, j = numpy.mgrid[: image.shape[0], : image.shape[1]]
    numpy.testing.assert_almost_equal(image_out.pixel_data[(i > 5) | (j < 5)], 0)
    numpy.testing.assert_almost_equal(image_out.pixel_data[(i < 5) & (j >= 5)], 1)


def test_minimum():
    numpy.random.seed(0)
    images_and_masks = [
        (numpy.random.uniform(size=(10, 10)).astype(numpy.float32), None)
        for i in range(3)
    ]
    expected = numpy.ones((10, 10), numpy.float32)
    for image, mask in images_and_masks:
        expected = numpy.minimum(expected, image)
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_MINIMUM, images_and_masks
    )
    assert not image.has_mask
    assert numpy.all(numpy.abs(image.pixel_data - expected) < numpy.finfo(float).eps)


def test_minimum_mask():
    numpy.random.seed(72)
    images_and_masks = [
        (
            numpy.random.uniform(size=(100, 100)).astype(numpy.float32),
            numpy.random.uniform(size=(100, 100)) > 0.3,
        )
        for i in range(3)
    ]
    expected = numpy.ones((100, 100), numpy.float32)
    expected_mask = numpy.zeros((100, 100), bool)
    for image, mask in images_and_masks:
        expected[mask] = numpy.minimum(expected[mask], image[mask])
        expected_mask = mask | expected_mask
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_MINIMUM, images_and_masks
    )
    assert image.has_mask
    assert numpy.any(image.mask == False)
    assert numpy.all(expected_mask == image.mask)
    assert numpy.all(
        numpy.abs(image.pixel_data[image.mask] - expected[expected_mask])
        < numpy.finfo(float).eps
    )
    assert numpy.all(image.pixel_data[~image.mask] == 0)


def test_minimum_color():
    numpy.random.seed(0)
    images_and_masks = [
        (numpy.random.uniform(size=(10, 10, 3)).astype(numpy.float32), None)
        for i in range(3)
    ]
    expected = numpy.ones((10, 10, 3), numpy.float32)
    for image, mask in images_and_masks:
        expected = numpy.minimum(expected, image)
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_MINIMUM, images_and_masks
    )
    assert not image.has_mask
    assert numpy.all(numpy.abs(image.pixel_data - expected) < numpy.finfo(float).eps)


def test_mask_unmasked():
    numpy.random.seed(81)
    images_and_masks = [(numpy.random.uniform(size=(10, 10)), None) for i in range(3)]
    image = run_image_set(cellprofiler.modules.makeprojection.P_MASK, images_and_masks)
    assert tuple(image.pixel_data.shape) == (10, 10)
    assert numpy.all(image.pixel_data == True)
    assert not image.has_mask


def test_mask():
    numpy.random.seed(81)
    images_and_masks = [
        (numpy.random.uniform(size=(10, 10)), numpy.random.uniform(size=(10, 10)) > 0.3)
        for i in range(3)
    ]
    expected = numpy.ones((10, 10), bool)
    for _, mask in images_and_masks:
        expected = expected & mask
    image = run_image_set(cellprofiler.modules.makeprojection.P_MASK, images_and_masks)
    assert numpy.all(image.pixel_data == expected)


def test_filtered():
    """Make sure the image shows up in the image set even if filtered

    This is similar to issue # 310 - the last image may be filtered before
    the projection is done and the aggregate image is then missing
    from the image set.
    """
    numpy.random.seed(81)
    images_and_masks = [(numpy.random.uniform(size=(10, 10)), None) for i in range(3)]
    image = run_image_set(
        cellprofiler.modules.makeprojection.P_AVERAGE, images_and_masks, run_last=False
    )
    numpy.testing.assert_array_almost_equal(
        image.pixel_data, (images_and_masks[0][0] + images_and_masks[1][0]) / 2
    )
