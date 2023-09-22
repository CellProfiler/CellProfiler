import numpy
import six.moves
import os

import cellprofiler_core.image
import cellprofiler_core.measurement


import cellprofiler.modules.graytocolor
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

OUTPUT_IMAGE_NAME = "outputimage"


def make_workspace(scheme, images, adjustments=None, colors=None, weights=None):
    module = cellprofiler.modules.graytocolor.GrayToColor()
    module.scheme_choice.value = scheme
    module.wants_rescale.value = False
    if scheme not in (
        cellprofiler.modules.graytocolor.SCHEME_COMPOSITE,
        cellprofiler.modules.graytocolor.SCHEME_STACK,
    ):
        image_names = [
            "image%d" % i
            if images[i] is not None
            else cellprofiler.modules.graytocolor.LEAVE_THIS_BLACK
            for i in range(7)
        ]
        for image_name_setting, image_name, adjustment_setting, adjustment in zip(
            (
                module.red_image_name,
                module.green_image_name,
                module.blue_image_name,
                module.cyan_image_name,
                module.magenta_image_name,
                module.yellow_image_name,
                module.gray_image_name,
            ),
            image_names,
            (
                module.red_adjustment_factor,
                module.green_adjustment_factor,
                module.blue_adjustment_factor,
                module.cyan_adjustment_factor,
                module.magenta_adjustment_factor,
                module.yellow_adjustment_factor,
                module.gray_adjustment_factor,
            ),
            adjustments,
        ):
            image_name_setting.value = image_name
            adjustment_setting.value = adjustment
    else:
        while len(module.stack_channels) < len(images):
            module.add_stack_channel_cb()
        image_names = []
        if weights is None:
            weights = [1.0] * len(images)
        if colors is None:
            colors = [
                cellprofiler.modules.graytocolor.DEFAULT_COLORS[
                    i % len(cellprofiler.modules.graytocolor.DEFAULT_COLORS)
                ]
                for i in range(len(images))
            ]
        for i, (image, color, weight) in enumerate(zip(images, colors, weights)):
            image_name = "image%d" % (i + 1)
            image_names.append(image_name)
            module.stack_channels[i].image_name.value = image_name
            module.stack_channels[i].color.value = color
            module.stack_channels[i].weight.value = weight

    module.rgb_image_name.value = OUTPUT_IMAGE_NAME
    module.set_module_num(1)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    for image, image_name in zip(images, image_names):
        if image is not None:
            image_set.add(image_name, cellprofiler_core.image.Image(image))
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("graytocolor/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.graytocolor.GrayToColor)
    assert module.scheme_choice == cellprofiler.modules.graytocolor.SCHEME_COMPOSITE
    assert module.rgb_image_name == "myimage"
    assert module.red_image_name == "1"
    assert module.green_image_name == "2"
    assert module.blue_image_name == "3"
    assert module.cyan_image_name == "4"
    assert module.magenta_image_name == "5"
    assert module.yellow_image_name == "6"
    assert module.gray_image_name == "7"
    assert numpy.round(numpy.abs(module.red_adjustment_factor.value - 2.1), 7) == 0
    assert numpy.round(numpy.abs(module.green_adjustment_factor.value - 2.2), 7) == 0
    assert numpy.round(numpy.abs(module.blue_adjustment_factor.value - 2.3), 7) == 0
    assert numpy.round(numpy.abs(module.cyan_adjustment_factor.value - 1.1), 7) == 0
    assert numpy.round(numpy.abs(module.magenta_adjustment_factor.value - 1.2), 7) == 0
    assert numpy.round(numpy.abs(module.yellow_adjustment_factor.value - 1.3), 7) == 0
    assert numpy.round(numpy.abs(module.gray_adjustment_factor.value - 1.4), 7) == 0
    assert len(module.stack_channels) == 2
    assert module.stack_channels[0].image_name == "DNA"
    assert module.stack_channels[1].image_name == "GFP"
    assert module.stack_channels[0].color.to_rgb() == (127, 0, 255)
    assert module.stack_channels[1].color.to_rgb() == (127, 255, 0)


def test_rgb():
    numpy.random.seed(0)
    for combination in (
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
    ):
        adjustments = numpy.random.uniform(size=7)
        images = [
            numpy.random.uniform(size=(10, 15)) if combination[i] else None
            for i in range(3)
        ]
        images += [None] * 4
        workspace, module = make_workspace(
            cellprofiler.modules.graytocolor.SCHEME_RGB, images, adjustments
        )
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data

        expected = numpy.dstack(
            [
                image if image is not None else numpy.zeros((10, 15))
                for image in images[:3]
            ]
        )
        for i in range(3):
            expected[:, :, i] *= adjustments[i]
        assert numpy.all(numpy.abs(expected - pixel_data) <= 0.00001)


def test_cmyk():
    numpy.random.seed(0)
    for combination in [[(i & 2 ^ j) != 0 for j in range(4)] for i in range(1, 16)]:
        adjustments = numpy.random.uniform(size=7)
        images = [
            numpy.random.uniform(size=(10, 15)) if combination[i] else None
            for i in range(4)
        ]
        images = [None] * 3 + images
        workspace, module = make_workspace(
            cellprofiler.modules.graytocolor.SCHEME_CMYK, images, adjustments
        )
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data

        expected = numpy.array(
            [
                numpy.dstack(
                    [image * adjustment if image is not None else numpy.zeros((10, 15))]
                    * 3
                )
                * numpy.array(multiplier)
                / numpy.sum(multiplier)
                for image, multiplier, adjustment in (
                    (images[3], (0, 1, 1), adjustments[3]),
                    (images[4], (1, 0, 1), adjustments[4]),
                    (images[5], (1, 1, 0), adjustments[5]),
                    (images[6], (1, 1, 1), adjustments[6]),
                )
            ]
        )
        expected = numpy.sum(expected, 0)
        assert numpy.all(numpy.abs(expected - pixel_data) <= 0.00001)


def test_stack():
    r = numpy.random.RandomState()
    r.seed(41)
    images = [r.uniform(size=(11, 13)) for _ in range(5)]
    workspace, module = make_workspace(
        cellprofiler.modules.graytocolor.SCHEME_STACK, images
    )
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    assert output.shape[:2] == images[0].shape
    assert output.shape[2] == len(images)
    for i, image in enumerate(images):
        numpy.testing.assert_array_almost_equal(output[:, :, i], image)


def test_composite():
    r = numpy.random.RandomState()
    r.seed(41)
    images = [r.uniform(size=(11, 13)) for _ in range(5)]
    colors = [r.randint(0, 255, size=3) for _ in range(5)]
    weights = r.uniform(low=1.0 / 255, high=1.5, size=5).tolist()
    color_names = ["#%02x%02x%02x" % tuple(color.tolist()) for color in colors]
    workspace, module = make_workspace(
        cellprofiler.modules.graytocolor.SCHEME_COMPOSITE,
        images,
        colors=color_names,
        weights=weights,
    )
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    assert output.shape[:2] == images[0].shape
    assert output.shape[2] == 3
    for i in range(3):
        channel = sum(
            [
                image * weight * float(color[i]) / 255
                for image, color, weight in zip(images, colors, weights)
            ]
        )
        numpy.testing.assert_array_almost_equal(output[:, :, i], channel)


def test_rgb_rescale():
    numpy.random.seed(0)
    for combination in (
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
    ):
        adjustments = numpy.random.uniform(size=7)
        images = [
            numpy.random.uniform(size=(10, 15)) if combination[i] else None
            for i in range(3)
        ]
        images += [None] * 4
        workspace, module = make_workspace(
            cellprofiler.modules.graytocolor.SCHEME_RGB, images, adjustments
        )
        module.wants_rescale.value = True
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data

        expected = numpy.dstack(
            [
                image if image is not None else numpy.zeros((10, 15))
                for image in images[:3]
            ]
        )
        for i in range(3):
            plane = expected[:, :, i]
            if plane.max() > 0:
                plane = plane / numpy.max(plane)
            plane *= adjustments[i]
            expected[:, :, i] = plane
        assert numpy.all(numpy.abs(expected - pixel_data) <= 0.00001)


def test_cmyk_rescale():
    numpy.random.seed(0)
    for combination in [[(i & 2 ^ j) != 0 for j in range(4)] for i in range(1, 16)]:
        adjustments = numpy.random.uniform(size=7)
        images = [
            numpy.random.uniform(size=(10, 15)) if combination[i] else None
            for i in range(4)
        ]
        images = [None] * 3 + images
        workspace, module = make_workspace(
            cellprofiler.modules.graytocolor.SCHEME_CMYK, images, adjustments
        )
        module.wants_rescale.value = True
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data

        expected = numpy.array(
            [
                numpy.dstack(
                    [(image/numpy.max(image)) * adjustment if image is not None else numpy.zeros((10, 15))]
                    * 3
                )
                * numpy.array(multiplier)
                / numpy.sum(multiplier)
                for image, multiplier, adjustment in (
                    (images[3], (0, 1, 1), adjustments[3]),
                    (images[4], (1, 0, 1), adjustments[4]),
                    (images[5], (1, 1, 0), adjustments[5]),
                    (images[6], (1, 1, 1), adjustments[6]),
                )
            ]
        )
        expected = numpy.sum(expected, 0)
        assert numpy.all(numpy.abs(expected - pixel_data) <= 0.00001)


def test_stack_rescale():
    # Shouldn't do anything to the result, setting not supported.
    r = numpy.random.RandomState()
    r.seed(41)
    images = [r.uniform(size=(11, 13)) for _ in range(5)]
    workspace, module = make_workspace(
        cellprofiler.modules.graytocolor.SCHEME_STACK, images
    )
    module.wants_rescale.value = True
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    assert output.shape[:2] == images[0].shape
    assert output.shape[2] == len(images)
    for i, image in enumerate(images):
        numpy.testing.assert_array_almost_equal(output[:, :, i], image)


def test_composite_rescale():
    r = numpy.random.RandomState()
    r.seed(41)
    images = [r.uniform(size=(11, 13)) for _ in range(5)]
    colors = [r.randint(0, 255, size=3) for _ in range(5)]
    weights = r.uniform(low=1.0 / 255, high=1.5, size=5).tolist()
    color_names = ["#%02x%02x%02x" % tuple(color.tolist()) for color in colors]
    workspace, module = make_workspace(
        cellprofiler.modules.graytocolor.SCHEME_COMPOSITE,
        images,
        colors=color_names,
        weights=weights,
    )
    module.wants_rescale.value = True
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
    assert output.shape[:2] == images[0].shape
    assert output.shape[2] == 3
    for i in range(3):
        channel = sum(
            [
                (image / image.max()) * weight * float(color[i]) / 255
                for image, color, weight in zip(images, colors, weights)
            ]
        )
        channel = numpy.where(channel >1, 1, channel)

        numpy.testing.assert_array_almost_equal(output[:, :, i], channel)