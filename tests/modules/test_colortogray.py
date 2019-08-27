import numpy
import six.moves

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.colortogray
import cellprofiler.modules.injectimage
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace
import tests.modules

IMAGE_NAME = "image"
OUTPUT_IMAGE_F = "outputimage%d"


def get_my_image():
    """A color image with red in the upper left, green in the lower left and blue in the upper right"""
    img = numpy.zeros((50, 50, 3))
    img[0:25, 0:25, 0] = 1
    img[0:25, 25:50, 1] = 1
    img[25:50, 0:25, 2] = 1
    return img


def test_init():
    x = cellprofiler.modules.colortogray.ColorToGray()


def test_combine():
    img = get_my_image()
    inj = cellprofiler.modules.injectimage.InjectImage("my_image", img)
    inj.set_module_num(1)
    ctg = cellprofiler.modules.colortogray.ColorToGray()
    ctg.set_module_num(2)
    ctg.image_name.value = "my_image"
    ctg.combine_or_split.value = cellprofiler.modules.colortogray.COMBINE
    ctg.red_contribution.value = 1
    ctg.green_contribution.value = 2
    ctg.blue_contribution.value = 3
    ctg.grayscale_name.value = "my_grayscale"
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(inj)
    pipeline.add_module(ctg)
    pipeline.test_valid()

    measurements = cellprofiler.measurement.Measurements()
    object_set = cellprofiler.object.ObjectSet()
    image_set_list = cellprofiler.image.ImageSetList()
    workspace = cellprofiler.workspace.Workspace(
        pipeline, inj, None, None, measurements, image_set_list, None
    )
    inj.prepare_run(workspace)
    inj.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    inj.run(
        cellprofiler.workspace.Workspace(
            pipeline, inj, image_set, object_set, measurements, None
        )
    )
    ctg.run(
        cellprofiler.workspace.Workspace(
            pipeline, ctg, image_set, object_set, measurements, None
        )
    )
    grayscale = image_set.get_image("my_grayscale")
    assert grayscale
    img = grayscale.image
    numpy.testing.assert_almost_equal(img[0, 0], 1.0 / 6.0)
    numpy.testing.assert_almost_equal(img[0, 25], 1.0 / 3.0)
    numpy.testing.assert_almost_equal(img[25, 0], 1.0 / 2.0)
    numpy.testing.assert_almost_equal(img[25, 25], 0)


def test_split_all():
    img = get_my_image()
    inj = cellprofiler.modules.injectimage.InjectImage("my_image", img)
    inj.set_module_num(1)
    ctg = cellprofiler.modules.colortogray.ColorToGray()
    ctg.set_module_num(2)
    ctg.image_name.value = "my_image"
    ctg.combine_or_split.value = cellprofiler.modules.colortogray.SPLIT
    ctg.use_red.value = True
    ctg.use_blue.value = True
    ctg.use_green.value = True
    ctg.red_name.value = "my_red"
    ctg.green_name.value = "my_green"
    ctg.blue_name.value = "my_blue"
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.add_module(inj)
    pipeline.add_module(ctg)
    pipeline.test_valid()

    measurements = cellprofiler.measurement.Measurements()
    object_set = cellprofiler.object.ObjectSet()
    image_set_list = cellprofiler.image.ImageSetList()
    workspace = cellprofiler.workspace.Workspace(
        pipeline, inj, None, None, measurements, image_set_list, None
    )
    inj.prepare_run(workspace)
    inj.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    inj.run(
        cellprofiler.workspace.Workspace(
            pipeline, inj, image_set, object_set, measurements, None
        )
    )
    ctg.run(
        cellprofiler.workspace.Workspace(
            pipeline, ctg, image_set, object_set, measurements, None
        )
    )
    red = image_set.get_image("my_red")
    assert red
    img = red.image
    numpy.testing.assert_almost_equal(img[0, 0], 1)
    numpy.testing.assert_almost_equal(img[0, 25], 0)
    numpy.testing.assert_almost_equal(img[25, 0], 0)
    numpy.testing.assert_almost_equal(img[25, 25], 0)
    green = image_set.get_image("my_green")
    assert green
    img = green.image
    numpy.testing.assert_almost_equal(img[0, 0], 0)
    numpy.testing.assert_almost_equal(img[0, 25], 1)
    numpy.testing.assert_almost_equal(img[25, 0], 0)
    numpy.testing.assert_almost_equal(img[25, 25], 0)
    blue = image_set.get_image("my_blue")
    assert blue
    img = blue.image
    numpy.testing.assert_almost_equal(img[0, 0], 0)
    numpy.testing.assert_almost_equal(img[0, 25], 0)
    numpy.testing.assert_almost_equal(img[25, 0], 1)
    numpy.testing.assert_almost_equal(img[25, 25], 0)


def test_combine_channels():
    numpy.random.seed(13)
    image = numpy.random.uniform(size=(20, 10, 5))
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, cellprofiler.image.Image(image))

    module = cellprofiler.modules.colortogray.ColorToGray()
    module.set_module_num(1)
    module.image_name.value = IMAGE_NAME
    module.combine_or_split.value = cellprofiler.modules.colortogray.COMBINE
    module.grayscale_name.value = OUTPUT_IMAGE_F % 1
    module.rgb_or_channels.value = cellprofiler.modules.colortogray.CH_CHANNELS
    module.add_channel()
    module.add_channel()

    channel_indexes = numpy.array([2, 0, 3])
    factors = numpy.random.uniform(size=3)
    divisor = numpy.sum(factors)
    expected = numpy.zeros((20, 10))
    for i, channel_index in enumerate(channel_indexes):
        module.channels[i].channel_choice.value = channel_index + 1
        module.channels[i].contribution.value_text = "%.10f" % factors[i]
        expected += image[:, :, channel_index] * factors[i] / divisor

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler.object.ObjectSet(),
        cellprofiler.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    pixels = image_set.get_image(module.grayscale_name.value).pixel_data
    assert pixels.ndim == 2
    assert tuple(pixels.shape) == (20, 10)
    numpy.testing.assert_almost_equal(expected, pixels)


def test_split_channels():
    numpy.random.seed(13)
    image = numpy.random.uniform(size=(20, 10, 5))
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, cellprofiler.image.Image(image))

    module = cellprofiler.modules.colortogray.ColorToGray()
    module.set_module_num(1)
    module.image_name.value = IMAGE_NAME
    module.combine_or_split.value = cellprofiler.modules.colortogray.SPLIT
    module.rgb_or_channels.value = cellprofiler.modules.colortogray.CH_CHANNELS
    module.add_channel()
    module.add_channel()
    module.add_channel()
    module.add_channel()

    channel_indexes = numpy.array([1, 4, 2])
    for i, channel_index in enumerate(channel_indexes):
        module.channels[i].channel_choice.value = channel_index + 1
        module.channels[i].image_name.value = OUTPUT_IMAGE_F % i

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler.object.ObjectSet(),
        cellprofiler.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    for i, channel_index in enumerate(channel_indexes):
        pixels = image_set.get_image(module.channels[i].image_name.value).pixel_data
        assert pixels.ndim == 2
        assert tuple(pixels.shape) == (20, 10)
        numpy.testing.assert_almost_equal(image[:, :, channel_index], pixels)


def test_load_combine():
    data = "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"
    pipeline = tests.modules.load_pipeline(None, data)
    module = pipeline.module(1)
    assert module.image_name.value == "TestInput"
    assert module.should_combine
    assert module.grayscale_name.value == "TestGray"
    assert module.red_contribution.value == 1
    assert module.green_contribution.value == 2
    assert module.blue_contribution.value == 3


def test_load_split():
    data = "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"
    pipeline = tests.modules.load_pipeline(None, data)
    module = pipeline.module(1)
    assert isinstance(module, cellprofiler.modules.colortogray.ColorToGray)
    assert module.image_name.value == "TestInput"
    assert module.should_split
    assert module.use_red.value
    assert module.red_name.value == "TestRed"
    assert module.use_green.value
    assert module.green_name.value == "TestGreen"
    assert not module.use_blue.value
    assert module.rgb_or_channels == cellprofiler.modules.colortogray.CH_RGB


def test_load_v2():
    with open("./tests/resources/modules/colortogray/v2.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.colortogray.ColorToGray)
    assert module.image_name == "DNA"
    assert module.combine_or_split == cellprofiler.modules.colortogray.COMBINE
    assert module.rgb_or_channels == cellprofiler.modules.colortogray.CH_CHANNELS
    assert module.grayscale_name == "OrigGrayw"
    assert module.red_contribution == 1
    assert module.green_contribution == 3
    assert module.blue_contribution == 5
    assert module.use_red
    assert module.use_green
    assert module.use_blue
    assert module.red_name == "OrigRedx"
    assert module.green_name == "OrigGreeny"
    assert module.blue_name == "OrigBluez"
    assert module.channel_count.value == 3
    assert module.channels[0].channel_choice.value == 1
    assert module.channels[1].channel_choice.value == 3
    assert module.channels[2].channel_choice.value == 2
    assert module.channels[0].contribution == 1
    assert module.channels[1].contribution == 2
    assert module.channels[2].contribution == 3
    assert module.channels[0].image_name == "RedChannel1"
    assert module.channels[1].image_name == "GreenChannel2"
    assert module.channels[2].image_name == "BlueChannel3"


def test_load_v3():
    """
    Tests a pipeline that was produced with module revision 3.
    The channel names are named according to the schema:
    Channel(#new_imagenumber)_(#channel_number),
    e.g. Channel3_2 would be the image number 3 that contains channel
    number 2.
    Thus it can be easily checked via the new image name, if the channel
    number is correctly parsed.
    """
    with open("./tests/resources/modules/colortogray/v3.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.colortogray.ColorToGray)
    assert module.image_name == "DNA"
    assert module.channel_count.value == 8
    for i in range(module.channel_count.value):
        c = module.channels[i].image_name.value.split("_")[1]
        assert module.channels[i].channel_choice.value == int(c)
    assert module.channels[6].image_name.value == "Channel7_7"
