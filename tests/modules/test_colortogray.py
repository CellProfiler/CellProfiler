"""test_colortogray.py - test the ColorToGray module
"""

from StringIO import StringIO

import numpy
from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline
import cellprofiler.measurement
import cellprofiler.image
import cellprofiler.object

import cellprofiler.modules.injectimage
import cellprofiler.modules.colortogray

import tests.modules
from cellprofiler.workspace import Workspace

IMAGE_NAME = "image"
OUTPUT_IMAGE_F = "outputimage%d"


def get_my_image():
    """A color image with red in the upper left, green in the lower left and blue in the upper right"""
    img = numpy.zeros((50, 50, 3))
    img[0:25, 0:25, 0] = 1
    img[0:25, 25:50, 1] = 1
    img[25:50, 0:25, 2] = 1
    return img


def test_00_00_init():
    x = cellprofiler.modules.colortogray.ColorToGray()


def test_01_01_combine():
    img = get_my_image()
    inj = cellprofiler.modules.injectimage.InjectImage("my_image", img)
    inj.module_num = 1
    ctg = cellprofiler.modules.colortogray.ColorToGray()
    ctg.module_num = 2
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
    workspace = Workspace(pipeline, inj, None, None, measurements,
                          image_set_list, None)
    inj.prepare_run(workspace)
    inj.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    inj.run(Workspace(pipeline, inj, image_set, object_set, measurements, None))
    ctg.run(Workspace(pipeline, ctg, image_set, object_set, measurements, None))
    grayscale = image_set.get_image("my_grayscale")
    assert grayscale
    img = grayscale.image
    numpy.testing.assert_almost_equal(img[0, 0], 1.0 / 6.0)
    numpy.testing.assert_almost_equal(img[0, 25], 1.0 / 3.0)
    numpy.testing.assert_almost_equal(img[25, 0], 1.0 / 2.0)
    numpy.testing.assert_almost_equal(img[25, 25], 0)


def test_01_02_split_all():
    img = get_my_image()
    inj = cellprofiler.modules.injectimage.InjectImage("my_image", img)
    inj.module_num = 1
    ctg = cellprofiler.modules.colortogray.ColorToGray()
    ctg.module_num = 2
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
    workspace = Workspace(pipeline, inj, None, None, measurements,
                          image_set_list, None)
    inj.prepare_run(workspace)
    inj.prepare_group(workspace, {}, [1])
    image_set = image_set_list.get_image_set(0)
    inj.run(Workspace(pipeline, inj, image_set, object_set, measurements, None))
    ctg.run(Workspace(pipeline, ctg, image_set, object_set, measurements, None))
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


def test_01_03_combine_channels():
    numpy.random.seed(13)
    image = numpy.random.uniform(size=(20, 10, 5))
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, cellprofiler.image.Image(image))

    module = cellprofiler.modules.colortogray.ColorToGray()
    module.module_num = 1
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
        module.channels[i].channel_choice.value = channel_index+1
        module.channels[i].contribution.value_text = "%.10f" % factors[i]
        expected += image[:, :, channel_index] * factors[i] / divisor

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = Workspace(pipeline, module, image_set, cellprofiler.object.ObjectSet(),
                          cellprofiler.measurement.Measurements(), image_set_list)
    module.run(workspace)
    pixels = image_set.get_image(module.grayscale_name.value).pixel_data
    assert pixels.ndim == 2
    assert tuple(pixels.shape) == (20, 10)
    numpy.testing.assert_almost_equal(expected, pixels)


def test_01_04_split_channels():
    numpy.random.seed(13)
    image = numpy.random.uniform(size=(20, 10, 5))
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, cellprofiler.image.Image(image))

    module = cellprofiler.modules.colortogray.ColorToGray()
    module.module_num = 1
    module.image_name.value = IMAGE_NAME
    module.combine_or_split.value = cellprofiler.modules.colortogray.SPLIT
    module.rgb_or_channels.value = cellprofiler.modules.colortogray.CH_CHANNELS
    module.add_channel()
    module.add_channel()
    module.add_channel()
    module.add_channel()

    channel_indexes = numpy.array([1, 4, 2])
    for i, channel_index in enumerate(channel_indexes):
        module.channels[i].channel_choice.value = channel_index+1
        module.channels[i].image_name.value = OUTPUT_IMAGE_F % i

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = Workspace(pipeline, module, image_set, cellprofiler.object.ObjectSet(),
                          cellprofiler.measurement.Measurements(), image_set_list)
    module.run(workspace)
    for i, channel_index in enumerate(channel_indexes):
        pixels = image_set.get_image(module.channels[i].image_name.value).pixel_data
        assert pixels.ndim == 2
        assert tuple(pixels.shape) == (20, 10)
        numpy.testing.assert_almost_equal(image[:, :, channel_index], pixels)


def test_2_1_load_matlab_combine():
    data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDEyIDA5OjE3OjA0IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAdAEAAHiczZS9TsMwEIAvTdpSKlURA2JkZONvYUR0QAwU1EYVq6teI0uJHeUHUZ6Ax+rMU2G3TupEoUlDB05ynDvfd2efdR4AwFcXoCPmIzFasJG20g1tSH2CcUyZG7XBgjNlX4kxJSElMw+nxEswgkxS+xNbcGcZZEvPfJ54OCK+7ixklPgzDKOXRQqq5Vf6gd6EfiLkJXUb4zuNKGeKV/GL1iwvjwt5B/JjbutglNThRLNL/zvY+lsl/nocW+kORvFjSJZ1+G6Bl/qQ+zPKcJ3/qoI3c7wJ7JIcZN9VeY0cb8C1qte+3E1D7rYm17Q+Zfci6zPG+Zq/r+B7Bb6X1ReRKftf7ulBtOBAi7PvvDJ290Ef8n1Qdd7jwj6lTn3iohvyJPh9H1Vx7UJcOxf3nLI5BnXOe6g8/y1OALvvUe+bOnn1e7eVPuQeDx0un4Ysjt53nYq8LfHXL+Gavit18hkNOEuQ36cb7m3Pul7s8E8ltf8AGWV9Kw=='
    pipeline = tests.modules.load_pipeline(None, data)
    module = pipeline.module(1)
    assert module.image_name.value == "TestGray"
    assert module.should_combine
    assert module.grayscale_name.value == "TestGray"
    assert module.red_contribution.value == 1
    assert module.green_contribution.value == 2
    assert module.blue_contribution.value == 3


def test_2_2_load_matlab_split():
    data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDEyIDA5OjI3OjMwIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAaQEAAHiczVTNToQwEJ6ysBg3IcSD8ejRm38Xj0YPxoOrWcjGazd0SROghB+z69Pt2afxESzZwpYGAXFNnGRSZjrffMN0WgsAVibAmK8HXDXYiiFsJGlhOyTLaOSnBuhwIvwbrnOcULwIyBwHOUmhktL/GC2Zu46rrSfm5QGZ4lAO5jLNwwVJ0udlCRTbL3RFAoe+E6hLGTYjbzSlLBJ4kV/1VrwsU3gtrp/arg+ooQ9Hkr+Iv4FdvN4QL+exhe2SNHtI8LoP3lDwhe3EAc0E/0UHflTDjyA6x3upu4sX1fAILnvWq+KuBuKu/7g/ptIfU/RnRrxB9U7hd/N0x6+cJfxD1g1qn/sJ1Of+tqPOQ6XOwqYh9omfsDz+vo6uvLaS167lPaWRR+I+/7svnv+WJ4b2c5Tnrg+vfO62sO9ZwBKXFU9BlUee93EHr8a/Jg24oe9IHz40AKdz5MfxFvf6w76etcSXUvq/AErXefI='
    pipeline = tests.modules.load_pipeline(None, data)
    module = pipeline.module(1)
    assert module.image_name.value == "TestGray"
    assert module.should_split
    assert module.use_red.value
    assert module.red_name.value == "TestRed"
    assert not module.use_green.value
    assert module.use_blue.value
    assert module.blue_name.value == "TestBlue"


def test_2_3_load_combine():
    data = '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'
    pipeline = tests.modules.load_pipeline(None, data)
    module = pipeline.module(1)
    assert module.image_name.value == "TestInput"
    assert module.should_combine
    assert module.grayscale_name.value == "TestGray"
    assert module.red_contribution.value == 1
    assert module.green_contribution.value == 2
    assert module.blue_contribution.value == 3


def test_2_4_load_split():
    data = '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'
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


def test_2_5_load_v2():
    data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10308

ColorToGray:[module_num:1|svn_version:\'10300\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
Select the input image:DNA
Conversion method:Combine
Image type\x3A:Channels
Name the output image:OrigGrayw
Relative weight of the red channel:1
Relative weight of the green channel:3
Relative weight of the blue channel:5
Convert red to gray?:Yes
Name the output image:OrigRedx
Convert green to gray?:Yes
Name the output image:OrigGreeny
Convert blue to gray?:Yes
Name the output image:OrigBluez
Channel count:3
Channel number\x3A:Red\x3A 1
Relative weight of the channel:1
Image name\x3A:RedChannel1
Channel number\x3A:Blue\x3A 3
Relative weight of the channel:2
Image name\x3A:GreenChannel2
Channel number\x3A:Green\x3A 2
Relative weight of the channel:3
Image name\x3A:BlueChannel3
"""
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO(data))
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

def test_2_6_load_v3():
    """
    Tests a pipeline that was produced with module revision 3.
    The channel names are named according to the schema:
    Channel(#new_imagenumber)_(#channel_number),
    e.g. Channel3_2 would be the image number 3 that contains channel
    number 2.
    Thus it can be easily checked via the new image name, if the channel
    number is correctly parsed.
    """

    data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:4
DateRevision:315
GitHash:
ModuleCount:5
HasImagePlaneDetails:False

ColorToGray:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:\xff\xfeD\x00N\x00A\x00
    Conversion method:\xff\xfeS\x00p\x00l\x00i\x00t\x00
    Image type:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x00s\x00
    Name the output image:\xff\xfeO\x00r\x00i\x00g\x00G\x00r\x00a\x00y\x00
    Relative weight of the red channel:\xff\xfe1\x00.\x000\x00
    Relative weight of the green channel:\xff\xfe1\x00.\x000\x00
    Relative weight of the blue channel:\xff\xfe1\x00.\x000\x00
    Convert red to gray?:\xff\xfeY\x00e\x00s\x00
    Name the output image:\xff\xfeO\x00r\x00i\x00g\x00R\x00e\x00d\x00
    Convert green to gray?:\xff\xfeY\x00e\x00s\x00
    Name the output image:\xff\xfeO\x00r\x00i\x00g\x00G\x00r\x00e\x00e\x00n\x00
    Convert blue to gray?:\xff\xfeY\x00e\x00s\x00
    Name the output image:\xff\xfeO\x00r\x00i\x00g\x00B\x00l\x00u\x00e\x00
    Convert hue to gray?:\xff\xfeY\x00e\x00s\x00
    Name the output image:\xff\xfeO\x00r\x00i\x00g\x00H\x00u\x00e\x00
    Convert saturation to gray?:\xff\xfeY\x00e\x00s\x00
    Name the output image:\xff\xfeO\x00r\x00i\x00g\x00S\x00a\x00t\x00u\x00r\x00a\x00t\x00i\x00o\x00n\x00
    Convert value to gray?:\xff\xfeY\x00e\x00s\x00
    Name the output image:\xff\xfeO\x00r\x00i\x00g\x00V\x00a\x00l\x00u\x00e\x00
    Channel count:\xff\xfe8\x00
    Channel number:\xff\xfeG\x00r\x00e\x00e\x00n\x00\x3A\x00 \x002\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x001\x00_\x002\x00
    Channel number:\xff\xfeR\x00e\x00d\x00\x3A\x00 \x001\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x002\x00_\x001\x00
    Channel number:\xff\xfeB\x00l\x00u\x00e\x00\x3A\x00 \x003\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x003\x00_\x003\x00
    Channel number:\xff\xfeA\x00l\x00p\x00h\x00a\x00\x3A\x00 \x004\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x004\x00_\x004\x00
    Channel number:\xff\xfe5\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x005\x00_\x005\x00
    Channel number:\xff\xfe7\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x006\x00_\x007\x00
    Channel number:\xff\xfe7\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x007\x00_\x007\x00
    Channel number:\xff\xfe6\x00
    Relative weight of the channel:\xff\xfe1\x00.\x000\x00
    Image name:\xff\xfeC\x00h\x00a\x00n\x00n\x00e\x00l\x008\x00_\x006\x00
"""
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.colortogray.ColorToGray)
    assert module.image_name == "DNA"
    assert module.channel_count.value == 8
    for i in range(module.channel_count.value):
        c = module.channels[i].image_name.value.split('_')[1]
        assert module.channels[i].channel_choice.value == int(c)
    assert module.channels[6].image_name.value == 'Channel7_7'
