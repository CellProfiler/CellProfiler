import os.path
import StringIO

import numpy
import numpy.random
import numpy.testing
import pytest
import skimage.data
import skimage.util

import cellprofiler.image
import cellprofiler.modules.saveimages
import cellprofiler.pipeline
import cellprofiler.setting


instance = cellprofiler.modules.saveimages.SaveImages()


@pytest.fixture(
    scope="module",
    params=[
        skimage.data.camera(),
        skimage.data.astronaut()
    ],
    ids=[
        "grayscale_image",
        "multichannel_image"
    ]
)
def image(request):
    return cellprofiler.image.Image(image=request.param)


@pytest.fixture(scope="module")
def volume():
    data = numpy.random.rand(10, 10, 10)

    return cellprofiler.image.Image(image=data, dimensions=3)


def test_load_v11():
    pipeline_txt = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:300
GitHash:
ModuleCount:4
HasImagePlaneDetails:False

SaveImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the type of image to save:Image
    Select the image to save:DNA
    Select the objects to save:None
    Select the module display window to save:None
    Select method for constructing file names:From image filename
    Select image name for file prefix:DNA
    Enter single file name:OrigBlue
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:
    Saved file format:png
    Output file location:Default Output Folder\x7C
    Image bit depth:8-bit integer
    Overwrite existing files without warning?:No
    When to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:gray
    Record the file and path information to the saved image?:No
    Create subfolders in the output folder?:No
    Base image folder:Elsewhere...\x7C
    Saved movie format:avi

SaveImages:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the type of image to save:Image
    Select the image to save:DNA
    Select the objects to save:None
    Select the module display window to save:None
    Select method for constructing file names:From image filename
    Select image name for file prefix:DNA
    Enter single file name:OrigBlue
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:
    Saved file format:jpg
    Output file location:Default Output Folder\x7C
    Image bit depth:8-bit integer
    Overwrite existing files without warning?:No
    When to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:gray
    Record the file and path information to the saved image?:No
    Create subfolders in the output folder?:No
    Base image folder:Elsewhere...\x7C
    Saved movie format:avi

SaveImages:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the type of image to save:Image
    Select the image to save:DNA
    Select the objects to save:None
    Select the module display window to save:None
    Select method for constructing file names:From image filename
    Select image name for file prefix:DNA
    Enter single file name:OrigBlue
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:
    Saved file format:tif
    Output file location:Default Output Folder\x7C
    Image bit depth:8-bit integer
    Overwrite existing files without warning?:No
    When to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:gray
    Record the file and path information to the saved image?:No
    Create subfolders in the output folder?:No
    Base image folder:Elsewhere...\x7C
    Saved movie format:avi

SaveImages:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the type of image to save:Movie
    Select the image to save:DNA
    Select the objects to save:None
    Select the module display window to save:None
    Select method for constructing file names:From image filename
    Select image name for file prefix:DNA
    Enter single file name:OrigBlue
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:
    Saved file format:tif
    Output file location:Default Output Folder\x7C
    Image bit depth:8-bit integer
    Overwrite existing files without warning?:No
    When to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:gray
    Record the file and path information to the saved image?:No
    Create subfolders in the output folder?:No
    Base image folder:Elsewhere...\x7C
    Saved movie format:avi
"""

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline = cellprofiler.pipeline.Pipeline()

    pipeline.add_listener(callback)

    pipeline.load(StringIO.StringIO(pipeline_txt))

    module = pipeline.modules()[0]

    assert module.save_image_or_figure.value == cellprofiler.modules.saveimages.IF_IMAGE

    assert module.image_name.value == "DNA"

    assert module.file_name_method.value == cellprofiler.modules.saveimages.FN_FROM_IMAGE

    assert module.file_image_name.value == "DNA"

    assert module.single_file_name.value == "OrigBlue"

    assert module.number_of_digits.value == 4

    assert module.wants_file_name_suffix.value == False

    assert module.file_format.value == "png"

    assert module.pathname.value == "Default Output Folder|"

    assert module.bit_depth.value == cellprofiler.modules.saveimages.BIT_DEPTH_8

    assert module.overwrite.value == False

    assert module.when_to_save.value == cellprofiler.modules.saveimages.WS_EVERY_CYCLE

    assert module.update_file_names.value == False

    assert module.create_subdirectories.value == False

    assert module.root_dir.value == "Elsewhere...|"

    module = pipeline.modules()[1]

    assert module.file_format.value == "jpeg"

    module = pipeline.modules()[2]

    assert module.file_format.value == "tiff"

    module = pipeline.modules()[3]

    assert module.file_format.value == "tiff"


def test_save_image_png_8(tmpdir, image, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_PNG

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_8

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example.png"))

    data = skimage.io.imread(os.path.join(directory, "example.png"))

    assert data.dtype == numpy.uint8

    numpy.testing.assert_array_equal(data, skimage.util.img_as_ubyte(image.pixel_data))


def test_save_image_jpeg_8(tmpdir, image, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_JPEG

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_8

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example.jpeg"))

    data = skimage.io.imread(os.path.join(directory, "example.jpeg"))

    assert data.dtype == numpy.uint8

    # TODO: How best to test lossy format?


def test_save_image_tiff_uint8(tmpdir, image, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_8

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example.tiff"))

    data = skimage.io.imread(os.path.join(directory, "example.tiff"))

    assert data.dtype == numpy.uint8

    numpy.testing.assert_array_equal(data, skimage.util.img_as_ubyte(image.pixel_data))


def test_save_image_tiff_uint16(tmpdir, image, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_16

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example.tiff"))

    data = skimage.io.imread(os.path.join(directory, "example.tiff"))

    assert data.dtype == numpy.uint16

    numpy.testing.assert_array_equal(data, skimage.util.img_as_uint(image.pixel_data))


def test_save_image_tiff_float32(tmpdir, image, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_FLOAT

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example.tiff"))

    data = skimage.io.imread(os.path.join(directory, "example.tiff"))

    assert data.dtype == numpy.float32

    numpy.testing.assert_array_equal(data, skimage.util.img_as_float(image.pixel_data))


def test_save_image_npy(tmpdir, image, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_NPY

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example.npy"))

    data = numpy.load(os.path.join(directory, "example.npy"))

    numpy.testing.assert_array_equal(data, image.pixel_data)


def test_save_volume_tiff_uint8(tmpdir, volume, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    workspace.image_set.add("example_volume", volume)

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example_volume"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example_volume"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_8

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example_volume.tiff"))

    data = skimage.io.imread(os.path.join(directory, "example_volume.tiff"))

    assert data.dtype == numpy.uint8

    numpy.testing.assert_array_equal(data, skimage.util.img_as_ubyte(volume.pixel_data))


def test_save_volume_tiff_uint16(tmpdir, volume, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    workspace.image_set.add("example_volume", volume)

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example_volume"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example_volume"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_16

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example_volume.tiff"))

    data = skimage.io.imread(os.path.join(directory, "example_volume.tiff"))

    assert data.dtype == numpy.uint16

    numpy.testing.assert_array_equal(data, skimage.util.img_as_uint(volume.pixel_data))


def test_save_volume_tiff_float32(tmpdir, volume, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    workspace.image_set.add("example_volume", volume)

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example_volume"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example_volume"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_FLOAT

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example_volume.tiff"))

    data = skimage.io.imread(os.path.join(directory, "example_volume.tiff"))

    assert data.dtype == numpy.float32

    numpy.testing.assert_array_equal(data, skimage.util.img_as_float(volume.pixel_data))


def test_save_volume_npy(tmpdir, volume, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    workspace.image_set.add("example_volume", volume)

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example_volume"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example_volume"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_NPY

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example_volume.npy"))

    data = numpy.load(os.path.join(directory, "example_volume.npy"))

    numpy.testing.assert_array_equal(data, volume.pixel_data)


@pytest.mark.parametrize(
    "volume",
    [cellprofiler.image.Image(image=numpy.ones((100, 100, 5)))]
)
def test_save_5_plus_channel_tiff_uint16(tmpdir, volume, module, workspace):
    directory = str(tmpdir.mkdir("images"))

    workspace.image_set.add("example_volume", volume)

    module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

    module.image_name.value = "example_volume"

    module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME

    module.single_file_name.value = "example_volume"

    module.pathname.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

    module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_16

    module.run(workspace)

    assert os.path.exists(os.path.join(directory, "example_volume.tiff"))

    data = skimage.io.imread(os.path.join(directory, "example_volume.tiff"))

    assert data.dtype == numpy.uint16

    numpy.testing.assert_array_equal(data, numpy.transpose(skimage.util.img_as_uint(volume.pixel_data), (2, 0, 1)))

