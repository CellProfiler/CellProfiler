import StringIO

import cellprofiler.modules.saveimages
import cellprofiler.pipeline


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
