import StringIO
import base64
import logging
import os
import tempfile
import unittest
import zlib

import bioformats
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.applythreshold
import cellprofiler.modules.createbatchfiles
import cellprofiler.modules.loadimages
import cellprofiler.modules.saveimages
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.setting
import cellprofiler.workspace
import matplotlib
import numpy
import scipy.sparse
import tests.modules

logger = logging.getLogger(__name__)
cellprofiler.preferences.set_headless()

IMAGE_NAME = 'inputimage'
OBJECTS_NAME = 'inputobjects'
FILE_IMAGE_NAME = 'fileimage'
FILE_NAME = 'filenm'


class TestSaveImages(unittest.TestCase):

    def setUp(self):
        # Change the default image directory to a temporary file
        cellprofiler.preferences.set_headless()
        self.new_image_directory = os.path.normcase(tempfile.mkdtemp())
        cellprofiler.preferences.set_default_image_directory(self.new_image_directory)
        self.new_output_directory = os.path.normcase(tempfile.mkdtemp())
        cellprofiler.preferences.set_default_output_directory(self.new_output_directory)
        self.custom_directory = os.path.normcase(tempfile.mkdtemp())

    def tearDown(self):
        for subdir in (self.new_image_directory, self.new_output_directory,
                       self.custom_directory):
            for filename in os.listdir(subdir):
                try:
                    os.remove(os.path.join(subdir, filename))
                except:
                    logger.warn("Failed to remove %s" % filename,
                                exc_info=True)
            try:
                os.rmdir(subdir)
            except:
                logger.warn("Failed to remove %s directory" % subdir,
                            exc_info=True)

    def on_event(self, pipeline, event):
        self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

    def test_00_03_load_v2(self):
        data = ('eJztVsFu0zAYdrJ0MCohNC47+ogQVNlQ0eiFdZSKSms70WriiJc6wZITR45T'
                'Vk48Ao/HY+wRiCNnSaywJK3EhVmy4t/+Pn+/v9iWp8PlxfAc9ns2nA6Xr11C'
                'MbykSLiM+wMYiFfwA8dI4BVkwQCOOYFzR0D7BB6/HRz3B2/68MS234HtijGZ'
                'Pk0+t10A9pPv46SaaqijYqNQZbzAQpDAizrAAkeq/3dSrxAn6JriK0RjHOUS'
                'Wf8kcNlyE94NTdkqpniG/CI4KbPYv8Y8mrsZUQ1fkhtMF+QH1paQwT7jNYkI'
                'CxRfza/33ukyoekuvrHvY56ko80v/flp5f4YFf4cFvol/hTkeKsC3yngn6l4'
                '4iMPK75dw98r8ffAaDbcifephvdcy1fGY858SGTSUO7ZIHOuyfofafPJeM6J'
                'l+WzK/+shv9E48t4xGDABIwjtQG28TMMvEZ+HoGyvoxH2EUxFZDFIowFXBGO'
                'HcH4pkkeVmk+C8xYgJvwjBLPSH1vwjNLPDPRA41872rrlvHHNeYb6Gwcmu+f'
                'tnlf7Jh3W389jhr9l231/sY7bXkPFXX2NXxWMvxBgXdWk1/V+UmvAo+zONxd'
                '/3/L+4H3wPsXvF8FXtX9UbxXJf4ruP88vQTl8yRjB1MacibfBLznp4+tqBeh'
                'NU4PWtRbJM30rRNVr+egQqeYl5m0Dmt80Nef+3L7fhs9s0KvW8Oz1Eta8r6A'
                'dr6/uAcPKvBt1yPbfwCfYqjK')
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.image_name.value, "DNA")
        self.assertEqual(module.file_image_name.value, "OrigDNA")
        self.assertEqual(module.file_name_method.value, cellprofiler.modules.saveimages.FN_FROM_IMAGE)
        self.assertEqual(module.pathname.dir_choice,
                         cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.when_to_save.value, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap.value, cellprofiler.modules.saveimages.CM_GRAY)
        self.assertFalse(module.overwrite)

    def test_00_04_00_load_v3(self):
        data = ('eJztVsFu0zAYdrJ0MCohNC47+ogQVNlQ0eiFdZSKSms70WriiJc6wZITR45T'
                'Vk48Ao/HY+wRiCNnSaywJK3EhVmy4t/+Pn+/v9iWp8PlxfAc9ns2nA6Xr11C'
                'MbykSLiM+wMYiFfwA8dI4BVkwQCOOYFzR0D7BB6/HRz3B2/68MS234HtijGZ'
                'Pk0+t10A9pPv46SaaqijYqNQZbzAQpDAizrAAkeq/3dSrxAn6JriK0RjHOUS'
                'Wf8kcNlyE94NTdkqpniG/CI4KbPYv8Y8mrsZUQ1fkhtMF+QH1paQwT7jNYkI'
                'CxRfza/33ukyoekuvrHvY56ko80v/flp5f4YFf4cFvol/hTkeKsC3yngn6l4'
                '4iMPK75dw98r8ffAaDbcifephvdcy1fGY858SGTSUO7ZIHOuyfofafPJeM6J'
                'l+WzK/+shv9E48t4xGDABIwjtQG28TMMvEZ+HoGyvoxH2EUxFZDFIowFXBGO'
                'HcH4pkkeVmk+C8xYgJvwjBLPSH1vwjNLPDPRA41872rrlvHHNeYb6Gwcmu+f'
                'tnlf7Jh3W389jhr9l231/sY7bXkPFXX2NXxWMvxBgXdWk1/V+UmvAo+zONxd'
                '/3/L+4H3wPsXvF8FXtX9UbxXJf4ruP88vQTl8yRjB1MacibfBLznp4+tqBeh'
                'NU4PWtRbJM30rRNVr+egQqeYl5m0Dmt80Nef+3L7fhs9s0KvW8Oz1Eta8r6A'
                'dr6/uAcPKvBt1yPbfwCfYqjK')
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.image_name.value, "DNA")
        self.assertEqual(module.file_image_name.value, "OrigDNA")
        self.assertEqual(module.file_name_method.value, cellprofiler.modules.saveimages.FN_FROM_IMAGE)
        self.assertEqual(module.pathname.dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.when_to_save.value, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap.value, cellprofiler.modules.saveimages.CM_GRAY)
        self.assertFalse(module.overwrite)

    def test_00_04_01_load_v4(self):
        '''Regression test of IMG-759 - load v4 SaveImages'''
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9438

SaveImages:[module_num:60|svn_version:\'9438\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    Select the type of image to save:Image
    Select the image to save:ColorOutlineImage
    Select the module display window to save:Fig
    Select method for constructing file names:From image filename
    Select image name for file prefix:OrigRGB
    Enter single file name:OrigBlue
    Text to append to the image name:_outlines
    Select file format to use:png
    Select location to save file:Custom with metadata
    Pathname for the saved file:&/\\g<Directory>/\\g<Subdirectory>
    Image bit depth:8
    Overwrite existing files without warning?:Yes
    Select how often to save:Every cycle
    Select how often to save:Last cycle
    Rescale the images? :No
    Select colormap:gray
    Update file names within CellProfiler?:No
    Create subfolders in the output folder?:No
'''
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.save_image_or_figure,
                         cellprofiler.modules.saveimages.IF_IMAGE)
        self.assertEqual(module.image_name, "ColorOutlineImage")
        self.assertEqual(module.figure_name, "Fig")
        self.assertEqual(module.file_name_method, cellprofiler.modules.saveimages.FN_FROM_IMAGE)
        self.assertEqual(module.file_image_name, "OrigRGB")
        self.assertEqual(module.single_file_name, "OrigBlue")
        self.assertEqual(module.wants_file_name_suffix, True)
        self.assertEqual(module.file_name_suffix, "_outlines")
        self.assertEqual(module.file_format, cellprofiler.modules.saveimages.FF_PNG)
        self.assertEqual(module.pathname.dir_choice,
                         cellprofiler.preferences.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(module.pathname.custom_path,
                         "./\\g<Directory>/\\g<Subdirectory>")
        self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
        self.assertTrue(module.overwrite)
        self.assertEqual(module.when_to_save, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertEqual(module.rescale, False)
        self.assertEqual(module.colormap, "gray")
        self.assertEqual(module.update_file_names, False)
        self.assertEqual(module.create_subdirectories, False)

    def test_00_05_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9514

SaveImages:[module_num:1|svn_version:\'9507\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Image
    Select the image to save:Img1
    Select the module display window to save:Mdw1
    Select method for constructing file names:From image filename
    Select image name for file prefix:Pfx1
    Enter single file name:Sfn1
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:A1
    Select file format to use:bmp
    Select location to save file:Default Output Folder\x7Ccp1
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Select colormap:gray
    Update file names within CellProfiler?:Yes
    Create subfolders in the output folder?:No

SaveImages:[module_num:2|svn_version:\'9507\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Mask
    Select the image to save:Img2
    Select the module display window to save:Mdw2
    Select method for constructing file names:Sequential numbers
    Select image name for file prefix:Pfx2
    Enter file prefix:Sfn2
    Do you want to add a suffix to the image file name?:Yes
    Text to append to the image name:A2
    Select file format to use:png
    Select location to save file:Default Input Folder\x7Ccp2
    Image bit depth:8
    Overwrite existing files without warning?:Yes
    Select how often to save:First cycle
    Rescale the images? :No
    Select colormap:copper
    Update file names within CellProfiler?:No
    Create subfolders in the output folder?:Yes

SaveImages:[module_num:3|svn_version:\'9507\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Cropping
    Select the image to save:Img3
    Select the module display window to save:Mdw3
    Select method for constructing file names:Single name
    Select image name for file prefix:Pfx3
    Enter single file name:Sfn3
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:A3
    Select file format to use:jpg
    Select location to save file:Same folder as image\x7Ccp3
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Last cycle
    Rescale the images? :Yes
    Select colormap:gray
    Update file names within CellProfiler?:Yes
    Create subfolders in the output folder?:No

SaveImages:[module_num:4|svn_version:\'9507\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Movie
    Select the image to save:Img4
    Select the module display window to save:Mdw4
    Select method for constructing file names:Name with metadata
    Select image name for file prefix:Pfx4
    Enter file name with metadata:Sfn4
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:A4
    Select file format to use:jpg
    Select location to save file:Elsewhere...\x7Ccp4
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Last cycle
    Rescale the images? :No
    Select colormap:gray
    Update file names within CellProfiler?:No
    Create subfolders in the output folder?:No

SaveImages:[module_num:5|svn_version:\'9507\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Module window
    Select the image to save:Img5
    Select the module display window to save:Mdw5
    Select method for constructing file names:Image filename with metadata
    Select image name for file prefix:Pfx5
    Enter file name with metadata:Sfn5
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:A5
    Select file format to use:png
    Select location to save file:Default Output Folder sub-folder\x7Ccp5
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Select colormap:gray
    Update file names within CellProfiler?:No
    Create subfolders in the output folder?:No

SaveImages:[module_num:6|svn_version:\'9507\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Image
    Select the image to save:Img6
    Select the module display window to save:Mdw6
    Select method for constructing file names:From image filename
    Select image name for file prefix:Pfx6
    Enter file name with metadata:Sfn6
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:A6
    Select file format to use:png
    Select location to save file:Default Input Folder sub-folder\x7Ccp6
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Select colormap:gray
    Update file names within CellProfiler?:No
    Create subfolders in the output folder?:Yes
"""
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 6)
        sif = [cellprofiler.modules.saveimages.IF_IMAGE, cellprofiler.modules.saveimages.IF_MASK, cellprofiler.modules.saveimages.IF_CROPPING,
               cellprofiler.modules.saveimages.IF_MOVIE, cellprofiler.modules.saveimages.IF_FIGURE, cellprofiler.modules.saveimages.IF_IMAGE]
        fnm = [cellprofiler.modules.saveimages.FN_FROM_IMAGE, cellprofiler.modules.saveimages.FN_SEQUENTIAL,
               cellprofiler.modules.saveimages.FN_SINGLE_NAME, cellprofiler.modules.saveimages.FN_SINGLE_NAME,
               cellprofiler.modules.saveimages.FN_FROM_IMAGE,
               cellprofiler.modules.saveimages.FN_FROM_IMAGE]
        suf = [ False, True, False, False, True, False]
        ff = [cellprofiler.modules.saveimages.FF_BMP, cellprofiler.modules.saveimages.FF_PNG, cellprofiler.modules.saveimages.FF_JPG,
              cellprofiler.modules.saveimages.FF_JPG, cellprofiler.modules.saveimages.FF_PNG, cellprofiler.modules.saveimages.FF_PNG]
        ov = [ False, True, False, False, False, False]
        wts = [cellprofiler.modules.saveimages.WS_EVERY_CYCLE, cellprofiler.modules.saveimages.WS_FIRST_CYCLE,
               cellprofiler.modules.saveimages.WS_LAST_CYCLE, cellprofiler.modules.saveimages.WS_LAST_CYCLE,
               cellprofiler.modules.saveimages.WS_EVERY_CYCLE, cellprofiler.modules.saveimages.WS_EVERY_CYCLE]
        dir_choice = [cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME
                      cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME,
                      cellprofiler.modules.saveimages.PC_WITH_IMAGE,
                      cellprofiler.preferences.ABSOLUTE_FOLDER_NAME,
                      cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                      cellprofiler.preferences.DEFAULT_INPUT_SUBFOLDER_NAME]
        rescale = [ False, False, True, False, False, False]
        cm = [ "gray", "copper", "gray", "gray", "gray", "gray" ]
        up = [ True, False, True, False, False, False]
        cre = [ False, True, False, False, False, True]
        for i, module in enumerate(pipeline.modules()):
            self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
            self.assertEqual(module.save_image_or_figure, sif[i])
            self.assertEqual(module.image_name, "Img%d" % (i+1))
            self.assertEqual(module.figure_name, "Mdw%d" % (i+1))
            self.assertEqual(module.file_name_method, fnm[i])
            self.assertEqual(module.file_image_name, "Pfx%d" % (i+1))
            self.assertEqual(module.single_file_name, "Sfn%d" % (i+1))
            self.assertEqual(module.wants_file_name_suffix, suf[i])
            if i == 4:
                # Single file name got copied into file name suffix
                self.assertEqual(module.file_name_suffix, "Sfn%d" %(i+1))
            else:
                self.assertEqual(module.file_name_suffix, "A%d" % (i+1))
            self.assertEqual(module.file_format, ff[i])
            self.assertEqual(module.pathname.dir_choice, dir_choice[i])
            self.assertEqual(module.pathname.custom_path, "cp%d" %(i+1))
            self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
            self.assertEqual(module.when_to_save, wts[i])
            self.assertEqual(module.rescale, rescale[i])
            self.assertEqual(module.colormap, cm[i])
            self.assertEqual(module.update_file_names, up[i])
            self.assertEqual(module.create_subdirectories, cre[i])

    def test_00_06_load_v6(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10237

SaveImages:[module_num:1|svn_version:\'10244\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Image
    Select the image to save:DNA
    Select the module display window to save:MyFigure
    Select method for constructing file names:Single name
    Select image name for file prefix:MyImage
    Enter single file name:DNA_\\g<WellColumn>
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:MySuffix
    Select file format to use:bmp
    Output file location:Default Output Folder sub-folder\x7CDNA_\\g<WellRow>
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Select colormap:gray
    Update file names within CellProfiler?:No
    Create subfolders in the output folder?:No
"""
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.save_image_or_figure, cellprofiler.modules.saveimages.IF_IMAGE)
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.figure_name, "MyFigure")
        self.assertEqual(module.file_name_method, cellprofiler.modules.saveimages.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "MyImage")
        self.assertEqual(module.single_file_name, r"DNA_\g<WellColumn>")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "MySuffix")
        self.assertEqual(module.file_format, cellprofiler.modules.saveimages.FF_BMP)
        self.assertEqual(module.pathname.dir_choice, cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"DNA_\g<WellRow>")
        self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap, "gray")
        self.assertFalse(module.update_file_names)
        self.assertFalse(module.create_subdirectories)

    def test_00_07_load_v7(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10782

SaveImages:[module_num:1|svn_version:\'10581\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Objects
    Select the image to save:None
    Select the objects to save:Nuclei
    Select the module display window to save:None
    Select method for constructing file names:Single name
    Select image name for file prefix:None
    Enter single file name:\\g<Well>_Nuclei
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:Whatever
    Select file format to use:png
    Output file location:Default Output Folder\x7CNone
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:Default
    Store file and path information to the saved image?:No
    Create subfolders in the output folder?:No

SaveImages:[module_num:2|svn_version:\'10581\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Objects
    Select the image to save:None
    Select the objects to save:Nuclei
    Select the module display window to save:None
    Select method for constructing file names:Single name
    Select image name for file prefix:None
    Enter single file name:\\g<Well>_Nuclei
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:Whatever
    Select file format to use:png
    Output file location:Default Output Folder\x7CNone
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Color
    Select colormap:Default
    Store file and path information to the saved image?:No
    Create subfolders in the output folder?:No
"""
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.save_image_or_figure, cellprofiler.modules.saveimages.IF_OBJECTS)
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.file_name_method, cellprofiler.modules.saveimages.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "None")
        self.assertEqual(module.single_file_name, r"\g<Well>_Nuclei")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "Whatever")
        self.assertEqual(module.file_format, cellprofiler.modules.saveimages.FF_PNG)
        self.assertEqual(module.pathname.dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"None")
        self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertFalse(module.rescale)
        self.assertEqual(module.gray_or_color, cellprofiler.modules.saveimages.GC_GRAYSCALE)
        self.assertEqual(module.colormap, "Default")
        self.assertFalse(module.update_file_names)
        self.assertFalse(module.create_subdirectories)
        self.assertEqual(module.root_dir.dir_choice,
                         cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.gray_or_color, cellprofiler.modules.saveimages.GC_COLOR)

    def test_00_08_load_v8(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        image_folder_text = pipeline.encode_txt(
            "%s|%s" % (cellprofiler.preferences.ABSOLUTE_FOLDER_NAME,
                       tests.modules.example_images_directory()))
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10782

SaveImages:[module_num:1|svn_version:\'10581\'|variable_revision_number:8|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Objects
    Select the image to save:None
    Select the objects to save:Nuclei
    Select the module display window to save:None
    Select method for constructing file names:Single name
    Select image name for file prefix:None
    Enter single file name:\\g<Well>_Nuclei
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:Whatever
    Select file format to use:tif
    Output file location:Default Output Folder\x7CNone
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:Default
    Store file and path information to the saved image?:No
    Create subfolders in the output folder?:Yes
    Image folder:%s
""" % image_folder_text
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.save_image_or_figure, cellprofiler.modules.saveimages.IF_OBJECTS)
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.file_name_method, cellprofiler.modules.saveimages.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "None")
        self.assertEqual(module.single_file_name, r"\g<Well>_Nuclei")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "Whatever")
        self.assertEqual(module.file_format, cellprofiler.modules.saveimages.FF_TIFF)
        self.assertEqual(module.pathname.dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"None")
        self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertFalse(module.rescale)
        self.assertEqual(module.gray_or_color, cellprofiler.modules.saveimages.GC_GRAYSCALE)
        self.assertEqual(module.colormap, "Default")
        self.assertFalse(module.update_file_names)
        self.assertTrue(module.create_subdirectories)
        self.assertEqual(module.root_dir.dir_choice,
                         cellprofiler.preferences.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.root_dir.custom_path,
                         tests.modules.example_images_directory())

    def test_00_09_load_v9(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        image_folder_text = pipeline.encode_txt(
            "%s|%s" % (cellprofiler.preferences.ABSOLUTE_FOLDER_NAME,
                       tests.modules.example_images_directory()))
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10782

SaveImages:[module_num:1|svn_version:\'10581\'|variable_revision_number:9|show_window:True|notes:\x5B\x5D]
    Select the type of image to save:Objects
    Select the image to save:None
    Select the objects to save:Nuclei
    Select the module display window to save:None
    Select method for constructing file names:Single name
    Select image name for file prefix:None
    Enter single file name:\\g<Well>_Nuclei
    Do you want to add a suffix to the image file name?:No
    Text to append to the image name:Whatever
    Select file format to use:tif
    Output file location:Default Output Folder\x7CNone
    Image bit depth:8
    Overwrite existing files without warning?:No
    Select how often to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:Default
    Store file and path information to the saved image?:No
    Create subfolders in the output folder?:Yes
    Image folder:%s
""" % image_folder_text
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.save_image_or_figure, cellprofiler.modules.saveimages.IF_OBJECTS)
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.file_name_method, cellprofiler.modules.saveimages.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "None")
        self.assertEqual(module.single_file_name, r"\g<Well>_Nuclei")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "Whatever")
        self.assertEqual(module.file_format, cellprofiler.modules.saveimages.FF_TIF)
        self.assertEqual(module.pathname.dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"None")
        self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertFalse(module.rescale)
        self.assertEqual(module.gray_or_color, cellprofiler.modules.saveimages.GC_GRAYSCALE)
        self.assertEqual(module.colormap, "Default")
        self.assertFalse(module.update_file_names)
        self.assertTrue(module.create_subdirectories)
        self.assertEqual(module.root_dir.dir_choice,
                         cellprofiler.preferences.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.root_dir.custom_path,
                         tests.modules.example_images_directory())
        self.assertEqual(module.movie_format, cellprofiler.modules.saveimages.FF_AVI)

    def test_00_10_load_v10(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        image_folder_text = pipeline.encode_txt(
            "%s|%s" % (cellprofiler.preferences.ABSOLUTE_FOLDER_NAME,
                       cellprofiler.pipeline.utf16encode(tests.modules.example_images_directory())))
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140128180905
GitHash:b9e9c97
ModuleCount:1
HasImagePlaneDetails:False

SaveImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the type of image to save:Objects
    Select the image to save:None
    Select the objects to save:Nuclei
    Select the module display window to save:None
    Select method for constructing file names:Single name
    Select image name for file prefix:None
    Enter single file name:\\\\g<Well>_Nuclei
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:Whatever
    Saved file format:tif
    Output file location:Default Output Folder\x7CNone
    Image bit depth:8
    Overwrite existing files without warning?:No
    When to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:Default
    Record the file and path information to the saved image?:No
    Create subfolders in the output folder?:Yes
    Base image folder:%s
""" % image_folder_text
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.save_image_or_figure, cellprofiler.modules.saveimages.IF_OBJECTS)
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.file_name_method, cellprofiler.modules.saveimages.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "None")
        self.assertEqual(module.single_file_name, r"\g<Well>_Nuclei")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "Whatever")
        self.assertEqual(module.file_format, cellprofiler.modules.saveimages.FF_TIF)
        self.assertEqual(module.pathname.dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"None")
        self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertFalse(module.rescale)
        self.assertEqual(module.gray_or_color, cellprofiler.modules.saveimages.GC_GRAYSCALE)
        self.assertEqual(module.colormap, "Default")
        self.assertFalse(module.update_file_names)
        self.assertTrue(module.create_subdirectories)
        self.assertEqual(module.root_dir.dir_choice,
                         cellprofiler.preferences.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.root_dir.custom_path,
                         tests.modules.example_images_directory())
        self.assertEqual(module.movie_format, cellprofiler.modules.saveimages.FF_AVI)

    def test_00_11_load_v11(self):
        pipeline = cellprofiler.pipeline.Pipeline()
        image_folder_text = pipeline.encode_txt(
            "%s|%s" % (cellprofiler.preferences.ABSOLUTE_FOLDER_NAME,
                       cellprofiler.pipeline.utf16encode(tests.modules.example_images_directory())))
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140128180905
GitHash:b9e9c97
ModuleCount:1
HasImagePlaneDetails:False

SaveImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the type of image to save:Objects
    Select the image to save:None
    Select the objects to save:Nuclei
    Select the module display window to save:None
    Select method for constructing file names:Single name
    Select image name for file prefix:None
    Enter single file name:\\\\g<Well>_Nuclei
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:Whatever
    Saved file format:tif
    Output file location:Default Output Folder\x7CNone
    Image bit depth:8-bit integer
    Overwrite existing files without warning?:No
    When to save:Every cycle
    Rescale the images? :No
    Save as grayscale or color image?:Grayscale
    Select colormap:Default
    Record the file and path information to the saved image?:No
    Create subfolders in the output folder?:Yes
    Base image folder:%s
    Saved movie format:tif
""" % image_folder_text
        def callback(caller,event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        self.assertEqual(module.save_image_or_figure, cellprofiler.modules.saveimages.IF_OBJECTS)
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.file_name_method, cellprofiler.modules.saveimages.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "None")
        self.assertEqual(module.single_file_name, r"\g<Well>_Nuclei")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "Whatever")
        self.assertEqual(module.file_format, cellprofiler.modules.saveimages.FF_TIF)
        self.assertEqual(module.pathname.dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"None")
        self.assertEqual(module.bit_depth, cellprofiler.modules.saveimages.BIT_DEPTH_8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cellprofiler.modules.saveimages.WS_EVERY_CYCLE)
        self.assertFalse(module.rescale)
        self.assertEqual(module.gray_or_color, cellprofiler.modules.saveimages.GC_GRAYSCALE)
        self.assertEqual(module.colormap, "Default")
        self.assertFalse(module.update_file_names)
        self.assertTrue(module.create_subdirectories)
        self.assertEqual(module.root_dir.dir_choice,
                         cellprofiler.preferences.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.root_dir.custom_path,
                         tests.modules.example_images_directory())
        self.assertEqual(module.movie_format, cellprofiler.modules.saveimages.FF_TIF)

    def test_01_01_save_first_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_ABOVE_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 1.0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        save_images.pathname.dir_choice = cellprofiler.modules.saveimages.PC_WITH_IMAGE
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_FIRST_CYCLE
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        self.assertFalse(os.path.isfile(img2_out_filename))
        pn,fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image','FileName_Derived')
        pathnames = measurements.get_all_measurements('Image','PathName_Derived')
        self.assertEqual(filenames[0],fn)
        self.assertEqual(pathnames[0],pn)
        data = bioformats.load_image(img1_out_filename, rescale=False)
        expected_data = bioformats.load_image(img1_filename, rescale=False)
        self.assertTrue(numpy.all(data[expected_data < 255] ==
                                  expected_data[expected_data < 255]))
        self.assertTrue(numpy.all(data[expected_data == 255] == 0))

    def test_01_02_save_all_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        save_images.pathname.dir_choice = cellprofiler.modules.saveimages.PC_WITH_IMAGE
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_EVERY_CYCLE
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        pn,fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image','FileName_Derived')
        pathnames = measurements.get_all_measurements('Image','PathName_Derived')
        self.assertEqual(filenames[0],fn)
        self.assertEqual(pathnames[0],pn)
        data = bioformats.load_image(img1_out_filename)
        expected_data = bioformats.load_image(img1_filename)
        self.assertTrue(numpy.all(data == expected_data))
        data = bioformats.load_image(img2_out_filename)
        expected_data = bioformats.load_image(img2_filename)
        self.assertTrue(numpy.all(data == expected_data))

    def test_01_03_save_last_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        save_images.pathname.dir_choice = cellprofiler.modules.saveimages.PC_WITH_IMAGE
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_LAST_CYCLE
        save_images.update_file_names.value = False
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertFalse(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        data = bioformats.load_image(img2_out_filename)
        expected_data = bioformats.load_image(img2_filename)
        self.assertTrue(numpy.all(data == expected_data))

    def test_01_04_save_all_to_output_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_output_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.new_output_directory,'img2OUT.tif')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        save_images.pathname.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_EVERY_CYCLE
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        pn,fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image','FileName_Derived')
        pathnames = measurements.get_all_measurements('Image','PathName_Derived')
        self.assertEqual(filenames[0],fn)
        self.assertEqual(pathnames[0],pn)
        data = bioformats.load_image(img1_out_filename)
        expected_data = bioformats.load_image(img1_filename)
        self.assertTrue(numpy.all(data == expected_data))
        data = bioformats.load_image(img2_out_filename)
        expected_data = bioformats.load_image(img2_filename)
        self.assertTrue(numpy.all(data == expected_data))

    def test_01_05_save_all_to_custom_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.tif')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        save_images.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_EVERY_CYCLE
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        pn,fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image','FileName_Derived')
        pathnames = measurements.get_all_measurements('Image','PathName_Derived')
        self.assertEqual(filenames[0],fn)
        self.assertEqual(pathnames[0],pn)
        data = bioformats.load_image(img1_out_filename)
        expected_data = bioformats.load_image(img1_filename)
        self.assertTrue(numpy.all(data == expected_data))
        data = bioformats.load_image(img2_out_filename)
        expected_data = bioformats.load_image(img2_filename)
        self.assertTrue(numpy.all(data == expected_data))

    def test_01_06_save_all_to_custom_png(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.png')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.png')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_PNG
        save_images.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_EVERY_CYCLE
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        pn,fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image','FileName_Derived')
        pathnames = measurements.get_all_measurements('Image','PathName_Derived')
        self.assertEqual(filenames[0],fn)
        self.assertEqual(pathnames[0],pn)
        data = bioformats.load_image(img1_out_filename)
        expected_data = bioformats.load_image(img1_filename)
        self.assertTrue(numpy.all(data == expected_data))
        data = bioformats.load_image(img2_out_filename)
        expected_data = bioformats.load_image(img2_filename)
        self.assertTrue(numpy.all(data == expected_data))

    def test_01_07_save_all_to_custom_jpg(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.jpg')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.jpg')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_JPG
        save_images.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_EVERY_CYCLE
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        pn,fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image','FileName_Derived')
        pathnames = measurements.get_all_measurements('Image','PathName_Derived')
        self.assertEqual(filenames[0],fn)
        self.assertEqual(pathnames[0],pn)
        data = bioformats.load_image(img1_out_filename)
        expected_data = bioformats.load_image(img1_filename)
        self.assertTrue(numpy.all(numpy.abs(data.astype(int) -
                                            expected_data.astype(int)) <= 4))
        data = bioformats.load_image(img2_out_filename)
        expected_data = bioformats.load_image(img2_filename)
        self.assertTrue(numpy.all(numpy.abs(data.astype(int) -
                                            expected_data.astype(int)) <= 4))

    def test_01_09_save_single_to_custom_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.tif')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        save_images.single_file_name.value ='img1OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        save_images.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_FIRST_CYCLE
        save_images.update_file_names.value = False
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        data = bioformats.load_image(img1_out_filename)
        expected_data = bioformats.load_image(img1_filename)
        self.assertTrue(numpy.all(data == expected_data))

    def test_01_10_save_all_to_custom_png_rgb(self):
        '''Tests the path of saving an image with a colormap other than gray'''
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.png')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif')
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.png')
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_BELOW_THRESHOLD
        apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_PNG
        save_images.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cellprofiler.modules.saveimages.WS_EVERY_CYCLE
        #
        # Use Jet to force saving rgb images
        #
        save_images.colormap.value = 'jet'
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        pn,fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image','FileName_Derived')
        pathnames = measurements.get_all_measurements('Image','PathName_Derived')
        self.assertEqual(filenames[0],fn)
        self.assertEqual(pathnames[0],pn)
        data = bioformats.load_image(img1_out_filename, rescale=False)
        image = bioformats.load_image(img1_filename)
        mapper = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.jet)
        expected_data = mapper.to_rgba(image, bytes=True)[:, :, :3]
        self.assertTrue(numpy.all(data == expected_data))
        data = bioformats.load_image(img2_out_filename, rescale=False)
        image = bioformats.load_image(img2_filename)
        mapper = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.jet)
        expected_data = mapper.to_rgba(image, bytes=True)[:, :, :3]
        self.assertTrue(numpy.all(data == expected_data))

    def test_01_11_save_to_image_subfolder(self):
        '''Test saving to a subfolder of the image folder

        Regression test of IMG-978
        '''
        img_filename = os.path.join(self.new_image_directory, "test", 'img1.tiff')
        workspace, module = self.make_workspace(numpy.zeros((10, 10)))
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = "img1"
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF
        module.pathname.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_SUBFOLDER_NAME
        module.pathname.custom_path = "test"
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))

    def test_01_12_save_to_output_subfolder(self):
        '''Test saving to a subfolder of the image folder

        Regression test of IMG-978
        '''
        img_filename = os.path.join(self.new_output_directory, "test", 'img1.tiff')
        workspace, module = self.make_workspace(numpy.zeros((10, 10)))
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = "img1"
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF
        module.pathname.dir_choice = cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.pathname.custom_path = "test"
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))

    def test_01_13_save_with_metadata(self):
        '''Test saving to a custom folder with metadata in the path'''
        img_filename = os.path.join(self.new_output_directory, "test", 'img1.tif')
        workspace, module = self.make_workspace(numpy.zeros((10, 10)))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        m.add_image_measurement("Metadata_T","test")
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = "img1"
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        module.pathname.dir_choice = cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.pathname.custom_path = "\\g<T>"
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))

    def test_01_14_01_create_subdirectories(self):
        img_path = os.path.join(self.new_output_directory, "test")
        input_path = os.path.join(self.new_image_directory, "test")
        # Needed for relpath
        os.mkdir(input_path)
        img_filename = os.path.join(img_path, 'img1.tif')
        workspace, module = self.make_workspace(numpy.zeros((10, 10)))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        m.add_image_measurement("FileName_"+FILE_IMAGE_NAME, "img1.tif")
        m.add_image_measurement("PathName_"+FILE_IMAGE_NAME, input_path)
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        module.file_image_name.value = FILE_IMAGE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        module.pathname.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
        module.create_subdirectories.value = True
        module.root_dir.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))

    def test_01_14_02_create_subdirectories_custom_path(self):
        #
        # Use something other than the default input directory for
        # the root
        #
        root_path, subfolder = os.path.split(self.new_image_directory)
        img_path = os.path.join(self.new_output_directory, subfolder, "test")
        input_path = os.path.join(self.new_image_directory, "test")
        # Needed for relpath
        os.makedirs(input_path)
        img_filename = os.path.join(img_path, 'img1.tif')
        workspace, module = self.make_workspace(numpy.zeros((10, 10)))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        m.add_image_measurement("FileName_"+FILE_IMAGE_NAME, "img1.tif")
        m.add_image_measurement("PathName_"+FILE_IMAGE_NAME, input_path)
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        module.file_image_name.value = FILE_IMAGE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        module.pathname.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
        module.create_subdirectories.value = True
        module.root_dir.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.root_dir.custom_path = root_path
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))

    def test_01_15_create_subdirectories_inherit_path(self):
        img_path1 = os.path.join(self.new_image_directory, "test1")
        img_path2 = os.path.join(self.new_image_directory, "test2")
        img1_filename = os.path.join(img_path1, 'img1.tif')
        img2_filename = os.path.join(img_path2, 'img2.tif')
        img1_out_filename = os.path.join(self.new_output_directory, "test1",
                                         'TEST0001.tif')
        img2_out_filename = os.path.join(self.new_output_directory, "test2",
                                         'TEST0002.tif')
        os.mkdir(img_path1)
        os.mkdir(img_path2)
        make_file(img1_filename, tests.modules.tif_8_1)
        make_file(img2_filename, tests.modules.tif_8_2)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cellprofiler.modules.loadimages.LoadImages()
        load_images.location.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        load_images.file_types.value = cellprofiler.modules.loadimages.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cellprofiler.modules.loadimages.MS_EXACT_MATCH
        load_images.descend_subdirectories.value = cellprofiler.modules.loadimages.SUB_ALL
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1

        apply_threshold = cellprofiler.modules.applythreshold.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cellprofiler.modules.applythreshold.TH_ABOVE_THRESHOLD
        apply_threshold.threshold_method.value = cellprofiler.modules.applythreshold.TM_MANUAL
        apply_threshold.manual_threshold.value = 1.0
        apply_threshold.binary.value = cellprofiler.modules.applythreshold.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cellprofiler.modules.saveimages.SaveImages()
        save_images.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cellprofiler.modules.saveimages.FN_SEQUENTIAL
        save_images.single_file_name.value = 'TEST'
        save_images.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        save_images.pathname.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
        save_images.create_subdirectories.value = True
        save_images.root_dir.dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
        save_images.update_file_names.value = True
        save_images.module_num = 3

        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        pn, fn = os.path.split(img1_out_filename)
        filenames = measurements.get_all_measurements('Image', 'FileName_Derived')
        pathnames = measurements.get_all_measurements('Image', 'PathName_Derived')
        self.assertEqual(filenames[0], fn)
        self.assertEqual(pathnames[0], pn)
        self.assertTrue(os.path.isfile(img1_out_filename), img1_out_filename + " does not exist")
        self.assertTrue(os.path.isfile(img2_out_filename), img2_out_filename + " does not exist")

    def test_02_01_prepare_to_create_batch(self):
        '''Test the "prepare_to_create_batch" method'''
        orig_path = '/foo/bar'
        def fn_alter_path(path, **varargs):
            self.assertEqual(path, orig_path)
            return '/imaging/analysis'
        module = cellprofiler.modules.saveimages.SaveImages()
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = orig_path
        module.prepare_to_create_batch(None, fn_alter_path)
        self.assertEqual(module.pathname.custom_path, '/imaging/analysis')

    def test_02_02_regression_prepare_to_create_batch(self):
        '''Make sure that "prepare_to_create_batch" handles metadata

        This is a regression test for IMG-200
        '''
        cmodule = cellprofiler.modules.createbatchfiles.CreateBatchFiles()
        module = cellprofiler.modules.saveimages.SaveImages()
        module.pathname.custom_path = '.\\\\\\g<Test>Outlines\\\\g<Run>_\\g<Plate>'
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.prepare_to_create_batch(None, cmodule.alter_path)
        self.assertEqual(module.pathname.custom_path, './\\g<Test>Outlines/g<Run>_\\g<Plate>')

    def test_02_03_create_batch_root_dir(self):
        # regression test of issue #813 - root_dir needs conversion
        orig_path = '/foo/bar'
        def fn_alter_path(path, **varargs):
            if path == orig_path:
                return '/imaging/analysis'
        module = cellprofiler.modules.saveimages.SaveImages()
        module.root_dir.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.root_dir.custom_path = orig_path
        module.create_subdirectories.value = True
        module.prepare_to_create_batch(None, fn_alter_path)
        self.assertEqual(module.root_dir.custom_path, '/imaging/analysis')


    def test_03_01_get_measurement_columns(self):
        module = cellprofiler.modules.saveimages.SaveImages()
        module.image_name.value = "MyImage"
        module.update_file_names.value = False
        self.assertEqual(len(module.get_measurement_columns(None)), 0)
        module.update_file_names.value = True
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns),2)
        for column in columns:
            self.assertEqual(column[0], "Image")
            self.assertTrue(column[1] in ("PathName_MyImage","FileName_MyImage"))

    def make_workspace(self, image, filename = None, path = None,
                       convert=True, save_objects=False, shape=None,
                       mask=None, cropping = None):
        '''Make a workspace and module appropriate for running saveimages'''
        module = cellprofiler.modules.saveimages.SaveImages()
        module.module_num = 1
        module.image_name.value = IMAGE_NAME
        module.objects_name.value = OBJECTS_NAME
        module.file_image_name.value = IMAGE_NAME

        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.ObjectSet()
        if save_objects:
            objects = cellprofiler.region.Objects()
            if save_objects == "ijv":
                objects.ijv = image
                if shape is not None:
                    objects.parent_image = cellprofiler.image.Image(numpy.zeros(shape))
            else:
                objects.segmented = image
            object_set.add_objects(objects, OBJECTS_NAME)
            module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_OBJECTS
        else:
            img = cellprofiler.image.Image(image, mask = mask, crop_mask= cropping,
                                           convert=convert)
            image_set.add(IMAGE_NAME, img)
            module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE

        m = cellprofiler.measurement.Measurements()
        if filename is not None:
            m.add_image_measurement('_'.join(("FileName", IMAGE_NAME)), filename)
        if path is not None:
            m.add_image_measurement('_'.join(("PathName", IMAGE_NAME)), path)

        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)

        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                     object_set, m, image_set_list)
        return workspace, module

    def test_04_01_save_with_image_name_and_metadata(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(30, 40))
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_FROM_IMAGE
        module.wants_file_name_suffix.value = True
        module.file_name_suffix.value = '\\g<Well>'
        module.file_format.value = cellprofiler.modules.saveimages.FF_PNG

        m = workspace.measurements
        m.add_image_measurement('Metadata_Well','C08')

        module.run(workspace)
        filename = os.path.join(cellprofiler.preferences.get_default_output_directory(),
                                "%sC08.%s" % (FILE_NAME, cellprofiler.modules.saveimages.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = bioformats.load_image(filename)
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(numpy.all(numpy.abs(image - pixel_data) < .02))

    def test_04_02_save_with_metadata(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(30, 40))
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = 'metadatatest\\g<Well>'
        module.file_format.value = cellprofiler.modules.saveimages.FF_PNG

        m = workspace.measurements
        m.add_image_measurement('Metadata_Well','C08')

        module.run(workspace)
        filename = os.path.join(cellprofiler.preferences.get_default_output_directory(),
                                "metadatatestC08.%s" % (cellprofiler.modules.saveimages.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = bioformats.load_image(filename)
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(numpy.all(numpy.abs(image - pixel_data) < .02))

    def test_04_03_clip(self):
        """Regression test of IMG-720: clip images with values outside of 0-1"""
        numpy.random.seed(43)
        image = numpy.random.uniform(size=(40, 30)) * 1.2 - .1
        expected = image.copy()
        expected[expected < 0] = 0
        expected[expected > 1] = 1
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = "foo"
        module.file_format.value = cellprofiler.modules.saveimages.FF_PNG
        module.rescale.value = False
        module.run(workspace)
        filename = os.path.join(cellprofiler.preferences.get_default_output_directory(),
                                "foo.%s" % (cellprofiler.modules.saveimages.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = bioformats.load_image(filename)
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(numpy.all(numpy.abs(expected - pixel_data) < .02))

    def test_04_04_rescale_gray(self):
        """Test rescaling a grayscale image

        Regression test of IMG-943
        """

        numpy.random.seed(44)
        expected = numpy.random.uniform(size=(10, 20))
        image = expected * .5
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = "foo"
        module.file_format.value = cellprofiler.modules.saveimages.FF_PNG
        module.rescale.value = True
        module.run(workspace)
        filename = os.path.join(cellprofiler.preferences.get_default_output_directory(),
                                "foo.%s" % (cellprofiler.modules.saveimages.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = bioformats.load_image(filename)
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(numpy.all(numpy.abs(expected - pixel_data) < .02))

    def test_04_05_rescale_color(self):
        """Test rescaling a color image"""

        numpy.random.seed(44)
        expected = numpy.random.uniform(size=(10, 20, 3))
        image = expected * .5
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
        module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = "foo"
        module.file_format.value = cellprofiler.modules.saveimages.FF_PNG
        module.rescale.value = True
        module.run(workspace)
        filename = os.path.join(cellprofiler.preferences.get_default_output_directory(),
                                "foo.%s" % (cellprofiler.modules.saveimages.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = bioformats.load_image(filename)
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(numpy.all(numpy.abs(expected - pixel_data) < .02))

    def run_movie(self, groupings=None, fn = None, color=False):
        '''Run a pipeline that produces a movie

        Returns a list containing the movie frames
        '''
        image_set_list = cellprofiler.image.ImageSetList()
        if groupings is None:
            nframes = 5
            key_names = []
            group_list = (({}, numpy.arange(nframes) + 1),)
        else:
            key_names, group_list = groupings
            nframes = sum([len(g[1]) for g in group_list])
        for i in range(nframes):
            image_set_list.get_image_set(i)

        numpy.random.seed(0)
        frames = [numpy.random.uniform(size=(128, 128, 3) if color else (128, 128))
                  for i in range(nframes)]
        measurements = cellprofiler.measurement.Measurements()
        pipeline = cellprofiler.pipeline.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))
        pipeline.add_listener(callback)
        module = cellprofiler.modules.saveimages.SaveImages()
        module.module_num = 1
        module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_MOVIE
        module.image_name.value = IMAGE_NAME
        module.file_image_name.value = IMAGE_NAME
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = FILE_NAME
        module.rescale.value = False

        if fn is not None:
            fn(module)
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(
            pipeline, module, None, None, measurements, image_set_list)
        self.assertTrue(module.prepare_run(workspace))
        is_first = True
        frame_iterator = iter(frames)
        first_image_set = True
        for group in group_list:
            self.assertTrue(module.prepare_group(workspace, group[0], group[1]))
            for image_number in group[1]:
                if not first_image_set:
                    measurements.next_image_set()
                else:
                    first_image_set = False
                image_set = image_set_list.get_image_set(image_number-1)
                img = cellprofiler.image.Image(frame_iterator.next())
                image_set.add(IMAGE_NAME, img)
                for key, value in group[0].iteritems():
                    measurements.add_image_measurement(key, value)
                workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set,
                                                             cellprofiler.region.ObjectSet(),
                                                             measurements, image_set_list)
                module.run(workspace)
            module.post_group(workspace, group)
        module.post_run(workspace)
        return frames

    def test_05_01_save_movie(self):
        frames = self.run_movie()
        for i, frame in enumerate(frames):
            path = os.path.join(self.custom_directory, FILE_NAME + ".avi")
            frame_out = bioformats.load_image(path, index=i)
            self.assertTrue(numpy.all(numpy.abs(frame - frame_out) < .05))

    def test_05_02_save_two_movies(self):
        '''Use metadata grouping to write two movies'''
        grouping = (('Metadata_test',),
                    (({'Metadata_test':"foo"}, [1,2,3,4,5]),
                     ({'Metadata_test':"bar"}, [6,7,8,9])))
        def fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
            module.single_file_name.value = r"\g<test>"
            module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        frames = self.run_movie(grouping, fn)
        for group in grouping[1]:
            path = os.path.join(self.custom_directory, group[0]["Metadata_test"] + ".avi")
            self.assertTrue(os.path.exists(path))
            for t,image_number in enumerate(group[1]):
                frame = frames[image_number-1]
                frame_out = bioformats.load_image(path, t=t)
                self.assertTrue(numpy.all(numpy.abs(frame - frame_out) < .05))

    def test_05_03_save_color_movie(self):
        '''Regression test of img-1227 - save a color movie saved in b/w

        also BioFormats crashed when saving in color, requiring update of
        loci_tools.jar
        '''
        frames = self.run_movie(color=True)
        for i, frame in enumerate(frames):
            path = os.path.join(self.custom_directory, FILE_NAME + ".avi")
            frame_out = bioformats.load_image(path, t=i)
            self.assertTrue(numpy.all(numpy.abs(frame - frame_out) < .05))

    def test_05_04_save_tif_movie(self):
        def fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
            module.movie_format.value = cellprofiler.modules.saveimages.FF_TIF
            module.overwrite.value = True

        for wants_color in False, True:
            frames = self.run_movie(fn=fn, color=wants_color)
            for i, frame in enumerate(frames):
                path = os.path.join(self.custom_directory, FILE_NAME + ".tif")
                frame_out = bioformats.load_image(path, index=i)
                self.assertTrue(numpy.all(numpy.abs(frame - frame_out) < .05))

    def test_05_05_save_mov_movie(self):
        def fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.saveimages.SaveImages))
            module.movie_format.value = cellprofiler.modules.saveimages.FF_MOV

        frames = self.run_movie(fn=fn)
        for i, frame in enumerate(frames):
            path = os.path.join(self.custom_directory, FILE_NAME + ".mov")
            frame_out = bioformats.load_image(path, index=i)
            self.assertTrue(numpy.all(numpy.abs(frame - frame_out) < .05))

    def test_06_01_save_image(self):
        numpy.random.seed(61)
        image8 = (numpy.random.uniform(size=(100, 100)) * 255).astype(numpy.uint8)
        image16 = (numpy.random.uniform(size=(100, 100)) * 65535).astype(numpy.uint16)
        imagefloat = numpy.random.uniform(size=(100, 100))
        image8s = (numpy.random.uniform(size=(100, 100)) * 245).astype(numpy.uint8)
        image8s[0,0] = 245
        image8s[0,1] = 0
        image16s = (numpy.random.uniform(size=(100, 100)) * 64535).astype(numpy.uint16)
        image16s[0,0] = 64535
        image16s[0,1] = 0
        imagefloats = imagefloat.copy()
        imagefloats[0,0] = 1
        imagefloats[0,1] = 0

        test_settings = [
            # 16-bit TIF from all image types
            {'rescale'       : False,
             'file_format'   : cellprofiler.modules.saveimages.FF_TIF,
             'bit_depth'     : cellprofiler.modules.saveimages.BIT_DEPTH_16,
             'input_image'   : imagefloat,
             'expected'      : (imagefloat * 65535).astype(numpy.uint16)},
            {'rescale'       : False,
             'file_format'   : cellprofiler.modules.saveimages.FF_TIF,
             'bit_depth'     : cellprofiler.modules.saveimages.BIT_DEPTH_8,
             'input_image'   : imagefloat,
             'expected'      : (imagefloat * 255).astype(numpy.uint8)},
            {'rescale'       : False,
             'file_format'   : cellprofiler.modules.saveimages.FF_TIF,
             'bit_depth'     : cellprofiler.modules.saveimages.BIT_DEPTH_16,
             'input_image'   : image16s,
             'expected'      : image16s },

            # Rescaled 16-bit image
            {'rescale'       : True,
             'file_format'   : cellprofiler.modules.saveimages.FF_TIF,
             'bit_depth'     : cellprofiler.modules.saveimages.BIT_DEPTH_16,
             'input_image'   : imagefloats / 2,
             'expected'      : imagefloats*65535. },
            {'rescale'       : True,
             'file_format'   : cellprofiler.modules.saveimages.FF_TIF,
             'bit_depth'     : cellprofiler.modules.saveimages.BIT_DEPTH_8,
             'input_image'   : imagefloats / 2,
             'expected'      : imagefloats * 255 },
            # Rescaled 32-bit float
            {'rescale'       : True,
             'file_format'   : cellprofiler.modules.saveimages.FF_TIF,
             'bit_depth'     : cellprofiler.modules.saveimages.BIT_DEPTH_FLOAT,
             'input_image'   : imagefloats / 2,
             'expected'      : imagefloats },
            # Unscaled 32-bit float
            {'rescale'       : False,
             'file_format'   : cellprofiler.modules.saveimages.FF_TIF,
             'bit_depth'     : cellprofiler.modules.saveimages.BIT_DEPTH_FLOAT,
             'input_image'   : (imagefloats * 16).astype(numpy.float32),
             'expected'      : (imagefloats * 16).astype(numpy.float32)}
        ]

        for i, setting in enumerate(test_settings):
            # Adjust settings each round and retest
            workspace, module = self.make_workspace(setting['input_image'],
                                                    convert=False)

            module.module_num = 1
            module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
            module.image_name.value = IMAGE_NAME
            module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
            module.pathname.custom_path = self.custom_directory
            module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
            module.single_file_name.value = FILE_NAME+str(i)

            module.rescale.value = setting['rescale']
            module.file_format.value = setting['file_format']
            module.bit_depth.value = setting['bit_depth']


            module.save_image(workspace)

            expected = setting['expected']
            filename = module.get_filename(workspace,
                                           make_dirs = False,
                                           check_overwrite = False)
            if expected.ndim == 2:
                expected = expected.reshape(expected.shape[0],
                                            expected.shape[1], 1)
            for index in range(expected.shape[2]):
                im = bioformats.load_image(filename, rescale=False)
                self.assertTrue(numpy.all(numpy.abs(im - expected[:, :, index]) <= 1))
            if os.path.isfile(filename):
                try:
                    os.remove(filename)
                except:
                    logger.warn(
                        "Not ideal, Bioformats still holding onto file handle.",
                        exc_info=True)

    def test_06_02_save_bmp(self):
        # Special code for saving bitmaps
        r = numpy.random.RandomState()
        r.seed(62)
        images = [
            r.uniform(size=(16, 20)),
            r.uniform(size=(15, 20)),
            r.uniform(size=(16, 20, 3)),
            r.uniform(size=(15, 20, 3)) ]
        for i, image in enumerate(images):
            # Adjust settings each round and retest
            workspace, module = self.make_workspace(image,
                                                    convert=False)

            module.module_num = 1
            module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_IMAGE
            module.image_name.value = IMAGE_NAME
            module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
            module.pathname.custom_path = self.custom_directory
            module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
            module.single_file_name.value = FILE_NAME+str(i)

            module.rescale.value = False
            module.file_format.value = cellprofiler.modules.saveimages.FF_BMP
            module.bit_depth.value = cellprofiler.modules.saveimages.BIT_DEPTH_8

            module.save_image(workspace)

            expected = (image * 255).astype(numpy.uint8)
            filename = module.get_filename(workspace,
                                           make_dirs = False,
                                           check_overwrite = False)
            im = bioformats.load_image(filename, rescale=False)
            numpy.testing.assert_array_equal(im, expected)
            if os.path.isfile(filename):
                try:
                    os.remove(filename)
                except:
                    logger.warn(
                        "Not ideal, Bioformats still holding onto file handle.",
                        exc_info=True)

    def test_06_03_save_mask(self):
        # regression test of issue #1215
        r = numpy.random.RandomState()
        r.seed(63)
        for image_type in cellprofiler.modules.saveimages.IF_MASK, cellprofiler.modules.saveimages.IF_CROPPING:
            image = r.uniform(size=(11, 15)) > .5
            if image_type == cellprofiler.modules.saveimages.IF_MASK:
                workspace, module = self.make_workspace(
                    numpy.zeros(image.shape), mask=image)
            else:
                workspace, module = self.make_workspace(
                    numpy.zeros(image.shape), cropping=image)

            assert isinstance(module, cellprofiler.modules.saveimages.SaveImages)
            module.save_image_or_figure.value = image_type
            module.image_name.value = IMAGE_NAME
            module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
            module.pathname.custom_path = self.custom_directory
            module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
            module.single_file_name.value = FILE_NAME
            module.file_format.value = cellprofiler.modules.saveimages.FF_TIF
            #
            # bug would make it throw an exception here
            #
            module.save_image(workspace)
            filename = module.get_filename(workspace,
                                           make_dirs = False,
                                           check_overwrite = False)
            im = bioformats.load_image(filename, rescale=False)
            numpy.testing.assert_array_equal(im > 0, image)
            if os.path.isfile(filename):
                try:
                    os.remove(filename)
                except:
                    logger.warn(
                        "Not ideal, Bioformats still holding onto file handle.",
                        exc_info=True)


    def test_07_01_save_objects_grayscale8_tiff(self):
        r = numpy.random.RandomState()
        r.seed(71)
        labels = r.randint(0, 10, size=(30,20))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cellprofiler.modules.saveimages.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cellprofiler.modules.saveimages.GC_GRAYSCALE
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        feature = cellprofiler.modules.saveimages.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cellprofiler.modules.saveimages.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        im = bioformats.load_image(filename, rescale=False)
        self.assertTrue(numpy.all(labels == im))

    def test_07_02_save_objects_grayscale_16_tiff(self):
        r = numpy.random.RandomState()
        labels = r.randint(0, 300, size=(300,300))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cellprofiler.modules.saveimages.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cellprofiler.modules.saveimages.GC_GRAYSCALE
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        feature = cellprofiler.modules.saveimages.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cellprofiler.modules.saveimages.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        im = bioformats.load_image(filename, rescale=False)
        self.assertTrue(numpy.all(labels == im))

    def test_07_03_save_objects_grayscale_png(self):
        r = numpy.random.RandomState()
        labels = r.randint(0, 10, size=(30,20))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cellprofiler.modules.saveimages.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cellprofiler.modules.saveimages.GC_GRAYSCALE
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_PNG

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        feature = cellprofiler.modules.saveimages.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cellprofiler.modules.saveimages.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        im = bioformats.load_image(filename, rescale=False)
        self.assertTrue(numpy.all(labels == im))

    def test_07_04_save_objects_color_png(self):
        r = numpy.random.RandomState()
        labels = r.randint(0, 10, size=(30,20))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cellprofiler.modules.saveimages.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cellprofiler.modules.saveimages.GC_COLOR
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_PNG

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        feature = cellprofiler.modules.saveimages.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cellprofiler.modules.saveimages.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        im = bioformats.load_image(filename, rescale=False)
        im.shape = (im.shape[0] * im.shape[1], im.shape[2])
        order = numpy.lexsort(im.transpose())
        different = numpy.hstack(([False], numpy.any(im[order[:-1], :] != im[order[1:], :], 1)))
        indices = numpy.cumsum(different)
        im = numpy.zeros(labels.shape, int)
        im.ravel()[order] = indices
        #
        # There should be a 1-1 correspondence between label #s and indices
        #
        x = scipy.sparse.coo.coo_matrix((numpy.ones(len(indices)),
                                         (labels.ravel(), im.ravel()))).toarray()
        self.assertEqual(numpy.sum(x != 0), 10)

    def test_07_05_save_overlapping_objects(self):
        r = numpy.random.RandomState()
        i,j = numpy.mgrid[0:20, 0:25]
        o1 = (i-10) ** 2 + (j - 10) **2 < 64
        o2 = (i-10) ** 2 + (j - 15) **2 < 64
        ijv = numpy.vstack(
            [numpy.column_stack((i[o], j[o], numpy.ones(numpy.sum(o), int) * (n + 1)))
             for n, o in enumerate((o1, o2))])
        workspace, module = self.make_workspace(ijv, save_objects = "ijv",
                                                shape = o1.shape)
        assert isinstance(module, cellprofiler.modules.saveimages.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cellprofiler.modules.saveimages.GC_GRAYSCALE
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        feature = cellprofiler.modules.saveimages.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cellprofiler.modules.saveimages.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        feature = cellprofiler.modules.loadimages.C_OBJECTS_URL + "_" + OBJECTS_NAME
        m_url = m.get_current_image_measurement(feature)
        self.assertEqual(m_url, cellprofiler.modules.loadimages.pathname2url(filename))
        for i in range(2):
            im = bioformats.load_image(filename, index=i, rescale=False)
            o = o1 if 1 in numpy.unique(im) else o2
            self.assertEqual(tuple(im.shape), tuple(o.shape))
            numpy.testing.assert_array_equal(
                o, im.astype(bool))

    def test_07_06_save_three_planes(self):
        #
        # A constant source of confusion: if an image has three planes,
        # isn't it RGB?
        #
        ijv = numpy.array([[5, 6, 1],
                           [ 5, 6, 2],
                           [ 5, 6, 3]])
        workspace, module = self.make_workspace(ijv, save_objects = "ijv",
                                                shape = (10, 15))
        assert isinstance(module, cellprofiler.modules.saveimages.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cellprofiler.modules.saveimages.GC_GRAYSCALE
        module.pathname.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIFF

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        metadata = bioformats.get_omexml_metadata(filename)
        planes = []
        for i in range(3):
            planes.append(bioformats.load_image(
                filename, index=i, rescale=False))
        img = numpy.dstack(planes)
        mask = numpy.ones(img.shape, bool)
        mask[5, 6, :] = False
        self.assertTrue(numpy.all(img[mask] == 0))
        objects = img[~mask]
        self.assertEqual(len(objects), 3)
        self.assertEqual((1, 2, 3), tuple(sorted(objects)))

    def test_07_07_save_objects_last_cycle(self):
        # Regression test of issue #1296
        m = cellprofiler.measurement.Measurements()
        object_set = cellprofiler.region.ObjectSet()
        l = numpy.zeros((11, 17), numpy.int32)
        l[2:-2, 2:-2]  = 1
        objects = cellprofiler.region.Objects()
        objects.segmented = l
        object_set.add_objects(objects, OBJECTS_NAME)
        module = cellprofiler.modules.saveimages.SaveImages()
        module.save_image_or_figure.value = cellprofiler.modules.saveimages.IF_OBJECTS
        module.file_name_method.value = cellprofiler.modules.saveimages.FN_SINGLE_NAME
        module.single_file_name.value = FILE_NAME
        module.file_format.value = cellprofiler.modules.saveimages.FF_TIF
        module.pathname.dir_choice = cellprofiler.modules.saveimages.DEFAULT_INPUT_FOLDER_NAME
        module.when_to_save.value = cellprofiler.modules.saveimages.WS_LAST_CYCLE
        module.objects_name.value = OBJECTS_NAME
        module.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, m, object_set, m, None)
        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.post_group(workspace)
        pixel_data = bioformats.load_image(filename, rescale=False)
        numpy.testing.assert_array_equal(pixel_data, l)

def make_array(encoded, shape, dtype=numpy.uint8):
    data = base64.b64decode(encoded)
    a = numpy.fromstring(data, dtype)
    return a.reshape(shape)

def make_file(filename, encoded):
    data = base64.b64decode(encoded)
    fid = open(filename,'wb')
    fid.write(data)
    fid.close()
