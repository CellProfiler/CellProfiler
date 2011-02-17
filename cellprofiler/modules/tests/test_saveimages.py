"""test_saveimages - test the saveimages module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import base64
import matplotlib.image
import numpy as np
import os
import sys
import Image as PILImage
from StringIO import StringIO
import unittest
import tempfile
import traceback
import zlib
from scipy.sparse import coo

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.modules.saveimages as cpm_si
import cellprofiler.modules.loadimages as cpm_li
import cellprofiler.modules.applythreshold as cpm_a
import cellprofiler.cpimage as cpi
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs
import cellprofiler.modules.createbatchfiles as cpm_c
from cellprofiler.cpmath.filter import stretch
from cellprofiler.utilities.get_proper_case_filename import get_proper_case_filename

import cellprofiler.modules.tests as cpmt
IMAGE_NAME = 'inputimage'
OBJECTS_NAME = 'inputobjects'
FILE_IMAGE_NAME = 'fileimage'
FILE_NAME = 'filenm'

from cellprofiler.utilities import jutil
import bioformats
jutil.attach()
env = jutil.get_env()
klass = env.find_class('loci/formats/FormatTools')
bioformats_revision = jutil.get_static_field(klass, 'SVN_REVISION', 'Ljava/lang/String;')
BIOFORMATS_CANT_WRITE = (bioformats_revision == '6181' and sys.platform=='darwin')


class TestSaveImages(unittest.TestCase):
    def setUp(self):
        # Change the default image directory to a temporary file
        cpprefs.set_headless()
        self.new_image_directory = get_proper_case_filename(tempfile.mkdtemp())
        cpprefs.set_default_image_directory(self.new_image_directory)
        self.new_output_directory = get_proper_case_filename(tempfile.mkdtemp())
        cpprefs.set_default_output_directory(self.new_output_directory)
        self.custom_directory = get_proper_case_filename(tempfile.mkdtemp())
    
    def tearDown(self):
        for subdir in (self.new_image_directory, self.new_output_directory,
                       self.custom_directory):
            for filename in os.listdir(subdir):
                try:
                    os.remove(os.path.join(subdir, filename))
                except:
                    sys.stderr.write("Failed to remove %s" % filename)
                    traceback.print_exc()
            try:
                os.rmdir(subdir)
            except:
                sys.stderr.write("Failed to remove %s directory" % subdir)
                traceback.print_exc()
    
    def on_event(self, pipeline, event):
        self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        
    def test_00_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwTy5RMDBSMDS1Mja2MrBUMDIAEiQDBkZPX34GBoYsRgaG'
                'ijl7J5/1OuwgcFw981aX0+uLk7/cmCy/XU701cIFKyU0q+5cy4zw2KQof/SH'
                'jVwNS3+tiXnupbxpF9Z0WxioGFvvn/tw/25bBokyRqs/1Ufutqzd2n1DwKZf'
                '1kdIakehtI6F9qfsd8ZCgUbfC9z2iCTfiJw5n8X725zW3UvZv02ryhCO13mW'
                'tvyfzOKKXToHk35O3Lf4+QX+llVTU/13LDJMdTwo/vv0zdj4aR611Xf2R1XL'
                '9kjJ/nKyW7+/qXZvaPB9oVf+lSbb8s3vrGh8HbYj16Z3RfQnc94/G488/ziD'
                'l2kazyWJr8/5mcM7jbXmMIp3/U3JW2L5XNs+WnSun8rcTz/yWgPNIlK4+aeW'
                'Tnq+L/zJGa70prNXLFYfinzgpvL7fPVC6+166vPzCzzN7pjL1K1Pso+tXeEf'
                'U6I8ra1+v/8Ng/0V60t+W6W0Tt5Tvue++5Xdly9cf1L/V8rvqWxM9rfXmQVi'
                '6vbnt985rV8qK7dCf+2Z/wwneDJMAawzzdI=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        self.assertEqual(module.image_name.value, "DNA")
        self.assertEqual(module.file_image_name.value, "OrigDNA")
        self.assertEqual(module.file_name_method.value, cpm_si.FN_FROM_IMAGE)
        self.assertEqual(module.pathname.dir_choice, 
                         cps.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.when_to_save.value, cpm_si.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap.value, cpm_si.CM_GRAY)
        self.assertFalse(module.overwrite)
    
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
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        self.assertEqual(module.image_name.value, "DNA")
        self.assertEqual(module.file_image_name.value, "OrigDNA")
        self.assertEqual(module.file_name_method.value, cpm_si.FN_FROM_IMAGE)
        self.assertEqual(module.pathname.dir_choice, 
                         cps.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.when_to_save.value, cpm_si.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap.value, cpm_si.CM_GRAY)
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
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))        
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        self.assertEqual(module.image_name.value, "DNA")
        self.assertEqual(module.file_image_name.value, "OrigDNA")
        self.assertEqual(module.file_name_method.value, cpm_si.FN_FROM_IMAGE)
        self.assertEqual(module.pathname.dir_choice, cps.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.when_to_save.value, cpm_si.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap.value, cpm_si.CM_GRAY)
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
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))        
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        self.assertEqual(module.save_image_or_figure,
                         cpm_si.IF_IMAGE)
        self.assertEqual(module.image_name, "ColorOutlineImage")
        self.assertEqual(module.figure_name, "Fig")
        self.assertEqual(module.file_name_method, cpm_si.FN_FROM_IMAGE)
        self.assertEqual(module.file_image_name, "OrigRGB")
        self.assertEqual(module.single_file_name, "OrigBlue")
        self.assertEqual(module.wants_file_name_suffix, True)
        self.assertEqual(module.file_name_suffix, "_outlines")
        self.assertEqual(module.file_format, cpm_si.FF_PNG)
        self.assertEqual(module.pathname.dir_choice, 
                         cps.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, 
                         "./\\g<Directory>/\\g<Subdirectory>")
        self.assertEqual(module.bit_depth, "8")
        self.assertTrue(module.overwrite)
        self.assertEqual(module.when_to_save, cpm_si.WS_EVERY_CYCLE)
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
    Select file format to use:gif
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
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))        
        self.assertEqual(len(pipeline.modules()), 6)
        sif = [ cpm_si.IF_IMAGE, cpm_si.IF_MASK, cpm_si.IF_CROPPING,
                cpm_si.IF_MOVIE, cpm_si.IF_FIGURE, cpm_si.IF_IMAGE]
        fnm = [ cpm_si.FN_FROM_IMAGE, cpm_si.FN_SEQUENTIAL,
                cpm_si.FN_SINGLE_NAME, cpm_si.FN_SINGLE_NAME,
                cpm_si.FN_FROM_IMAGE,
                cpm_si.FN_FROM_IMAGE]
        suf = [ False, True, False, False, True, False]
        ff = [cpm_si.FF_BMP, cpm_si.FF_GIF, cpm_si.FF_JPG,
              cpm_si.FF_JPG, cpm_si.FF_PNG, cpm_si.FF_PNG]
        ov = [ False, True, False, False, False, False]
        wts = [ cpm_si.WS_EVERY_CYCLE, cpm_si.WS_FIRST_CYCLE,
                cpm_si.WS_LAST_CYCLE, cpm_si.WS_LAST_CYCLE,
                cpm_si.WS_EVERY_CYCLE, cpm_si.WS_EVERY_CYCLE]
        dir_choice = [cps.DEFAULT_OUTPUT_FOLDER_NAME,
                      cps.DEFAULT_INPUT_FOLDER_NAME,
                      cpm_si.PC_WITH_IMAGE,
                      cps.ABSOLUTE_FOLDER_NAME,
                      cps.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                      cps.DEFAULT_INPUT_SUBFOLDER_NAME]
        rescale = [ False, False, True, False, False, False]
        cm = [ "gray", "copper", "gray", "gray", "gray", "gray" ]
        up = [ True, False, True, False, False, False]
        cre = [ False, True, False, False, False, True]
        for i, module in enumerate(pipeline.modules()):
            self.assertTrue(isinstance(module, cpm_si.SaveImages))
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
            self.assertEqual(module.bit_depth, "8")
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
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))        
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        self.assertEqual(module.save_image_or_figure, cpm_si.IF_IMAGE)
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.figure_name, "MyFigure")
        self.assertEqual(module.file_name_method, cpm_si.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "MyImage")
        self.assertEqual(module.single_file_name, r"DNA_\g<WellColumn>")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "MySuffix")
        self.assertEqual(module.file_format, cpm_si.FF_BMP)
        self.assertEqual(module.pathname.dir_choice, cps.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"DNA_\g<WellRow>")
        self.assertEqual(module.bit_depth, 8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cpm_si.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap, "gray")
        self.assertFalse(module.update_file_names)
        self.assertFalse(module.create_subdirectories)
        
    def test_00_07_load_v7(self):
        data = """CellProfiler Pipeline: http://www.cellprofiler.org
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
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))        
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        self.assertEqual(module.save_image_or_figure, cpm_si.IF_OBJECTS)
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.file_name_method, cpm_si.FN_SINGLE_NAME)
        self.assertEqual(module.file_image_name, "None")
        self.assertEqual(module.single_file_name, r"\g<Well>_Nuclei")
        self.assertFalse(module.wants_file_name_suffix)
        self.assertEqual(module.file_name_suffix, "Whatever")
        self.assertEqual(module.file_format, cpm_si.FF_PNG)
        self.assertEqual(module.pathname.dir_choice, cps.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.pathname.custom_path, r"None")
        self.assertEqual(module.bit_depth, 8)
        self.assertFalse(module.overwrite)
        self.assertEqual(module.when_to_save, cpm_si.WS_EVERY_CYCLE)
        self.assertFalse(module.rescale)
        self.assertEqual(module.gray_or_color, cpm_si.GC_GRAYSCALE)
        self.assertEqual(module.colormap, "Default")
        self.assertFalse(module.update_file_names)
        self.assertFalse(module.create_subdirectories)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        self.assertEqual(module.gray_or_color, cpm_si.GC_COLOR)
        
    def test_01_01_save_first_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tiff')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tiff')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_ABOVE_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 1.0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname.dir_choice = cpm_si.PC_WITH_IMAGE
        save_images.when_to_save.value = cpm_si.WS_FIRST_CYCLE
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
        data = matplotlib.image.imread(img1_out_filename)
        expected_data = matplotlib.image.imread(img1_filename) 
        self.assertTrue(np.all(data[expected_data < 255] ==
                                  expected_data[expected_data < 255]))
        self.assertTrue(np.all(data[expected_data == 255] == 0))

    def test_01_02_save_all_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tiff')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tiff')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname.dir_choice = cpm_si.PC_WITH_IMAGE
        save_images.when_to_save.value = cpm_si.WS_EVERY_CYCLE
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
        data = matplotlib.image.imread(img1_out_filename)
        expected_data = matplotlib.image.imread(img1_filename) 
        self.assertTrue(np.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(np.all(data==expected_data))

    def test_01_03_save_last_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tiff')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tiff')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname.dir_choice = cpm_si.PC_WITH_IMAGE
        save_images.when_to_save.value = cpm_si.WS_LAST_CYCLE
        save_images.update_file_names.value = False
        save_images.module_num = 3
        
        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertFalse(os.path.isfile(img1_out_filename))
        self.assertTrue(os.path.isfile(img2_out_filename))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(np.all(data==expected_data))

    def test_01_04_save_all_to_output_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_output_directory,'img1OUT.tiff')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_output_directory,'img2OUT.tiff')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname.dir_choice = cps.DEFAULT_OUTPUT_FOLDER_NAME
        save_images.when_to_save.value = cpm_si.WS_EVERY_CYCLE
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
        data = matplotlib.image.imread(img1_out_filename)
        expected_data = matplotlib.image.imread(img1_filename)
        self.assertTrue(np.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(np.all(data==expected_data))

    def test_01_05_save_all_to_custom_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.tiff')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.tiff')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cpm_si.WS_EVERY_CYCLE
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
        data = matplotlib.image.imread(img1_out_filename)
        expected_data = matplotlib.image.imread(img1_filename) 
        self.assertTrue(np.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(np.all(data==expected_data))

    def test_01_06_save_all_to_custom_png(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.png')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.png')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_PNG
        save_images.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cpm_si.WS_EVERY_CYCLE
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
        pil = PILImage.open(img1_out_filename)
        data = matplotlib.image.pil_to_array(pil)
        pil = PILImage.open(img1_filename)
        expected_data = matplotlib.image.pil_to_array(pil) 
        self.assertTrue(np.all(data==expected_data))
        pil = PILImage.open(img2_out_filename)
        data = matplotlib.image.pil_to_array(pil)
        pil = PILImage.open(img2_filename)
        expected_data = matplotlib.image.pil_to_array(pil) 
        self.assertTrue(np.all(data==expected_data))

    def test_01_07_save_all_to_custom_jpg(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.jpeg')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.jpeg')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_JPG
        save_images.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cpm_si.WS_EVERY_CYCLE
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
        data = matplotlib.image.imread(img1_out_filename)
        expected_data = matplotlib.image.imread(img1_filename) 
        #crud - no lossless jpeg in PIL
        self.assertTrue(np.all(np.abs(data.astype(int)-
                                            expected_data.astype(int))<=4))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(np.all(np.abs(data.astype(int)-
                                            expected_data.astype(int))<=4))

    def test_01_08_save_all_to_custom_gif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.gif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.gif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_GIF
        save_images.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cpm_si.WS_EVERY_CYCLE
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
        data = matplotlib.image.imread(img1_out_filename)
        expected_data = matplotlib.image.imread(img1_filename) 
        self.assertTrue(np.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(np.all(data==expected_data))

    def test_01_09_save_single_to_custom_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.tiff')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.tiff')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_SINGLE_NAME
        save_images.single_file_name.value ='img1OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cpm_si.WS_FIRST_CYCLE
        save_images.update_file_names.value = False
        save_images.module_num = 3
        
        pipeline.add_module(load_images)
        pipeline.add_module(apply_threshold)
        pipeline.add_module(save_images)
        pipeline.test_valid()
        measurements = pipeline.run()
        self.assertTrue(os.path.isfile(img1_out_filename))
        data = matplotlib.image.imread(img1_out_filename)
        expected_data = matplotlib.image.imread(img1_filename) 
        self.assertTrue(np.all(data==expected_data))
        
    def test_01_10_save_all_to_custom_png_rgb(self):
        '''Tests the path of saving an image with a colormap other than gray'''
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.png')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.png')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0].common_text.value = '.tif'
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low_or_high.value = cpm_a.TH_BELOW_THRESHOLD
        apply_threshold.threshold_method.value = cpm_a.TM_MANUAL
        apply_threshold.manual_threshold.value = 0
        apply_threshold.binary.value = cpm_a.GRAYSCALE
        apply_threshold.module_num = 2

        save_images = cpm_si.SaveImages()
        save_images.save_image_or_figure.value = cpm_si.IF_IMAGE
        save_images.image_name.value = 'Derived'
        save_images.file_image_name.value = 'Orig'
        save_images.file_name_method.value = cpm_si.FN_FROM_IMAGE
        save_images.wants_file_name_suffix.value = True
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_PNG
        save_images.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        save_images.pathname.custom_path = self.custom_directory
        save_images.when_to_save.value = cpm_si.WS_EVERY_CYCLE
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
        pil = PILImage.open(img1_out_filename)
        data = matplotlib.image.pil_to_array(pil)
        pil = PILImage.open(img1_filename)
        mapper = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.jet)
        expected_data = mapper.to_rgba(matplotlib.image.pil_to_array(pil), 
                                       bytes=True)
        self.assertTrue(np.all(data==expected_data))
        pil = PILImage.open(img2_out_filename)
        data = matplotlib.image.pil_to_array(pil)
        pil = PILImage.open(img2_filename)
        mapper = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.jet)
        expected_data = mapper.to_rgba(matplotlib.image.pil_to_array(pil), 
                                       bytes=True)
        self.assertTrue(np.all(data==expected_data))
        
    def test_01_11_save_to_image_subfolder(self):
        '''Test saving to a subfolder of the image folder
        
        Regression test of IMG-978
        '''
        img_filename = os.path.join(self.new_image_directory, "test", 'img1.tiff')
        workspace, module = self.make_workspace(np.zeros((10,10)))
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = "img1"
        module.file_format.value = cpm_si.FF_TIFF
        module.pathname.dir_choice = cps.DEFAULT_INPUT_SUBFOLDER_NAME
        module.pathname.custom_path = "test"
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))
        
    def test_01_12_save_to_output_subfolder(self):
        '''Test saving to a subfolder of the image folder
        
        Regression test of IMG-978
        '''
        img_filename = os.path.join(self.new_output_directory, "test", 'img1.tiff')
        workspace, module = self.make_workspace(np.zeros((10,10)))
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = "img1"
        module.file_format.value = cpm_si.FF_TIFF
        module.pathname.dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.pathname.custom_path = "test"
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))
        
    def test_01_13_save_with_metadata(self):
        '''Test saving to a custom folder with metadata in the path'''
        img_filename = os.path.join(self.new_output_directory, "test", 'img1.tiff')
        workspace, module = self.make_workspace(np.zeros((10,10)))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpm.Measurements))
        m.add_image_measurement("Metadata_T","test")
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = "img1"
        module.file_format.value = cpm_si.FF_TIFF
        module.pathname.dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.pathname.custom_path = "\\g<T>"
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))
        
    def test_01_14_create_subdirectories(self):
        img_path = os.path.join(self.new_output_directory, "test")
        input_path = os.path.join(self.new_image_directory, "test")
        # Needed for relpath
        os.mkdir(input_path)
        img_filename = os.path.join(img_path, 'img1.tiff')
        workspace, module = self.make_workspace(np.zeros((10,10)))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpm.Measurements))
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        m.add_image_measurement("FileName_"+FILE_IMAGE_NAME, "img1.tiff")
        m.add_image_measurement("PathName_"+FILE_IMAGE_NAME, input_path)
        module.file_name_method.value = cpm_si.FN_FROM_IMAGE
        module.file_image_name.value = FILE_IMAGE_NAME
        module.file_format.value = cpm_si.FF_TIFF
        module.pathname.dir_choice = cps.DEFAULT_OUTPUT_FOLDER_NAME
        module.create_subdirectories.value = True
        module.run(workspace)
        self.assertTrue(os.path.exists(img_filename))
        
    def test_02_01_prepare_to_create_batch(self):
        '''Test the "prepare_to_create_batch" method'''
        orig_path = '/foo/bar'
        def fn_alter_path(path, **varargs):
            self.assertEqual(path, orig_path)
            return '/imaging/analysis'
        module = cpm_si.SaveImages()
        module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = orig_path
        module.prepare_to_create_batch(None,None, fn_alter_path)
        self.assertEqual(module.pathname.custom_path, '/imaging/analysis')
    
    def test_02_02_regression_prepare_to_create_batch(self):
        '''Make sure that "prepare_to_create_batch" handles metadata

        This is a regression test for IMG-200
        '''
        cmodule = cpm_c.CreateBatchFiles()
        module = cpm_si.SaveImages()
        module.pathname.custom_path = '.\\\\\\g<Test>Outlines\\\\g<Run>_\\g<Plate>'
        module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.prepare_to_create_batch(None,None, cmodule.alter_path)
        self.assertEqual(module.pathname.custom_path, './\\g<Test>Outlines/g<Run>_\\g<Plate>')
    
    def test_03_01_get_measurement_columns(self):
        module = cpm_si.SaveImages()
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
                       convert=True, save_objects=False):
        '''Make a workspace and module appropriate for running saveimages'''
        module = cpm_si.SaveImages()
        module.module_num = 1
        module.image_name.value = IMAGE_NAME
        module.objects_name.value = OBJECTS_NAME
        module.file_image_name.value = IMAGE_NAME

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        if save_objects:
            objects = cpo.Objects()
            objects.segmented = image
            object_set.add_objects(objects, OBJECTS_NAME)
            module.save_image_or_figure.value = cpm_si.IF_OBJECTS
        else:
            img = cpi.Image(image, convert=convert)
            image_set.add(IMAGE_NAME, img)
            module.save_image_or_figure.value = cpm_si.IF_IMAGE
        
        m = cpm.Measurements()
        if filename is not None:
            m.add_image_measurement('_'.join(("FileName", IMAGE_NAME)), filename)
        if path is not None:
            m.add_image_measurement('_'.join(("PathName", IMAGE_NAME)), path)
            
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, m, image_set_list)
        return workspace, module
    
    def test_04_01_save_with_image_name_and_metadata(self):
        np.random.seed(0)
        image = np.random.uniform(size=(30,40))
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.save_image_or_figure.value = cpm_si.IF_IMAGE
        module.file_name_method.value = cpm_si.FN_FROM_IMAGE
        module.wants_file_name_suffix.value = True
        module.file_name_suffix.value = '\\g<Well>'
        module.file_format.value = cpm_si.FF_PNG
        
        m = workspace.measurements
        m.add_image_measurement('Metadata_Well','C08')
        
        module.run(workspace)
        filename = os.path.join(cpprefs.get_default_output_directory(),
                                "%sC08.%s" %(FILE_NAME, cpm_si.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = cpm_li.load_using_PIL(filename)
        pixel_data = pixel_data.astype(float) / 255.0
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(np.all(np.abs(image - pixel_data) < .02))
        
    def test_04_02_save_with_metadata(self):
        np.random.seed(0)
        image = np.random.uniform(size=(30,40))
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.save_image_or_figure.value = cpm_si.IF_IMAGE
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = 'metadatatest\\g<Well>'
        module.file_format.value = cpm_si.FF_PNG
        
        m = workspace.measurements
        m.add_image_measurement('Metadata_Well','C08')
        
        module.run(workspace)
        filename = os.path.join(cpprefs.get_default_output_directory(),
                                "metadatatestC08.%s" %(cpm_si.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = cpm_li.load_using_PIL(filename)
        pixel_data = pixel_data.astype(float) / 255.0
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(np.all(np.abs(image - pixel_data) < .02))
        
    def test_04_03_clip(self):
        """Regression test of IMG-720: clip images with values outside of 0-1"""
        np.random.seed(43)
        image = np.random.uniform(size=(40,30)) * 1.2 - .1
        expected = image.copy()
        expected[expected < 0] = 0
        expected[expected > 1] = 1
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.save_image_or_figure.value = cpm_si.IF_IMAGE
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = "foo"
        module.file_format.value = cpm_si.FF_PNG
        module.rescale.value = False
        module.run(workspace)
        filename = os.path.join(cpprefs.get_default_output_directory(),
                                "foo.%s"%(cpm_si.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = cpm_li.load_using_PIL(filename)
        pixel_data = pixel_data.astype(float) / 255.0
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(np.all(np.abs(expected - pixel_data) < .02))
        
    def test_04_04_rescale_gray(self):
        """Test rescaling a grayscale image
        
        Regression test of IMG-943
        """
        
        np.random.seed(44)
        expected = np.random.uniform(size=(10,20))
        image = expected * .5
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.save_image_or_figure.value = cpm_si.IF_IMAGE
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = "foo"
        module.file_format.value = cpm_si.FF_PNG
        module.rescale.value = True
        module.run(workspace)
        filename = os.path.join(cpprefs.get_default_output_directory(),
                                "foo.%s"%(cpm_si.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = cpm_li.load_using_PIL(filename)
        pixel_data = pixel_data.astype(float) / 255.0
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(np.all(np.abs(expected - pixel_data) < .02))

    def test_04_05_rescale_color(self):
        """Test rescaling a color image"""
        
        np.random.seed(44)
        expected = np.random.uniform(size=(10,20,3))
        image = expected * .5
        workspace, module = self.make_workspace(image, FILE_NAME)
        self.assertTrue(isinstance(module, cpm_si.SaveImages))
        module.save_image_or_figure.value = cpm_si.IF_IMAGE
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = "foo"
        module.file_format.value = cpm_si.FF_PNG
        module.rescale.value = True
        module.run(workspace)
        filename = os.path.join(cpprefs.get_default_output_directory(),
                                "foo.%s"%(cpm_si.FF_PNG))
        self.assertTrue(os.path.isfile(filename))
        pixel_data = cpm_li.load_using_PIL(filename)
        pixel_data = pixel_data.astype(float) / 255.0
        self.assertEqual(pixel_data.shape, image.shape)
        self.assertTrue(np.all(np.abs(expected - pixel_data) < .02))

    def run_movie(self, groupings=None, fn = None, color=False):
        '''Run a pipeline that produces a movie
        
        Returns a list containing the movie frames
        '''
        image_set_list = cpi.ImageSetList()
        if groupings == None:
            nframes = 5
            key_names = []
            group_list = (({}, np.arange(nframes) + 1),)
        else:
            key_names, group_list = groupings
            nframes = sum([len(g[1]) for g in group_list])
        for i in range(nframes):
            image_set_list.get_image_set(i)
            
        np.random.seed(0)
        frames = [ np.random.uniform(size=(128,128,3) if color else (128,128)) 
                   for i in range(nframes)]
        measurements = cpm.Measurements()
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        module = cpm_si.SaveImages()
        module.module_num = 1
        module.save_image_or_figure.value = cpm_si.IF_MOVIE
        module.image_name.value = IMAGE_NAME
        module.file_image_name.value = IMAGE_NAME
        module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.single_file_name.value = FILE_NAME
        module.rescale.value = False
        
        if fn is not None:
            fn(module)
        pipeline.add_module(module)
        self.assertTrue(module.prepare_run(pipeline, image_set_list))
        is_first = True
        frame_iterator = iter(frames)
        first_image_set = True
        for group in group_list:
            self.assertTrue(module.prepare_group(pipeline, image_set_list,
                                                 group[0], group[1]))
            for image_number in group[1]:
                if not first_image_set:
                    measurements.next_image_set()
                else:
                    first_image_set = False
                image_set = image_set_list.get_image_set(image_number-1)
                img = cpi.Image(frame_iterator.next())
                image_set.add(IMAGE_NAME, img)
                for key, value in group[0].iteritems():
                    measurements.add_image_measurement(key, value)
                workspace = cpw.Workspace(pipeline, module, image_set, 
                                          cpo.ObjectSet(),
                                          measurements, image_set_list)
                module.run(workspace)
            module.post_group(workspace, group)
        module.post_run(workspace)
        return frames
        
    def test_05_01_save_movie(self):
        if BIOFORMATS_CANT_WRITE:
            print "WARNING: Skipping test. The current version of bioformats can't be used for writing images on MacOS X."
            return
        frames = self.run_movie()
        for i, frame in enumerate(frames):
            path = os.path.join(self.custom_directory, FILE_NAME + ".avi")
            frame_out = cpm_li.load_using_bioformats(path, t=i)
            self.assertTrue(np.all(np.abs(frame - frame_out) < .05))
            
    def test_05_02_save_two_movies(self):
        '''Use metadata grouping to write two movies'''
        if BIOFORMATS_CANT_WRITE:
            print "WARNING: Skipping test. The current version of bioformats can't be used for writing images on MacOS X."
            return
        grouping = (('Metadata_test',),
                    (({'Metadata_test':"foo"}, [1,2,3,4,5]),
                     ({'Metadata_test':"bar"}, [6,7,8,9])))
        def fn(module):
            self.assertTrue(isinstance(module, cpm_si.SaveImages))
            module.single_file_name.value = r"\g<test>"
            module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        frames = self.run_movie(grouping, fn)
        for group in grouping[1]:
            path = os.path.join(self.custom_directory, group[0]["Metadata_test"] + ".avi")
            self.assertTrue(os.path.exists(path))
            for t,image_number in enumerate(group[1]):
                frame = frames[image_number-1]
                frame_out = cpm_li.load_using_bioformats(path, t=t)
                self.assertTrue(np.all(np.abs(frame - frame_out) < .05))
                
    def test_05_03_save_color_movie(self):
        '''Regression test of img-1227 - save a color movie saved in b/w
        
        also BioFormats crashed when saving in color, requiring update of
        loci_tools.jar
        '''
        frames = self.run_movie(color=True)
        for i, frame in enumerate(frames):
            path = os.path.join(self.custom_directory, FILE_NAME + ".avi")
            frame_out = cpm_li.load_using_bioformats(path, t=i)
            self.assertTrue(np.all(np.abs(frame - frame_out) < .05))
                
    def test_06_01_save_image_with_bioformats(self):
        if BIOFORMATS_CANT_WRITE:
            print "WARNING: Skipping test. The current version of bioformats can't be used for writing images on MacOS X."
            return
        np.random.seed(61)
        image8 = (np.random.uniform(size=(100,100))*255).astype(np.uint8)
        image16 = (np.random.uniform(size=(100,100))*65535).astype(np.uint16)
        imagefloat = np.random.uniform(size=(100,100))
        image8s = (np.random.uniform(size=(100,100))*245).astype(np.uint8)
        image8s[0,0] = 245
        image8s[0,1] = 0
        image16s = (np.random.uniform(size=(100,100))*64535).astype(np.uint16)
        image16s[0,0] = 64535
        image16s[0,1] = 0
        imagefloats = imagefloat.copy()
        imagefloats[0,0] = 1
        imagefloats[0,1] = 0
        
        test_settings = [
            # 16-bit TIF from all image types
            {'rescale'       : False, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image8,
             'expected'      : image8.astype(np.uint16) },
            {'rescale'       : False, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : imagefloat,
             'expected'      : (imagefloat * 65535).astype(np.uint16) },
            {'rescale'       : False, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image16,
             'expected'      : image16 },

            # Rescaled 16-bit image
            {'rescale'       : True, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image8s + 10,
             'expected'      : image8s.astype(np.float32)*65535.0/245.0 },
            {'rescale'       : True, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image16s + 1000,
             'expected'      : image16s.astype(np.float32)*65535.0/64535.0 },
            {'rescale'       : True, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : imagefloats / 2 + .1,
             'expected'      : (imagefloats * 65535.0).astype(np.uint16) }
        ]

        for i, setting in enumerate(test_settings):
            # Adjust settings each round and retest
            workspace, module = self.make_workspace(setting['input_image'],
                                                    convert=False)

            module.module_num = 1
            module.save_image_or_figure.value = cpm_si.IF_IMAGE
            module.image_name.value = IMAGE_NAME
            module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
            module.pathname.custom_path = self.custom_directory
            module.file_name_method.value = cpm_si.FN_SINGLE_NAME
            module.single_file_name.value = FILE_NAME+str(i)
            
            module.rescale.value = setting['rescale']
            module.file_format.value = setting['file_format']
            module.bit_depth.value = setting['bit_depth']
            
        
            module.save_image_with_bioformats(workspace)

            expected = setting['expected']
            filename = module.get_filename(workspace,
                                           make_dirs = False,
                                           check_overwrite = False)
            im = cpm_li.load_using_bioformats(filename)
            self.assertTrue(np.all(np.abs(im*65535 - expected) <= 1))
            if os.path.isfile(filename):
                try:
                    os.remove(filename)
                except:
                    sys.stderr.write("Not ideal, Bioformats still holding onto file handle.\n")
                    traceback.print_exc()

    def test_06_02_save_image_with_libtiff(self):
        try:
            import libtiff
        except:
            sys.stderr.write("Failed to import libtiff.\n")
            traceback.print_exc()
            return
            
        image = np.ones((255,255)).astype(np.uint8)
        for i in range(image.shape[0]):
            image[i,:] = i
        image2 = np.ones((100,100)).astype(np.uint8)
        for i in range(image2.shape[0]):
            image2[i,:] = i
            
        test_settings = [
            # 16-bit TIF from all image types
            {'rescale'       : False, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image},
            {'rescale'       : False, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image.astype(np.float) / 255.},
            {'rescale'       : False, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image.astype(np.uint16) * 255},

            # Rescaled 16-bit image
            {'rescale'       : True, 
             'file_format'   : cpm_si.FF_TIF, 
             'bit_depth'     : '16',
             'input_image'   : image2},
        ]

        for setting in test_settings:
            # Adjust settings each round and retest
            workspace, module = self.make_workspace(setting['input_image'])

            module.save_image_or_figure.value = cpm_si.IF_IMAGE
            module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
            module.pathname.custom_path = self.custom_directory
            module.file_name_method.value = cpm_si.FN_SINGLE_NAME
            module.single_file_name.value = FILE_NAME
            
            module.rescale.value = setting['rescale']
            module.file_format.value = setting['file_format']
            module.bit_depth.value = setting['bit_depth']
            
            filename = module.get_filename(workspace, make_dirs = False,
                                           check_overwrite = False)
            if os.path.isfile(filename):
                os.remove(filename)
        
            module.save_image_with_libtiff(workspace)

            # Convert original image to float to compare it to the saved image
            if setting['input_image'].dtype == np.uint8:
                expected = setting['input_image'] / 255.
            elif setting['input_image'].dtype == np.uint16:
                expected = setting['input_image'] / 65535.
            elif issubclass(setting['input_image'].dtype.type, np.floating):
                expected = setting['input_image']

            if setting['rescale']:
                expected = stretch(expected)                
            im = cpm_li.load_using_bioformats(filename)
            
            self.assertTrue (np.allclose(im, expected, atol=.001), 
                    'Saved image did not match original when reloaded.\n'
                    'Settings were: \n'
                    '%s\n'
                    'Original: \n'
                    '%s\n'
                    'Expected: \n'
                    '%s\n'
                    %(setting, im[:,0], expected[:,0]))
            
    def test_07_01_save_objects_grayscale8_tiff(self):
        if BIOFORMATS_CANT_WRITE:
            print "WARNING: Skipping test. The current version of bioformats can't be used for writing images on MacOS X."
            try:
                import libtiff
            except:
                sys.stderr.write("Failed to import libtiff.\n")
                traceback.print_exc()
                return
        r = np.random.RandomState()
        labels = r.randint(0, 10, size=(30,20))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cpm_si.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cpm_si.GC_GRAYSCALE
        module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.file_format.value = cpm_si.FF_TIFF

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpm.Measurements)
        feature = cpm_si.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cpm_si.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        image = PILImage.open(filename)
        im = cpm_li.load_using_bioformats(filename, rescale=False)
        self.assertTrue(np.all(labels == im))
        
    def test_07_02_save_objects_grayscale_16_tiff(self):
        if BIOFORMATS_CANT_WRITE:
            print "WARNING: Skipping test. The current version of bioformats can't be used for writing images on MacOS X."
            try:
                import libtiff
            except:
                sys.stderr.write("Failed to import libtiff.\n")
                traceback.print_exc()
                return
        r = np.random.RandomState()
        labels = r.randint(0, 300, size=(300,300))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cpm_si.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cpm_si.GC_GRAYSCALE
        module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.file_format.value = cpm_si.FF_TIFF

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpm.Measurements)
        feature = cpm_si.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cpm_si.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        image = PILImage.open(filename)
        im = cpm_li.load_using_bioformats(filename, rescale=False)
        self.assertTrue(np.all(labels == im))
        
    def test_07_03_save_objects_grayscale_png(self):
        r = np.random.RandomState()
        labels = r.randint(0, 10, size=(30,20))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cpm_si.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cpm_si.GC_GRAYSCALE
        module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.file_format.value = cpm_si.FF_PNG

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpm.Measurements)
        feature = cpm_si.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cpm_si.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        image = PILImage.open(filename)
        im = cpm_li.load_using_bioformats(filename, rescale=False)
        self.assertTrue(np.all(labels == im))
        
    def test_07_04_save_objects_color_png(self):
        r = np.random.RandomState()
        labels = r.randint(0, 10, size=(30,20))
        workspace, module = self.make_workspace(labels, save_objects = True)
        assert isinstance(module, cpm_si.SaveImages)
        module.update_file_names.value = True
        module.gray_or_color.value = cpm_si.GC_COLOR
        module.pathname.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        module.pathname.custom_path = self.custom_directory
        module.file_name_method.value = cpm_si.FN_SINGLE_NAME
        module.file_format.value = cpm_si.FF_PNG

        filename = module.get_filename(workspace, make_dirs = False,
                                       check_overwrite = False)
        if os.path.isfile(filename):
            os.remove(filename)
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpm.Measurements)
        feature = cpm_si.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
        m_filename = m.get_current_image_measurement(feature)
        self.assertEqual(m_filename, os.path.split(filename)[1])
        feature = cpm_si.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
        m_pathname = m.get_current_image_measurement(feature)
        self.assertEqual(m_pathname, os.path.split(filename)[0])
        im = cpm_li.load_using_bioformats(filename)
        im.shape = (im.shape[0] * im.shape[1], im.shape[2])
        order = np.lexsort(im.transpose())
        different = np.hstack(([False], np.any(im[order[:-1],:] != im[order[1:],:], 1)))
        indices = np.cumsum(different)
        im = np.zeros(labels.shape, int)
        im.ravel()[order] = indices
        #
        # There should be a 1-1 correspondence between label #s and indices
        #
        x = coo.coo_matrix((np.ones(len(indices)),
                            (labels.ravel(), im.ravel()))).toarray()
        self.assertEqual(np.sum(x != 0), 10)
        
def make_array(encoded,shape,dtype=np.uint8):
    data = base64.b64decode(encoded)
    a = np.fromstring(data,dtype)
    return a.reshape(shape)

def make_file(filename, encoded):
    data = base64.b64decode(encoded)
    fid = open(filename,'wb')
    fid.write(data)
    fid.close()
