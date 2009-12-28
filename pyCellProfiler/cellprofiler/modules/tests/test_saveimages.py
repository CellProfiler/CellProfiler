"""test_saveimages - test the saveimages module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import base64
import matplotlib.image
import numpy
import os
import sys
import Image as PILImage
from StringIO import StringIO
import unittest
import tempfile
import zlib

import cellprofiler.modules.saveimages as cpm_si
import cellprofiler.modules.loadimages as cpm_li
import cellprofiler.modules.applythreshold as cpm_a
import cellprofiler.pipeline as cpp
import cellprofiler.preferences as cpprefs
import cellprofiler.modules.createbatchfiles as cpm_c
from cellprofiler.utilities.get_proper_case_filename import get_proper_case_filename

import cellprofiler.modules.tests as cpmt

class TestSaveImages(unittest.TestCase):
    def setUp(self):
        # Change the default image directory to a temporary file
        self.old_image_directory = cpprefs.get_default_image_directory()
        self.new_image_directory = get_proper_case_filename(tempfile.mkdtemp())
        cpprefs.set_default_image_directory(self.new_image_directory)
        self.old_output_directory = cpprefs.get_default_output_directory()
        self.new_output_directory = get_proper_case_filename(tempfile.mkdtemp())
        cpprefs.set_default_output_directory(self.new_output_directory)
        self.custom_directory = get_proper_case_filename(tempfile.mkdtemp())
    
    def tearDown(self):
        for subdir in (self.new_image_directory, self.new_output_directory,
                       self.custom_directory):
            for filename in os.listdir(subdir):
                os.remove(os.path.join(subdir, filename))
            os.rmdir(subdir)
        if os.path.isdir(self.old_image_directory):
            cpprefs.set_default_image_directory(self.old_image_directory)
        if os.path.isdir(self.old_output_directory):
            cpprefs.set_default_output_directory(self.old_output_directory)
    
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
        self.assertEqual(module.pathname_choice.value, cpm_si.PC_DEFAULT)
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
        self.assertEqual(module.pathname_choice.value, cpm_si.PC_DEFAULT)
        self.assertEqual(module.when_to_save.value, cpm_si.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap.value, cpm_si.CM_GRAY)
        self.assertFalse(module.overwrite)
        
    def test_00_04_load_v3(self):
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
        self.assertEqual(module.pathname_choice.value, cpm_si.PC_DEFAULT)
        self.assertEqual(module.when_to_save.value, cpm_si.WS_EVERY_CYCLE)
        self.assertEqual(module.colormap.value, cpm_si.CM_GRAY)
        self.assertFalse(module.overwrite)
    
    def test_01_01_save_first_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname_choice.value = cpm_si.PC_WITH_IMAGE
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
        self.assertTrue(numpy.all(data[expected_data < 255] ==
                                  expected_data[expected_data < 255]))
        self.assertTrue(numpy.all(data[expected_data == 255] == 0))

    def test_01_02_save_all_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname_choice.value = cpm_si.PC_WITH_IMAGE
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
        self.assertTrue(numpy.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(numpy.all(data==expected_data))

    def test_01_03_save_last_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname_choice.value = cpm_si.PC_WITH_IMAGE
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
        self.assertTrue(numpy.all(data==expected_data))

    def test_01_04_save_all_to_output_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_output_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_output_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname_choice.value = cpm_si.PC_DEFAULT
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
        self.assertTrue(numpy.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(numpy.all(data==expected_data))

    def test_01_05_save_all_to_custom_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_TIF
        save_images.pathname_choice.value = cpm_si.PC_CUSTOM
        save_images.pathname.value = self.custom_directory
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
        self.assertTrue(numpy.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(numpy.all(data==expected_data))

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
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_PNG
        save_images.pathname_choice.value = cpm_si.PC_CUSTOM
        save_images.pathname.value = self.custom_directory
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
        self.assertTrue(numpy.all(data==expected_data))
        pil = PILImage.open(img2_out_filename)
        data = matplotlib.image.pil_to_array(pil)
        pil = PILImage.open(img2_filename)
        expected_data = matplotlib.image.pil_to_array(pil) 
        self.assertTrue(numpy.all(data==expected_data))

    def test_01_07_save_all_to_custom_jpg(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.jpg')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.jpg')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_JPG
        save_images.pathname_choice.value = cpm_si.PC_CUSTOM
        save_images.pathname.value = self.custom_directory
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
        self.assertTrue(numpy.all(numpy.abs(data.astype(int)-
                                            expected_data.astype(int))<=4))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(numpy.all(numpy.abs(data.astype(int)-
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
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.file_name_suffix.value ='OUT'
        save_images.file_format.value = cpm_si.FF_GIF
        save_images.pathname_choice.value = cpm_si.PC_CUSTOM
        save_images.pathname.value = self.custom_directory
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
        self.assertTrue(numpy.all(data==expected_data))
        data = matplotlib.image.imread(img2_out_filename)
        expected_data = matplotlib.image.imread(img2_filename) 
        self.assertTrue(numpy.all(data==expected_data))

    def test_01_09_save_single_to_custom_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.custom_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.custom_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.on_event)
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images[0][cpm_li.FD_COMMON_TEXT].value = '.tif'
        load_images.images[0][cpm_li.FD_IMAGE_NAME].value = 'Orig'
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
        save_images.pathname_choice.value = cpm_si.PC_CUSTOM
        save_images.pathname.value = self.custom_directory
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
        self.assertTrue(numpy.all(data==expected_data))
    
    def test_02_01_prepare_to_create_batch(self):
        '''Test the "prepare_to_create_batch" method'''
        orig_path = '/foo/bar'
        def fn_alter_path(path, **varargs):
            self.assertEqual(path, orig_path)
            return '/imaging/analysis'
        module = cpm_si.SaveImages()
        module.pathname.value = orig_path
        module.prepare_to_create_batch(None,None, fn_alter_path)
        self.assertEqual(module.pathname.value, '/imaging/analysis')
    
    def test_02_02_regression_prepare_to_create_batch(self):
        '''Make sure that "prepare_to_create_batch" handles metadata

        This is a regression test for IMG-200
        '''
        cmodule = cpm_c.CreateBatchFiles()
        module = cpm_si.SaveImages()
        module.pathname.value = '.\\\\\\g<Test>Outlines\\\\g<Run>_\\g<Plate>'
        module.pathname_choice.value = cpm_si.PC_WITH_METADATA
        module.prepare_to_create_batch(None,None, cmodule.alter_path)
        self.assertEqual(module.pathname.value, './\\g<Test>Outlines/g<Run>_\\g<Plate>')
    
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
        
    
def make_array(encoded,shape,dtype=numpy.uint8):
    data = base64.b64decode(encoded)
    a = numpy.fromstring(data,dtype)
    return a.reshape(shape)

def make_file(filename, encoded):
    data = base64.b64decode(encoded)
    fid = open(filename,'wb')
    fid.write(data)
    fid.close()
    

        
