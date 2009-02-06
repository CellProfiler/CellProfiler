"""test_saveimages - test the saveimages module

"""
__version__="$Revision 1 $"

import matplotlib.image
import numpy
import os
import PIL.Image
import unittest
import base64
import tempfile

import cellprofiler.modules.saveimages as cpm_si
import cellprofiler.modules.loadimages as cpm_li
import cellprofiler.modules.applythreshold as cpm_a
import cellprofiler.pipeline as cpp
import cellprofiler.preferences as cpprefs

import cellprofiler.modules.tests as cpmt

class TestSaveImages(unittest.TestCase):
    def setUp(self):
        # Change the default image directory to a temporary file
        self.old_image_directory = cpprefs.get_default_image_directory()
        self.new_image_directory = tempfile.mkdtemp()
        cpprefs.set_default_image_directory(self.new_image_directory)
        self.old_output_directory = cpprefs.get_default_output_directory()
        self.new_output_directory = tempfile.mkdtemp()
        cpprefs.set_default_output_directory(self.new_output_directory)
        self.custom_directory = tempfile.mkdtemp()
    
    def tearDown(self):
        for subdir in (self.new_image_directory, self.new_output_directory,
                       self.custom_directory):
            for filename in os.listdir(subdir):
                os.remove(os.path.join(subdir, filename))
            os.rmdir(subdir)
        cpprefs.set_default_image_directory(self.old_image_directory)
        cpprefs.set_default_output_directory(self.old_output_directory)
    
    def test_01_01_save_first_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        self.assertTrue(numpy.all(data==expected_data))

    def test_01_02_save_all_to_same_tif(self):
        img1_filename = os.path.join(self.new_image_directory,'img1.tif')
        img1_out_filename = os.path.join(self.new_image_directory,'img1OUT.tif')
        img2_filename = os.path.join(self.new_image_directory,'img2.tif') 
        img2_out_filename = os.path.join(self.new_image_directory,'img2OUT.tif')
        make_file(img1_filename, cpmt.tif_8_1)
        make_file(img2_filename, cpmt.tif_8_2)
        pipeline = cpp.Pipeline()
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        pil = PIL.Image.open(img1_out_filename)
        data = matplotlib.image.pil_to_array(pil)
        pil = PIL.Image.open(img1_filename)
        expected_data = matplotlib.image.pil_to_array(pil) 
        self.assertTrue(numpy.all(data==expected_data))
        pil = PIL.Image.open(img2_out_filename)
        data = matplotlib.image.pil_to_array(pil)
        pil = PIL.Image.open(img2_filename)
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
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
        load_images = cpm_li.LoadImages()
        load_images.file_types.value = cpm_li.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = cpm_li.MS_EXACT_MATCH
        load_images.images_common_text[0].value = '.tif'
        load_images.image_names[0].value = 'Orig'
        load_images.module_num = 1
        
        apply_threshold = cpm_a.ApplyThreshold()
        apply_threshold.image_name.value = 'Orig'
        apply_threshold.thresholded_image_name.value = 'Derived'
        apply_threshold.low.value = False
        apply_threshold.high.value = False
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
    
def make_array(encoded,shape,dtype=numpy.uint8):
    data = base64.b64decode(encoded)
    a = numpy.fromstring(data,dtype)
    return a.reshape(shape)

def make_file(filename, encoded):
    data = base64.b64decode(encoded)
    fid = open(filename,'wb')
    fid.write(data)
    fid.close()
    

        