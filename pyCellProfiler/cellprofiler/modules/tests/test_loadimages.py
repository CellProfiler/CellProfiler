"""Test the LoadImages module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import base64
import numpy
import os
import unittest
import tempfile

import cellprofiler.cpmodule as CPM
import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.tests as T
import cellprofiler.cpimage as I
import cellprofiler.pipeline as P

class testLoadImages(unittest.TestCase):
    def test_00_00init(self):
        x=LI.LoadImages()
    
    def test_00_01version(self):
        self.assertEqual(LI.LoadImages().variable_revision_number,1,"LoadImages' version number has changed")
    
    def test_01_01load_image_text_match(self):
        l=LI.LoadImages()
        l.settings()[l.SLOT_MATCH_METHOD].set_value(LI.MS_EXACT_MATCH)
        l.settings()[l.SLOT_LOCATION].value = LI.DIR_OTHER
        l.settings()[l.SLOT_LOCATION_OTHER].value =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_COMMON_TEXT].set_value("1-01-A-01.tif")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.prepare_run(pipeline, image_set_list)
        self.assertEqual(image_set_list.count(),1,"Expected one image set in the list")
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),1)
        self.assertEqual(image_set.get_names()[0],"my_image")
        self.assertTrue(image_set.get_image("my_image"))
        
    def test_01_02load_image_text_match_many(self):
        l=LI.LoadImages()
        l.settings()[l.SLOT_MATCH_METHOD].set_value(LI.MS_EXACT_MATCH)
        l.settings()[l.SLOT_LOCATION].value = LI.DIR_OTHER
        l.settings()[l.SLOT_LOCATION_OTHER].value =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        for i in range(0,4):
            ii = i+1
            if i:
                l.add_imagecb()
            idx = l.SLOT_FIRST_IMAGE+l.SLOT_IMAGE_FIELD_COUNT * i 
            l.settings()[idx+l.SLOT_OFFSET_COMMON_TEXT].set_value("1-0%(ii)d-A-0%(ii)d.tif"%(locals()))
            l.settings()[idx+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image%(i)d"%(locals()))
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.prepare_run(pipeline, image_set_list)
        self.assertEqual(image_set_list.count(),1,"Expected one image set, there were %d"%(image_set_list.count()))
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),4)
        for i in range(0,4):
            self.assertTrue("my_image%d"%(i) in image_set.get_names())
            self.assertTrue(image_set.get_image("my_image%d"%(i)))
        
    def test_02_01load_image_regex_match(self):
        l=LI.LoadImages()
        l.settings()[l.SLOT_MATCH_METHOD].set_value(LI.MS_REGEXP)
        l.settings()[l.SLOT_LOCATION].value = LI.DIR_OTHER
        l.settings()[l.SLOT_LOCATION_OTHER].value =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_COMMON_TEXT].set_value("Channel1-[0-1][0-9]-A-01")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.prepare_run(pipeline, image_set_list)
        self.assertEqual(image_set_list.count(),1,"Expected one image set in the list")
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),1)
        self.assertEqual(image_set.get_names()[0],"my_image")
        self.assertTrue(image_set.get_image("my_image"))
    
    def test_03_01_load_version_2(self):
        data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDA1IDEwOjMwOjQ0IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAkAEAAHic7ZPNTsJAEMen5UOUZIPxwpEXEInEwJHEiyQKBgj3xS7NJku32bYEPXn0NXwLjz6au7iFbWkoEm+6yWQ70/n/djo7RQDwcQJQlntFmg3fq6R9yzDlj0kYUs8NSlCEuo5/SptiQfGMkSlmEQlgs+J435vzybO/efXAnYiRAV6YyXINosWMiGA4j4X69SNdETamLwSSK04bkSUNKPe0XvPT0Zi/gwck7a2w7YOV0QdkxNXzHWzzixn5dSO/pv0JWYWXI+JGDIsGWfmCBKrAQPF6ObzTFE/5T5xx0Qzp3KjrGM5QUPdWsQxOK4djJTgWXP3rflXXhsPm7ByS96l86jl0SZ0IswZdYDcx53l12AmeDQN+XP3NA88rJHQF8K7wWvdq7f8fq0YcZey9nHNrqb4pWzfLFTzyG7KFxP/LvHj3Yf89mPd+SB1nqTqUf8+x0zcGVXG6BqecwSkbHFv7qIoQquzqs+owv6em/VazfXPd6XTTc5t1vvndtnyyYXfe83RFqXq/+LlOnac0X7WHgow='
        pipeline = T.load_pipeline(self, data)
        self.assertEqual(len(pipeline.modules()),1)
        module = pipeline.module(1)
        self.assertEqual(module.load_choice(),LI.MS_REGEXP)
        self.assertTrue(module.load_images())
        self.assertFalse(module.load_movies())
        self.assertTrue(module.text_to_exclude(), 'Do not use')
        self.assertEqual(len(module.image_name_vars()),1)
        self.assertEqual(module.image_name_vars()[0].value,'OrigColor')
        self.assertEqual(module.text_to_find_vars()[0].value,'color.tif')
        self.assertFalse(module.analyze_sub_dirs())
        
    def test_03_02_load_version_4(self):
        data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDA1IDExOjA2OjM5IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAApwEAAHic5VTNTsJAEJ42BdEDwXjQY49eJCIXj8b4A4mCAUK8mYUudZO22/QHwafyEbj5Wu5KC8um0qV6c5LNdGZnvpn9OrtVAFhUAMpMMwU6LKWU2JqwuN3HUUQ8OyyBASeJf8HWEAUEjRw8RE6MQ1hJ6m97EzqY+6utR2rFDu4gVwxm0ondEQ7C7iRNTLafyAw7ffKOYVPSsB6ekpBQL8lP8GXvqi6NpLpVtlrGmgctg4cjwc/jr2Adb2TE14T4WrIGeBad3c7QODJdFI1fVXD2JRxuh42Xt0Z90L4T+rnMwdmTcLjdDYjdw5bSeX7q40LqowgO7+M+wNj7JQ7vp7kjLxUJp5L0c81GWaWPAymf2zfU9GhkxiFWP48qznkOjraBo0Hzj+u3cnAOJRxuE88iU2LFyDGJi+zV7VM5j76Bp0OHFuOhrshD1lzZAZqHY+Sk709VQX9o298Tkaes/Lw+s96Xb3LtgMa+ySjH/n/GK6p92P7fxLkqeq8eKLLawkWQ57mcU1dnX7WMPJV70ChYzyiQZ7DMz+Nl3vOOvJ5uiU8l9X8BnJqT/A=='
        pipeline = T.load_pipeline(self, data)
        self.assertEqual(len(pipeline.modules()),1)
        module = pipeline.module(1)
        self.assertEqual(module.load_choice(),LI.MS_EXACT_MATCH)
        self.assertTrue(module.load_images())
        self.assertFalse(module.load_movies())
        self.assertTrue(module.text_to_exclude(), 'Do not use')
        self.assertEqual(len(module.image_name_vars()),3)
        self.assertEqual(module.image_name_vars()[0].value,'OrigRed')
        self.assertEqual(module.text_to_find_vars()[0].value,'s1_w1.TIF')
        self.assertEqual(module.image_name_vars()[1].value,'OrigGreen')
        self.assertEqual(module.text_to_find_vars()[1].value,'s1_w2.TIF')
        self.assertEqual(module.image_name_vars()[2].value,'OrigBlue')
        self.assertEqual(module.text_to_find_vars()[2].value,'s1_w3.TIF')
        self.assertFalse(module.analyze_sub_dirs())
        
    def test_04_01_load_save_and_load(self):
        data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDA1IDExOjA2OjM5IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAApwEAAHic5VTNTsJAEJ42BdEDwXjQY49eJCIXj8b4A4mCAUK8mYUudZO22/QHwafyEbj5Wu5KC8um0qV6c5LNdGZnvpn9OrtVAFhUAMpMMwU6LKWU2JqwuN3HUUQ8OyyBASeJf8HWEAUEjRw8RE6MQ1hJ6m97EzqY+6utR2rFDu4gVwxm0ondEQ7C7iRNTLafyAw7ffKOYVPSsB6ekpBQL8lP8GXvqi6NpLpVtlrGmgctg4cjwc/jr2Adb2TE14T4WrIGeBad3c7QODJdFI1fVXD2JRxuh42Xt0Z90L4T+rnMwdmTcLjdDYjdw5bSeX7q40LqowgO7+M+wNj7JQ7vp7kjLxUJp5L0c81GWaWPAymf2zfU9GhkxiFWP48qznkOjraBo0Hzj+u3cnAOJRxuE88iU2LFyDGJi+zV7VM5j76Bp0OHFuOhrshD1lzZAZqHY+Sk709VQX9o298Tkaes/Lw+s96Xb3LtgMa+ySjH/n/GK6p92P7fxLkqeq8eKLLawkWQ57mcU1dnX7WMPJV70ChYzyiQZ7DMz+Nl3vOOvJ5uiU8l9X8BnJqT/A=='
        pipeline = T.load_pipeline(self, data)
        (matfd,matpath) = tempfile.mkstemp('.mat')
        matfh = os.fdopen(matfd,'wb')
        pipeline.save(matfh)
        matfh.flush()
        pipeline = P.Pipeline()
        pipeline.load(matpath)
        matfh.close()
        self.assertEqual(len(pipeline.modules()),1)
        module = pipeline.module(1)
        self.assertEqual(module.load_choice(),LI.MS_EXACT_MATCH)
        self.assertTrue(module.load_images())
        self.assertFalse(module.load_movies())
        self.assertTrue(module.text_to_exclude(), 'Do not use')
        self.assertEqual(len(module.image_name_vars()),3)
        self.assertEqual(module.image_name_vars()[0].value,'OrigRed')
        self.assertEqual(module.text_to_find_vars()[0].value,'s1_w1.TIF')
        self.assertEqual(module.image_name_vars()[1].value,'OrigGreen')
        self.assertEqual(module.text_to_find_vars()[1].value,'s1_w2.TIF')
        self.assertEqual(module.image_name_vars()[2].value,'OrigBlue')
        self.assertEqual(module.text_to_find_vars()[2].value,'s1_w3.TIF')
        self.assertFalse(module.analyze_sub_dirs())
    
    def test_05_01_load_PNG(self):
        """Test loading of a .PNG file
        
        Regression test a bug in PIL that flips the image
        """
        data = base64.b64decode(T.png_8_1)
        (matfd,matpath) = tempfile.mkstemp('.png')
        matfh = os.fdopen(matfd,'wb')
        matfh.write(data)
        matfh.flush()
        path,filename = os.path.split(matpath)
        load_images = LI.LoadImages()
        load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = LI.MS_EXACT_MATCH
        load_images.images_common_text[0].value = filename
        load_images.image_names[0].value = 'Orig'
        load_images.location.value = LI.DIR_OTHER
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig')
                matfh.close()
                pixel_data = image.pixel_data
                pixel_data = (pixel_data * 255+.5).astype(numpy.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = numpy.fromstring(check_data,numpy.uint8).reshape(T.raw_8_1_shape)
                outer_self.assertTrue(numpy.all(pixel_data ==check_image))
        check_image = CheckImage()
        check_image.module_num = 2
        pipeline = P.Pipeline()
        pipeline.add_module(load_images)
        pipeline.add_module(check_image)
        pipeline.run()

    def test_05_02_load_GIF(self):
        """Test loading of a .GIF file
        
        """
        data = base64.b64decode(T.gif_8_1)
        (matfd,matpath) = tempfile.mkstemp('.gif')
        matfh = os.fdopen(matfd,'wb')
        matfh.write(data)
        matfh.flush()
        path,filename = os.path.split(matpath)
        load_images = LI.LoadImages()
        load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = LI.MS_EXACT_MATCH
        load_images.images_common_text[0].value = filename
        load_images.image_names[0].value = 'Orig'
        load_images.location.value = LI.DIR_OTHER
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig')
                matfh.close()
                pixel_data = image.pixel_data
                pixel_data = (pixel_data * 255+.5).astype(numpy.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = numpy.fromstring(check_data,numpy.uint8).reshape(T.raw_8_1_shape)
                outer_self.assertTrue(numpy.all(pixel_data ==check_image))
        check_image = CheckImage()
        check_image.module_num = 2
        pipeline = P.Pipeline()
        pipeline.add_module(load_images)
        pipeline.add_module(check_image)
        pipeline.run()

    def test_05_03_load_TIF(self):
        """Test loading of a .TIF file
        
        """
        data = base64.b64decode(T.tif_8_1)
        (matfd,matpath) = tempfile.mkstemp('.tif')
        matfh = os.fdopen(matfd,'wb')
        matfh.write(data)
        matfh.flush()
        path,filename = os.path.split(matpath)
        load_images = LI.LoadImages()
        load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = LI.MS_EXACT_MATCH
        load_images.images_common_text[0].value = filename
        load_images.image_names[0].value = 'Orig'
        load_images.location.value = LI.DIR_OTHER
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig')
                matfh.close()
                pixel_data = image.pixel_data
                pixel_data = (pixel_data * 255+.5).astype(numpy.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = numpy.fromstring(check_data,numpy.uint8).reshape(T.raw_8_1_shape)
                outer_self.assertTrue(numpy.all(pixel_data ==check_image))
        check_image = CheckImage()
        check_image.module_num = 2
        pipeline = P.Pipeline()
        pipeline.add_module(load_images)
        pipeline.add_module(check_image)
        pipeline.run()

if __name__=="main":
    unittest.main()
