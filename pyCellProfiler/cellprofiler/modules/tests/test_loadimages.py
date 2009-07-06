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

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as CPM
import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.tests as T
import cellprofiler.cpimage as I
import cellprofiler.objects as cpo
import cellprofiler.measurements as measurements
import cellprofiler.pipeline as P
import cellprofiler.workspace as W

class testLoadImages(unittest.TestCase):
    def error_callback(self, calller, event):
        if isinstance(event, P.RunExceptionEvent):
            self.fail(event.error.message)

    def test_00_00init(self):
        x=LI.LoadImages()
    
    def test_00_01version(self):
        self.assertEqual(LI.LoadImages().variable_revision_number,2,"LoadImages' version number has changed")
    
    def test_01_01load_image_text_match(self):
        l=LI.LoadImages()
        l.settings()[l.SLOT_MATCH_METHOD].set_value(LI.MS_EXACT_MATCH)
        l.settings()[l.SLOT_LOCATION].value = LI.DIR_OTHER
        l.settings()[l.SLOT_LOCATION_OTHER].value =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        l.settings()[l.SLOT_FIRST_IMAGE_V2+l.SLOT_OFFSET_COMMON_TEXT].set_value("1-01-A-01.tif")
        l.settings()[l.SLOT_FIRST_IMAGE_V2+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        l.prepare_run(pipeline, image_set_list, None)
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
            idx = l.SLOT_FIRST_IMAGE_V2+l.SLOT_IMAGE_FIELD_COUNT * i 
            l.settings()[idx+l.SLOT_OFFSET_COMMON_TEXT].set_value("1-0%(ii)d-A-0%(ii)d.tif"%(locals()))
            l.settings()[idx+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image%(i)d"%(locals()))
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        l.prepare_run(pipeline, image_set_list, None)
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
        l.settings()[l.SLOT_FIRST_IMAGE_V2+l.SLOT_OFFSET_COMMON_TEXT].set_value("Channel1-[0-1][0-9]-A-01")
        l.settings()[l.SLOT_FIRST_IMAGE_V2+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image")
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        l.prepare_run(pipeline, image_set_list,None)
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
        pipeline.add_listener(self.error_callback)
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
        
    def test_03_03_load_new_version_2(self):
        #XXX: in progress
        data = 'eJztV91u2jAUNjRURZOm7qLaLn1ZtoIga6UWTbQMJo2NMFRYt6rqVhcMWHJiFJwONlXaI+yR9ih7hD3CbOpA8CJC6S42jUhOfI7Pd36+EzmOVWxWi8/hXiYLrWIz3SEUwzpFvMNcOw8dvgNLLkYctyFz8rDZ8+Arj8KsCXN7+V0zLyZmNnsAlrtiFeu+fIrbunhsiBFXSwklxwJDyg3MOXG6gwQwwCOl/y7GCXIJuqT4BFEPD6YhfH3F6bDmqD9Zsljbo7iG7KCxuGqefYndwZuOD1TLdTLEtEE+Y60E3+wYX5EBYY7CK/+6dhKXcS2u5GF/fcpDLISHrYBe2r8EU3sjxP5BwH5TycRpkyvS9hCFxEbdSRbS31GEv03NnxxNPOTpF0PU4tBGvNWTfrIRfmIzfmLgqV9/BC6hxZdypVp9ayl8VNz4DD4OamwxHh9qcaVcxh3kUQ4rkkRYJi5uceaOfstjXfPnX76/ZID/qPzXZvJYA6eie3fBRfG9AWbrlnKphxwHU3OZuOVacaF89fcjtyA/xgzOEP11sMR9jcC91uqU8oftw/ozuRHiQuZJ6qOU3mFKj9mnwlkxXT9P+ZoSo57tFM6y6YPzL7kd8/rGuEEEcqxMjf3KPHoReexreUhZ+jrFyFUBdq9TaamymMN7SmcqXRmNppo79je3yH6Q1PBSLo0461M0sJV+mX6bq34v1e8fxu2+H39in1rh/h/cEZj/PoedD8aHjK7LvD4URw/c/5fqXfH7d+K+BXBh+1zweyLtL8B8Xh+DWV6l3BJbfd9l8n/IzdjjQ/sgQxlq35yaM1UxrQQO0Ho9yZA4wbziYrYVwYNe/5SXn4fLxIuHxLsXgTPUH5nEvQe34317jj3Q7H8BnRn8NQ=='
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
    
    def test_03_03_load_new_version_3(self):
        #XXX: in progress
        data = 'eJztWW9v20QYv7Rp1TI0DV4A0iR0L5uusezQwlqhLFnDINBk0Ro2pq6Uq3NpDp19ln3uGhASL3nNJ+IlH4ePwJ1jx86tqR2njTQprhz7efz8nn93z/3pterdo/pTuKfpsFXvlvuEYtihiPeZax1Ah3nkagceuhhx3IPMPoDdgQ+/9ymEFWjsHuztHxg6rOj6PshxFZqt++Lx0+cArIvnhrhXwk9rIV1I3JI+xpwT+8JbA0XwWcj/R9wvkUvQOcUvEfWxF5uI+E27z7pDZ/ypxXo+xW1kJYXF1fatc+x6z/sRMPzcIVeYHpPfsBJCJPYCXxKPMDvEh/pV7tgu44pdmYdf7sd5KCh5kHl5mOBL+e9ALF+8Jm8fJeQfhDSxe+SS9HxEIbHQxdgLqU9P0bc6oW8VNNr1AFdLwT1Q/JB3F1/x8jdXyOTQQtwcSD2PU/SsK3ok3fZNikk2/wsT+AL4ImPcxQlcEXy5s6tn8XdN8VfSzaOjH1s58/1atFYW3MoEbgW0WTZ703Bp/exTJU5JN3Af+ZTDpuxksEFcbHLmDm8t7nUFF10RbjN8Zumf9xT/Jf2cez78lrJzRMd67qq9VJyh6XPZyzMu6JoeXDtG+JLI313FrdaV8MHIUlcbYNJ/SR8OkG1jamRp700FL+mmzbHtET5MxJ1Hz+GQM4ciz5pTz3X+zDq+GRlxat0ben6/n8kFhC2m1Tn8/irE/ZuC+1uxL+mft550vpYLGFx15K/2qFSWrIbkbJ3o5f3T33f/KI9eKvFLabt0JuVeYUpfsLfVk3q5c1qKOIeM+pZdHQkbO0I8ED4mQqkXcEtn2qOzNyeS+xS5JuvhqrZdenOqbWfPw7Q6GqTgHit5kLR05DVGbjUKeZSGFrP5oDoOOkzNMOZkafcPFHuSbjBoMw59D8fxzlHHlUWOt9E65q7sqf27AhY7rraZjfPY08N56M8U3A9gsh0lnahDUYI5S2tUWYHeeeJedP3cxvpslvXNotdTs/ejvYWv37pvGTTFPOyFO5x57OdZT73C5GIg98yXcoNom9Pmw9vMw3Xj8jPm4guX+XYv1jP4cLZ95iL9DTal0mFnfvt52o2d/yp2KoEDUOyVsXMLeVjilrgl7v3D1RK4rP/Xisev0fDxPsW7xOWbRz4Gk/1A0sznlNj4nYlk2Y+XuAhXAze3y12vj5a4JW4RuI3C9P2Guh8OzsHAzXWxDSbrQtImptRxmTy/dDUrOGTzNMpQb3TKpR2J12biwEvacVLs1BQ7tWl2SA/bnPSHjius+ZxZiBNTa4bcjuDWI66ax81r7CbzsSL+Pnl4c/7VvMft8d+TPPZWV9+1dy8FVwwzKHF/gdnae+sG+Si2vPL/A2yLmoM='
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        
        
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
        load_images.images[0][LI.FD_COMMON_TEXT].value = filename
        load_images.images[0][LI.FD_IMAGE_NAME].value = 'Orig'
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
        pipeline.add_listener(self.error_callback)
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
        load_images.images[0][LI.FD_COMMON_TEXT].value = filename
        load_images.images[0][LI.FD_IMAGE_NAME].value = 'Orig'
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
        pipeline.add_listener(self.error_callback)
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
        load_images.images[0][LI.FD_COMMON_TEXT].value = filename
        load_images.images[0][LI.FD_IMAGE_NAME].value = 'Orig'
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
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(load_images)
        pipeline.add_module(check_image)
        pipeline.run()

    def test_05_04_load_JPG(self):
        """Test loading of a .JPG file
        
        """
        data = base64.b64decode(T.jpg_8_1)
        (matfd,matpath) = tempfile.mkstemp('.jpg')
        matfh = os.fdopen(matfd,'wb')
        matfh.write(data)
        matfh.flush()
        path,filename = os.path.split(matpath)
        load_images = LI.LoadImages()
        load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = LI.MS_EXACT_MATCH
        load_images.images[0][LI.FD_COMMON_TEXT].value = filename
        load_images.images[0][LI.FD_IMAGE_NAME].value = 'Orig'
        load_images.location.value = LI.DIR_OTHER
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig',
                                                      must_be_grayscale=True)
                pixel_data = image.pixel_data
                matfh.close()
                pixel_data = (pixel_data * 255).astype(numpy.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = numpy.fromstring(check_data,numpy.uint8).reshape(T.raw_8_1_shape)
                # JPEG is lossy, apparently even when you ask for no compression
                epsilon = 1
                outer_self.assertTrue(numpy.all(numpy.abs(pixel_data.astype(int) 
                                                          - check_image.astype(int) <=
                                                          epsilon)))
        check_image = CheckImage()
        check_image.module_num = 2
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(load_images)
        pipeline.add_module(check_image)
        pipeline.run()
    
    def test_06_01_file_metadata(self):
        """Test file metadata on two sets of two files
        
        """
        directory = tempfile.mkdtemp()
        data = base64.b64decode(T.tif_8_1)
        filenames = ["MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s1_w2_[EFBB8532-9A90-4040-8974-477FE1E0F3CA].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w1_[138B5A19-2515-4D46-9AB7-F70CE4D56631].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w2_[59784AC1-6A66-44DE-A87E-E4BDC1A33A54].tif"
                     ]
        for filename in filenames:
            fd = open(os.path.join(directory, filename),"wb")
            fd.write(data)
            fd.close()
        try:
            load_images = LI.LoadImages()
            load_images.add_imagecb()
            load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            load_images.match_method.value = LI.MS_REGEXP
            load_images.location.value = LI.DIR_OTHER
            load_images.location_other.value = directory
            load_images.group_by_metadata.value = True
            load_images.images[0][LI.FD_COMMON_TEXT].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1][LI.FD_COMMON_TEXT].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.images[0][LI.FD_IMAGE_NAME].value = "Channel1"
            load_images.images[1][LI.FD_IMAGE_NAME].value = "Channel2"
            load_images.images[0][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[1][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[0][LI.FD_FILE_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1][LI.FD_FILE_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_listener(self.error_callback)
            pipeline.add_module(load_images)
            image_set_list = I.ImageSetList()
            load_images.prepare_run(pipeline, image_set_list, None)
            self.assertEqual(image_set_list.count(),2)
            image_set = image_set_list.get_image_set(0)
            self.assertEqual(image_set.get_image_provider("Channel1").get_filename(),
                             filenames[0])
            self.assertEqual(image_set.get_image_provider("Channel2").get_filename(),
                             filenames[1])
            m = measurements.Measurements()
            w = W.Workspace(pipeline, load_images, image_set, cpo.ObjectSet(),m,
                            image_set_list)
            load_images.run(w)
            self.assertEqual(m.get_current_measurement("Image", "Metadata_plate"),
                             "MMD-ControlSet-plateA-2008-08-06")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_row"),
                             "A")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_col"),
                             "12")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_site"),
                             "1")
            image_set = image_set_list.get_image_set(1)
            self.assertEqual(image_set.get_image_provider("Channel1").get_filename(),
                             filenames[2])
            self.assertEqual(image_set.get_image_provider("Channel2").get_filename(),
                             filenames[3])
            m = measurements.Measurements()
            w = W.Workspace(pipeline, load_images, image_set, cpo.ObjectSet(),m,
                            image_set_list)
            load_images.run(w)
            self.assertEqual(m.get_current_measurement("Image", "Metadata_plate"),
                             "MMD-ControlSet-plateA-2008-08-06")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_row"),
                             "A")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_col"),
                             "12")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_site"),
                             "2")
        finally:
            for filename in filenames:
                os.remove(os.path.join(directory,filename))
            os.rmdir(directory)
    
    def test_06_02_path_metadata(self):
        """Test recovery of path metadata"""
        directory = tempfile.mkdtemp()
        data = base64.b64decode(T.tif_8_1)
        path_and_file = [("MMD-ControlSet-plateA-2008-08-06_A12_s1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE]","w1.tif"),
                         ("MMD-ControlSet-plateA-2008-08-06_A12_s1_[EFBB8532-9A90-4040-8974-477FE1E0F3CA]","w2.tif"),
                         ("MMD-ControlSet-plateA-2008-08-06_A12_s2_[138B5A19-2515-4D46-9AB7-F70CE4D56631]","w1.tif"),
                         ("MMD-ControlSet-plateA-2008-08-06_A12_s2_[59784AC1-6A66-44DE-A87E-E4BDC1A33A54]","w2.tif")
                         ]
        for path,filename in path_and_file:
            os.mkdir(os.path.join(directory,path))
            fd = open(os.path.join(directory, path,filename),"wb")
            fd.write(data)
            fd.close()
        try:
            load_images = LI.LoadImages()
            load_images.add_imagecb()
            load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            load_images.descend_subdirectories.value = True
            load_images.match_method.value = LI.MS_EXACT_MATCH
            load_images.location.value = LI.DIR_OTHER
            load_images.location_other.value = directory
            load_images.group_by_metadata.value = True
            load_images.images[0][LI.FD_COMMON_TEXT].value = "w1.tif"
            load_images.images[1][LI.FD_COMMON_TEXT].value = "w2.tif"
            load_images.images[0][LI.FD_IMAGE_NAME].value = "Channel1"
            load_images.images[1][LI.FD_IMAGE_NAME].value = "Channel2"
            load_images.images[0][LI.FD_METADATA_CHOICE].value = LI.M_PATH
            load_images.images[1][LI.FD_METADATA_CHOICE].value = LI.M_PATH
            load_images.images[0][LI.FD_PATH_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)"
            load_images.images[1][LI.FD_PATH_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)"
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_listener(self.error_callback)
            pipeline.add_module(load_images)
            image_set_list = I.ImageSetList()
            load_images.prepare_run(pipeline, image_set_list, None)
            self.assertEqual(image_set_list.count(),2)
            image_set = image_set_list.get_image_set(0)
            self.assertEqual(image_set.get_image_provider("Channel1").get_filename(),
                             os.path.join(*path_and_file[0]))
            self.assertEqual(image_set.get_image_provider("Channel2").get_filename(),
                             os.path.join(*path_and_file[1]))
            m = measurements.Measurements()
            w = W.Workspace(pipeline, load_images, image_set, cpo.ObjectSet(),m,
                            image_set_list)
            load_images.run(w)
            self.assertEqual(m.get_current_measurement("Image", "Metadata_plate"),
                             "MMD-ControlSet-plateA-2008-08-06")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_row"),
                             "A")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_col"),
                             "12")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_site"),
                             "1")
            image_set = image_set_list.get_image_set(1)
            self.assertEqual(image_set.get_image_provider("Channel1").get_filename(),
                             os.path.join(*path_and_file[2]))
            self.assertEqual(image_set.get_image_provider("Channel2").get_filename(),
                             os.path.join(*path_and_file[3]))
            m = measurements.Measurements()
            w = W.Workspace(pipeline, load_images, image_set, cpo.ObjectSet(),m,
                            image_set_list)
            load_images.run(w)
            self.assertEqual(m.get_current_measurement("Image", "Metadata_plate"),
                             "MMD-ControlSet-plateA-2008-08-06")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_row"),
                             "A")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_well_col"),
                             "12")
            self.assertEqual(m.get_current_measurement("Image", "Metadata_site"),
                             "2")
        finally:
            for path, filename in path_and_file:
                os.remove(os.path.join(directory,path,filename))
                os.rmdir(os.path.join(directory,path))
            os.rmdir(directory)
    
    def test_06_03_missing_image(self):
        """Test expected failure when an image is missing from the set"""
        directory = tempfile.mkdtemp()
        data = base64.b64decode(T.tif_8_1)
        filename = "MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif"
        fd = open(os.path.join(directory, filename),"wb")
        fd.write(data)
        fd.close()
        try:
            load_images = LI.LoadImages()
            load_images.add_imagecb()
            load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            load_images.match_method.value = LI.MS_REGEXP
            load_images.location.value = LI.DIR_OTHER
            load_images.location_other.value = directory
            load_images.group_by_metadata.value = True
            load_images.check_images.value = True
            load_images.images[0][LI.FD_COMMON_TEXT].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1][LI.FD_COMMON_TEXT].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.images[0][LI.FD_IMAGE_NAME].value = "Channel1"
            load_images.images[1][LI.FD_IMAGE_NAME].value = "Channel2"
            load_images.images[0][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[1][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[0][LI.FD_FILE_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1][LI.FD_FILE_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_listener(self.error_callback)
            pipeline.add_module(load_images)
            image_set_list = I.ImageSetList()
            self.assertRaises(ValueError, load_images.prepare_run, pipeline, 
                              image_set_list, None)
        finally:
            os.remove(os.path.join(directory, filename))
            os.rmdir(directory)
    def test_06_04_conflict(self):
        """Test expected failure when two images have the same metadata"""
        directory = tempfile.mkdtemp()
        data = base64.b64decode(T.tif_8_1)
        filenames = ["MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s1_w2_[EFBB8532-9A90-4040-8974-477FE1E0F3CA].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[138B5A19-2515-4D46-9AB7-F70CE4D56631].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w1_[138B5A19-2515-4D46-9AB7-F70CE4D56631].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w2_[59784AC1-6A66-44DE-A87E-E4BDC1A33A54].tif"
                     ]
        for filename in filenames:
            fd = open(os.path.join(directory, filename),"wb")
            fd.write(data)
            fd.close()
        try:
            load_images = LI.LoadImages()
            load_images.add_imagecb()
            load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            load_images.match_method.value = LI.MS_REGEXP
            load_images.location.value = LI.DIR_OTHER
            load_images.location_other.value = directory
            load_images.group_by_metadata.value = True
            load_images.images[0][LI.FD_COMMON_TEXT].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1][LI.FD_COMMON_TEXT].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.images[0][LI.FD_IMAGE_NAME].value = "Channel1"
            load_images.images[1][LI.FD_IMAGE_NAME].value = "Channel2"
            load_images.images[0][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[1][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[0][LI.FD_FILE_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1][LI.FD_FILE_METADATA].value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_module(load_images)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            self.assertRaises(ValueError,load_images.prepare_run, pipeline, 
                              image_set_list, None)
        finally:
            for filename in filenames:
                os.remove(os.path.join(directory,filename))
            os.rmdir(directory)

if __name__=="main":
    unittest.main()
