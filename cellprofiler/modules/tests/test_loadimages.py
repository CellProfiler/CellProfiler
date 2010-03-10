"""Test the LoadImages module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import base64
import gc
import hashlib
import numpy as np
import os
import re
import unittest
import tempfile
import time
import sys
import zlib
from StringIO import StringIO

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as CPM
import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.tests as T
import cellprofiler.cpimage as I
import cellprofiler.objects as cpo
import cellprofiler.measurements as measurements
import cellprofiler.pipeline as P
import cellprofiler.workspace as W
from cellprofiler.modules.tests import example_images_directory

class testLoadImages(unittest.TestCase):
    def error_callback(self, calller, event):
        if isinstance(event, P.RunExceptionEvent):
            self.fail(event.error.message)

    def test_00_00init(self):
        x=LI.LoadImages()
    
    def test_00_01version(self):
        self.assertEqual(LI.LoadImages().variable_revision_number,4,"LoadImages' version number has changed")
    
    def test_01_01load_image_text_match(self):
        l=LI.LoadImages()
        l.settings()[l.SLOT_MATCH_METHOD].set_value(LI.MS_EXACT_MATCH)
        l.settings()[l.SLOT_LOCATION].value = LI.ABSOLUTE_FOLDER_NAME
        l.settings()[l.SLOT_LOCATION_OTHER].value =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_COMMON_TEXT].set_value("1-01-A-01.tif")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image")
        l.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(l)
        l.prepare_run(pipeline, image_set_list, None)
        self.assertEqual(image_set_list.count(),1,"Expected one image set in the list")
        l.prepare_group(pipeline, image_set_list, (), [1])
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),1)
        self.assertEqual(image_set.get_names()[0],"my_image")
        self.assertTrue(image_set.get_image("my_image"))
        
    def test_01_02load_image_text_match_many(self):
        l=LI.LoadImages()
        l.settings()[l.SLOT_MATCH_METHOD].set_value(LI.MS_EXACT_MATCH)
        l.settings()[l.SLOT_LOCATION].value = LI.ABSOLUTE_FOLDER_NAME
        l.settings()[l.SLOT_LOCATION_OTHER].value =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        for i in range(0,4):
            ii = i+1
            if i:
                l.add_imagecb()
            idx = l.SLOT_FIRST_IMAGE+l.SLOT_IMAGE_FIELD_COUNT * i 
            l.settings()[idx+l.SLOT_OFFSET_COMMON_TEXT].set_value("1-0%(ii)d-A-0%(ii)d.tif"%(locals()))
            l.settings()[idx+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image%(i)d"%(locals()))
        l.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_module(l)
        pipeline.add_listener(self.error_callback)
        l.prepare_run(pipeline, image_set_list, None)
        self.assertEqual(image_set_list.count(),1,"Expected one image set, there were %d"%(image_set_list.count()))
        image_set = image_set_list.get_image_set(0)
        l.prepare_group(pipeline, image_set_list, (), [1])
        self.assertEqual(len(image_set.get_names()),4)
        for i in range(0,4):
            self.assertTrue("my_image%d"%(i) in image_set.get_names())
            self.assertTrue(image_set.get_image("my_image%d"%(i)))
        
    def test_02_01load_image_regex_match(self):
        l=LI.LoadImages()
        l.settings()[l.SLOT_MATCH_METHOD].set_value(LI.MS_REGEXP)
        l.settings()[l.SLOT_LOCATION].value = LI.ABSOLUTE_FOLDER_NAME
        l.settings()[l.SLOT_LOCATION_OTHER].value =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_COMMON_TEXT].set_value("Channel1-[0-1][0-9]-A-01")
        l.settings()[l.SLOT_FIRST_IMAGE+l.SLOT_OFFSET_IMAGE_NAME].set_value("my_image")
        l.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_module(l)
        l.prepare_run(pipeline, image_set_list,None)
        self.assertEqual(image_set_list.count(),1,"Expected one image set in the list")
        l.prepare_group(pipeline, image_set_list, (), [1])
        image_set = image_set_list.get_image_set(0)
        self.assertEqual(len(image_set.get_names()),1)
        self.assertEqual(image_set.get_names()[0],"my_image")
        self.assertTrue(image_set.get_image("my_image"))
        
    def test_03_00_load_matlab_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:1234
FromMatlab:True

LoadImages:[module_num:1|svn_version:\'8913\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    How do you want to load these files?:Text-Exact match
    Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:Channel1-
    What do you want to call these images within CellProfiler?:MyImages
    Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:Channel2-
    What do you want to call these images within CellProfiler?:OtherImages
    Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:/
    What do you want to call these images within CellProfiler?:/
    Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:/
    What do you want to call these images within CellProfiler?:/
    If using ORDER, how many images are there in each group (i.e. each field of view)?:5
    Are you loading image or movie files?:Image
    If you are loading a movie, what is the extension?:stk
    Analyze all subfolders within the selected folder?:Yes
    Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder.:./Images
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(len(module.images), 2)
        self.assertEqual(module.images[0][LI.FD_IMAGE_NAME], "MyImages")
        self.assertEqual(module.images[1][LI.FD_IMAGE_NAME], "OtherImages")
        self.assertEqual(module.order_group_size, 5)
        self.assertTrue(module.analyze_sub_dirs())
        self.assertEqual(module.location, LI.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.location_other, "./Images")
        
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
        data = 'eJztV91u2jAUNjRURZOm7qLaLn1ZtoIga6UWTbQMJo2NMFRYt6rqVhcMWHJiFJwONlXaI+yR9ih7hD3CbOpA8CJC6S42jUhOfI7Pd36+EzmOVWxWi8/hXiYLrWIz3SEUwzpFvMNcOw8dvgNLLkYctyFz8rDZ8+Arj8KsCXN7+V0zLyZmNnsAlrtiFeu+fIrbunhsiBFXSwklxwJDyg3MOXG6gwQwwCOl/y7GCXIJuqT4BFEPD6YhfH3F6bDmqD9Zsljbo7iG7KCxuGqefYndwZuOD1TLdTLEtEE+Y60E3+wYX5EBYY7CK/+6dhKXcS2u5GF/fcpDLISHrYBe2r8EU3sjxP5BwH5TycRpkyvS9hCFxEbdSRbS31GEv03NnxxNPOTpF0PU4tBGvNWTfrIRfmIzfmLgqV9/BC6hxZdypVp9ayl8VNz4DD4OamwxHh9qcaVcxh3kUQ4rkkRYJi5uceaOfstjXfPnX76/ZID/qPzXZvJYA6eie3fBRfG9AWbrlnKphxwHU3OZuOVacaF89fcjtyA/xgzOEP11sMR9jcC91uqU8oftw/ozuRHiQuZJ6qOU3mFKj9mnwlkxXT9P+ZoSo57tFM6y6YPzL7kd8/rGuEEEcqxMjf3KPHoReexreUhZ+jrFyFUBdq9TaamymMN7SmcqXRmNppo79je3yH6Q1PBSLo0461M0sJV+mX6bq34v1e8fxu2+H39in1rh/h/cEZj/PoedD8aHjK7LvD4URw/c/5fqXfH7d+K+BXBh+1zweyLtL8B8Xh+DWV6l3BJbfd9l8n/IzdjjQ/sgQxlq35yaM1UxrQQO0Ho9yZA4wbziYrYVwYNe/5SXn4fLxIuHxLsXgTPUH5nEvQe34317jj3Q7H8BnRn8NQ=='
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),1)
        module = pipeline.module(1)
        self.assertEqual(module.load_choice(), LI.MS_EXACT_MATCH)
        self.assertTrue(module.load_images())
        self.assertFalse(module.load_movies())
        self.assertTrue(module.exclude.value)
        self.assertEqual(module.text_to_exclude(), 'ILLUM')
        self.assertEqual(module.images[0][LI.FD_IMAGE_NAME], 'DNA')
        self.assertEqual(module.images[0][LI.FD_COMMON_TEXT], 'Channel2')
        self.assertEqual(module.images[1][LI.FD_IMAGE_NAME], 'Cytoplasm')
        self.assertEqual(module.images[1][LI.FD_COMMON_TEXT], 'Channel1')
        
    def test_03_03_load_new_version_3(self):
        data = 'eJztV+1u0zAUdT+1CgmNP2M/vX/boFHawdgqtK20IIqaUm1lYkIgvNZtLTlxlDhbC9o78Eg8Eo9AnLlNaqKmK0iA1Ehucq/vPef6OLUdo9ppVl/Ap5oOjWqn2CcUwzZFvM8cswJt5pLRY1hzMOK4B5lVgR0PwzcehfAZLOmV8n5lbx+Wdf0QLHGlGsZ9/3bi/+T9+5rf0rIrJ+1UpAn7DHNOrIGbA1mwKf3f/XaOHIIuKT5H1MNuSDHxN6w+64ztaZfBeh7FLWRGg/2r5ZmX2HHf9ieJsrtNRpiekS9YGcIk7BRfEZcwS+ZLfNU75WVc4Q10yIc6pGJ02Ij4RfxrEMZnY+IfROLXpU2sHrkiPQ9RSEw0mFYR8CfgrSt4onXwiBdfjlCXQxPx7lDg6Ak4qRmcFNiT/AcJeTmFX9iNZvOdIfOTeNMz+WnQYovp+FDhFXYd95FHOWwIEWGdOLjLmTP+pY68gje5JniFiP5J9Wdm6siAC3/2/kZe0jytgVm9hF0bIsvCtLwMb71VXahe9b0qgcXe64JSr7BfiYXQ8pcH6Rc47xNwthQcYX/Sdovbx+3np+z6SHu0EzzXGD36oBcPP34t3+xE8IcJ+AcKvrAF3gVGjgR8cnNLYTCLD0OSwFdH49Dzm/NYWlbX2pgzmyLXjIz7rvNaBqt5nTevm7m77SN/Yr1a5a3ykvJOwPz/Qdz5IjikDBzm2dA/umD7fxrvSt9/M+9bJC9ufYzuNyL+M5iv6y6Y1VXYXUyp7TDxPeVoZnDodzXKUO/21K01/cdG5ACujqcQwxOtK+0/bSTooI4/1OXH8TJ8mRi+ewl5WflFp+6zi+i+PSceKPE/AfCf5eY='
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),1)
        module = pipeline.module(1)
        self.assertEqual(module.load_choice(), LI.MS_EXACT_MATCH)
        self.assertTrue(module.load_images())
        self.assertFalse(module.load_movies())
        self.assertTrue(module.exclude.value)
        self.assertEqual(module.text_to_exclude(), 'ILLUM')
        self.assertEqual(module.images[0][LI.FD_IMAGE_NAME], 'DNA')
        self.assertEqual(module.images[0][LI.FD_COMMON_TEXT], 'Channel2')
        self.assertEqual(module.images[0][LI.FD_FILE_METADATA], '^.*-(?P<Row>.+)-(?P<Col>[0-9]{2})')
        self.assertEqual(module.images[1][LI.FD_IMAGE_NAME], 'Cytoplasm')
        self.assertEqual(module.images[1][LI.FD_COMMON_TEXT], 'Channel1')
        self.assertEqual(module.images[1][LI.FD_FILE_METADATA], '^.*-(?P<Row>.+)-(?P<Col>[0-9]{2})')

    def test_03_04_load_new_version_4(self):
        data = ('eJztVt1O2zAUdn9A65AmuBqXvgS0VGnpNqgmILRDVOqfoGJDVaeZ1m0tOXGV'
                'OKjdxDvsco/DI/EIi0vSpF7WpN24mIQlKznH53zf8Zf4p6a1qtopfJtVYU1r'
                'KX1CMWxSxPvM1IvQ4G9gycSI4x5kRhGemQRq9gCq72GuUMztFwsHMK+qh2C1'
                'lqjUXjmPnxsArDvPF05PukNrrp0IdGFfYs6JMbDWQBpsu/57p18hk6Abiq8Q'
                'tbHlU3j+itFnrcloNlRjPZviOtKDwU6r2/oNNq1G30t0h5tkjOkl+YalKXhh'
                'F/iWWIQZbr6LL3tnvIxLvEKH+7SvQyJEh62AX8SfAz8+HRG/6drE6JFb0rMR'
                'hURHg1kVAu8kAm9TwhO9hcdc+ThGXQ51xLtDgaNG4CTmcBJgPyb/S4lf2GUG'
                'DcahbWF/HlH8yTmcJKizeHq+lviFXcZ9ZFMOK0JMWCYm7nJmTn6rY13C85qH'
                'lwHx60/N1ZEC185XfMq8P+m1LN9F41Os75yRdBZ2aYgMA9Oc8hc6letarDz5'
                '/8yBeP9nWN1nYkM1nG0mUPcwAuedhCPsL64A7Vy+o7RV5bDzPX+n7Bw3Pziq'
                'HrU15byzOzVLjeqRN74bj+9A4hO2gLrGyHSxCneP6DVm8KGPP/WV0cT3CL6H'
                '1HL72L9YJ895z3lPlXcCFq+fsHNxergOTGaPoHPk4tH/NN9V834E8sLWfXBf'
                'FfFfwWJd98C8rsLuYkpHJhP3VDOrTy9TVpYy1Hu8zWSrzmslcLGR55MJ4QnW'
                'lXTetiJ0kOfv6/JwvApfOoRvIyIv7d6URd5nsJzuOwvigRT/CxrVwC0=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        assert isinstance(module, LI.LoadImages)
        self.assertEqual(len(module.metadata_fields.selections), 1)
        self.assertEqual(module.metadata_fields.selections[0], "ROW")
        self.assertEqual(len(module.images), 1)
        self.assertEqual(module.images[0][LI.FD_FILE_METADATA], '^Channel[12]-[0-9]{2}-(?P<ROW>[A-H])-(?P<COL>[0-9]{2})')
        
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
        load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig')
                matfh.close()
                pixel_data = image.pixel_data
                pixel_data = (pixel_data * 255+.5).astype(np.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = np.fromstring(check_data,np.uint8).reshape(T.raw_8_1_shape)
                outer_self.assertTrue(np.all(pixel_data ==check_image))
                digest = hashlib.md5()
                digest.update((check_image.astype(float)/255).data)
                hexdigest = workspace.measurements.get_current_image_measurement('MD5Digest_Orig')
                outer_self.assertEqual(hexdigest, digest.hexdigest())
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
        load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig')
                matfh.close()
                pixel_data = image.pixel_data
                pixel_data = (pixel_data * 255+.5).astype(np.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = np.fromstring(check_data,np.uint8).reshape(T.raw_8_1_shape)
                outer_self.assertTrue(np.all(pixel_data ==check_image))
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
        load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig')
                matfh.close()
                pixel_data = image.pixel_data
                pixel_data = (pixel_data * 255+.5).astype(np.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = np.fromstring(check_data,np.uint8).reshape(T.raw_8_1_shape)
                outer_self.assertTrue(np.all(pixel_data ==check_image))
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
        load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
        load_images.location_other.value = path
        load_images.module_num = 1
        outer_self = self
        class CheckImage(CPM.CPModule):
            def run(self,workspace):
                image = workspace.image_set.get_image('Orig',
                                                      must_be_grayscale=True)
                pixel_data = image.pixel_data
                matfh.close()
                pixel_data = (pixel_data * 255).astype(np.uint8)
                check_data = base64.b64decode(T.raw_8_1)
                check_image = np.fromstring(check_data,np.uint8).reshape(T.raw_8_1_shape)
                # JPEG is lossy, apparently even when you ask for no compression
                epsilon = 1
                outer_self.assertTrue(np.all(np.abs(pixel_data.astype(int) 
                                                          - check_image.astype(int) <=
                                                          epsilon)))
        check_image = CheckImage()
        check_image.module_num = 2
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(load_images)
        pipeline.add_module(check_image)
        pipeline.run()
    
    def test_05_05_load_url(self):
        lip = LI.LoadImagesImageProvider(
            "broad", 
            "http://www.cellprofiler.org/linked_files",
            "broad-logo.gif")
        logo = lip.provide_image(None)
        self.assertEqual(logo.pixel_data.shape, (38, 150, 3))
        lip.release_memory()
        
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
            load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
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
            load_images.prepare_group(pipeline, image_set_list, (), [1,2])
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
            self.assertEqual(m.get_current_image_measurement("Metadata_Well"),
                             "A12")
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
            load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
            load_images.location_other.value = directory
            load_images.group_by_metadata.value = True
            load_images.images[0][LI.FD_COMMON_TEXT].value = "w1.tif"
            load_images.images[1][LI.FD_COMMON_TEXT].value = "w2.tif"
            load_images.images[0][LI.FD_IMAGE_NAME].value = "Channel1"
            load_images.images[1][LI.FD_IMAGE_NAME].value = "Channel2"
            load_images.images[0][LI.FD_METADATA_CHOICE].value = LI.M_PATH
            load_images.images[1][LI.FD_METADATA_CHOICE].value = LI.M_PATH
            load_images.images[0][LI.FD_PATH_METADATA].value = "(?P<plate>MMD.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)"
            load_images.images[1][LI.FD_PATH_METADATA].value = "(?P<plate>MMD.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)"
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_listener(self.error_callback)
            pipeline.add_module(load_images)
            image_set_list = I.ImageSetList()
            load_images.prepare_run(pipeline, image_set_list, None)
            self.assertEqual(image_set_list.count(),2)
            load_images.prepare_group(pipeline, image_set_list, {}, [1,2])
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
            load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
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
            load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
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
            
    def test_06_05_hierarchy(self):
        """Regression test a file applicable to multiple files
        
        The bug is documented in IMG-202
        """
        directory = tempfile.mkdtemp()
        data = base64.b64decode(T.tif_8_1)
        filenames = ["2008-08-06-run1-plateA_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-06-run1-plateA_A12_s2_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-07-run1-plateA_A12_s3_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-06-run1-plateB_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-06-run1-plateB_A12_s2_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-07-run1-plateB_A12_s3_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-06-run2-plateA_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-06-run2-plateA_A12_s2_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-07-run2-plateA_A12_s3_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-06-run2-plateB_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-06-run2-plateB_A12_s2_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "2008-08-07-run2-plateB_A12_s3_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "illum_run1-plateA.tif",
                     "illum_run1-plateB.tif",
                     "illum_run2-plateA.tif",
                     "illum_run2-plateB.tif",
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
            load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
            load_images.location_other.value = directory
            load_images.group_by_metadata.value = True
            load_images.images[0][LI.FD_COMMON_TEXT].value = "_w1_"
            load_images.images[1][LI.FD_COMMON_TEXT].value = "^illum"
            load_images.images[0][LI.FD_IMAGE_NAME].value = "Channel1"
            load_images.images[1][LI.FD_IMAGE_NAME].value = "Illum"
            load_images.images[0][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[1][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
            load_images.images[0][LI.FD_FILE_METADATA].value =\
                       ("^(?P<Date>[0-9]{4}-[0-9]{2}-[0-9]{2})-"
                        "run(?P<Run>[0-9])-(?P<plate>.*?)_"
                        "(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_"
                        "s(?P<site>[0-9]+)_w1_")
            load_images.images[1][LI.FD_FILE_METADATA].value =\
                       "^illum_run(?P<Run>[0-9])-(?P<plate>.*?)\\."
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_module(load_images)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            load_images.prepare_run(pipeline, image_set_list, None)
            for i in range(12):
                iset = image_set_list.legacy_fields["LoadImages:1"][i]
                ctags = re.search(load_images.images[0][LI.FD_FILE_METADATA].value,
                                  iset["Channel1"][3]).groupdict()
                itags = re.search(load_images.images[1][LI.FD_FILE_METADATA].value,
                                  iset["Illum"][3]).groupdict()
                self.assertEqual(ctags["Run"], itags["Run"])
                self.assertEqual(ctags["plate"], itags["plate"])
        finally:
            for filename in filenames:
                os.remove(os.path.join(directory,filename))
            os.rmdir(directory)
            
    def test_06_06_allowed_conflict(self):
        """Test choice of newest file when there is a conflict"""
        filenames = ["MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s1_w2_[EFBB8532-9A90-4040-8974-477FE1E0F3CA].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[138B5A19-2515-4D46-9AB7-F70CE4D56631].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w1_[138B5A19-2515-4D46-9AB7-F70CE4D56631].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w2_[59784AC1-6A66-44DE-A87E-E4BDC1A33A54].tif"
                     ]
        for chosen, order in ((2,(0,1,2,3,4)),(0,(2,1,0,3,4))):
            #
            # LoadImages should choose the file that was written last
            #
            directory = tempfile.mkdtemp()
            data = base64.b64decode(T.tif_8_1)
            for i in range(len(filenames)):
                filename = filenames[order[i]]
                fd = open(os.path.join(directory, filename),"wb")
                fd.write(data)
                fd.close()
                # make sure times are different
                if os.stat_float_times():
                    time.sleep(.01)
                else:
                    time.sleep(1)
            try:
                load_images = LI.LoadImages()
                load_images.add_imagecb()
                load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
                load_images.match_method.value = LI.MS_REGEXP
                load_images.location.value = LI.ABSOLUTE_FOLDER_NAME
                load_images.location_other.value = directory
                load_images.group_by_metadata.value = True
                load_images.metadata_fields.value = ["plate", "well_row", 
                                                     "well_col", "site"]
                load_images.check_images.value = False
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
                load_images.prepare_run(pipeline, image_set_list, None)
                d = dict(plate = "MMD-ControlSet-plateA-2008-08-06",
                         well_row = "A",
                         well_col = "12",
                         Well = "A12",
                         site = "1")
                key_names, groupings = load_images.get_groupings(image_set_list)
                self.assertEqual(len(groupings), 2)
                my_groups = [x for x in groupings
                             if all([d[key_name] == x[0][key_name]
                                     for key_name in key_names])]
                self.assertEqual(len(my_groups), 1)
                load_images.prepare_group(pipeline, image_set_list,
                                          d,  my_groups[0][1])
                image_set = image_set_list.get_image_set(d)
                image = image_set.get_image("Channel1")
                self.assertEqual(image.file_name, filenames[chosen])
            finally:
                for filename in filenames:
                    os.remove(os.path.join(directory,filename))
                os.rmdir(directory)
            
    def test_07_01_get_measurement_columns(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9157

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    What type of files are you loading?:individual images
    How do you want to load these files?:Text-Exact match
    How many images are there in each group?:3
    Type the text that the excluded images have in common:ILLUM
    Analyze all subfolders within the selected folder?:No
    Image location:Default Image Folder
    Enter the full path to the images:
    Do you want to check image sets for missing or duplicate files?:Yes
    Do you want to group image sets by metadata?:Yes
    Do you want to exclude certain files?:Yes
    What metadata fields do you want to group by?:
    Type the text that these images have in common (case-sensitive):Channel2
    What do you want to call this image in CellProfiler?:DNA
    What is the position of this image in each group?:1
    Do you want to extract metadata from the file name, the subfolder path or both?:File name
    Type the regular expression that finds metadata in the file name\x3A:^.*-(?P<WellRow>.+)-(?P<WellCol>\x5B0-9\x5D{2})
    Type the regular expression that finds metadata in the subfolder path\x3A:(?P<Year>\x5B0-9\x5D{4})-(?P<Month>\x5B0-9\x5D{2})-(?P<Day>\x5B0-9\x5D{2})
    Type the text that these images have in common (case-sensitive):Channel1
    What do you want to call this image in CellProfiler?:Cytoplasm
    What is the position of this image in each group?:2
    Do you want to extract metadata from the file name, the subfolder path or both?:File name
    Type the regular expression that finds metadata in the file name\x3A:^.*-(?P<Row>.+)-(?P<Col>\x5B0-9\x5D{2})
    Type the regular expression that finds metadata in the subfolder path\x3A:(?P<Year>\x5B0-9\x5D{4})-(?P<Month>\x5B0-9\x5D{2})-(?P<Day>\x5B0-9\x5D{2})
'''
        fd = StringIO(data)
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        module = pipeline.module(1)
        expected_cols = [('Image', 'FileName_DNA', 'varchar(128)'), 
                         ('Image', 'PathName_DNA', 'varchar(256)'),
                         ('Image', 'MD5Digest_DNA', 'varchar(32)'),
                         ('Image', 'Metadata_WellRow', 'varchar(128)'), 
                         ('Image', 'Metadata_WellCol', 'varchar(128)'), 
                         ('Image', 'FileName_Cytoplasm', 'varchar(128)'), 
                         ('Image', 'PathName_Cytoplasm', 'varchar(256)'), 
                         ('Image', 'MD5Digest_Cytoplasm', 'varchar(32)'),
                         ('Image', 'Metadata_Well', 'varchar(128)')]
        returned_cols = module.get_measurement_columns(pipeline)
        # check for duplicates
        assert len(returned_cols) == len(set(returned_cols))
        # check what was returned was expected
        for c in expected_cols: 
            assert c in returned_cols
        for c in returned_cols: 
            assert c in expected_cols
    
    def test_08_01_get_groupings(self):
        '''Get groupings for the SBS image set'''
        sbs_path = os.path.join(T.example_images_directory(),'ExampleSBSImages')
        module = LI.LoadImages()
        module.location.value = LI.ABSOLUTE_FOLDER_NAME
        module.location_other.value = sbs_path
        module.group_by_metadata.value = True
        module.images[0][LI.FD_COMMON_TEXT].value = 'Channel1-'
        module.images[0][LI.FD_IMAGE_NAME].value = 'MyImage'
        module.images[0][LI.FD_METADATA_CHOICE].value = LI.M_FILE_NAME
        module.images[0][LI.FD_FILE_METADATA].value = '^Channel1-[0-9]{2}-(?P<ROW>[A-H])-(?P<COL>[0-9]{2})'
        module.metadata_fields.value = "ROW"
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = pipeline.prepare_run(None)
        self.assertTrue(isinstance(image_set_list, I.ImageSetList))
        keys, groupings = module.get_groupings(image_set_list)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], "ROW")
        self.assertEqual(len(groupings), 8)
        self.assertTrue(all([g[0]["ROW"] == row for g, row in zip(groupings, 'ABCDEFGH')]))
        for grouping in groupings:
            row = grouping[0]["ROW"]
            module.prepare_group(pipeline, image_set_list, grouping[0],
                                 grouping[1])
            for image_number in grouping[1]:
                image_set = image_set_list.get_image_set(image_number-1)
                self.assertEqual(image_set.keys["ROW"], row)
                provider = image_set.get_image_provider("MyImage")
                self.assertTrue(isinstance(provider, LI.LoadImagesImageProvider))
                match = re.search(module.images[0][LI.FD_FILE_METADATA].value,
                                  provider.get_filename())
                self.assertTrue(match)
                self.assertEqual(row, match.group("ROW"))
    
    def test_09_01_load_avi(self):
        if LI.FF_AVI_MOVIES not in LI.FF:
            sys.stderr.write("WARNING: AVI movies not supported\n")
        avi_path = os.path.join(T.example_images_directory(), 
                                'ExampleTrackObjects')
        module = LI.LoadImages()
        module.file_types.value = LI.FF_AVI_MOVIES
        module.images[0][LI.FD_COMMON_TEXT].value = 'avi'
        module.images[0][LI.FD_IMAGE_NAME].value = 'MyImage'
        module.location.value = LI.ABSOLUTE_FOLDER_NAME
        module.location_other.value = avi_path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        module.prepare_group(pipeline, image_set_list, (), [1,2,3])
        image_set = image_set_list.get_image_set(0)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.get_names())
        image = image_set.get_image('MyImage')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (264,542,3))
        image_set = image_set_list.get_image_set(1)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.get_names())
        image = image_set.get_image('MyImage')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (264,542,3))
        self.assertTrue(np.any(img1!=img2))
    
    def test_09_02_load_stk(self):
        path = '//iodine/imaging_analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM'
        if not os.path.isdir(path):
            path = '/imaging/analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM'
            if not os.path.isdir(path):
                sys.stderr.write("WARNING: unknown path to stk file. Test not run.\n")
                return
        module = LI.LoadImages()
        module.file_types.value = LI.FF_STK_MOVIES
        module.images[0][LI.FD_COMMON_TEXT].value = 'stk'
        module.images[0][LI.FD_IMAGE_NAME].value = 'MyImage'
        module.location.value = LI.ABSOLUTE_FOLDER_NAME
        module.location_other.value = path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        module.prepare_group(pipeline, image_set_list, (), [1,2,3])
        image_set = image_set_list.get_image_set(0)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.get_names())
        image = image_set.get_image('MyImage')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (1040,1388))
        image_set = image_set_list.get_image_set(1)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.get_names())
        image = image_set.get_image('MyImage')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (1040,1388))
        self.assertTrue(np.any(img1!=img2))
    
    def test_10_1_load_many(self):
        '''Load an image many times to ensure that memory is freed each time'''
        path = os.path.join(example_images_directory(), "ExampleSBSImages")
        for i in range(3):
            module = LI.LoadImages()
            module.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            module.images[0][LI.FD_COMMON_TEXT].value = 'Channel1-'
            module.images[0][LI.FD_IMAGE_NAME].value = 'MyImage'
            module.location.value = LI.ABSOLUTE_FOLDER_NAME
            module.location_other.value = path
            module.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_module(module)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            module.prepare_run(pipeline, image_set_list, None)
            module.prepare_group(pipeline, image_set_list, (), np.arange(96))
            for j in range(96):
                image_set = image_set_list.get_image_set(j)
                m = measurements.Measurements()
                workspace = W.Workspace(pipeline, module, image_set,
                                        cpo.ObjectSet(), m,
                                        image_set_list)
                module.run(workspace)
                self.assertTrue('MyImage' in image_set.get_names())
                image = image_set.get_image('MyImage')
                self.assertEqual(image.pixel_data.shape[0], 640)
                self.assertEqual(image.pixel_data.shape[1], 640)
                image_set_list.purge_image_set(j)
                gc.collect()

if __name__=="main":
    unittest.main()
