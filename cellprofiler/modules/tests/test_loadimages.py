"""Test the LoadImages module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import base64
import gc
import glob
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
import traceback
import PIL.Image

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

IMAGE_NAME = "image"
OBJECTS_NAME = "objects"
OUTLINES_NAME = "outlines"

class testLoadImages(unittest.TestCase):
    def setUp(self):
        self.directory = None
        
    def tearDown(self):
        if self.directory is not None:
            try:
                for path in (os.path.sep.join((self.directory, "*","*")),
                             os.path.sep.join((self.directory, "*"))):
                    files = glob.glob(path)
                    for filename in files:
                        if os.path.isfile(filename):
                            os.remove(filename)
                        else:
                            os.rmdir(filename)
                os.rmdir(self.directory)
            except:
                sys.stderr.write("Failed during file delete / teardown\n")
                traceback.print_exc()
        
    def error_callback(self, calller, event):
        if isinstance(event, P.RunExceptionEvent):
            self.fail(event.error.message)

    def test_00_00init(self):
        x=LI.LoadImages()
    
    def test_00_01version(self):
        self.assertEqual(LI.LoadImages().variable_revision_number, 11,
                         "LoadImages' version number has changed")
    
    def test_01_01load_image_text_match(self):
        l=LI.LoadImages()
        l.match_method.value = LI.MS_EXACT_MATCH
        l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        l.location.custom_path =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        l.images[0].common_text.value = "1-01-A-01.tif"
        l.images[0].channels[0].image_name.value = "my_image"
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
        l.match_method.value = LI.MS_EXACT_MATCH
        l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        l.location.custom_path =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        for i in range(0,4):
            ii = i+1
            if i:
                l.add_imagecb()
            l.images[i].common_text.value = "1-0%(ii)d-A-0%(ii)d.tif" % locals()
            l.images[i].channels[0].image_name.value = "my_image%(i)d" % locals()
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
        
    def test_02_01_load_image_regex_match(self):
        l=LI.LoadImages()
        l.match_method.value = LI.MS_REGEXP
        l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        l.location.custom_path =\
            os.path.join(T.example_images_directory(),"ExampleSBSImages")
        l.images[0].common_text.value = "Channel1-[0-1][0-9]-A-01"
        l.images[0].channels[0].image_name.value = "my_image"
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
        
    def test_02_02_load_image_by_order(self):
        #
        # Make a list of 12 files
        #
        directory = tempfile.mkdtemp()
        self.directory = directory
        data = base64.b64decode(T.tif_8_1)
        tiff_fmt = "image%02d.tif"
        for i in range(12):
            path = os.path.join(directory, tiff_fmt % i)
            fd = open(path, "wb")
            fd.write(data)
            fd.close()
        #
        # Code for permutations taken from 
        # http://docs.python.org/library/itertools.html#itertools.permutations
        # which has the Python copyright
        #
        def permutations(iterable, r=None):
            # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
            # permutations(range(3)) --> 012 021 102 120 201 210
            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            if r > n:
                return
            indices = range(n)
            cycles = range(n, n-r, -1)
            yield tuple(pool[i] for i in indices[:r])
            while n:
                for i in reversed(range(r)):
                    cycles[i] -= 1
                    if cycles[i] == 0:
                        indices[i:] = indices[i+1:] + indices[i:i+1]
                        cycles[i] = n - i
                    else:
                        j = cycles[i]
                        indices[i], indices[-j] = indices[-j], indices[i]
                        yield tuple(pool[i] for i in indices[:r])
                        break
                else:
                    return
    
        #
        # Run through group sizes = 2-4, # of images 2-4
        #
        for group_size in range(2, 5):
            for image_count in range(2, group_size+1):
                #
                # For each possible permutation of image numbers
                #
                for indexes in permutations(range(group_size), image_count):
                    l = LI.LoadImages()
                    l.match_method.value = LI.MS_ORDER
                    l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
                    l.location.custom_path = directory
                    l.order_group_size.value = group_size
                    l.images[0].channels[0].image_name.value = "image%d" % 1
                    l.images[0].order_position.value = indexes[0] + 1
                    for i, index in enumerate(indexes[1:]):
                        l.add_imagecb()
                        l.images[i+1].order_position.value = index + 1
                        l.images[i+1].channels[0].image_name.value = "image%d" % (i+2)
                    l.module_num = 1
                    image_set_list = I.ImageSetList()
                    pipeline = P.Pipeline()
                    pipeline.add_module(l)
                    l.prepare_run(pipeline, image_set_list,None)
                    nsets = 12 / group_size
                    self.assertEqual(image_set_list.count(), nsets)
                    l.prepare_group(pipeline, image_set_list, (), 
                                    list(range(1, nsets+1)))
                    m = measurements.Measurements()
                    for i in range(0, nsets):
                        if i > 0:
                            m.next_image_set(i + 1)
                        image_set = image_set_list.get_image_set(i)
                        workspace = W.Workspace(pipeline, l, image_set,
                                                cpo.ObjectSet(), m,
                                                image_set_list)
                        l.run(workspace)
                        for j in range(image_count):
                            feature = LI.C_FILE_NAME + ("_image%d" % (j+1))
                            idx = i * group_size + indexes[j]
                            expected = tiff_fmt % idx
                            value = m.get_current_image_measurement(feature)
                            self.assertEqual(expected, value)
                    
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
        self.assertEqual(module.images[0].channels[0].image_name, "MyImages")
        self.assertEqual(module.images[1].channels[0].image_name, "OtherImages")
        self.assertEqual(module.order_group_size, 5)
        self.assertTrue(module.analyze_sub_dirs())
        self.assertEqual(module.location.dir_choice, 
                         LI.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(module.location.custom_path, "./Images")
        
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
        self.assertEqual(module.image_name_vars()[0],'OrigColor')
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
        self.assertEqual(module.image_name_vars()[0],'OrigRed')
        self.assertEqual(module.text_to_find_vars()[0].value,'s1_w1.TIF')
        self.assertEqual(module.image_name_vars()[1],'OrigGreen')
        self.assertEqual(module.text_to_find_vars()[1].value,'s1_w2.TIF')
        self.assertEqual(module.image_name_vars()[2],'OrigBlue')
        self.assertEqual(module.text_to_find_vars()[2].value,'s1_w3.TIF')
        self.assertFalse(module.analyze_sub_dirs())
        self.assertEqual(module.location.dir_choice, 
                         LI.DEFAULT_INPUT_FOLDER_NAME)
        
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
        self.assertEqual(module.images[0].channels[0].image_name, 'DNA')
        self.assertEqual(module.images[0].common_text, 'Channel2')
        self.assertEqual(module.images[1].channels[0].image_name, 'Cytoplasm')
        self.assertEqual(module.images[1].common_text, 'Channel1')
        self.assertEqual(module.location.dir_choice, 
                         LI.DEFAULT_INPUT_FOLDER_NAME)
        
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
        self.assertEqual(module.images[0].channels[0].image_name, 'DNA')
        self.assertEqual(module.images[0].common_text, 'Channel2')
        self.assertEqual(module.images[0].file_metadata, '^.*-(?P<Row>.+)-(?P<Col>[0-9]{2})')
        self.assertEqual(module.images[1].channels[0].image_name, 'Cytoplasm')
        self.assertEqual(module.images[1].common_text, 'Channel1')
        self.assertEqual(module.images[1].file_metadata, '^.*-(?P<Row>.+)-(?P<Col>[0-9]{2})')
        self.assertEqual(module.location.dir_choice, 
                         LI.DEFAULT_INPUT_FOLDER_NAME)

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
        self.assertEqual(module.images[0].file_metadata, '^Channel[12]-[0-9]{2}-(?P<ROW>[A-H])-(?P<COL>[0-9]{2})')
        self.assertEqual(module.location.dir_choice, 
                         LI.DEFAULT_INPUT_FOLDER_NAME)
        
    def test_03_05_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9497

LoadImages:[module_num:1|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Thumb
    Analyze all subfolders within the selected folder?:No
    Image location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:Yes
    Specify metadata fields to group by:
    Text that these images have in common (case-sensitive):Foo
    Name of this image in CellProfiler:DNA
    Position of this image in each group:1
    Select from where to extract metadata?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)$
    Text that these images have in common (case-sensitive):Bar
    Name of this image in CellProfiler:Cytoplasm
    Position of this image in each group:2
    Select from where to extract metadata?:File name
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Text that these images have in common (case-sensitive):Baz
    Name of this image in CellProfiler:Other
    Position of this image in each group:3
    Select from where to extract metadata?:Path
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$

LoadImages:[module_num:2|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    File type to be loaded:stk movies
    File selection method:Text-Regular expressions
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:Yes
    Image location:Default Output Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:Yes
    Exclude certain files?:No
    Specify metadata fields to group by:Plate,Run
    Text that these images have in common (case-sensitive):Whatever
    Name of this image in CellProfiler:DNA
    Position of this image in each group:1
    Select from where to extract metadata?:Both
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$

LoadImages:[module_num:3|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    File type to be loaded:avi movies
    File selection method:Order
    Number of images in each group?:5
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Image location:Elsewhere...\x7C/imaging/analysis/People/Lee
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Text that these images have in common (case-sensitive):
    Name of this image in CellProfiler:DNA
    Position of this image in each group:2
    Select from where to extract metadata?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Text that these images have in common (case-sensitive):
    Name of this image in CellProfiler:Actin
    Position of this image in each group:1
    Select from where to extract metadata?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$

LoadImages:[module_num:4|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    File type to be loaded:tif,tiff,flex movies
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Image location:Default Input Folder sub-folder\x7Cfoo
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Text that these images have in common (case-sensitive):
    Name of this image in CellProfiler:DNA
    Position of this image in each group:1
    Select from where to extract metadata?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$

LoadImages:[module_num:5|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Image location:Default Output Folder sub-folder\x7Cbar
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Text that these images have in common (case-sensitive):
    Name of this image in CellProfiler:DNA
    Position of this image in each group:1
    Select from where to extract metadata?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 5)
        
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(module.order_group_size, 3)
        self.assertEqual(module.match_exclude, "Thumb")
        self.assertEqual(module.descend_subdirectories,LI.SUB_NONE)
        self.assertEqual(module.location.dir_choice, LI.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.check_images)
        self.assertFalse(module.group_by_metadata)
        self.assertTrue(module.exclude)
        self.assertEqual(len(module.images), 3)
        self.assertEqual(module.images[0].channels[0].image_name, "DNA")
        self.assertEqual(module.images[0].order_position, 1)
        self.assertEqual(module.images[0].common_text, "Foo")
        self.assertEqual(module.images[0].metadata_choice, LI.M_NONE)
        self.assertEqual(module.images[0].file_metadata, "^(?P<Plate>.*)")
        self.assertEqual(module.images[0].path_metadata,r".*[\\/](?P<Date>.*)$")
        self.assertEqual(module.images[1].channels[0].image_name, "Cytoplasm")
        self.assertEqual(module.images[1].common_text, "Bar")
        self.assertEqual(module.images[1].metadata_choice, LI.M_FILE_NAME)
        self.assertEqual(module.images[2].channels[0].image_name, "Other")
        self.assertEqual(module.images[2].common_text, "Baz")
        self.assertEqual(module.images[2].metadata_choice, LI.M_PATH)
        
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_STK_MOVIES)
        self.assertEqual(module.match_method, LI.MS_REGEXP)
        self.assertEqual(module.location.dir_choice, LI.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertTrue(module.group_by_metadata)
        self.assertEqual(module.descend_subdirectories, LI.SUB_ALL)
        self.assertEqual(len(module.metadata_fields.selections), 2)
        self.assertEqual(module.metadata_fields.selections[0], "Plate")
        self.assertEqual(module.metadata_fields.selections[1], "Run")
        self.assertEqual(module.images[0].metadata_choice, LI.M_BOTH)
        
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_AVI_MOVIES)
        self.assertEqual(module.match_method, LI.MS_ORDER)
        self.assertEqual(module.location.dir_choice, LI.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.location.custom_path, "/imaging/analysis/People/Lee")

        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
        self.assertEqual(module.location.dir_choice, LI.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(module.location.custom_path, "foo")
        
        module = pipeline.modules()[4]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.location.dir_choice, LI.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(module.location.custom_path, "bar")

    def test_03_06_load_v6(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9801

LoadImages:[module_num:1|svn_version:\'9799\'|variable_revision_number:6|show_window:True|notes:\x5B\'A flex file\'\x5D]
    File type to be loaded:tif,tiff,flex movies
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Thumb
    Analyze all subfolders within the selected folder?:No
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:Yes
    Exclude certain files?:No
    Specify metadata fields to group by:Series,T,Z
    Image count:1
    Text that these images have in common (case-sensitive):.flex
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:foo
    Type the regular expression that finds metadata in the subfolder path:bar
    Channel count:2
    Name this loaded image:DNA
    Channel number\x3A:1
    Name this loaded image:Protein
    Channel number\x3A:3
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        
        module = pipeline.modules()[0]
        module.notes = "A flex file"
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(module.order_group_size, 3)
        self.assertEqual(module.match_exclude, "Thumb")
        self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
        self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.check_images)
        self.assertTrue(module.group_by_metadata)
        self.assertFalse(module.exclude)
        self.assertEqual(len(module.metadata_fields.selections), 3)
        self.assertEqual(module.metadata_fields.selections[0], LI.M_SERIES)
        self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
        self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
        self.assertEqual(module.image_count.value, 1)
        self.assertEqual(len(module.images), 1)
        image = module.images[0]
        self.assertEqual(image.common_text, ".flex")
        self.assertEqual(image.order_position, 1)
        self.assertEqual(image.metadata_choice, LI.M_NONE)
        self.assertEqual(image.file_metadata, "foo")
        self.assertEqual(image.path_metadata, "bar")
        self.assertEqual(image.channel_count.value, 2)
        self.assertEqual(len(image.channels), 2)
        for channel, channel_number, image_name in (
            (image.channels[0], 1, "DNA"),
            (image.channels[1], 3, "Protein")):
            self.assertEqual(channel.channel_number, channel_number)
            self.assertEqual(channel.image_name, image_name)
            
    def test_03_07_load_v7(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10021

LoadImages:[module_num:1|svn_version:\'9976\'|variable_revision_number:7|show_window:True|notes:\x5B\'A flex file\'\x5D]
    File type to be loaded:tif,tiff,flex movies
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Thumb
    Analyze all subfolders within the selected folder?:No
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:Yes
    Exclude certain files?:No
    Specify metadata fields to group by:Series,T,Z
    Image count:1
    Text that these images have in common (case-sensitive):.flex
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:foo
    Type the regular expression that finds metadata in the subfolder path:bar
    Channel count:2
    Group movie frames?:No
    Interleaving\x3A:Interleaved
    Channels per group\x3A:2
    Name this loaded image:DNA
    Channel number\x3A:1
    Name this loaded image:Protein
    Channel number\x3A:3
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        
        module = pipeline.modules()[0]
        module.notes = "A flex file"
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(module.order_group_size, 3)
        self.assertEqual(module.match_exclude, "Thumb")
        self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
        self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.check_images)
        self.assertTrue(module.group_by_metadata)
        self.assertFalse(module.exclude)
        self.assertEqual(len(module.metadata_fields.selections), 3)
        self.assertEqual(module.metadata_fields.selections[0], LI.M_SERIES)
        self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
        self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
        self.assertEqual(module.image_count.value, 1)
        self.assertEqual(len(module.images), 1)
        image = module.images[0]
        self.assertEqual(image.common_text, ".flex")
        self.assertEqual(image.order_position, 1)
        self.assertEqual(image.metadata_choice, LI.M_NONE)
        self.assertEqual(image.file_metadata, "foo")
        self.assertEqual(image.path_metadata, "bar")
        self.assertEqual(image.channel_count.value, 2)
        self.assertEqual(len(image.channels), 2)
        for channel, channel_number, image_name in (
            (image.channels[0], 1, "DNA"),
            (image.channels[1], 3, "Protein")):
            self.assertEqual(channel.channel_number, channel_number)
            self.assertEqual(channel.image_name, image_name)

    def test_03_08_load_v8(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10530

LoadImages:[module_num:1|svn_version:\'10503\'|variable_revision_number:8|show_window:True|notes:\x5B\'A flex file\'\x5D]
    File type to be loaded:tif,tiff,flex movies, zvi movies
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Thumb
    Analyze all subfolders within the selected folder?:No
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:Yes
    Exclude certain files?:No
    Specify metadata fields to group by:Series,T,Z
    Image count:1
    Text that these images have in common (case-sensitive):.flex
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:foo
    Type the regular expression that finds metadata in the subfolder path:bar
    Channel count:2
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Name this loaded image:DNA
    Channel number:1
    Rescale image?:Yes
    Name this loaded image:Protein
    Channel number:3
    Rescale image?:No
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        
        module = pipeline.modules()[0]
        module.notes = "A flex file"
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(module.order_group_size, 3)
        self.assertEqual(module.match_exclude, "Thumb")
        self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
        self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.check_images)
        self.assertTrue(module.group_by_metadata)
        self.assertFalse(module.exclude)
        self.assertEqual(len(module.metadata_fields.selections), 3)
        self.assertEqual(module.metadata_fields.selections[0], LI.M_SERIES)
        self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
        self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
        self.assertEqual(module.image_count.value, 1)
        self.assertEqual(len(module.images), 1)
        image = module.images[0]
        self.assertEqual(image.common_text, ".flex")
        self.assertEqual(image.order_position, 1)
        self.assertEqual(image.metadata_choice, LI.M_NONE)
        self.assertEqual(image.file_metadata, "foo")
        self.assertEqual(image.path_metadata, "bar")
        self.assertEqual(image.channel_count.value, 2)
        self.assertEqual(len(image.channels), 2)
        for channel, channel_number, image_name, rescale in (
            (image.channels[0], 1, "DNA", True),
            (image.channels[1], 3, "Protein", False)):
            self.assertEqual(channel.channel_number, channel_number)
            self.assertEqual(channel.image_name, image_name)
            self.assertEqual(channel.rescale.value, rescale)
            self.assertEqual(channel.image_object_choice, LI.IO_IMAGES)

    def test_03_09_load_v9(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10530

LoadImages:[module_num:1|svn_version:\'10503\'|variable_revision_number:9|show_window:True|notes:\x5B\'A flex file\'\x5D]
    File type to be loaded:tif,tiff,flex movies, zvi movies
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Thumb
    Analyze all subfolders within the selected folder?:No
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:Yes
    Exclude certain files?:No
    Specify metadata fields to group by:Series,T,Z
    Image count:1
    Text that these images have in common (case-sensitive):.flex
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:foo
    Type the regular expression that finds metadata in the subfolder path:bar
    Channel count:2
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load images or objects?:Images
    Name this loaded image:DNA
    Name this loaded object:Nuclei
    Channel number:1
    Rescale image?:Yes
    Load images or objects?:Objects
    Name this loaded image:Protein
    Name this loaded object:Cytoplasm
    Channel number:3
    Rescale image?:No
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        
        module = pipeline.modules()[0]
        module.notes = "A flex file"
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(module.order_group_size, 3)
        self.assertEqual(module.match_exclude, "Thumb")
        self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
        self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.check_images)
        self.assertTrue(module.group_by_metadata)
        self.assertFalse(module.exclude)
        self.assertEqual(len(module.metadata_fields.selections), 3)
        self.assertEqual(module.metadata_fields.selections[0], LI.M_SERIES)
        self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
        self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
        self.assertEqual(module.image_count.value, 1)
        self.assertEqual(len(module.images), 1)
        image = module.images[0]
        self.assertEqual(image.common_text, ".flex")
        self.assertEqual(image.order_position, 1)
        self.assertEqual(image.metadata_choice, LI.M_NONE)
        self.assertEqual(image.file_metadata, "foo")
        self.assertEqual(image.path_metadata, "bar")
        self.assertEqual(image.channel_count.value, 2)
        self.assertEqual(len(image.channels), 2)
        for channel, choice, channel_number, image_name, object_name, rescale in (
            (image.channels[0], LI.IO_IMAGES, 1, "DNA", "Nuclei", True),
            (image.channels[1], LI.IO_OBJECTS, 3, "Protein", "Cytoplasm", False)):
            self.assertEqual(channel.image_object_choice, choice)
            self.assertEqual(channel.channel_number, channel_number)
            self.assertEqual(channel.image_name, image_name)
            self.assertEqual(channel.object_name, object_name)
            self.assertEqual(channel.rescale.value, rescale)
    
    def test_03_10_load_v10(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10809

LoadImages:[module_num:1|svn_version:\'10807\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Whatever
    Analyze all subfolders within the selected folder?:No
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Image count:2
    Text that these images have in common (case-sensitive):_w1_
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:w1
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:Yes
    Name the outline image:MyOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):_w2_
    Position of this image in each group:2
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:w2
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:MyNucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(module.order_group_size, 3)
        self.assertEqual(module.match_exclude, "Whatever")
        self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
        self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.check_images)
        self.assertFalse(module.group_by_metadata)
        self.assertFalse(module.exclude)
        self.assertEqual(module.image_count.value, 2)
        self.assertEqual(len(module.images), 2)
        image = module.images[0]
        self.assertEqual(image.common_text, "_w1_")
        self.assertEqual(image.order_position, 1)
        self.assertEqual(image.metadata_choice, LI.M_NONE)
        self.assertEqual(image.channel_count.value, 1)
        self.assertEqual(len(image.channels), 1)
        channel = image.channels[0]
        self.assertEqual(channel.image_object_choice, LI.IO_IMAGES)
        self.assertEqual(channel.channel_number, 1)
        self.assertEqual(channel.image_name, "w1")
        self.assertEqual(channel.object_name, "Nuclei")
        self.assertTrue(channel.rescale)
        self.assertTrue(channel.wants_outlines)
        self.assertEqual(channel.outlines_name, "MyOutlines")
        self.assertFalse(module.images[1].channels[0].wants_outlines)
        
    def test_03_11_load_v11(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10809

LoadImages:[module_num:1|svn_version:\'10807\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Whatever
    Analyze subfolders within the selected folder?:All
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:hello/kitty,fubar
    Image count:1
    Text that these images have in common (case-sensitive):_w1_
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:w1
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:Yes
    Name the outline image:MyOutlines
    Channel number:1
    Rescale intensities?:Yes

LoadImages:[module_num:2|svn_version:\'10807\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Whatever
    Analyze subfolders within the selected folder?:Some
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:hello/kitty,fubar
    Image count:1
    Text that these images have in common (case-sensitive):_w1_
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:w1
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:Yes
    Name the outline image:MyOutlines
    Channel number:1
    Rescale intensities?:Yes

LoadImages:[module_num:3|svn_version:\'10807\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Whatever
    Analyze subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:hello/kitty,fubar
    Image count:1
    Text that these images have in common (case-sensitive):_w1_
    Position of this image in each group:1
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:w1
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:Yes
    Name the outline image:MyOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
        self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
        self.assertEqual(module.order_group_size, 3)
        self.assertEqual(module.match_exclude, "Whatever")
        self.assertEqual(module.descend_subdirectories, LI.SUB_ALL)
        self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
        self.assertTrue(module.check_images)
        self.assertFalse(module.group_by_metadata)
        self.assertFalse(module.exclude)
        self.assertEqual(module.image_count.value, 1)
        self.assertEqual(len(module.images), 1)
        image = module.images[0]
        self.assertEqual(image.common_text, "_w1_")
        self.assertEqual(image.order_position, 1)
        self.assertEqual(image.metadata_choice, LI.M_NONE)
        self.assertEqual(image.channel_count.value, 1)
        self.assertEqual(len(image.channels), 1)
        channel = image.channels[0]
        self.assertEqual(channel.image_object_choice, LI.IO_IMAGES)
        self.assertEqual(channel.channel_number, 1)
        self.assertEqual(channel.image_name, "w1")
        self.assertEqual(channel.object_name, "Nuclei")
        self.assertTrue(channel.rescale)
        self.assertTrue(channel.wants_outlines)
        self.assertEqual(channel.outlines_name, "MyOutlines")
        
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.descend_subdirectories, LI.SUB_SOME)
        
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, LI.LoadImages))
        self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
        
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
        self.assertEqual(module.image_name_vars()[0],'OrigRed')
        self.assertEqual(module.text_to_find_vars()[0].value,'s1_w1.TIF')
        self.assertEqual(module.image_name_vars()[1],'OrigGreen')
        self.assertEqual(module.text_to_find_vars()[1].value,'s1_w2.TIF')
        self.assertEqual(module.image_name_vars()[2],'OrigBlue')
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
        load_images.images[0].common_text.value = filename
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        load_images.location.custom_path = path
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
                digest.update((check_image.astype(np.float32)/255).data)
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
        load_images.images[0].common_text.value = filename
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        load_images.location.custom_path = path
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
        load_images.images[0].common_text.value = filename
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        load_images.location.custom_path = path
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
        m = pipeline.run()
        self.assertTrue(isinstance(m, measurements.Measurements))
        fn = m.get_all_measurements(measurements.IMAGE, 'FileName_Orig')
        self.assertEqual(len(fn), 1)
        self.assertEqual(fn[0], filename)
        p = m.get_all_measurements(measurements.IMAGE, 'PathName_Orig')
        self.assertEqual(p[0], path)
        scale = m.get_all_measurements(measurements.IMAGE, 'Scaling_Orig')
        self.assertEqual(scale[0], 255)

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
        load_images.images[0].common_text.value = filename
        load_images.images[0].channels[0].image_name.value = 'Orig'
        load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        load_images.location.custom_path = path
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
            "broad-logo.gif", True)
        logo = lip.provide_image(None)
        self.assertEqual(logo.pixel_data.shape, (38, 150, 4))
        lip.release_memory()
        
    def test_05_06_load_Nikon_tif(self):
        '''This is the Nikon format TIF file from IMG-838'''
        nikon_path = os.path.join(T.testimages_directory(), "NikonTIF.tif")
        image = LI.load_using_bioformats(nikon_path)
        self.assertEqual(tuple(image.shape), (731, 805, 3))
        self.assertAlmostEqual(np.sum(image.astype(np.float64)), 560730.83, 0)
        
    def test_05_07_load_Metamorph_tif(self):
        '''Regression test of IMG-883
        
        This file generated a null-pointer exception in the MetamorphReader
        '''
        metamorph_path = os.path.join(
            T.testimages_directory(), 
            "IXMtest_P24_s9_w560D948A4-4D16-49D0-9080-7575267498F9.tif")
        image = LI.load_using_bioformats(metamorph_path)
        self.assertEqual(tuple(image.shape), (520, 696))
        self.assertAlmostEqual(np.sum(image.astype(np.float64)), 2071.93, 0)
        
    def test_05_08_load_5channel_tif(self):
        '''Load a 5-channel image'''
        
        path = T.testimages_directory()
        module = LI.LoadImages()
        module.module_num = 1
        module.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        module.match_method.value = LI.MS_EXACT_MATCH
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = path
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.images[0].common_text.value = "5channel.tif"
        
        pipeline = P.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, P.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        
        image_set_list = I.ImageSetList()
        self.assertTrue(module.prepare_run(pipeline, image_set_list, None))
        self.assertEqual(image_set_list.count(), 1)
        key_names, group_list = pipeline.get_groupings(image_set_list)
        self.assertEqual(len(group_list), 1)
        grouping, image_numbers = group_list[0]
        self.assertEqual(len(image_numbers), 1)
        module.prepare_group(pipeline, image_set_list, grouping, image_numbers)
        
        image_set = image_set_list.get_image_set(0)
        workspace = W.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                measurements.Measurements(), image_set_list)
        module.run(workspace)
        image = image_set.get_image(IMAGE_NAME)
        pixels = image.pixel_data
        self.assertEqual(pixels.ndim, 3)
        self.assertEqual(tuple(pixels.shape), (64, 64, 5))

    def test_05_09_load_C01(self):
        """IMG-457: Test loading of a .c01 file"""
        c01_path = os.path.join(T.testimages_directory(), "icd002235_090127090001_a01f00d1.c01")
        image = LI.load_using_bioformats(c01_path)
        self.assertEqual(tuple(image.shape), (512,512))
        m = hashlib.md5()
        m.update((image * 65535).astype(np.uint16))
        self.assertEqual(m.digest(), 'SER\r\xc4\xd5\x02\x13@P\x12\x99\xe2(e\x85')
        
    def test_06_01_file_metadata(self):
        """Test file metadata on two sets of two files
        
        """
        directory = tempfile.mkdtemp()
        self.directory = directory
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
        load_images = LI.LoadImages()
        load_images.add_imagecb()
        load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = LI.MS_REGEXP
        load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        load_images.location.custom_path = directory
        load_images.group_by_metadata.value = True
        load_images.images[0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
        load_images.images[0].channels[0].image_name.value = "Channel1"
        load_images.images[1].channels[0].image_name.value = "Channel2"
        load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
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
    
    def test_06_02_path_metadata(self):
        """Test recovery of path metadata"""
        directory = tempfile.mkdtemp()
        self.directory = directory
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
    
    def test_06_03_missing_image(self):
        """Test expected failure when an image is missing from the set"""
        directory = tempfile.mkdtemp()
        self.directory = directory
        data = base64.b64decode(T.tif_8_1)
        filename = "MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif"
        fd = open(os.path.join(directory, filename),"wb")
        fd.write(data)
        fd.close()
        load_images = LI.LoadImages()
        load_images.add_imagecb()
        load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = LI.MS_REGEXP
        load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        load_images.location.custom_path = directory
        load_images.group_by_metadata.value = True
        load_images.check_images.value = True
        load_images.images[0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
        load_images.images[0].channels[0].image_name.value = "Channel1"
        load_images.images[1].channels[0].image_name.value = "Channel2"
        load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
        load_images.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(load_images)
        image_set_list = I.ImageSetList()
        self.assertRaises(ValueError, load_images.prepare_run, pipeline, 
                          image_set_list, None)
            
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
            load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            load_images.location.custom_path = directory
            load_images.group_by_metadata.value = True
            load_images.images[0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.images[0].channels[0].image_name.value = "Channel1"
            load_images.images[1].channels[0].image_name.value = "Channel2"
            load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
            load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
            load_images.images[0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
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
            load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            load_images.location.custom_path = directory
            load_images.group_by_metadata.value = True
            load_images.images[0].common_text.value = "_w1_"
            load_images.images[1].common_text.value = "^illum"
            load_images.images[0].channels[0].image_name.value = "Channel1"
            load_images.images[1].channels[0].image_name.value = "Illum"
            load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
            load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
            load_images.images[0].file_metadata.value =\
                       ("^(?P<Date>[0-9]{4}-[0-9]{2}-[0-9]{2})-"
                        "run(?P<Run>[0-9])-(?P<plate>.*?)_"
                        "(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_"
                        "s(?P<site>[0-9]+)_w1_")
            load_images.images[1].file_metadata.value =\
                       "^illum_run(?P<Run>[0-9])-(?P<plate>.*?)\\."
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_module(load_images)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            load_images.prepare_run(pipeline, image_set_list, None)
            for i in range(12):
                iset = image_set_list.legacy_fields["LoadImages:1"][i]
                ctags = re.search(load_images.images[0].file_metadata.value,
                                  iset["Channel1"][3]).groupdict()
                itags = re.search(load_images.images[1].file_metadata.value,
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
                # The Mac claims to save float times, but stat returns
                # a float whose fractional part is always 0
                #
                # Also happens on at least one Centos build.
                #
                if os.stat_float_times() and not sys.platform in ("darwin", "linux2"):
                    time.sleep(.1)
                else:
                    time.sleep(1)
            try:
                load_images = LI.LoadImages()
                load_images.add_imagecb()
                load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
                load_images.match_method.value = LI.MS_REGEXP
                load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
                load_images.location.custom_path = directory
                load_images.group_by_metadata.value = True
                load_images.metadata_fields.value = ["plate", "well_row", 
                                                     "well_col", "site"]
                load_images.check_images.value = False
                load_images.images[0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
                load_images.images[1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
                load_images.images[0].channels[0].image_name.value = "Channel1"
                load_images.images[1].channels[0].image_name.value = "Channel2"
                load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
                load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
                load_images.images[0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
                load_images.images[1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
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
                    p = os.path.join(directory,filename)
                    try:
                        os.remove(p)
                    except:
                        print "Failed to remove %s" % p
                try:
                    os.rmdir(directory)
                except:
                    print "Failed to remove " + directory
                    
    def test_06_07_subfolders(self):
        '''Test recursion down the list of subfolders'''
        directory = tempfile.mkdtemp()
        filenames = [ ("d1","bar.tif"),
                      ("d1","foo.tif"),
                      (os.path.join("d2","d3"), "foo.tif"),
                      (os.path.join("d2","d4"), "bar.tif")]
        data = base64.b64decode(T.tif_8_1)
        try:
            for path, file_name in filenames:
                d = os.path.join(directory, path)
                if not os.path.isdir(d):
                    os.makedirs(d)
                fd = open(os.path.join(directory, path, file_name),"wb")
                fd.write(data)
                fd.close()
            # test recursive symlinks
            try:
                os.symlink(os.path.join(directory, filenames[0][0]),
                           os.path.join(directory, filenames[-1][0], filenames[0][0]))
            except Exception, e:
                print "ignoring symlink exception:", e
            load_images = LI.LoadImages()
            load_images.module_num = 1
            load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            load_images.match_method.value = LI.MS_EXACT_MATCH
            load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            load_images.descend_subdirectories.value = LI.SUB_ALL
            load_images.location.custom_path = directory
            load_images.images[0].common_text.value = ".tif"
            load_images.images[0].channels[0].image_name.value = "my_image"
            load_images.check_images.value = False
            pipeline = P.Pipeline()
            pipeline.add_module(load_images)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            self.assertTrue(load_images.prepare_run(pipeline, image_set_list, None))
            self.assertEqual(image_set_list.count(), len(filenames))
            m = measurements.Measurements()
            load_images.prepare_group(pipeline, image_set_list, {}, np.arange(1, len(filenames)+1))
            for i, (path, file_name) in enumerate(filenames):
                if i > 0:
                    m.next_image_set()
                image_set = image_set_list.get_image_set(i)
                w = W.Workspace(pipeline, load_images, image_set,
                                cpo.ObjectSet(), m, image_set_list)
                load_images.run(w)
                image_provider = image_set.get_image_provider("my_image")
                self.assertEqual(image_provider.get_pathname(), directory)
                self.assertEqual(image_provider.get_filename(), os.path.join(path, file_name))
                f = m.get_current_image_measurement("FileName_my_image")
                self.assertEqual(f, file_name)
                p = m.get_current_image_measurement("PathName_my_image")
                self.assertEqual(os.path.join(directory, path), p)
        finally:
            for path, directories, file_names in os.walk(directory, False):
                for file_name in file_names:
                    p = os.path.join(path, file_name)
                    try:
                        os.remove(p)
                    except:
                        print "Failed to remove " + p
                        traceback.print_exc()
                try:
                    os.rmdir(path)
                except:
                    print "Failed to remove " + path
                    traceback.print_exc()
            
    def test_06_08_some_subfolders(self):
        '''Test recursion down the list of subfolders, some folders filtered'''
        directory = tempfile.mkdtemp()
        filenames = [ ("d1","bar.tif"),
                      ("d1","foo.tif"),
                      (os.path.join("d2","d3"), "foo.tif"),
                      (os.path.join("d2","d4"), "bar.tif"),
                      (os.path.join("d5","d6","d7"), "foo.tif")]
        exclusions = [ os.path.join("d2","d3"),
                       os.path.join("d5","d6") ]
        expected_filenames = filenames[:2] + [filenames[3]]

        data = base64.b64decode(T.tif_8_1)
        try:
            for path, file_name in filenames:
                d = os.path.join(directory, path)
                if not os.path.isdir(d):
                    os.makedirs(d)
                fd = open(os.path.join(directory, path, file_name),"wb")
                fd.write(data)
                fd.close()
            load_images = LI.LoadImages()
            load_images.module_num = 1
            load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            load_images.match_method.value = LI.MS_EXACT_MATCH
            load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            load_images.descend_subdirectories.value = LI.SUB_SOME
            load_images.subdirectory_filter.value = \
                       load_images.subdirectory_filter.get_value_string(
                           exclusions)
            load_images.location.custom_path = directory
            load_images.images[0].common_text.value = ".tif"
            load_images.images[0].channels[0].image_name.value = "my_image"
            load_images.check_images.value = False
            pipeline = P.Pipeline()
            pipeline.add_module(load_images)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            self.assertTrue(load_images.prepare_run(pipeline, image_set_list, None))
            self.assertEqual(image_set_list.count(), len(expected_filenames))
            m = measurements.Measurements()
            load_images.prepare_group(pipeline, image_set_list, {}, 
                                      np.arange(1, len(expected_filenames)+1))
            for i, (path, file_name) in enumerate(expected_filenames):
                if i > 0:
                    m.next_image_set()
                image_set = image_set_list.get_image_set(i)
                w = W.Workspace(pipeline, load_images, image_set,
                                cpo.ObjectSet(), m, image_set_list)
                load_images.run(w)
                image_provider = image_set.get_image_provider("my_image")
                self.assertEqual(image_provider.get_pathname(), directory)
                self.assertEqual(image_provider.get_filename(), os.path.join(path, file_name))
                f = m.get_current_image_measurement("FileName_my_image")
                self.assertEqual(f, file_name)
                p = m.get_current_image_measurement("PathName_my_image")
                self.assertEqual(os.path.join(directory, path), p)
        finally:
            for path, directories, file_names in os.walk(directory, False):
                for file_name in file_names:
                    p = os.path.join(path, file_name)
                    try:
                        os.remove(p)
                    except:
                        print "Failed to remove " + p
                        traceback.print_exc()
                try:
                    os.rmdir(path)
                except:
                    print "Failed to remove " + path
                    traceback.print_exc()
            
    def get_example_pipeline_data(self):
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
            Type the regular expression that finds metadata in the file name\x3A:^.*-(?P<WellRow>.+)-(?P<WellCol>\x5B0-9\x5D{2})
            Type the regular expression that finds metadata in the subfolder path\x3A:(?P<Year>\x5B0-9\x5D{4})-(?P<Month>\x5B0-9\x5D{2})-(?P<Day>\x5B0-9\x5D{2})
        '''
        return data
    
    def test_07_01_get_measurement_columns(self):
        data = self.get_example_pipeline_data()
        fd = StringIO(data)
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        module = pipeline.module(1)
        expected_cols = [('Image', 'FileName_DNA', 'varchar(128)'), 
                         ('Image', 'PathName_DNA', 'varchar(256)'),
                         ('Image', 'MD5Digest_DNA', 'varchar(32)'),
                         ('Image', 'Scaling_DNA', 'float'),
                         ('Image', 'Metadata_WellRow', 'varchar(128)'), 
                         ('Image', 'Metadata_WellCol', 'varchar(128)'), 
                         ('Image', 'FileName_Cytoplasm', 'varchar(128)'), 
                         ('Image', 'PathName_Cytoplasm', 'varchar(256)'), 
                         ('Image', 'MD5Digest_Cytoplasm', 'varchar(32)'),
                         ('Image', 'Scaling_Cytoplasm', 'float'),
                         ('Image', 'Metadata_Well', 'varchar(128)'),
                         ('Image', 'Height_DNA', 'integer'),
                         ('Image', 'Height_Cytoplasm', 'integer'),
                         ('Image', 'Width_DNA', 'integer'),
                         ('Image', 'Width_Cytoplasm', 'integer')]
        returned_cols = module.get_measurement_columns(pipeline)
        # check for duplicates
        assert len(returned_cols) == len(set(returned_cols))
        # check what was returned was expected
        for c in expected_cols: 
            assert c in returned_cols
        for c in returned_cols: 
            assert c in expected_cols
        #
        # Run with file and path metadata
        #
        module.images[0].metadata_choice.value = LI.M_BOTH
        expected_cols += [('Image', 'Metadata_Year', 'varchar(256)'),
                          ('Image', 'Metadata_Month', 'varchar(256)'),
                          ('Image', 'Metadata_Day', 'varchar(256)')]
        returned_cols = module.get_measurement_columns(pipeline)
        # check for duplicates
        assert len(returned_cols) == len(set(returned_cols))
        # check what was returned was expected
        for c in expected_cols: 
            assert c in returned_cols
        for c in returned_cols: 
            assert c in expected_cols
            
    def test_07_02_get_measurements(self):
        data = self.get_example_pipeline_data()
        fd = StringIO(data)
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        module = pipeline.module(1)
        categories = {'FileName' : ['DNA', 'Cytoplasm'], 
                      'PathName' : ['DNA', 'Cytoplasm'], 
                      'MD5Digest': ['DNA', 'Cytoplasm'], 
                      'Metadata' : ['WellRow', 'WellCol','Well']}
        for cat, expected in categories.items():
            assert set(expected) == set(module.get_measurements(pipeline, 
                                                    measurements.IMAGE, cat))
        module.images[0].metadata_choice.value = LI.M_BOTH
        categories['Metadata'] += ['Year','Month','Day']
        for cat, expected in categories.items():
            assert set(expected) == set(module.get_measurements(
                pipeline, measurements.IMAGE, cat))
        
    def test_07_03_get_categories(self):
        data = self.get_example_pipeline_data()
        fd = StringIO(data)
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        module = pipeline.module(1)
        results = module.get_categories(pipeline, measurements.IMAGE)
        expected = ['FileName', 'PathName', 'MD5Digest', 'Metadata', 'Scaling', 'Height', 'Width']
        assert set(results) == set(expected)
        
    def test_07_04_get_movie_measurements(self):
        # AVI movies should have time metadata
        module = LI.LoadImages()
        base_expected_cols = [('Image', 'FileName_DNA', 'varchar(128)'), 
                              ('Image', 'PathName_DNA', 'varchar(256)'),
                              ('Image', 'MD5Digest_DNA', 'varchar(32)'),
                              ('Image', 'Scaling_DNA', 'float'),
                              ('Image', 'Height_DNA', 'integer'),
                              ('Image', 'Width_DNA', 'integer'),
                              ('Image', 'Metadata_T', 'integer')]
        file_expected_cols = [
            ('Image', 'Metadata_WellRow', 'varchar(128)'), 
            ('Image', 'Metadata_WellCol', 'varchar(128)'),
            ('Image', 'Metadata_Well', 'varchar(128)')]
        path_expected_cols = [
            ('Image', 'Metadata_Year', 'varchar(256)'),
            ('Image', 'Metadata_Month', 'varchar(256)'),
            ('Image', 'Metadata_Day', 'varchar(256)')]
        for ft in (LI.FF_AVI_MOVIES, LI.FF_STK_MOVIES):
            module.file_types.value = ft
            module.images[0].channels[0].image_name.value = "DNA"
            module.images[0].file_metadata.value = "^.*-(?P<WellRow>.+)-(?P<WellCol>[0-9]{2})"
            module.images[0].path_metadata.value = "(?P<Year>[0-9]{4})-(?P<Month>[0-9]{2})-(?P<Day>[0-9]{2})"
            for metadata_choice, expected_cols in (
                (LI.M_NONE, base_expected_cols),
                (LI.M_FILE_NAME, base_expected_cols + file_expected_cols),
                (LI.M_PATH, base_expected_cols + path_expected_cols),
                (LI.M_BOTH, base_expected_cols + file_expected_cols + path_expected_cols)):
                module.images[0].metadata_choice.value = metadata_choice
                columns = module.get_measurement_columns(None)
                self.assertEqual(len(columns), len(set(columns)))
                self.assertEqual(len(columns), len(expected_cols))
                for column in columns:
                    self.assertTrue(column in expected_cols)
                categories = module.get_categories(None, measurements.IMAGE)
                self.assertEqual(len(categories), 7)
                category_dict = {}
                for column in expected_cols:
                    category, feature = column[1].split("_",1)
                    if not category_dict.has_key(category):
                        category_dict[category] = []
                    category_dict[category].append(feature)
                for category in category_dict.keys():
                    self.assertTrue(category in categories)
                    expected_features = category_dict[category]
                    features = module.get_measurements(None, measurements.IMAGE,
                                                       category)
                    self.assertEqual(len(features), len(expected_features))
                    self.assertEqual(len(features), len(set(features)))
                    self.assertTrue(all([feature in expected_features
                                         for feature in features]))
        
    def test_07_05_get_flex_measurements(self):
        # AVI movies should have time metadata
        module = LI.LoadImages()
        base_expected_cols = [('Image', 'FileName_DNA', 'varchar(128)'), 
                              ('Image', 'PathName_DNA', 'varchar(256)'),
                              ('Image', 'MD5Digest_DNA', 'varchar(32)'),
                              ('Image', 'Scaling_DNA', 'float'),
                              ('Image', 'Metadata_T', 'integer'),
                              ('Image', 'Metadata_Z', 'integer'),
                              ('Image', 'Height_DNA', 'integer'),
                              ('Image', 'Width_DNA', 'integer'),
                              ('Image', 'Metadata_Series', 'integer')]
        file_expected_cols = [
            ('Image', 'Metadata_WellRow', 'varchar(128)'), 
            ('Image', 'Metadata_WellCol', 'varchar(128)'),
            ('Image', 'Metadata_Well', 'varchar(128)')]
        path_expected_cols = [
            ('Image', 'Metadata_Year', 'varchar(256)'),
            ('Image', 'Metadata_Month', 'varchar(256)'),
            ('Image', 'Metadata_Day', 'varchar(256)')]
        module.file_types.value = LI.FF_OTHER_MOVIES
        module.images[0].channels[0].image_name.value = "DNA"
        module.images[0].file_metadata.value = "^.*-(?P<WellRow>.+)-(?P<WellCol>[0-9]{2})"
        module.images[0].path_metadata.value = "(?P<Year>[0-9]{4})-(?P<Month>[0-9]{2})-(?P<Day>[0-9]{2})"
        for metadata_choice, expected_cols in (
            (LI.M_NONE, base_expected_cols),
            (LI.M_FILE_NAME, base_expected_cols + file_expected_cols),
            (LI.M_PATH, base_expected_cols + path_expected_cols),
            (LI.M_BOTH, base_expected_cols + file_expected_cols + path_expected_cols)):
            module.images[0].metadata_choice.value = metadata_choice
            columns = module.get_measurement_columns(None)
            self.assertEqual(len(columns), len(set(columns)))
            self.assertEqual(len(columns), len(expected_cols))
            for column in columns:
                self.assertTrue(column in expected_cols)
            categories = module.get_categories(None, measurements.IMAGE)
            self.assertEqual(len(categories), 7)
            category_dict = {}
            for column in expected_cols:
                category, feature = column[1].split("_",1)
                if not category_dict.has_key(category):
                    category_dict[category] = []
                category_dict[category].append(feature)
            for category in category_dict.keys():
                self.assertTrue(category in categories)
                expected_features = category_dict[category]
                features = module.get_measurements(None, measurements.IMAGE,
                                                   category)
                self.assertEqual(len(features), len(expected_features))
                self.assertEqual(len(features), len(set(features)))
                self.assertTrue(all([feature in expected_features
                                     for feature in features]))
                
    def test_07_06_get_object_measurement_columns(self):
        module = LI.LoadImages()
        channel = module.images[0].channels[0]
        channel.image_object_choice.value = LI.IO_OBJECTS
        channel.object_name.value = OBJECTS_NAME
        columns = module.get_measurement_columns(None)
        for object_name, feature in (
            (measurements.IMAGE, LI.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME),
            (measurements.IMAGE, LI.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME),
            (measurements.IMAGE, LI.I.C_COUNT + "_" + OBJECTS_NAME),
            (OBJECTS_NAME, LI.I.M_LOCATION_CENTER_X),
            (OBJECTS_NAME, LI.I.M_LOCATION_CENTER_Y),
            (OBJECTS_NAME, LI.I.M_NUMBER_OBJECT_NUMBER)):
            self.assertTrue(any([True for column in columns
                                 if column[0] == object_name and
                                 column[1] == feature]))

    def test_07_07_get_object_categories(self):
        module = LI.LoadImages()
        channel = module.images[0].channels[0]
        channel.image_object_choice.value = LI.IO_OBJECTS
        channel.object_name.value = OBJECTS_NAME
        for object_name, expected_categories in (
            (measurements.IMAGE, 
             (LI.C_OBJECTS_FILE_NAME, LI.C_OBJECTS_PATH_NAME, LI.I.C_COUNT)),
            (OBJECTS_NAME, (LI.I.C_LOCATION, LI.I.C_NUMBER)),
            ("Foo", [])):
            categories = module.get_categories(None, object_name)
            for expected_category in expected_categories:
                self.assertTrue(expected_category in categories)
            for category in categories:
                self.assertTrue(category in expected_categories)
                
    def test_07_08_get_object_measurements(self):
        module = LI.LoadImages()
        channel = module.images[0].channels[0]
        channel.image_object_choice.value = LI.IO_OBJECTS
        channel.object_name.value = OBJECTS_NAME
        for object_name, expected in (
            ( measurements.IMAGE, (
                ( LI.C_OBJECTS_FILE_NAME, [ OBJECTS_NAME ]),
                ( LI.C_OBJECTS_PATH_NAME, [ OBJECTS_NAME ]),
                ( LI.I.C_COUNT, [ OBJECTS_NAME ]))),
            ( OBJECTS_NAME, (
                (LI.I.C_LOCATION, [ LI.I.FTR_CENTER_X, LI.I.FTR_CENTER_Y ]),
                (LI.I.C_NUMBER, [ LI.I.FTR_OBJECT_NUMBER ])))):
            for category, expected_features in expected:
                features = module.get_measurements(None, object_name, category)
                for feature in features:
                    self.assertTrue(feature in expected_features)
                for expected_feature in expected_features:
                    self.assertTrue(expected_feature in features)

    def test_08_01_get_groupings(self):
        '''Get groupings for the SBS image set'''
        sbs_path = os.path.join(T.example_images_directory(),'ExampleSBSImages')
        module = LI.LoadImages()
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = sbs_path
        module.group_by_metadata.value = True
        module.images[0].common_text.value = 'Channel1-'
        module.images[0].channels[0].image_name.value = 'MyImage'
        module.images[0].metadata_choice.value = LI.M_FILE_NAME
        module.images[0].file_metadata.value = '^Channel1-[0-9]{2}-(?P<ROW>[A-H])-(?P<COL>[0-9]{2})'
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
                match = re.search(module.images[0].file_metadata.value,
                                  provider.get_filename())
                self.assertTrue(match)
                self.assertEqual(row, match.group("ROW"))
    
    def test_09_01_load_avi(self):
        if LI.FF_AVI_MOVIES not in LI.FF:
            sys.stderr.write("WARNING: AVI movies not supported\n")
            return
        avi_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_AVI_MOVIES
        module.images[0].common_text.value = 'DrosophilaEmbryo_GFPHistone.avi'
        module.images[0].channels[0].image_name.value = 'MyImage'
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = avi_path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        self.assertEqual(image_set_list.count(), 65)
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
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 0)
        image_set = image_set_list.get_image_set(1)
        m.next_image_set()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.get_names())
        image = image_set.get_image('MyImage')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (264,542,3))
        self.assertTrue(np.any(img1!=img2))
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 1)
    
    def test_09_02_load_stk(self):
        path = '//iodine/imaging_analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM'
        if not os.path.isdir(path):
            path = '/imaging/analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM'
            if not os.path.isdir(path):
                path = '/Volumes/imaging_analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM'
                if not os.path.isdir(path):                
                    sys.stderr.write("WARNING: unknown path to stk file. Test not run.\n")
                    return
        module = LI.LoadImages()
        module.file_types.value = LI.FF_STK_MOVIES
        module.images[0].common_text.value = 'stk'
        module.images[0].channels[0].image_name.value = 'MyImage'
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = path
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
    
    def test_09_03_load_flex(self):
        flex_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_OTHER_MOVIES
        module.images[0].common_text.value = 'RLM1 SSN3 300308 008015000.flex'
        module.images[0].channels[0].image_name.value = 'Green'
        module.images[0].channels[0].channel_number.value = "2"
        module.add_channel(module.images[0])
        module.images[0].channels[1].image_name.value = 'Red'
        module.images[0].channels[1].channel_number.value = "1"
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = flex_path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        keys, groupings = module.get_groupings(image_set_list)
        self.assertTrue("FileName" in keys)
        self.assertTrue("Series" in keys)
        self.assertEqual(len(groupings), 4)
        m = measurements.Measurements()
        for grouping, image_numbers in groupings:
            module.prepare_group(pipeline, image_set_list, grouping, image_numbers)
            for image_number in image_numbers:
                image_set = image_set_list.get_image_set(image_number-1)
                workspace = W.Workspace(pipeline, module, image_set,
                                        cpo.ObjectSet(), m,
                                        image_set_list)
                module.run(workspace)
                for feature, expected in ((LI.M_SERIES, grouping[LI.M_SERIES]),
                                          (LI.M_Z, 0),
                                          (LI.M_T, 0)):
                    value = m.get_current_image_measurement(
                        measurements.C_METADATA + "_" + feature)
                    self.assertEqual(value, expected)
                red_image = image_set.get_image("Red")
                green_image = image_set.get_image("Green")
                self.assertEqual(tuple(red_image.pixel_data.shape),
                                 tuple(green_image.pixel_data.shape))
                m.next_image_set()
    
    def test_09_04_group_interleaved_avi_frames(self):
        #
        # Test interleaved grouping by movie frames
        #
        if LI.FF_AVI_MOVIES not in LI.FF:
            sys.stderr.write("WARNING: AVI movies not supported\n")
            return
        avi_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_AVI_MOVIES
        image = module.images[0]
        image.common_text.value = 'DrosophilaEmbryo_GFPHistone.avi'
        image.wants_movie_frame_grouping.value = True
        image.interleaving.value = LI.I_INTERLEAVED
        image.channels_per_group.value = 5
        channel = image.channels[0]
        channel.image_name.value = 'Channel01'
        channel.channel_number.value = "1"
        module.add_channel(image)
        channel = module.images[0].channels[1]
        channel.channel_number.value = "3"
        channel.image_name.value = 'Channel03'
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = avi_path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        self.assertEqual(image_set_list.count(), 13)
        module.prepare_group(pipeline, image_set_list, (), np.arange(1,16))
        image_set = image_set_list.get_image_set(0)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.get_names())
        image = image_set.get_image('Channel01')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (264,542,3))
        self.assertAlmostEqual(np.mean(img1), .07897, 3)
        self.assertTrue('Channel03' in image_set.get_names())
        provider = image_set.get_image_provider("Channel03")
        self.assertTrue(isinstance(provider, LI.LoadImagesMovieFrameProvider))
        self.assertEqual(provider.get_frame(), 2)
        image = image_set.get_image('Channel03')
        img3 = image.pixel_data
        self.assertEqual(tuple(img3.shape), (264,542,3))
        self.assertAlmostEqual(np.mean(img3), .07781, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 0)
        image_set = image_set_list.get_image_set(1)
        m.next_image_set()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.get_names())
        image = image_set.get_image('Channel01')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (264,542,3))
        self.assertAlmostEqual(np.mean(img2), .07860, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 1)
        provider = image_set.get_image_provider("Channel03")
        self.assertTrue(isinstance(provider, LI.LoadImagesMovieFrameProvider))
        self.assertEqual(provider.get_frame(), 7)
        
    def test_09_05_group_separated_avi_frames(self):
        #
        # Test separated grouping by movie frames
        #
        if LI.FF_AVI_MOVIES not in LI.FF:
            sys.stderr.write("WARNING: AVI movies not supported\n")
            return
        avi_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_AVI_MOVIES
        image = module.images[0]
        image.common_text.value = 'DrosophilaEmbryo_GFPHistone.avi'
        image.wants_movie_frame_grouping.value = True
        image.interleaving.value = LI.I_SEPARATED
        image.channels_per_group.value = 5
        channel = image.channels[0]
        channel.image_name.value = 'Channel01'
        channel.channel_number.value = "1"
        module.add_channel(image)
        channel = module.images[0].channels[1]
        channel.channel_number.value = "3"
        channel.image_name.value = 'Channel03'
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = avi_path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        self.assertEqual(image_set_list.count(), 13)
        module.prepare_group(pipeline, image_set_list, (), np.arange(1,16))
        image_set = image_set_list.get_image_set(0)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.get_names())
        image = image_set.get_image('Channel01')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (264,542,3))
        self.assertAlmostEqual(np.mean(img1), .07897, 3)
        self.assertTrue('Channel03' in image_set.get_names())
        provider = image_set.get_image_provider("Channel03")
        self.assertTrue(isinstance(provider, LI.LoadImagesMovieFrameProvider))
        self.assertEqual(provider.get_frame(), 26)
        image = image_set.get_image('Channel03')
        img3 = image.pixel_data
        self.assertEqual(tuple(img3.shape), (264,542,3))
        self.assertAlmostEqual(np.mean(img3), .073312, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 0)
        image_set = image_set_list.get_image_set(1)
        m.next_image_set()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.get_names())
        provider = image_set.get_image_provider("Channel01")
        self.assertTrue(isinstance(provider, LI.LoadImagesMovieFrameProvider))
        self.assertEqual(provider.get_frame(), 1)
        image = image_set.get_image('Channel01')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (264,542,3))
        self.assertAlmostEqual(np.mean(img2), .079923, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 1)
        provider = image_set.get_image_provider("Channel03")
        self.assertTrue(isinstance(provider, LI.LoadImagesMovieFrameProvider))
        self.assertEqual(provider.get_frame(), 27)
        
    def test_09_06_load_flex_interleaved(self):
        # needs better test case file
        flex_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_OTHER_MOVIES
        module.images[0].common_text.value = 'RLM1 SSN3 300308 008015000.flex'
        module.images[0].channels[0].image_name.value = 'Green'
        module.images[0].channels[0].channel_number.value = "2"
        module.add_channel(module.images[0])
        module.images[0].channels[1].image_name.value = 'Red'
        module.images[0].channels[1].channel_number.value = "1"
        module.images[0].wants_movie_frame_grouping.value = True
        module.images[0].channels_per_group.value = 2
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = flex_path
        module.images[0].interleaving.value = LI.I_INTERLEAVED
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        keys, groupings = module.get_groupings(image_set_list)
        self.assertTrue("FileName" in keys)
        self.assertTrue("Series" in keys)
        self.assertEqual(len(groupings), 4)
        m = measurements.Measurements()
        for group_number, (grouping, image_numbers) in enumerate(groupings):
            module.prepare_group(pipeline, image_set_list, grouping, image_numbers)
            for group_index, image_number in enumerate(image_numbers):
                image_set = image_set_list.get_image_set(image_number-1)
                m.add_image_measurement(cpp.GROUP_INDEX, group_index)
                m.add_image_measurement(cpp.GROUP_NUMBER, group_number)
                workspace = W.Workspace(pipeline, module, image_set,
                                        cpo.ObjectSet(), m,
                                        image_set_list)
                module.run(workspace)
                for feature, expected in ((LI.M_SERIES, grouping[LI.M_SERIES]),
                                          (LI.M_Z, 0),
                                          (LI.M_T, 0)):
                    value = m.get_current_image_measurement(
                        measurements.C_METADATA + "_" + feature)
                    self.assertEqual(value, expected)
                
                red_image_provider = image_set.get_image_provider("Red")
                green_image_provider = image_set.get_image_provider("Green")
                self.assertEqual(red_image_provider.get_c(), 0)
                self.assertEqual(green_image_provider.get_c(), 1)
                self.assertEqual(red_image_provider.get_t(), 0)
                self.assertEqual(green_image_provider.get_t(), 0)
                red_image = image_set.get_image("Red")
                green_image = image_set.get_image("Green")
                self.assertEqual(tuple(red_image.pixel_data.shape),
                                 tuple(green_image.pixel_data.shape))
                m.next_image_set()
                
    def test_09_07_load_flex_separated(self):
        # Needs better test case file
        flex_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_OTHER_MOVIES
        module.images[0].common_text.value = 'RLM1 SSN3 300308 008015000.flex'
        module.images[0].channels[0].image_name.value = 'Green'
        module.images[0].channels[0].channel_number.value = "2"
        module.add_channel(module.images[0])
        module.images[0].channels[1].image_name.value = 'Red'
        module.images[0].channels[1].channel_number.value = "1"
        module.images[0].wants_movie_frame_grouping.value = True
        module.images[0].channels_per_group.value = 2
        module.images[0].interleaving.value = LI.I_SEPARATED
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = flex_path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        keys, groupings = module.get_groupings(image_set_list)
        self.assertTrue("FileName" in keys)
        self.assertTrue("Series" in keys)
        self.assertEqual(len(groupings), 4)
        m = measurements.Measurements()
        for group_number, (grouping, image_numbers) in enumerate(groupings):
            module.prepare_group(pipeline, image_set_list, grouping, image_numbers)
            
            for group_index, image_number in enumerate(image_numbers):
                image_set = image_set_list.get_image_set(image_number-1)
                workspace = W.Workspace(pipeline, module, image_set,
                                        cpo.ObjectSet(), m,
                                        image_set_list)
                m.add_image_measurement(cpp.GROUP_INDEX, group_index)
                m.add_image_measurement(cpp.GROUP_NUMBER, group_number)
                module.run(workspace)
                for feature, expected in ((LI.M_SERIES, grouping[LI.M_SERIES]),
                                          (LI.M_Z, 0),
                                          (LI.M_T, 0)):
                    value = m.get_current_image_measurement(
                        measurements.C_METADATA + "_" + feature)
                    self.assertEqual(value, expected)
                
                red_image_provider = image_set.get_image_provider("Red")
                green_image_provider = image_set.get_image_provider("Green")
                self.assertEqual(red_image_provider.get_c(), 0)
                self.assertEqual(green_image_provider.get_c(), 1)
                self.assertEqual(red_image_provider.get_t(), 0)
                self.assertEqual(green_image_provider.get_t(), 0)
                red_image = image_set.get_image("Red")
                green_image = image_set.get_image("Green")
                self.assertEqual(tuple(red_image.pixel_data.shape),
                                 tuple(green_image.pixel_data.shape))
                m.next_image_set()
                
    def test_10_01_load_unscaled(self):
        '''Load a image with and without rescaling'''
        path = os.path.join(example_images_directory(), 
                            "ExampleSpecklesImages")
        module = LI.LoadImages()
        module.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        module.images[0].common_text.value = '1-162hrh2ax2'
        module.images[0].channels[0].image_name.value = 'MyImage'
        module.images[0].channels[0].rescale.value = False
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        module.prepare_group(pipeline, image_set_list, (), [1])
        image_set = image_set_list.get_image_set(0)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        scales = m.get_all_measurements(measurements.IMAGE,
                                        LI.C_SCALING + "_MyImage")
        self.assertEqual(len(scales), 1)
        self.assertEqual(scales[0], 4095)
        image = image_set.get_image("MyImage")
        self.assertTrue(np.all(image.pixel_data <= 1.0/16.0))
        pixel_data = image.pixel_data
        module.images[0].channels[0].rescale.value = True
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        module.prepare_group(pipeline, image_set_list, (), [1])
        image_set = image_set_list.get_image_set(0)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        image = image_set.get_image("MyImage")
        np.testing.assert_almost_equal(pixel_data * 65535.0 / 4095.0 , 
                                       image.pixel_data)
        
    def test_11_01_load_many(self):
        '''Load an image many times to ensure that memory is freed each time'''
        path = os.path.join(example_images_directory(), "ExampleSBSImages")
        for i in range(3):
            module = LI.LoadImages()
            module.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            module.images[0].common_text.value = 'Channel1-'
            module.images[0].channels[0].image_name.value = 'MyImage'
            module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            module.location.custom_path = path
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
    
    def make_objects_workspace(self, image, mode = "L", filename="myfile.tif"):
        directory = tempfile.mkdtemp()
        self.directory = directory
        pilimage = PIL.Image.fromarray(image.astype(np.uint8), mode)
        pilimage.save(os.path.join(directory, filename))
        module = LI.LoadImages()
        module.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        module.images[0].common_text.value = filename
        module.images[0].channels[0].image_object_choice.value = LI.IO_OBJECTS
        module.images[0].channels[0].object_name.value = OBJECTS_NAME
        module.images[0].channels[0].wants_outlines.value = True
        module.images[0].channels[0].outlines_name.value = OUTLINES_NAME
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = directory
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        module.prepare_run(pipeline, image_set_list, None)
        module.prepare_group(pipeline, image_set_list, (), [0])
        workspace = W.Workspace(pipeline, module, image_set_list.get_image_set(0), 
                                cpo.ObjectSet(), measurements.Measurements(),
                                image_set_list)
        return workspace, module
    
    def test_12_01_load_empty_objects(self):
        workspace, module = self.make_objects_workspace(np.zeros((20,30), int))
        module.run(workspace)
        assert isinstance(module, LI.LoadImages)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == 0))
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name, measurement in (
            (measurements.IMAGE, LI.I.FF_COUNT % OBJECTS_NAME),
            (OBJECTS_NAME, LI.I.M_LOCATION_CENTER_X),
            (OBJECTS_NAME, LI.I.M_LOCATION_CENTER_Y),
            (OBJECTS_NAME, LI.I.M_NUMBER_OBJECT_NUMBER)):
            self.assertTrue(any(
                [True for column in columns
                 if column[0] == object_name and column[1] == measurement]))
        m = workspace.measurements
        assert isinstance(m, measurements.Measurements)
        self.assertEqual(m.get_current_image_measurement(LI.I.FF_COUNT %OBJECTS_NAME), 0)

    def test_12_02_load_indexed_objects(self):
        r = np.random.RandomState()
        r.seed(1202)
        image = r.randint(0, 10, size=(20,30))
        workspace, module = self.make_objects_workspace(image)
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))
        m = workspace.measurements
        assert isinstance(m, measurements.Measurements)
        self.assertEqual(m.get_current_image_measurement(LI.I.FF_COUNT %OBJECTS_NAME), 9)
        i,j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        c = np.bincount(image.ravel())[1:].astype(float)
        x = np.bincount(image.ravel(), j.ravel())[1:].astype(float) / c
        y = np.bincount(image.ravel(), i.ravel())[1:].astype(float) / c
        v = m.get_current_measurement(OBJECTS_NAME, LI.I.M_NUMBER_OBJECT_NUMBER)
        self.assertTrue(np.all(v == np.arange(1,10)))
        v = m.get_current_measurement(OBJECTS_NAME, LI.I.M_LOCATION_CENTER_X)
        self.assertTrue(np.all(v == x))
        v = m.get_current_measurement(OBJECTS_NAME, LI.I.M_LOCATION_CENTER_Y)
        self.assertTrue(np.all(v == y))
        
    def test_12_03_load_sparse_objects(self):
        r = np.random.RandomState()
        r.seed(1203)
        image = r.randint(0, 10, size=(20,30))
        workspace, module = self.make_objects_workspace(image * 10)
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))
        
    def test_12_04_load_color_objects(self):
        r = np.random.RandomState()
        r.seed(1203)
        image = r.randint(0, 10, size=(20,30))
        colors = np.array([[0,0,0], [1, 4, 2], [1, 5, 0],
                           [2, 0, 0], [3, 0, 0], [4, 0, 0],
                           [5, 0, 0], [6, 0, 0], [7, 0, 0],
                           [8, 0, 0], [9, 0, 0]])
        cimage = colors[image]
        workspace, module = self.make_objects_workspace(cimage,mode="RGB", 
                                                        filename="myimage.png")
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))
        
    def test_12_05_object_outlines(self):
        image = np.zeros((30,40), int)
        image[10:15, 20:30] = 1
        workspace, module = self.make_objects_workspace(image)
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))
        expected_outlines = image != 0
        expected_outlines[11:14,21:29] = False
        image_set = workspace.get_image_set()
        outlines = image_set.get_image(OUTLINES_NAME)
        np.testing.assert_equal(outlines.pixel_data, expected_outlines)
        
    def test_13_01_batch_images(self):
        module = LI.LoadImages()
        module.match_method.value = LI.MS_REGEXP
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        orig_path = os.path.join(T.example_images_directory(),"ExampleSBSImages")
        module.location.custom_path = orig_path
        target_path = orig_path.replace("ExampleSBSImages", "ExampleTrackObjects")
            
        file_regexp = "^Channel1-[0-9]{2}-[A-P]-[0-9]{2}.tif$"
        module.images[0].common_text.value = file_regexp
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(module)
        module.prepare_run(pipeline, image_set_list, None)
        def fn_alter_path(pathname, **varargs):
            is_path = (pathname == orig_path)
            is_file = re.match(file_regexp, pathname) is not None
            self.assertTrue(is_path or is_file)
            if is_path:
                return target_path
            else:
                return pathname
        module.prepare_to_create_batch(pipeline, image_set_list, fn_alter_path)
        key_names, group_list = pipeline.get_groupings(image_set_list)
        self.assertEqual(len(group_list), 1)
        group_keys, image_numbers = group_list[0]
        self.assertEqual(len(image_numbers), 96)
        module.prepare_group(pipeline, image_set_list, group_keys, image_numbers)
        for image_number in image_numbers:
            image_set = image_set_list.get_image_set(image_number-1)
            image_provider = image_set.get_image_provider(IMAGE_NAME)
            self.assertTrue(isinstance(image_provider, LI.LoadImagesImageProvider))
            self.assertEqual(image_provider.get_pathname(), target_path)
    
    def test_13_02_batch_movies(self):
        module = LI.LoadImages()
        module.match_method.value = LI.MS_EXACT_MATCH
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.file_types.value = LI.FF_AVI_MOVIES
        orig_path = T.testimages_directory()
        module.location.custom_path = orig_path
        target_path = os.path.join(orig_path, "Images")
            
        file_name = "DrosophilaEmbryo_GFPHistone.avi"
        module.images[0].common_text.value = file_name
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(module)
        module.prepare_run(pipeline, image_set_list, None)
        def fn_alter_path(pathname, **varargs):
            is_fullpath = (os.path.join(orig_path, file_name) == pathname)
            is_path = (orig_path == pathname)
            self.assertTrue(is_fullpath or is_path)
            if is_fullpath:
                return os.path.join(target_path, file_name)
            else:
                return target_path
        module.prepare_to_create_batch(pipeline, image_set_list, fn_alter_path)
        key_names, group_list = pipeline.get_groupings(image_set_list)
        self.assertEqual(len(group_list), 1)
        group_keys, image_numbers = group_list[0]
        self.assertEqual(len(image_numbers), 65)
        module.prepare_group(pipeline, image_set_list, group_keys, image_numbers)
        for image_number in image_numbers:
            image_set = image_set_list.get_image_set(image_number-1)
            image_provider = image_set.get_image_provider(IMAGE_NAME)
            self.assertTrue(isinstance(image_provider, LI.LoadImagesMovieFrameProvider))
            self.assertEqual(image_provider.get_pathname(), target_path)
            self.assertEqual(image_provider.get_t(), image_number-1)
    
    def test_13_03_batch_flex(self):
        module = LI.LoadImages()
        module.match_method.value = LI.MS_EXACT_MATCH
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.file_types.value = LI.FF_OTHER_MOVIES
        orig_path = T.testimages_directory()
        module.location.custom_path = orig_path
        target_path = os.path.join(orig_path, "Images")
            
        file_name = "RLM1 SSN3 300308 008015000.flex"
        module.images[0].common_text.value = file_name
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(module)
        module.prepare_run(pipeline, image_set_list, None)
        def fn_alter_path(pathname, **varargs):
            is_fullpath = (os.path.join(orig_path, file_name) == pathname)
            is_path = (orig_path == pathname)
            self.assertTrue(is_fullpath or is_path)
            if is_fullpath:
                return os.path.join(target_path, file_name)
            else:
                return target_path
        module.prepare_to_create_batch(pipeline, image_set_list, fn_alter_path)
        key_names, group_list = pipeline.get_groupings(image_set_list)
        self.assertEqual(len(group_list), 4)
        for i, (group_keys, image_numbers) in enumerate(group_list):
            self.assertEqual(len(image_numbers), 1)
            module.prepare_group(pipeline, image_set_list, group_keys, 
                                 image_numbers)
            for image_number in image_numbers:
                image_set = image_set_list.get_image_set(image_number-1)
                image_provider = image_set.get_image_provider(IMAGE_NAME)
                self.assertTrue(isinstance(image_provider, LI.LoadImagesFlexFrameProvider))
                self.assertEqual(image_provider.get_pathname(), target_path)
                self.assertEqual(image_provider.get_series(), i)
                self.assertEqual(image_provider.get_t(), 0)
                self.assertEqual(image_provider.get_z(), 0)
                self.assertEqual(image_provider.get_c(), 0)
    
    def test_14_01_load_unicode(self):
        '''Load an image from a unicode - encoded location'''
        self.directory = tempfile.mkdtemp()
        directory = os.path.join(self.directory, u"\u2211a")
        os.mkdir(directory)
        filename = u"\u03b1\u00b2.jpg"
        path = os.path.join(directory, filename)
        data = base64.b64decode(T.jpg_8_1)
        fd = open(path,'wb')
        fd.write(data)
        fd.close()
        module = LI.LoadImages()
        module.module_num = 1
        module.match_method.value = LI.MS_EXACT_MATCH
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = directory
        module.images[0].common_text.value = ".jpg"
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        image_set_list = I.ImageSetList()
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        m = measurements.Measurements()
        self.assertTrue(module.prepare_run(pipeline, image_set_list, None))
        self.assertEqual(image_set_list.count(), 1)
        key_names, group_list = pipeline.get_groupings(image_set_list)
        self.assertEqual(len(group_list), 1)
        group_keys, image_numbers = group_list[0]
        self.assertEqual(len(image_numbers), 1)
        module.prepare_group(pipeline, image_set_list, group_keys, image_numbers)
        image_set = image_set_list.get_image_set(image_numbers[0]-1)
        image_provider = image_set.get_image_provider(IMAGE_NAME)
        self.assertEqual(image_provider.get_filename(), filename)
        workspace = W.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                m, image_set_list)
        module.run(workspace)
        pixel_data = image_set.get_image(IMAGE_NAME).pixel_data
        self.assertEqual(tuple(pixel_data.shape[:2]), tuple(T.raw_8_1_shape))
        file_feature = '_'.join((LI.C_FILE_NAME, IMAGE_NAME))
        file_measurement = m.get_current_image_measurement(file_feature)
        self.assertEqual(file_measurement, filename)
        path_feature = '_'.join((LI.C_PATH_NAME, IMAGE_NAME))
        path_measurement = m.get_current_image_measurement(path_feature)
        self.assertEqual(os.path.split(directory)[1],
                         os.path.split(path_measurement)[1])
            
if __name__=="main":
    unittest.main()
