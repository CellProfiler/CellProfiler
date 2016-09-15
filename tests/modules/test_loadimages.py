"""Test the LoadImages module
"""

import base64
import glob
import hashlib
import os
import re
import sys
import tempfile
import time
import traceback
import unittest
import urllib
import zlib
from StringIO import StringIO

import cellprofiler.image as I
import cellprofiler.measurement as measurements
import cellprofiler.modules.loadimages as LI
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.pipeline as P
import cellprofiler.preferences as cpprefs
import cellprofiler.workspace as W
import numpy as np
import tests.modules as T
from bioformats.formatreader import clear_image_reader_cache
from bioformats.formatwriter import write_image
from bioformats.omexml import PT_UINT8
from cellprofiler.modules.namesandtypes import M_IMAGE_SET
from tests.modules import \
    example_images_directory, maybe_download_sbs, maybe_download_tesst_image, maybe_download_fly, \
    cp_logo_url_folder, cp_logo_url_filename, cp_logo_url_shape

IMAGE_NAME = "image"
ALT_IMAGE_NAME = "altimage"
OBJECTS_NAME = "objects"
OUTLINES_NAME = "outlines"


class ConvtesterMixin:
    '''Mixin class that supplies a generic legacy conversion tester method

    '''

    def convtester(self, pipeline_text, directory, fn_filter=(lambda x: True)):
        '''Test whether a converted pipeline yields the same output

        pipeline_text - the pipeline as a text file

        directory - the default input directory

        fn_filter - a function that returns True if a file should be included
                    in the workspace file list.
        '''
        cpprefs.set_default_image_directory(directory)
        pipeline = cpp.Pipeline()
        pipeline.load(StringIO(pipeline_text))

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        m = [m for m in pipeline.modules() if isinstance(m, LI.LoadImages)][0]
        m1 = measurements.Measurements()
        w1 = W.Workspace(pipeline, m, m1, None, m1, None)
        pipeline.prepare_run(w1)

        m2 = measurements.Measurements()
        w2 = W.Workspace(pipeline, m, m2, None, m2, None)
        urls = [LI.pathname2url(os.path.join(directory, filename))
                for filename in os.listdir(directory)
                if fn_filter(filename)]
        w2.file_list.add_files_to_filelist(urls)
        pipeline.add_urls(urls, False)
        pipeline.convert_legacy_input_modules()
        pipeline.prepare_run(w2)

        ff1 = m1.get_feature_names(measurements.IMAGE)
        ffexpected = [f.replace("IMAGE_FOR_", "") for f in ff1
                      if not f.startswith(measurements.C_METADATA)]
        ff2 = [x for x in m2.get_feature_names(measurements.IMAGE)
               if not any([x.startswith(y) for y in (
                measurements.C_FRAME, measurements.C_SERIES,
                measurements.C_OBJECTS_FRAME,
                measurements.C_OBJECTS_SERIES,
                measurements.C_CHANNEL,
                measurements.C_OBJECTS_CHANNEL,
                measurements.C_METADATA,
                M_IMAGE_SET
            )])]
        self.assertItemsEqual(ffexpected, ff2)
        for feature in ff1:
            if feature.startswith(measurements.C_METADATA):
                self.assertTrue(m2.has_feature(measurements.IMAGE, feature))
        ff1a = filter((lambda x: not x.startswith(measurements.C_METADATA)),
                      ff1)
        self.assertEqual(m1.image_set_count, m2.image_set_count)
        image_numbers = m1.get_image_numbers()
        #
        # Order images by URL
        #
        m_url1 = sorted(ff1, key=lambda f: f.replace("IMAGE_FOR_", ""))
        m_url2 = sorted(ff2)
        order1, order2 = [np.lexsort(
                [mm.get_measurement(measurements.IMAGE, f, image_numbers)
                 for f in m_url]) for mm, m_url in ((m1, m_url1), (m2, m_url2))]
        image_numbers1 = image_numbers[order1]
        image_numbers2 = image_numbers[order2]
        for f1, f2 in zip(ff1a, ff2):
            if f1 in (measurements.GROUP_INDEX, measurements.GROUP_NUMBER,
                      measurements.IMAGE_NUMBER):
                continue
            v1 = m1.get_measurement(measurements.IMAGE, f1,
                                    image_set_number=image_numbers1)
            v2 = m2.get_measurement(measurements.IMAGE, f2,
                                    image_set_number=image_numbers2)
            if (f1.startswith(measurements.C_PATH_NAME) or
                    f1.startswith(measurements.C_OBJECTS_PATH_NAME)):
                for p1, p2 in zip(v1, v2):
                    self.assertEqual(os.path.normcase(p1),
                                     os.path.normcase(p2))
            elif (f1.startswith(measurements.C_URL) or
                      f1.startswith(measurements.C_OBJECTS_URL)):
                for p1, p2 in zip(v1, v2):
                    self.assertEqual(
                            os.path.normcase(LI.url2pathname(p1.encode("utf-8"))),
                            os.path.normcase(LI.url2pathname(p2.encode("utf-8"))))
            else:
                np.testing.assert_array_equal(v1, v2)


class testLoadImages(unittest.TestCase, ConvtesterMixin):
    @classmethod
    def setUpClass(cls):
        maybe_download_sbs()

    def setUp(self):
        self.directory = None

    def tearDown(self):
        clear_image_reader_cache()
        if self.directory is not None:
            try:
                for path in (os.path.sep.join((self.directory, "*", "*")),
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

            # def test_00_00init(self):
            #     x=LI.LoadImages()

            # def test_00_01version(self):
            #     self.assertEqual(LI.LoadImages().variable_revision_number, 11,
            #                      "LoadImages' version number has changed")

            # def test_01_01load_image_text_match(self):
            #     l=LI.LoadImages()
            #     l.match_method.value = LI.MS_EXACT_MATCH
            #     l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            #     l.location.custom_path =\
            #         os.path.join(T.example_images_directory(),"ExampleSBSImages")
            #     l.images[0].common_text.value = "1-01-A-01.tif"
            #     l.images[0].channels[0].image_name.value = "my_image"
            #     l.module_num = 1
            #     image_set_list = I.ImageSetList()
            #     pipeline = P.Pipeline()
            #     pipeline.add_listener(self.error_callback)
            #     pipeline.add_module(l)
            #     m = measurements.Measurements()
            #     workspace = W.Workspace(pipeline, l, None, None, m, image_set_list)
            #     l.prepare_run(workspace)
            #     image_numbers = m.get_image_numbers()
            #     self.assertEqual(len(image_numbers), 1,
            #                      "Expected one image set in the list")
            #     l.prepare_group(workspace, (), [1])
            #     image_set = image_set_list.get_image_set(0)
            #     l.run(W.Workspace(pipeline, l, image_set, cpo.ObjectSet(),
            #                       m, image_set_list))
            #     self.assertEqual(len(image_set.names),1)
            #     self.assertEqual(image_set.names[0],"my_image")
            #     self.assertTrue(image_set.get_image("my_image"))
            #
            # def test_01_02load_image_text_match_many(self):
            #     l=LI.LoadImages()
            #     l.match_method.value = LI.MS_EXACT_MATCH
            #     l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            #     l.location.custom_path =\
            #         os.path.join(T.example_images_directory(),"ExampleSBSImages")
            #     for i in range(0,4):
            #         ii = i+1
            #         if i:
            #             l.add_imagecb()
            #         l.images[i].common_text.value = "1-0%(ii)d-A-0%(ii)d.tif" % locals()
            #         l.images[i].channels[0].image_name.value = "my_image%(i)d" % locals()
            #     l.module_num = 1
            #     image_set_list = I.ImageSetList()
            #     pipeline = P.Pipeline()
            #     pipeline.add_module(l)
            #     pipeline.add_listener(self.error_callback)
            #     m = measurements.Measurements()
            #     workspace = W.Workspace(pipeline, l, None, None, m, image_set_list)
            #     l.prepare_run(workspace)
            #     image_numbers = m.get_image_numbers()
            #     self.assertEqual(len(image_numbers), 1,
            #                      "Expected one image set, there were %d"
            #                      % (image_set_list.count()))
            #     l.prepare_group(workspace, (), image_numbers)
            #     image_set = image_set_list.get_image_set(0)
            #     l.run(W.Workspace(pipeline, l, image_set, cpo.ObjectSet(),
            #                       m, image_set_list))
            #     self.assertEqual(len(image_set.names),4)
            #     for i in range(0,4):
            #         self.assertTrue("my_image%d"%(i) in image_set.names)
            #         self.assertTrue(image_set.get_image("my_image%d"%(i)))
            #
            # def test_02_01_load_image_regex_match(self):
            #     l=LI.LoadImages()
            #     l.match_method.value = LI.MS_REGEXP
            #     l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            #     l.location.custom_path =\
            #         os.path.join(T.example_images_directory(),"ExampleSBSImages")
            #     l.images[0].common_text.value = "Channel1-[0-1][0-9]-A-01"
            #     l.images[0].channels[0].image_name.value = "my_image"
            #     l.module_num = 1
            #     image_set_list = I.ImageSetList()
            #     pipeline = P.Pipeline()
            #     m = measurements.Measurements()
            #     pipeline.add_module(l)
            #     workspace = W.Workspace(pipeline, l, None, None, m, image_set_list)
            #     l.prepare_run(workspace)
            #     image_numbers = m.get_image_numbers()
            #     self.assertEqual(image_numbers, 1, "Expected one image set in the list")
            #     l.prepare_group(workspace, (), image_numbers)
            #     image_set = image_set_list.get_image_set(0)
            #     l.run(W.Workspace(pipeline, l, image_set, cpo.ObjectSet(), m,
            #                       image_set_list))
            #     self.assertEqual(len(image_set.names),1)
            #     self.assertEqual(image_set.names[0],"my_image")
            #     self.assertTrue(image_set.get_image("my_image"))

            #     def test_02_02_load_image_by_order(self):
            #         #
            #         # Make a list of 12 files
            #         #
            #         directory = tempfile.mkdtemp()
            #         self.directory = directory
            #         data = base64.b64decode(T.tif_8_1)
            #         tiff_fmt = "image%02d.tif"
            #         for i in range(12):
            #             path = os.path.join(directory, tiff_fmt % i)
            #             fd = open(path, "wb")
            #             fd.write(data)
            #             fd.close()
            #         #
            #         # Code for permutations taken from
            #         # http://docs.python.org/library/itertools.html#itertools.permutations
            #         # which has the Python copyright
            #         #
            #         def permutations(iterable, r=None):
            #             # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
            #             # permutations(range(3)) --> 012 021 102 120 201 210
            #             pool = tuple(iterable)
            #             n = len(pool)
            #             r = n if r is None else r
            #             if r > n:
            #                 return
            #             indices = range(n)
            #             cycles = range(n, n-r, -1)
            #             yield tuple(pool[i] for i in indices[:r])
            #             while n:
            #                 for i in reversed(range(r)):
            #                     cycles[i] -= 1
            #                     if cycles[i] == 0:
            #                         indices[i:] = indices[i+1:] + indices[i:i+1]
            #                         cycles[i] = n - i
            #                     else:
            #                         j = cycles[i]
            #                         indices[i], indices[-j] = indices[-j], indices[i]
            #                         yield tuple(pool[i] for i in indices[:r])
            #                         break
            #                 else:
            #                     return
            #
            #         #
            #         # Run through group sizes = 2-4, # of images 2-4
            #         #
            #         for group_size in range(2, 5):
            #             for image_count in range(2, group_size+1):
            #                 #
            #                 # For each possible permutation of image numbers
            #                 #
            #                 for indexes in permutations(range(group_size), image_count):
            #                     l = LI.LoadImages()
            #                     l.match_method.value = LI.MS_ORDER
            #                     l.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            #                     l.location.custom_path = directory
            #                     l.order_group_size.value = group_size
            #                     l.images[0].channels[0].image_name.value = "image%d" % 1
            #                     l.images[0].order_position.value = indexes[0] + 1
            #                     for i, index in enumerate(indexes[1:]):
            #                         l.add_imagecb()
            #                         l.images[i+1].order_position.value = index + 1
            #                         l.images[i+1].channels[0].image_name.value = "image%d" % (i+2)
            #                     l.module_num = 1
            #                     image_set_list = I.ImageSetList()
            #                     pipeline = P.Pipeline()
            #                     pipeline.add_module(l)
            #                     m = measurements.Measurements()
            #                     workspace = W.Workspace(pipeline, l, None, None, m,
            #                                             image_set_list)
            #                     l.prepare_run(workspace)
            #                     image_numbers = m.get_image_numbers()
            #                     nsets = 12 / group_size
            #                     self.assertEqual(len(image_numbers), nsets)
            #                     l.prepare_group(workspace, (), list(range(1, nsets+1)))
            #                     for i in range(0, nsets):
            #                         if i > 0:
            #                             m.next_image_set(i + 1)
            #                         image_set = image_set_list.get_image_set(i)
            #                         workspace = W.Workspace(pipeline, l, image_set,
            #                                                 cpo.ObjectSet(), m,
            #                                                 image_set_list)
            #                         l.run(workspace)
            #                         for j in range(image_count):
            #                             feature = LI.C_FILE_NAME + ("_image%d" % (j+1))
            #                             idx = i * group_size + indexes[j]
            #                             expected = tiff_fmt % idx
            #                             value = m.get_current_image_measurement(feature)
            #                             self.assertEqual(expected, value)
            #
            #     def test_03_00_load_matlab_v1(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:1234
            # FromMatlab:True
            #
            # LoadImages:[module_num:1|svn_version:\'8913\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
            #     How do you want to load these files?:Text-Exact match
            #     Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:Channel1-
            #     What do you want to call these images within CellProfiler?:MyImages
            #     Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:Channel2-
            #     What do you want to call these images within CellProfiler?:OtherImages
            #     Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:/
            #     What do you want to call these images within CellProfiler?:/
            #     Type the text that one type of image has in common (for TEXT options), or their position in each group (for ORDER option)\x3A:/
            #     What do you want to call these images within CellProfiler?:/
            #     If using ORDER, how many images are there in each group (i.e. each field of view)?:5
            #     Are you loading image or movie files?:Image
            #     If you are loading a movie, what is the extension?:stk
            #     Analyze all subfolders within the selected folder?:Yes
            #     Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder.:./Images
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 1)
            #         module = pipeline.modules()[-1]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(len(module.images), 2)
            #         self.assertEqual(module.images[0].channels[0].image_name, "MyImages")
            #         self.assertEqual(module.images[1].channels[0].image_name, "OtherImages")
            #         self.assertEqual(module.order_group_size, 5)
            #         self.assertTrue(module.analyze_sub_dirs())
            #         self.assertEqual(module.location.dir_choice,
            #                          LI.DEFAULT_INPUT_SUBFOLDER_NAME)
            #         self.assertEqual(module.location.custom_path, "./Images")
            #
            #     def test_03_01_load_version_2(self):
            #         data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDA1IDEwOjMwOjQ0IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAkAEAAHic7ZPNTsJAEMen5UOUZIPxwpEXEInEwJHEiyQKBgj3xS7NJku32bYEPXn0NXwLjz6au7iFbWkoEm+6yWQ70/n/djo7RQDwcQJQlntFmg3fq6R9yzDlj0kYUs8NSlCEuo5/SptiQfGMkSlmEQlgs+J435vzybO/efXAnYiRAV6YyXINosWMiGA4j4X69SNdETamLwSSK04bkSUNKPe0XvPT0Zi/gwck7a2w7YOV0QdkxNXzHWzzixn5dSO/pv0JWYWXI+JGDIsGWfmCBKrAQPF6ObzTFE/5T5xx0Qzp3KjrGM5QUPdWsQxOK4djJTgWXP3rflXXhsPm7ByS96l86jl0SZ0IswZdYDcx53l12AmeDQN+XP3NA88rJHQF8K7wWvdq7f8fq0YcZey9nHNrqb4pWzfLFTzyG7KFxP/LvHj3Yf89mPd+SB1nqTqUf8+x0zcGVXG6BqecwSkbHFv7qIoQquzqs+owv6em/VazfXPd6XTTc5t1vvndtnyyYXfe83RFqXq/+LlOnac0X7WHgow='
            #         pipeline = T.load_pipeline(self, data)
            #         self.assertEqual(len(pipeline.modules()),1)
            #         module = pipeline.module(1)
            #         self.assertEqual(module.load_choice(),LI.MS_REGEXP)
            #         self.assertTrue(module.load_images())
            #         self.assertFalse(module.load_movies())
            #         self.assertTrue(module.text_to_exclude(), 'Do not use')
            #         self.assertEqual(len(module.image_name_vars()),1)
            #         self.assertEqual(module.image_name_vars()[0],'OrigColor')
            #         self.assertEqual(module.text_to_find_vars()[0].value,'color.tif')
            #         self.assertFalse(module.analyze_sub_dirs())
            #
            #     def test_03_02_load_version_4(self):
            #         data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDA1IDExOjA2OjM5IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAApwEAAHic5VTNTsJAEJ42BdEDwXjQY49eJCIXj8b4A4mCAUK8mYUudZO22/QHwafyEbj5Wu5KC8um0qV6c5LNdGZnvpn9OrtVAFhUAMpMMwU6LKWU2JqwuN3HUUQ8OyyBASeJf8HWEAUEjRw8RE6MQ1hJ6m97EzqY+6utR2rFDu4gVwxm0ondEQ7C7iRNTLafyAw7ffKOYVPSsB6ekpBQL8lP8GXvqi6NpLpVtlrGmgctg4cjwc/jr2Adb2TE14T4WrIGeBad3c7QODJdFI1fVXD2JRxuh42Xt0Z90L4T+rnMwdmTcLjdDYjdw5bSeX7q40LqowgO7+M+wNj7JQ7vp7kjLxUJp5L0c81GWaWPAymf2zfU9GhkxiFWP48qznkOjraBo0Hzj+u3cnAOJRxuE88iU2LFyDGJi+zV7VM5j76Bp0OHFuOhrshD1lzZAZqHY+Sk709VQX9o298Tkaes/Lw+s96Xb3LtgMa+ySjH/n/GK6p92P7fxLkqeq8eKLLawkWQ57mcU1dnX7WMPJV70ChYzyiQZ7DMz+Nl3vOOvJ5uiU8l9X8BnJqT/A=='
            #         pipeline = T.load_pipeline(self, data)
            #         pipeline.add_listener(self.error_callback)
            #         self.assertEqual(len(pipeline.modules()),1)
            #         module = pipeline.module(1)
            #         self.assertEqual(module.load_choice(),LI.MS_EXACT_MATCH)
            #         self.assertTrue(module.load_images())
            #         self.assertFalse(module.load_movies())
            #         self.assertTrue(module.text_to_exclude(), 'Do not use')
            #         self.assertEqual(len(module.image_name_vars()),3)
            #         self.assertEqual(module.image_name_vars()[0],'OrigRed')
            #         self.assertEqual(module.text_to_find_vars()[0].value,'s1_w1.TIF')
            #         self.assertEqual(module.image_name_vars()[1],'OrigGreen')
            #         self.assertEqual(module.text_to_find_vars()[1].value,'s1_w2.TIF')
            #         self.assertEqual(module.image_name_vars()[2],'OrigBlue')
            #         self.assertEqual(module.text_to_find_vars()[2].value,'s1_w3.TIF')
            #         self.assertFalse(module.analyze_sub_dirs())
            #         self.assertEqual(module.location.dir_choice,
            #                          LI.DEFAULT_INPUT_FOLDER_NAME)
            #
            #     def test_03_03_load_new_version_2(self):
            #         data = 'eJztV91u2jAUNjRURZOm7qLaLn1ZtoIga6UWTbQMJo2NMFRYt6rqVhcMWHJiFJwONlXaI+yR9ih7hD3CbOpA8CJC6S42jUhOfI7Pd36+EzmOVWxWi8/hXiYLrWIz3SEUwzpFvMNcOw8dvgNLLkYctyFz8rDZ8+Arj8KsCXN7+V0zLyZmNnsAlrtiFeu+fIrbunhsiBFXSwklxwJDyg3MOXG6gwQwwCOl/y7GCXIJuqT4BFEPD6YhfH3F6bDmqD9Zsljbo7iG7KCxuGqefYndwZuOD1TLdTLEtEE+Y60E3+wYX5EBYY7CK/+6dhKXcS2u5GF/fcpDLISHrYBe2r8EU3sjxP5BwH5TycRpkyvS9hCFxEbdSRbS31GEv03NnxxNPOTpF0PU4tBGvNWTfrIRfmIzfmLgqV9/BC6hxZdypVp9ayl8VNz4DD4OamwxHh9qcaVcxh3kUQ4rkkRYJi5uceaOfstjXfPnX76/ZID/qPzXZvJYA6eie3fBRfG9AWbrlnKphxwHU3OZuOVacaF89fcjtyA/xgzOEP11sMR9jcC91uqU8oftw/ozuRHiQuZJ6qOU3mFKj9mnwlkxXT9P+ZoSo57tFM6y6YPzL7kd8/rGuEEEcqxMjf3KPHoReexreUhZ+jrFyFUBdq9TaamymMN7SmcqXRmNppo79je3yH6Q1PBSLo0461M0sJV+mX6bq34v1e8fxu2+H39in1rh/h/cEZj/PoedD8aHjK7LvD4URw/c/5fqXfH7d+K+BXBh+1zweyLtL8B8Xh+DWV6l3BJbfd9l8n/IzdjjQ/sgQxlq35yaM1UxrQQO0Ho9yZA4wbziYrYVwYNe/5SXn4fLxIuHxLsXgTPUH5nEvQe34317jj3Q7H8BnRn8NQ=='
            #         fd = StringIO(zlib.decompress(base64.b64decode(data)))
            #         pipeline = cpp.Pipeline()
            #         pipeline.add_listener(self.error_callback)
            #         pipeline.load(fd)
            #         self.assertEqual(len(pipeline.modules()),1)
            #         module = pipeline.module(1)
            #         self.assertEqual(module.load_choice(), LI.MS_EXACT_MATCH)
            #         self.assertTrue(module.load_images())
            #         self.assertFalse(module.load_movies())
            #         self.assertTrue(module.exclude.value)
            #         self.assertEqual(module.text_to_exclude(), 'ILLUM')
            #         self.assertEqual(module.images[0].channels[0].image_name, 'DNA')
            #         self.assertEqual(module.images[0].common_text, 'Channel2')
            #         self.assertEqual(module.images[1].channels[0].image_name, 'Cytoplasm')
            #         self.assertEqual(module.images[1].common_text, 'Channel1')
            #         self.assertEqual(module.location.dir_choice,
            #                          LI.DEFAULT_INPUT_FOLDER_NAME)
            #
            #     def test_03_03_load_new_version_3(self):
            #         data = 'eJztV+1u0zAUdT+1CgmNP2M/vX/boFHawdgqtK20IIqaUm1lYkIgvNZtLTlxlDhbC9o78Eg8Eo9AnLlNaqKmK0iA1Ehucq/vPef6OLUdo9ppVl/Ap5oOjWqn2CcUwzZFvM8cswJt5pLRY1hzMOK4B5lVgR0PwzcehfAZLOmV8n5lbx+Wdf0QLHGlGsZ9/3bi/+T9+5rf0rIrJ+1UpAn7DHNOrIGbA1mwKf3f/XaOHIIuKT5H1MNuSDHxN6w+64ztaZfBeh7FLWRGg/2r5ZmX2HHf9ieJsrtNRpiekS9YGcIk7BRfEZcwS+ZLfNU75WVc4Q10yIc6pGJ02Ij4RfxrEMZnY+IfROLXpU2sHrkiPQ9RSEw0mFYR8CfgrSt4onXwiBdfjlCXQxPx7lDg6Ak4qRmcFNiT/AcJeTmFX9iNZvOdIfOTeNMz+WnQYovp+FDhFXYd95FHOWwIEWGdOLjLmTP+pY68gje5JniFiP5J9Wdm6siAC3/2/kZe0jytgVm9hF0bIsvCtLwMb71VXahe9b0qgcXe64JSr7BfiYXQ8pcH6Rc47xNwthQcYX/Sdovbx+3np+z6SHu0EzzXGD36oBcPP34t3+xE8IcJ+AcKvrAF3gVGjgR8cnNLYTCLD0OSwFdH49Dzm/NYWlbX2pgzmyLXjIz7rvNaBqt5nTevm7m77SN/Yr1a5a3ykvJOwPz/Qdz5IjikDBzm2dA/umD7fxrvSt9/M+9bJC9ufYzuNyL+M5iv6y6Y1VXYXUyp7TDxPeVoZnDodzXKUO/21K01/cdG5ACujqcQwxOtK+0/bSTooI4/1OXH8TJ8mRi+ewl5WflFp+6zi+i+PSceKPE/AfCf5eY='
            #         fd = StringIO(zlib.decompress(base64.b64decode(data)))
            #         pipeline = cpp.Pipeline()
            #         pipeline.add_listener(self.error_callback)
            #         pipeline.load(fd)
            #         self.assertEqual(len(pipeline.modules()),1)
            #         module = pipeline.module(1)
            #         self.assertEqual(module.load_choice(), LI.MS_EXACT_MATCH)
            #         self.assertTrue(module.load_images())
            #         self.assertFalse(module.load_movies())
            #         self.assertTrue(module.exclude.value)
            #         self.assertEqual(module.text_to_exclude(), 'ILLUM')
            #         self.assertEqual(module.images[0].channels[0].image_name, 'DNA')
            #         self.assertEqual(module.images[0].common_text, 'Channel2')
            #         self.assertEqual(module.images[0].file_metadata, '^.*-(?P<Row>.+)-(?P<Col>[0-9]{2})')
            #         self.assertEqual(module.images[1].channels[0].image_name, 'Cytoplasm')
            #         self.assertEqual(module.images[1].common_text, 'Channel1')
            #         self.assertEqual(module.images[1].file_metadata, '^.*-(?P<Row>.+)-(?P<Col>[0-9]{2})')
            #         self.assertEqual(module.location.dir_choice,
            #                          LI.DEFAULT_INPUT_FOLDER_NAME)
            #
            #     def test_03_04_load_new_version_4(self):
            #         data = ('eJztVt1O2zAUdn9A65AmuBqXvgS0VGnpNqgmILRDVOqfoGJDVaeZ1m0tOXGV'
            #                 'OKjdxDvsco/DI/EIi0vSpF7WpN24mIQlKznH53zf8Zf4p6a1qtopfJtVYU1r'
            #                 'KX1CMWxSxPvM1IvQ4G9gycSI4x5kRhGemQRq9gCq72GuUMztFwsHMK+qh2C1'
            #                 'lqjUXjmPnxsArDvPF05PukNrrp0IdGFfYs6JMbDWQBpsu/57p18hk6Abiq8Q'
            #                 'tbHlU3j+itFnrcloNlRjPZviOtKDwU6r2/oNNq1G30t0h5tkjOkl+YalKXhh'
            #                 'F/iWWIQZbr6LL3tnvIxLvEKH+7SvQyJEh62AX8SfAz8+HRG/6drE6JFb0rMR'
            #                 'hURHg1kVAu8kAm9TwhO9hcdc+ThGXQ51xLtDgaNG4CTmcBJgPyb/S4lf2GUG'
            #                 'DcahbWF/HlH8yTmcJKizeHq+lviFXcZ9ZFMOK0JMWCYm7nJmTn6rY13C85qH'
            #                 'lwHx60/N1ZEC185XfMq8P+m1LN9F41Os75yRdBZ2aYgMA9Oc8hc6letarDz5'
            #                 '/8yBeP9nWN1nYkM1nG0mUPcwAuedhCPsL64A7Vy+o7RV5bDzPX+n7Bw3Pziq'
            #                 'HrU15byzOzVLjeqRN74bj+9A4hO2gLrGyHSxCneP6DVm8KGPP/WV0cT3CL6H'
            #                 '1HL72L9YJ895z3lPlXcCFq+fsHNxergOTGaPoHPk4tH/NN9V834E8sLWfXBf'
            #                 'FfFfwWJd98C8rsLuYkpHJhP3VDOrTy9TVpYy1Hu8zWSrzmslcLGR55MJ4QnW'
            #                 'lXTetiJ0kOfv6/JwvApfOoRvIyIv7d6URd5nsJzuOwvigRT/CxrVwC0=')
            #         pipeline = cpp.Pipeline()
            #         def callback(caller,event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
            #         self.assertEqual(len(pipeline.modules()), 1)
            #         module = pipeline.modules()[0]
            #         assert isinstance(module, LI.LoadImages)
            #         self.assertEqual(len(module.metadata_fields.selections), 1)
            #         self.assertEqual(module.metadata_fields.selections[0], "ROW")
            #         self.assertEqual(len(module.images), 1)
            #         self.assertEqual(module.images[0].file_metadata, '^Channel[12]-[0-9]{2}-(?P<ROW>[A-H])-(?P<COL>[0-9]{2})')
            #         self.assertEqual(module.location.dir_choice,
            #                          LI.DEFAULT_INPUT_FOLDER_NAME)
            #
            #     def test_03_05_load_v5(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:9497
            #
            # LoadImages:[module_num:1|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:individual images
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Thumb
            #     Analyze all subfolders within the selected folder?:No
            #     Image location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:Yes
            #     Specify metadata fields to group by:
            #     Text that these images have in common (case-sensitive):Foo
            #     Name of this image in CellProfiler:DNA
            #     Position of this image in each group:1
            #     Select from where to extract metadata?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)$
            #     Text that these images have in common (case-sensitive):Bar
            #     Name of this image in CellProfiler:Cytoplasm
            #     Position of this image in each group:2
            #     Select from where to extract metadata?:File name
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #     Text that these images have in common (case-sensitive):Baz
            #     Name of this image in CellProfiler:Other
            #     Position of this image in each group:3
            #     Select from where to extract metadata?:Path
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #
            # LoadImages:[module_num:2|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:stk movies
            #     File selection method:Text-Regular expressions
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Do not use
            #     Analyze all subfolders within the selected folder?:Yes
            #     Image location:Default Output Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:Yes
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:Plate,Run
            #     Text that these images have in common (case-sensitive):Whatever
            #     Name of this image in CellProfiler:DNA
            #     Position of this image in each group:1
            #     Select from where to extract metadata?:Both
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #
            # LoadImages:[module_num:3|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:avi movies
            #     File selection method:Order
            #     Number of images in each group?:5
            #     Type the text that the excluded images have in common:Do not use
            #     Analyze all subfolders within the selected folder?:No
            #     Image location:Elsewhere...\x7C/imaging/analysis/People/Lee
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:
            #     Text that these images have in common (case-sensitive):
            #     Name of this image in CellProfiler:DNA
            #     Position of this image in each group:2
            #     Select from where to extract metadata?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #     Text that these images have in common (case-sensitive):
            #     Name of this image in CellProfiler:Actin
            #     Position of this image in each group:1
            #     Select from where to extract metadata?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #
            # LoadImages:[module_num:4|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:tif,tiff,flex movies
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Do not use
            #     Analyze all subfolders within the selected folder?:No
            #     Image location:Default Input Folder sub-folder\x7Cfoo
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:
            #     Text that these images have in common (case-sensitive):
            #     Name of this image in CellProfiler:DNA
            #     Position of this image in each group:1
            #     Select from where to extract metadata?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #
            # LoadImages:[module_num:5|svn_version:\'9497\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:individual images
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Do not use
            #     Analyze all subfolders within the selected folder?:No
            #     Image location:Default Output Folder sub-folder\x7Cbar
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:
            #     Text that these images have in common (case-sensitive):
            #     Name of this image in CellProfiler:DNA
            #     Position of this image in each group:1
            #     Select from where to extract metadata?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 5)
            #
            #         module = pipeline.modules()[0]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(module.order_group_size, 3)
            #         self.assertEqual(module.match_exclude, "Thumb")
            #         self.assertEqual(module.descend_subdirectories,LI.SUB_NONE)
            #         self.assertEqual(module.location.dir_choice, LI.DEFAULT_INPUT_FOLDER_NAME)
            #         self.assertTrue(module.check_images)
            #         self.assertFalse(module.group_by_metadata)
            #         self.assertTrue(module.exclude)
            #         self.assertEqual(len(module.images), 3)
            #         self.assertEqual(module.images[0].channels[0].image_name, "DNA")
            #         self.assertEqual(module.images[0].order_position, 1)
            #         self.assertEqual(module.images[0].common_text, "Foo")
            #         self.assertEqual(module.images[0].metadata_choice, LI.M_NONE)
            #         self.assertEqual(module.images[0].file_metadata, "^(?P<Plate>.*)")
            #         self.assertEqual(module.images[0].path_metadata,r".*[\\/](?P<Date>.*)$")
            #         self.assertEqual(module.images[1].channels[0].image_name, "Cytoplasm")
            #         self.assertEqual(module.images[1].common_text, "Bar")
            #         self.assertEqual(module.images[1].metadata_choice, LI.M_FILE_NAME)
            #         self.assertEqual(module.images[2].channels[0].image_name, "Other")
            #         self.assertEqual(module.images[2].common_text, "Baz")
            #         self.assertEqual(module.images[2].metadata_choice, LI.M_PATH)
            #
            #         module = pipeline.modules()[1]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_STK_MOVIES)
            #         self.assertEqual(module.match_method, LI.MS_REGEXP)
            #         self.assertEqual(module.location.dir_choice, LI.DEFAULT_OUTPUT_FOLDER_NAME)
            #         self.assertTrue(module.group_by_metadata)
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_ALL)
            #         self.assertEqual(len(module.metadata_fields.selections), 2)
            #         self.assertEqual(module.metadata_fields.selections[0], "Plate")
            #         self.assertEqual(module.metadata_fields.selections[1], "Run")
            #         self.assertEqual(module.images[0].metadata_choice, LI.M_BOTH)
            #
            #         module = pipeline.modules()[2]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_AVI_MOVIES)
            #         self.assertEqual(module.match_method, LI.MS_ORDER)
            #         self.assertEqual(module.location.dir_choice, LI.ABSOLUTE_FOLDER_NAME)
            #         self.assertEqual(module.location.custom_path, "/imaging/analysis/People/Lee")
            #
            #         module = pipeline.modules()[3]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
            #         self.assertEqual(module.location.dir_choice, LI.DEFAULT_INPUT_SUBFOLDER_NAME)
            #         self.assertEqual(module.location.custom_path, "foo")
            #
            #         module = pipeline.modules()[4]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.location.dir_choice, LI.DEFAULT_OUTPUT_SUBFOLDER_NAME)
            #         self.assertEqual(module.location.custom_path, "bar")
            #
            #     def test_03_06_load_v6(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:9801
            #
            # LoadImages:[module_num:1|svn_version:\'9799\'|variable_revision_number:6|show_window:True|notes:\x5B\'A flex file\'\x5D]
            #     File type to be loaded:tif,tiff,flex movies
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Thumb
            #     Analyze all subfolders within the selected folder?:No
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:Yes
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:Series,T,Z
            #     Image count:1
            #     Text that these images have in common (case-sensitive):.flex
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:foo
            #     Type the regular expression that finds metadata in the subfolder path:bar
            #     Channel count:2
            #     Name this loaded image:DNA
            #     Channel number\x3A:1
            #     Name this loaded image:Protein
            #     Channel number\x3A:3
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 1)
            #
            #         module = pipeline.modules()[0]
            #         module.notes = "A flex file"
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(module.order_group_size, 3)
            #         self.assertEqual(module.match_exclude, "Thumb")
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
            #         self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
            #         self.assertTrue(module.check_images)
            #         self.assertTrue(module.group_by_metadata)
            #         self.assertFalse(module.exclude)
            #         self.assertEqual(len(module.metadata_fields.selections), 3)
            #         self.assertEqual(module.metadata_fields.selections[0], LI.C_SERIES)
            #         self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
            #         self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
            #         self.assertEqual(module.image_count.value, 1)
            #         self.assertEqual(len(module.images), 1)
            #         image = module.images[0]
            #         self.assertEqual(image.common_text, ".flex")
            #         self.assertEqual(image.order_position, 1)
            #         self.assertEqual(image.metadata_choice, LI.M_NONE)
            #         self.assertEqual(image.file_metadata, "foo")
            #         self.assertEqual(image.path_metadata, "bar")
            #         self.assertEqual(image.channel_count.value, 2)
            #         self.assertEqual(len(image.channels), 2)
            #         for channel, channel_number, image_name in (
            #             (image.channels[0], 1, "DNA"),
            #             (image.channels[1], 3, "Protein")):
            #             self.assertEqual(channel.channel_number, channel_number)
            #             self.assertEqual(channel.image_name, image_name)
            #
            #     def test_03_07_load_v7(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:10021
            #
            # LoadImages:[module_num:1|svn_version:\'9976\'|variable_revision_number:7|show_window:True|notes:\x5B\'A flex file\'\x5D]
            #     File type to be loaded:tif,tiff,flex movies
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Thumb
            #     Analyze all subfolders within the selected folder?:No
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:Yes
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:Series,T,Z
            #     Image count:1
            #     Text that these images have in common (case-sensitive):.flex
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:foo
            #     Type the regular expression that finds metadata in the subfolder path:bar
            #     Channel count:2
            #     Group movie frames?:No
            #     Interleaving\x3A:Interleaved
            #     Channels per group\x3A:2
            #     Name this loaded image:DNA
            #     Channel number\x3A:1
            #     Name this loaded image:Protein
            #     Channel number\x3A:3
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 1)
            #
            #         module = pipeline.modules()[0]
            #         module.notes = "A flex file"
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(module.order_group_size, 3)
            #         self.assertEqual(module.match_exclude, "Thumb")
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
            #         self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
            #         self.assertTrue(module.check_images)
            #         self.assertTrue(module.group_by_metadata)
            #         self.assertFalse(module.exclude)
            #         self.assertEqual(len(module.metadata_fields.selections), 3)
            #         self.assertEqual(module.metadata_fields.selections[0], LI.C_SERIES)
            #         self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
            #         self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
            #         self.assertEqual(module.image_count.value, 1)
            #         self.assertEqual(len(module.images), 1)
            #         image = module.images[0]
            #         self.assertEqual(image.common_text, ".flex")
            #         self.assertEqual(image.order_position, 1)
            #         self.assertEqual(image.metadata_choice, LI.M_NONE)
            #         self.assertEqual(image.file_metadata, "foo")
            #         self.assertEqual(image.path_metadata, "bar")
            #         self.assertEqual(image.channel_count.value, 2)
            #         self.assertEqual(len(image.channels), 2)
            #         for channel, channel_number, image_name in (
            #             (image.channels[0], 1, "DNA"),
            #             (image.channels[1], 3, "Protein")):
            #             self.assertEqual(channel.channel_number, channel_number)
            #             self.assertEqual(channel.image_name, image_name)
            #
            #     def test_03_08_load_v8(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:10530
            #
            # LoadImages:[module_num:1|svn_version:\'10503\'|variable_revision_number:8|show_window:True|notes:\x5B\'A flex file\'\x5D]
            #     File type to be loaded:tif,tiff,flex movies, zvi movies
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Thumb
            #     Analyze all subfolders within the selected folder?:No
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:Yes
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:Series,T,Z
            #     Image count:1
            #     Text that these images have in common (case-sensitive):.flex
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:foo
            #     Type the regular expression that finds metadata in the subfolder path:bar
            #     Channel count:2
            #     Group the movie frames?:No
            #     Grouping method:Interleaved
            #     Number of channels per group:2
            #     Name this loaded image:DNA
            #     Channel number:1
            #     Rescale image?:Yes
            #     Name this loaded image:Protein
            #     Channel number:3
            #     Rescale image?:No
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 1)
            #
            #         module = pipeline.modules()[0]
            #         module.notes = "A flex file"
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(module.order_group_size, 3)
            #         self.assertEqual(module.match_exclude, "Thumb")
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
            #         self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
            #         self.assertTrue(module.check_images)
            #         self.assertTrue(module.group_by_metadata)
            #         self.assertFalse(module.exclude)
            #         self.assertEqual(len(module.metadata_fields.selections), 3)
            #         self.assertEqual(module.metadata_fields.selections[0], LI.C_SERIES)
            #         self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
            #         self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
            #         self.assertEqual(module.image_count.value, 1)
            #         self.assertEqual(len(module.images), 1)
            #         image = module.images[0]
            #         self.assertEqual(image.common_text, ".flex")
            #         self.assertEqual(image.order_position, 1)
            #         self.assertEqual(image.metadata_choice, LI.M_NONE)
            #         self.assertEqual(image.file_metadata, "foo")
            #         self.assertEqual(image.path_metadata, "bar")
            #         self.assertEqual(image.channel_count.value, 2)
            #         self.assertEqual(len(image.channels), 2)
            #         for channel, channel_number, image_name, rescale in (
            #             (image.channels[0], 1, "DNA", True),
            #             (image.channels[1], 3, "Protein", False)):
            #             self.assertEqual(channel.channel_number, channel_number)
            #             self.assertEqual(channel.image_name, image_name)
            #             self.assertEqual(channel.rescale.value, rescale)
            #             self.assertEqual(channel.image_object_choice, LI.IO_IMAGES)
            #
            #     def test_03_09_load_v9(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:10530
            #
            # LoadImages:[module_num:1|svn_version:\'10503\'|variable_revision_number:9|show_window:True|notes:\x5B\'A flex file\'\x5D]
            #     File type to be loaded:tif,tiff,flex movies, zvi movies
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Thumb
            #     Analyze all subfolders within the selected folder?:No
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:Yes
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:Series,T,Z
            #     Image count:1
            #     Text that these images have in common (case-sensitive):.flex
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:foo
            #     Type the regular expression that finds metadata in the subfolder path:bar
            #     Channel count:2
            #     Group the movie frames?:No
            #     Grouping method:Interleaved
            #     Number of channels per group:2
            #     Load images or objects?:Images
            #     Name this loaded image:DNA
            #     Name this loaded object:Nuclei
            #     Channel number:1
            #     Rescale image?:Yes
            #     Load images or objects?:Objects
            #     Name this loaded image:Protein
            #     Name this loaded object:Cytoplasm
            #     Channel number:3
            #     Rescale image?:No
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 1)
            #
            #         module = pipeline.modules()[0]
            #         module.notes = "A flex file"
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_OTHER_MOVIES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(module.order_group_size, 3)
            #         self.assertEqual(module.match_exclude, "Thumb")
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
            #         self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
            #         self.assertTrue(module.check_images)
            #         self.assertTrue(module.group_by_metadata)
            #         self.assertFalse(module.exclude)
            #         self.assertEqual(len(module.metadata_fields.selections), 3)
            #         self.assertEqual(module.metadata_fields.selections[0], LI.C_SERIES)
            #         self.assertEqual(module.metadata_fields.selections[1], LI.M_T)
            #         self.assertEqual(module.metadata_fields.selections[2], LI.M_Z)
            #         self.assertEqual(module.image_count.value, 1)
            #         self.assertEqual(len(module.images), 1)
            #         image = module.images[0]
            #         self.assertEqual(image.common_text, ".flex")
            #         self.assertEqual(image.order_position, 1)
            #         self.assertEqual(image.metadata_choice, LI.M_NONE)
            #         self.assertEqual(image.file_metadata, "foo")
            #         self.assertEqual(image.path_metadata, "bar")
            #         self.assertEqual(image.channel_count.value, 2)
            #         self.assertEqual(len(image.channels), 2)
            #         for channel, choice, channel_number, image_name, object_name, rescale in (
            #             (image.channels[0], LI.IO_IMAGES, 1, "DNA", "Nuclei", True),
            #             (image.channels[1], LI.IO_OBJECTS, 3, "Protein", "Cytoplasm", False)):
            #             self.assertEqual(channel.image_object_choice, choice)
            #             self.assertEqual(channel.channel_number, channel_number)
            #             self.assertEqual(channel.image_name, image_name)
            #             self.assertEqual(channel.object_name, object_name)
            #             self.assertEqual(channel.rescale.value, rescale)
            #
            #     def test_03_10_load_v10(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:10809
            #
            # LoadImages:[module_num:1|svn_version:\'10807\'|variable_revision_number:10|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:individual images
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Whatever
            #     Analyze all subfolders within the selected folder?:No
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:
            #     Image count:2
            #     Text that these images have in common (case-sensitive):_w1_
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #     Channel count:1
            #     Group the movie frames?:No
            #     Grouping method:Interleaved
            #     Number of channels per group:3
            #     Load the input as images or objects?:Images
            #     Name this loaded image:w1
            #     Name this loaded object:Nuclei
            #     Retain outlines of loaded objects?:Yes
            #     Name the outline image:MyOutlines
            #     Channel number:1
            #     Rescale intensities?:Yes
            #     Text that these images have in common (case-sensitive):_w2_
            #     Position of this image in each group:2
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #     Channel count:1
            #     Group the movie frames?:No
            #     Grouping method:Interleaved
            #     Number of channels per group:3
            #     Load the input as images or objects?:Images
            #     Name this loaded image:w2
            #     Name this loaded object:Nuclei
            #     Retain outlines of loaded objects?:No
            #     Name the outline image:MyNucleiOutlines
            #     Channel number:1
            #     Rescale intensities?:Yes
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 1)
            #
            #         module = pipeline.modules()[0]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(module.order_group_size, 3)
            #         self.assertEqual(module.match_exclude, "Whatever")
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
            #         self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
            #         self.assertTrue(module.check_images)
            #         self.assertFalse(module.group_by_metadata)
            #         self.assertFalse(module.exclude)
            #         self.assertEqual(module.image_count.value, 2)
            #         self.assertEqual(len(module.images), 2)
            #         image = module.images[0]
            #         self.assertEqual(image.common_text, "_w1_")
            #         self.assertEqual(image.order_position, 1)
            #         self.assertEqual(image.metadata_choice, LI.M_NONE)
            #         self.assertEqual(image.channel_count.value, 1)
            #         self.assertEqual(len(image.channels), 1)
            #         channel = image.channels[0]
            #         self.assertEqual(channel.image_object_choice, LI.IO_IMAGES)
            #         self.assertEqual(channel.channel_number, 1)
            #         self.assertEqual(channel.image_name, "w1")
            #         self.assertEqual(channel.object_name, "Nuclei")
            #         self.assertTrue(channel.rescale)
            #         self.assertTrue(channel.wants_outlines)
            #         self.assertEqual(channel.outlines_name, "MyOutlines")
            #         self.assertFalse(module.images[1].channels[0].wants_outlines)
            #
            #     def test_03_11_load_v11(self):
            #         data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
            # Version:1
            # SVNRevision:10809
            #
            # LoadImages:[module_num:1|svn_version:\'10807\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:individual images
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Whatever
            #     Analyze subfolders within the selected folder?:All
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:
            #     Select subfolders to analyze:hello/kitty,fubar
            #     Image count:1
            #     Text that these images have in common (case-sensitive):_w1_
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #     Channel count:1
            #     Group the movie frames?:No
            #     Grouping method:Interleaved
            #     Number of channels per group:3
            #     Load the input as images or objects?:Images
            #     Name this loaded image:w1
            #     Name this loaded object:Nuclei
            #     Retain outlines of loaded objects?:Yes
            #     Name the outline image:MyOutlines
            #     Channel number:1
            #     Rescale intensities?:Yes
            #
            # LoadImages:[module_num:2|svn_version:\'10807\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:individual images
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Whatever
            #     Analyze subfolders within the selected folder?:Some
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:
            #     Select subfolders to analyze:hello/kitty,fubar
            #     Image count:1
            #     Text that these images have in common (case-sensitive):_w1_
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #     Channel count:1
            #     Group the movie frames?:No
            #     Grouping method:Interleaved
            #     Number of channels per group:3
            #     Load the input as images or objects?:Images
            #     Name this loaded image:w1
            #     Name this loaded object:Nuclei
            #     Retain outlines of loaded objects?:Yes
            #     Name the outline image:MyOutlines
            #     Channel number:1
            #     Rescale intensities?:Yes
            #
            # LoadImages:[module_num:3|svn_version:\'10807\'|variable_revision_number:11|show_window:True|notes:\x5B\x5D]
            #     File type to be loaded:individual images
            #     File selection method:Text-Exact match
            #     Number of images in each group?:3
            #     Type the text that the excluded images have in common:Whatever
            #     Analyze subfolders within the selected folder?:None
            #     Input image file location:Default Input Folder\x7CNone
            #     Check image sets for missing or duplicate files?:Yes
            #     Group images by metadata?:No
            #     Exclude certain files?:No
            #     Specify metadata fields to group by:
            #     Select subfolders to analyze:hello/kitty,fubar
            #     Image count:1
            #     Text that these images have in common (case-sensitive):_w1_
            #     Position of this image in each group:1
            #     Extract metadata from where?:None
            #     Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
            #     Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Date>.*)\x5B\\\\/\x5D(?P<Run>.*)$
            #     Channel count:1
            #     Group the movie frames?:No
            #     Grouping method:Interleaved
            #     Number of channels per group:3
            #     Load the input as images or objects?:Images
            #     Name this loaded image:w1
            #     Name this loaded object:Nuclei
            #     Retain outlines of loaded objects?:Yes
            #     Name the outline image:MyOutlines
            #     Channel number:1
            #     Rescale intensities?:Yes
            # """
            #         pipeline = cpp.Pipeline()
            #         def callback(caller, event):
            #             self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            #         pipeline.add_listener(callback)
            #         pipeline.load(StringIO(data))
            #         self.assertEqual(len(pipeline.modules()), 3)
            #
            #         module = pipeline.modules()[0]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.file_types, LI.FF_INDIVIDUAL_IMAGES)
            #         self.assertEqual(module.match_method, LI.MS_EXACT_MATCH)
            #         self.assertEqual(module.order_group_size, 3)
            #         self.assertEqual(module.match_exclude, "Whatever")
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_ALL)
            #         self.assertEqual(module.location.dir_choice, LI.cps.DEFAULT_INPUT_FOLDER_NAME)
            #         self.assertTrue(module.check_images)
            #         self.assertFalse(module.group_by_metadata)
            #         self.assertFalse(module.exclude)
            #         self.assertEqual(module.image_count.value, 1)
            #         self.assertEqual(len(module.images), 1)
            #         image = module.images[0]
            #         self.assertEqual(image.common_text, "_w1_")
            #         self.assertEqual(image.order_position, 1)
            #         self.assertEqual(image.metadata_choice, LI.M_NONE)
            #         self.assertEqual(image.channel_count.value, 1)
            #         self.assertEqual(len(image.channels), 1)
            #         channel = image.channels[0]
            #         self.assertEqual(channel.image_object_choice, LI.IO_IMAGES)
            #         self.assertEqual(channel.channel_number, 1)
            #         self.assertEqual(channel.image_name, "w1")
            #         self.assertEqual(channel.object_name, "Nuclei")
            #         self.assertTrue(channel.rescale)
            #         self.assertTrue(channel.wants_outlines)
            #         self.assertEqual(channel.outlines_name, "MyOutlines")
            #
            #         module = pipeline.modules()[1]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_SOME)
            #
            #         module = pipeline.modules()[2]
            #         self.assertTrue(isinstance(module, LI.LoadImages))
            #         self.assertEqual(module.descend_subdirectories, LI.SUB_NONE)
            #
            #     def test_04_01_load_save_and_load(self):
            #         data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBNb24gSmFuIDA1IDExOjA2OjM5IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAApwEAAHic5VTNTsJAEJ42BdEDwXjQY49eJCIXj8b4A4mCAUK8mYUudZO22/QHwafyEbj5Wu5KC8um0qV6c5LNdGZnvpn9OrtVAFhUAMpMMwU6LKWU2JqwuN3HUUQ8OyyBASeJf8HWEAUEjRw8RE6MQ1hJ6m97EzqY+6utR2rFDu4gVwxm0ondEQ7C7iRNTLafyAw7ffKOYVPSsB6ekpBQL8lP8GXvqi6NpLpVtlrGmgctg4cjwc/jr2Adb2TE14T4WrIGeBad3c7QODJdFI1fVXD2JRxuh42Xt0Z90L4T+rnMwdmTcLjdDYjdw5bSeX7q40LqowgO7+M+wNj7JQ7vp7kjLxUJp5L0c81GWaWPAymf2zfU9GhkxiFWP48qznkOjraBo0Hzj+u3cnAOJRxuE88iU2LFyDGJi+zV7VM5j76Bp0OHFuOhrshD1lzZAZqHY+Sk709VQX9o298Tkaes/Lw+s96Xb3LtgMa+ySjH/n/GK6p92P7fxLkqeq8eKLLawkWQ57mcU1dnX7WMPJV70ChYzyiQZ7DMz+Nl3vOOvJ5uiU8l9X8BnJqT/A=='
            #         pipeline = T.load_pipeline(self, data)
            #         (matfd,matpath) = tempfile.mkstemp('.mat')
            #         matfh = os.fdopen(matfd,'wb')
            #         pipeline.save(matfh)
            #         matfh.flush()
            #         pipeline = P.Pipeline()
            #         pipeline.load(matpath)
            #         matfh.close()
            #         self.assertEqual(len(pipeline.modules()),1)
            #         module = pipeline.module(1)
            #         self.assertEqual(module.load_choice(),LI.MS_EXACT_MATCH)
            #         self.assertTrue(module.load_images())
            #         self.assertFalse(module.load_movies())
            #         self.assertTrue(module.text_to_exclude(), 'Do not use')
            #         self.assertEqual(len(module.image_name_vars()),3)
            #         self.assertEqual(module.image_name_vars()[0],'OrigRed')
            #         self.assertEqual(module.text_to_find_vars()[0].value,'s1_w1.TIF')
            #         self.assertEqual(module.image_name_vars()[1],'OrigGreen')
            #         self.assertEqual(module.text_to_find_vars()[1].value,'s1_w2.TIF')
            #         self.assertEqual(module.image_name_vars()[2],'OrigBlue')
            #         self.assertEqual(module.text_to_find_vars()[2].value,'s1_w3.TIF')
            #         self.assertFalse(module.analyze_sub_dirs())
            #
            #     def test_05_01_load_PNG(self):
            #         """Test loading of a .PNG file
            #
            #         Regression test a bug in PIL that flips the image
            #         """
            #         data = base64.b64decode(T.png_8_1)
            #         (matfd,matpath) = tempfile.mkstemp('.png')
            #         matfh = os.fdopen(matfd,'wb')
            #         matfh.write(data)
            #         matfh.flush()
            #         path,filename = os.path.split(matpath)
            #         load_images = LI.LoadImages()
            #         load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            #         load_images.match_method.value = LI.MS_EXACT_MATCH
            #         load_images.images[0].common_text.value = filename
            #         load_images.images[0].channels[0].image_name.value = 'Orig'
            #         load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            #         load_images.location.custom_path = path
            #         load_images.module_num = 1
            #
            #         class CheckImage(CPM.CPModule):
            #             variable_revision_number = 1
            #             module_name = "CheckImage"
            #             def settings(self):
            #                 return []
            #             def run(self,workspace):
            #                 image = workspace.image_set.get_image('Orig')
            #                 matfh.close()
            #                 pixel_data = image.pixel_data
            #                 check_data = base64.b64decode(T.raw_8_1)
            #                 check_image = np.fromstring(check_data,np.uint8).reshape(T.raw_8_1_shape)
            #                 self.expected_digest = hashlib.md5()
            #                 self.expected_digest.update(
            #                     (check_image.astype(np.float32)/255).tostring())
            #                 self.digest = hashlib.md5()
            #                 self.digest.update(
            #                     pixel_data.astype(np.float32).tostring())
            #         check_image = CheckImage()
            #         check_image.module_num = 2
            #         pipeline = P.Pipeline()
            #         pipeline.add_listener(self.error_callback)
            #         pipeline.add_module(load_images)
            #         pipeline.add_module(check_image)
            #         m = pipeline.run()
            #         self.assertEqual(check_image.digest.hexdigest(),
            #                          check_image.expected_digest.hexdigest())
            #         md5 = m[measurements.IMAGE, "_".join((LI.C_MD5_DIGEST, "Orig")), 1]
            #         expected_md5 = hashlib.md5()
            #         expected_md5.update(data)
            #         self.assertEqual(md5, expected_md5.hexdigest())
            #
            #     def test_05_02_load_GIF(self):
            #         """Test loading of a .GIF file
            #
            #         """
            #         data = base64.b64decode(T.gif_8_1)
            #         (matfd,matpath) = tempfile.mkstemp('.gif')
            #         matfh = os.fdopen(matfd,'wb')
            #         matfh.write(data)
            #         matfh.flush()
            #         path,filename = os.path.split(matpath)
            #         load_images = LI.LoadImages()
            #         load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            #         load_images.match_method.value = LI.MS_EXACT_MATCH
            #         load_images.images[0].common_text.value = filename
            #         load_images.images[0].channels[0].image_name.value = 'Orig'
            #         load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            #         load_images.location.custom_path = path
            #         load_images.module_num = 1
            #         outer_self = self
            #         class CheckImage(CPM.CPModule):
            #             variable_revision_number = 1
            #             module_name = "CheckImage"
            #             def settings(self):
            #                 return []
            #             def run(self,workspace):
            #                 image = workspace.image_set.get_image(
            #                     'Orig', must_be_grayscale=True)
            #                 matfh.close()
            #                 pixel_data = image.pixel_data
            #                 pixel_data = (pixel_data * 255+.5).astype(np.uint8)
            #                 check_data = base64.b64decode(T.raw_8_1)
            #                 check_image = np.fromstring(check_data,np.uint8).reshape(T.raw_8_1_shape)
            #                 outer_self.assertTrue(np.all(pixel_data ==check_image))
            #         check_image = CheckImage()
            #         check_image.module_num = 2
            #         pipeline = P.Pipeline()
            #         pipeline.add_listener(self.error_callback)
            #         pipeline.add_module(load_images)
            #         pipeline.add_module(check_image)
            #         pipeline.run()
            #
            #     def test_05_03_load_TIF(self):
            #         """Test loading of a .TIF file
            #
            #         """
            #         data = base64.b64decode(T.tif_8_1)
            #         (matfd,matpath) = tempfile.mkstemp('.tif')
            #         matfh = os.fdopen(matfd,'wb')
            #         matfh.write(data)
            #         matfh.flush()
            #         path,filename = os.path.split(matpath)
            #         load_images = LI.LoadImages()
            #         load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
            #         load_images.match_method.value = LI.MS_EXACT_MATCH
            #         load_images.images[0].common_text.value = filename
            #         load_images.images[0].channels[0].image_name.value = 'Orig'
            #         load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
            #         load_images.location.custom_path = path
            #         load_images.module_num = 1
            #         outer_self = self
            #         class CheckImage(CPM.CPModule):
            #             variable_revision_number = 1
            #             module_name = "CheckImage"
            #             def settings(self):
            #                 return []
            #             def run(self,workspace):
            #                 image = workspace.image_set.get_image('Orig')
            #                 matfh.close()
            #                 pixel_data = image.pixel_data
            #                 pixel_data = (pixel_data * 255+.5).astype(np.uint8)
            #                 check_data = base64.b64decode(T.raw_8_1)
            #                 check_image = np.fromstring(check_data,np.uint8).reshape(T.raw_8_1_shape)
            #                 outer_self.assertTrue(np.all(pixel_data ==check_image))
            #         check_image = CheckImage()
            #         check_image.module_num = 2
            #         pipeline = P.Pipeline()
            #         pipeline.add_listener(self.error_callback)
            #         pipeline.add_module(load_images)
            #         pipeline.add_module(check_image)
            #         m = pipeline.run()
            #         self.assertTrue(isinstance(m, measurements.Measurements))
            #         fn = m.get_all_measurements(measurements.IMAGE, 'FileName_Orig')
            #         self.assertEqual(len(fn), 1)
            #         self.assertEqual(fn[0], filename)
            #         p = m.get_all_measurements(measurements.IMAGE, 'PathName_Orig')
            #         self.assertEqual(p[0], path)
            #         scale = m.get_all_measurements(measurements.IMAGE, 'Scaling_Orig')
            #         self.assertEqual(scale[0], 255)

    # def test_05_04_load_JPG(self):
    #     """Test loading of a .JPG file
    #
    #     """
    #     data = base64.b64decode(T.jpg_8_1)
    #     (matfd, matpath) = tempfile.mkstemp('.jpg')
    #     matfh = os.fdopen(matfd, 'wb')
    #     matfh.write(data)
    #     matfh.flush()
    #     path, filename = os.path.split(matpath)
    #     load_images = LI.LoadImages()
    #     load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
    #     load_images.match_method.value = LI.MS_EXACT_MATCH
    #     load_images.images[0].common_text.value = filename
    #     load_images.images[0].channels[0].image_name.value = 'Orig'
    #     load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
    #     load_images.location.custom_path = path
    #     load_images.module_num = 1
    #     outer_self = self
    #
    #     class CheckImage(CPM.CPModule):
    #         variable_revision_number = 1
    #         module_name = "CheckImage"
    #
    #         def settings(self):
    #             return []
    #
    #         def run(self, workspace):
    #             image = workspace.image_set.get_image('Orig',
    #                                                   must_be_grayscale=True)
    #             pixel_data = image.pixel_data
    #             matfh.close()
    #             pixel_data = (pixel_data * 255).astype(np.uint8)
    #             check_data = base64.b64decode(T.raw_8_1)
    #             check_image = np.fromstring(check_data, np.uint8).reshape(T.raw_8_1_shape)
    #             # JPEG is lossy, apparently even when you ask for no compression
    #             epsilon = 1
    #             outer_self.assertTrue(np.all(np.abs(pixel_data.astype(int)
    #                                                 - check_image.astype(int) <=
    #                                                 epsilon)))
    #
    #     check_image = CheckImage()
    #     check_image.module_num = 2
    #     pipeline = P.Pipeline()
    #     pipeline.add_listener(self.error_callback)
    #     pipeline.add_module(load_images)
    #     pipeline.add_module(check_image)
    #     pipeline.run()

    def test_05_05_load_url(self):
        lip = LI.LoadImagesImageProvider(
                "broad",
                cp_logo_url_folder,
                cp_logo_url_filename, True)
        logo = lip.provide_image(None)
        self.assertEqual(logo.pixel_data.shape, cp_logo_url_shape)
        lip.release_memory()

    def test_05_06_load_Nikon_tif(self):
        '''This is the Nikon format TIF file from IMG-838'''
        maybe_download_tesst_image("NikonTIF.tif")
        lip = LI.LoadImagesImageProvider(
                "nikon",
                T.testimages_directory(),
                "NikonTIF.tif",
                True)
        image = lip.provide_image(None).pixel_data
        self.assertEqual(tuple(image.shape), (731, 805, 3))
        self.assertAlmostEqual(np.sum(image.astype(np.float64)), 560730.83, 0)

    def test_05_07_load_Metamorph_tif(self):
        '''Regression test of IMG-883

        This file generated a null-pointer exception in the MetamorphReader
        '''
        maybe_download_tesst_image(
                "IXMtest_P24_s9_w560D948A4-4D16-49D0-9080-7575267498F9.tif")
        lip = LI.LoadImagesImageProvider(
                "nikon",
                T.testimages_directory(),
                "IXMtest_P24_s9_w560D948A4-4D16-49D0-9080-7575267498F9.tif",
                True)
        image = lip.provide_image(None).pixel_data
        self.assertEqual(tuple(image.shape), (520, 696))
        self.assertAlmostEqual(np.sum(image.astype(np.float64)), 2071.93, 0)

    # With Subimager and the new file_ui framework, you'd load individual
    # planes.
    @unittest.skip
    def test_05_08_load_5channel_tif(self):
        '''Load a 5-channel image'''
        maybe_download_tesst_image("5channel.tif")
        path = T.testimages_directory()
        file_name = "5channel.tif"
        maybe_download_tesst_image(file_name)
        module = LI.LoadImages()
        module.module_num = 1
        module.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        module.match_method.value = LI.MS_EXACT_MATCH
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = path
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.images[0].common_text.value = file_name

        pipeline = P.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, P.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        image_set_list = I.ImageSetList()
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m, image_set_list)
        self.assertTrue(module.prepare_run(workspace))
        image_numbers = m.get_image_numbers()
        self.assertEqual(len(image_numbers), 1)
        key_names, group_list = pipeline.get_groupings(workspace)
        self.assertEqual(len(group_list), 1)
        grouping, image_numbers = group_list[0]
        self.assertEqual(len(image_numbers), 1)
        module.prepare_group(workspace, grouping, image_numbers)

        image_set = image_set_list.get_image_set(0)
        workspace = W.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                m, image_set_list)
        module.run(workspace)
        image = image_set.get_image(IMAGE_NAME)
        pixels = image.pixel_data
        self.assertEqual(pixels.ndim, 3)
        self.assertEqual(tuple(pixels.shape), (64, 64, 5))

    def test_05_09_load_C01(self):
        """IMG-457: Test loading of a .c01 file"""
        file_name = "icd002235_090127090001_a01f00d1.c01"
        maybe_download_tesst_image(file_name)
        lip = LI.LoadImagesImageProvider(
                "nikon",
                T.testimages_directory(),
                file_name,
                True)
        image = lip.provide_image(None).pixel_data
        self.assertEqual(tuple(image.shape), (512, 512))
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
            fd = open(os.path.join(directory, filename), "wb")
            fd.write(data)
            fd.close()
        load_images = LI.LoadImages()
        load_images.add_imagecb()
        load_images.file_types.value = LI.FF_INDIVIDUAL_IMAGES
        load_images.match_method.value = LI.MS_REGEXP
        load_images.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        load_images.location.custom_path = directory
        load_images.group_by_metadata.value = True
        load_images.images[
            0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[
            1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
        load_images.images[0].channels[0].image_name.value = "Channel1"
        load_images.images[1].channels[0].image_name.value = "Channel2"
        load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[
            0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[
            1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
        load_images.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(load_images)
        image_set_list = I.ImageSetList()
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, load_images, None, None, m,
                                image_set_list)
        load_images.prepare_run(workspace)
        self.assertEqual(m.image_set_count, 2)
        load_images.prepare_group(workspace, (), [1, 2])
        image_set = image_set_list.get_image_set(0)
        w = W.Workspace(pipeline, load_images, image_set, cpo.ObjectSet(), m,
                        image_set_list)
        load_images.run(w)
        self.assertEqual(image_set.get_image_provider("Channel1").get_filename(),
                         filenames[0])
        self.assertEqual(image_set.get_image_provider("Channel2").get_filename(),
                         filenames[1])
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
        m.next_image_set(2)
        w = W.Workspace(pipeline, load_images, image_set, cpo.ObjectSet(), m,
                        image_set_list)
        load_images.run(w)
        self.assertEqual(image_set.get_image_provider("Channel1").get_filename(),
                         filenames[2])
        self.assertEqual(image_set.get_image_provider("Channel2").get_filename(),
                         filenames[3])
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
        path_and_file = [("MMD-ControlSet-plateA-2008-08-06_A12_s1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE]", "w1.tif"),
                         ("MMD-ControlSet-plateA-2008-08-06_A12_s1_[EFBB8532-9A90-4040-8974-477FE1E0F3CA]", "w2.tif"),
                         ("MMD-ControlSet-plateA-2008-08-06_A12_s2_[138B5A19-2515-4D46-9AB7-F70CE4D56631]", "w1.tif"),
                         ("MMD-ControlSet-plateA-2008-08-06_A12_s2_[59784AC1-6A66-44DE-A87E-E4BDC1A33A54]", "w2.tif")
                         ]
        for path, filename in path_and_file:
            os.mkdir(os.path.join(directory, path))
            fd = open(os.path.join(directory, path, filename), "wb")
            fd.write(data)
            fd.close()

    def test_06_03_missing_image(self):
        """Test expected failure when an image is missing from the set"""
        directory = tempfile.mkdtemp()
        self.directory = directory
        data = base64.b64decode(T.tif_8_1)
        filename = "MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif"
        fd = open(os.path.join(directory, filename), "wb")
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
        load_images.images[
            0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[
            1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
        load_images.images[0].channels[0].image_name.value = "Channel1"
        load_images.images[1].channels[0].image_name.value = "Channel2"
        load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
        load_images.images[
            0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
        load_images.images[
            1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
        load_images.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(load_images)
        image_set_list = I.ImageSetList()
        self.assertFalse(load_images.prepare_run(
                W.Workspace(pipeline, load_images, None, None,
                            measurements.Measurements(),
                            image_set_list)))

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
            fd = open(os.path.join(directory, filename), "wb")
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
            load_images.images[
                0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[
                1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.images[0].channels[0].image_name.value = "Channel1"
            load_images.images[1].channels[0].image_name.value = "Channel2"
            load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
            load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
            load_images.images[
                0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
            load_images.images[
                1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_module(load_images)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            self.assertFalse(load_images.prepare_run(
                    W.Workspace(
                            pipeline, load_images, None, None,
                            measurements.Measurements(), image_set_list)))
        finally:
            clear_image_reader_cache()
            for filename in filenames:
                os.remove(os.path.join(directory, filename))
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
            fd = open(os.path.join(directory, filename), "wb")
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
            load_images.images[0].file_metadata.value = \
                ("^(?P<Date>[0-9]{4}-[0-9]{2}-[0-9]{2})-"
                 "run(?P<Run>[0-9])-(?P<plate>.*?)_"
                 "(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_"
                 "s(?P<site>[0-9]+)_w1_")
            load_images.images[1].file_metadata.value = \
                "^illum_run(?P<Run>[0-9])-(?P<plate>.*?)\\."
            load_images.module_num = 1
            pipeline = P.Pipeline()
            pipeline.add_module(load_images)
            pipeline.add_listener(self.error_callback)
            image_set_list = I.ImageSetList()
            m = measurements.Measurements()
            load_images.prepare_run(
                    W.Workspace(pipeline, load_images, None, None,
                                m, image_set_list))
            for i in range(12):
                channel1_filename = m.get_measurement(
                        measurements.IMAGE,
                        LI.C_FILE_NAME + "_" + "Channel1", i + 1)
                ctags = re.search(load_images.images[0].file_metadata.value,
                                  channel1_filename).groupdict()
                illum_filename = m.get_measurement(
                        measurements.IMAGE,
                        LI.C_FILE_NAME + "_" + "Illum", i + 1)
                itags = re.search(load_images.images[1].file_metadata.value,
                                  illum_filename).groupdict()
                self.assertEqual(ctags["Run"], itags["Run"])
                self.assertEqual(ctags["plate"], itags["plate"])
        finally:
            clear_image_reader_cache()
            for filename in filenames:
                os.remove(os.path.join(directory, filename))
            os.rmdir(directory)

    def test_06_06_allowed_conflict(self):
        """Test choice of newest file when there is a conflict"""
        filenames = ["MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s1_w2_[EFBB8532-9A90-4040-8974-477FE1E0F3CA].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s1_w1_[138B5A19-2515-4D46-9AB7-F70CE4D56631].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w1_[138B5A19-2515-4D46-9AB7-F70CE4D56631].tif",
                     "MMD-ControlSet-plateA-2008-08-06_A12_s2_w2_[59784AC1-6A66-44DE-A87E-E4BDC1A33A54].tif"
                     ]
        for chosen, order in ((2, (0, 1, 2, 3, 4)), (0, (2, 1, 0, 3, 4))):
            #
            # LoadImages should choose the file that was written last
            #
            directory = tempfile.mkdtemp()
            data = base64.b64decode(T.tif_8_1)
            for i in range(len(filenames)):
                filename = filenames[order[i]]
                fd = open(os.path.join(directory, filename), "wb")
                fd.write(data)
                fd.close()
                # make sure times are different
                # The Mac claims to save float times, but stat returns
                # a float whose fractional part is always 0
                #
                # Also happens on at least one Centos build.
                #
                if os.stat_float_times() and not any([
                                                         sys.platform.startswith(x) for x in ("darwin", "linux")]):
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
                load_images.images[
                    0].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
                load_images.images[
                    1].common_text.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
                load_images.images[0].channels[0].image_name.value = "Channel1"
                load_images.images[1].channels[0].image_name.value = "Channel2"
                load_images.images[0].metadata_choice.value = LI.M_FILE_NAME
                load_images.images[1].metadata_choice.value = LI.M_FILE_NAME
                load_images.images[
                    0].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w1_"
                load_images.images[
                    1].file_metadata.value = "^(?P<plate>.*?)_(?P<well_row>[A-P])(?P<well_col>[0-9]{2})_s(?P<site>[0-9]+)_w2_"
                load_images.module_num = 1
                pipeline = P.Pipeline()
                pipeline.add_module(load_images)
                pipeline.add_listener(self.error_callback)
                image_set_list = I.ImageSetList()
                m = measurements.Measurements()
                workspace = W.Workspace(pipeline, load_images, None, None,
                                        m, image_set_list)
                load_images.prepare_run(workspace)
                d = dict(plate="MMD-ControlSet-plateA-2008-08-06",
                         well_row="A",
                         well_col="12",
                         Well="A12",
                         site="1")
                key_names, groupings = load_images.get_groupings(workspace)
                self.assertEqual(len(groupings), 2)
                my_groups = [x for x in groupings
                             if all([d[key_name] == x[0][key_name]
                                     for key_name in key_names])]
                self.assertEqual(len(my_groups), 1)
                load_images.prepare_group(workspace, d, my_groups[0][1])
                image_set = image_set_list.get_image_set(d)
                load_images.run(W.Workspace(pipeline, load_images, image_set,
                                            cpo.ObjectSet(), m, image_set_list))
                image = image_set.get_image("Channel1")
                self.assertEqual(image.file_name, filenames[chosen])
            finally:
                clear_image_reader_cache()
                for filename in filenames:
                    p = os.path.join(directory, filename)
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
        filenames = [("d1", "bar.tif"),
                     ("d1", "foo.tif"),
                     (os.path.join("d2", "d3"), "foo.tif"),
                     (os.path.join("d2", "d4"), "bar.tif")]
        data = base64.b64decode(T.tif_8_1)
        try:
            for path, file_name in filenames:
                d = os.path.join(directory, path)
                if not os.path.isdir(d):
                    os.makedirs(d)
                fd = open(os.path.join(directory, path, file_name), "wb")
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
            m = measurements.Measurements()
            workspace = W.Workspace(pipeline, load_images, None, None, m,
                                    image_set_list)
            self.assertTrue(load_images.prepare_run(workspace))
            image_numbers = m.get_image_numbers()
            self.assertEqual(len(image_numbers), len(filenames))
            load_images.prepare_group(workspace, {}, image_numbers)
            for i, (path, file_name) in enumerate(filenames):
                if i > 0:
                    m.next_image_set()
                image_set = image_set_list.get_image_set(i)
                w = W.Workspace(pipeline, load_images, image_set,
                                cpo.ObjectSet(), m, image_set_list)
                load_images.run(w)
                image = image_set.get_image("my_image")
                self.assertEqual(tuple(image.pixel_data.shape), (48, 32))
                f = m.get_current_image_measurement("FileName_my_image")
                self.assertEqual(f, file_name)
                p = m.get_current_image_measurement("PathName_my_image")
                self.assertEqual(os.path.join(directory, path), p)
        finally:
            clear_image_reader_cache()
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
        filenames = [("d1", "bar.tif"),
                     ("d1", "foo.tif"),
                     (os.path.join("d2", "d3"), "foo.tif"),
                     (os.path.join("d2", "d4"), "bar.tif"),
                     (os.path.join("d5", "d6", "d7"), "foo.tif")]
        exclusions = [os.path.join("d2", "d3"),
                      os.path.join("d5", "d6")]
        expected_filenames = filenames[:2] + [filenames[3]]

        data = base64.b64decode(T.tif_8_1)
        try:
            for path, file_name in filenames:
                d = os.path.join(directory, path)
                if not os.path.isdir(d):
                    os.makedirs(d)
                fd = open(os.path.join(directory, path, file_name), "wb")
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
            m = measurements.Measurements()
            workspace = W.Workspace(pipeline, load_images, None, None,
                                    m, image_set_list)
            self.assertTrue(load_images.prepare_run(workspace))
            image_numbers = m.get_image_numbers()
            self.assertEqual(len(image_numbers), len(expected_filenames))
            load_images.prepare_group(workspace, {}, image_numbers)
            for i, (path, file_name) in enumerate(expected_filenames):
                if i > 0:
                    m.next_image_set()
                image_set = image_set_list.get_image_set(i)
                w = W.Workspace(pipeline, load_images, image_set,
                                cpo.ObjectSet(), m, image_set_list)
                load_images.run(w)
                image = image_set.get_image("my_image")
                self.assertEqual(tuple(image.pixel_data.shape), (48, 32))
                f = m.get_current_image_measurement("FileName_my_image")
                self.assertEqual(f, file_name)
                p = m.get_current_image_measurement("PathName_my_image")
                self.assertEqual(os.path.join(directory, path), p)
        finally:
            clear_image_reader_cache()
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
                         ('Image', 'URL_DNA', 'varchar(256)'),
                         ('Image', 'MD5Digest_DNA', 'varchar(32)'),
                         ('Image', 'Scaling_DNA', 'float'),
                         ('Image', 'Metadata_WellRow', 'varchar(128)'),
                         ('Image', 'Metadata_WellCol', 'varchar(128)'),
                         ('Image', 'FileName_Cytoplasm', 'varchar(128)'),
                         ('Image', 'PathName_Cytoplasm', 'varchar(256)'),
                         ('Image', 'URL_Cytoplasm', 'varchar(256)'),
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
        categories = {'FileName': ['DNA', 'Cytoplasm'],
                      'PathName': ['DNA', 'Cytoplasm'],
                      'MD5Digest': ['DNA', 'Cytoplasm'],
                      'Metadata': ['WellRow', 'WellCol', 'Well']}
        for cat, expected in categories.items():
            assert set(expected) == set(module.get_measurements(pipeline,
                                                                measurements.IMAGE, cat))
        module.images[0].metadata_choice.value = LI.M_BOTH
        categories['Metadata'] += ['Year', 'Month', 'Day']
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
        expected = ['FileName', 'PathName', 'URL', 'MD5Digest', 'Metadata',
                    'Scaling', 'Height', 'Width']
        assert set(results) == set(expected)

    def test_07_04_get_movie_measurements(self):
        # AVI movies should have time metadata
        module = LI.LoadImages()
        base_expected_cols = [('Image', 'FileName_DNA', 'varchar(128)'),
                              ('Image', 'PathName_DNA', 'varchar(256)'),
                              ('Image', 'URL_DNA', 'varchar(256)'),
                              ('Image', 'MD5Digest_DNA', 'varchar(32)'),
                              ('Image', 'Scaling_DNA', 'float'),
                              ('Image', 'Height_DNA', 'integer'),
                              ('Image', 'Width_DNA', 'integer'),
                              ('Image', 'Metadata_T', 'integer'),
                              ('Image', 'Frame_DNA', 'integer')]
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
                self.assertEqual(len(categories), 9)
                category_dict = {}
                for column in expected_cols:
                    category, feature = column[1].split("_", 1)
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
                              ('Image', 'URL_DNA', 'varchar(256)'),
                              ('Image', 'MD5Digest_DNA', 'varchar(32)'),
                              ('Image', 'Scaling_DNA', 'float'),
                              ('Image', 'Metadata_T', 'integer'),
                              ('Image', 'Metadata_Z', 'integer'),
                              ('Image', 'Height_DNA', 'integer'),
                              ('Image', 'Width_DNA', 'integer'),
                              ('Image', 'Series_DNA', 'integer'),
                              ('Image', 'Frame_DNA', 'integer')]
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
            self.assertEqual(len(categories), 10)
            category_dict = {}
            for column in expected_cols:
                category, feature = column[1].split("_", 1)
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
                 (LI.C_OBJECTS_FILE_NAME, LI.C_OBJECTS_PATH_NAME,
                  LI.C_OBJECTS_URL, LI.I.C_COUNT)),
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
                (measurements.IMAGE, (
                        (LI.C_OBJECTS_FILE_NAME, [OBJECTS_NAME]),
                        (LI.C_OBJECTS_PATH_NAME, [OBJECTS_NAME]),
                        (LI.I.C_COUNT, [OBJECTS_NAME]))),
                (OBJECTS_NAME, (
                        (LI.I.C_LOCATION, [LI.I.FTR_CENTER_X, LI.I.FTR_CENTER_Y]),
                        (LI.I.C_NUMBER, [LI.I.FTR_OBJECT_NUMBER])))):
            for category, expected_features in expected:
                features = module.get_measurements(None, object_name, category)
                for feature in features:
                    self.assertTrue(feature in expected_features)
                for expected_feature in expected_features:
                    self.assertTrue(expected_feature in features)

    def test_08_01_get_groupings(self):
        '''Get groupings for the SBS image set'''
        sbs_path = os.path.join(T.example_images_directory(), 'ExampleSBSImages')
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
        image_set_list = I.ImageSetList()
        m = measurements.Measurements()
        workspace = W.Workspace(
                pipeline, None, None, None, m,
                image_set_list)
        self.assertTrue(pipeline.prepare_run(workspace))
        keys, groupings = module.get_groupings(workspace)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], "ROW")
        self.assertEqual(len(groupings), 8)
        self.assertTrue(all([g[0]["ROW"] == row for g, row in zip(groupings, 'ABCDEFGH')]))
        for grouping in groupings:
            row = grouping[0]["ROW"]
            module.prepare_group(workspace, grouping[0], grouping[1])
            for image_number in grouping[1]:
                image_set = image_set_list.get_image_set(image_number - 1)
                m.next_image_set(image_number)
                module.run(W.Workspace(pipeline, module, image_set,
                                       cpo.ObjectSet(), m, image_set_list))
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
        file_name = 'DrosophilaEmbryo_GFPHistone.avi'
        avi_path = T.testimages_directory()
        maybe_download_tesst_image(file_name)
        module = LI.LoadImages()
        module.file_types.value = LI.FF_AVI_MOVIES
        module.images[0].common_text.value = file_name
        module.images[0].channels[0].image_name.value = 'MyImage'
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = avi_path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None,
                                m, image_set_list)
        module.prepare_run(workspace)
        self.assertEqual(m.image_set_count, 65)
        module.prepare_group(workspace, (), [1, 2, 3])
        image_set = image_set_list.get_image_set(0)
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.names)
        image = image_set.get_image('MyImage')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (264, 542, 3))
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 0)
        image_set = image_set_list.get_image_set(1)
        m.next_image_set()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.names)
        image = image_set.get_image('MyImage')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (264, 542, 3))
        self.assertTrue(np.any(img1 != img2))
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 1)

    def test_09_02_load_stk(self):
        for path in [
            '//iodine/imaging_analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM',
            '/imaging/analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM',
            '/Volumes/imaging_analysis/2009_03_12_CellCycle_WolthuisLab_RobWolthuis/2009_09_19/Images/09_02_11-OA 10nM',
            os.path.join(T.example_images_directory(), '09_02_11-OA 10nM')]:
            if os.path.isdir(path):
                break
        else:
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
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        module.prepare_group(workspace, (), [1, 2, 3])
        image_set = image_set_list.get_image_set(0)
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.names)
        image = image_set.get_image('MyImage')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (1040, 1388))
        image_set = image_set_list.get_image_set(1)
        m.next_image_set(2)
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('MyImage' in image_set.names)
        image = image_set.get_image('MyImage')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (1040, 1388))
        self.assertTrue(np.any(img1 != img2))

    def test_09_02_01_load_2_stk(self):
        # Regression test of bug 327
        path = T.testimages_directory()
        maybe_download_tesst_image("C0.stk")
        maybe_download_tesst_image("C1.stk")
        files = [os.path.join(path, x) for x in ("C0.stk", "C1.stk")]
        if not all([os.path.exists(f) for f in files]):
            sys.stderr.write("Warning, could not find test files for STK test: %s\n" % str(files))
            return

        module = LI.LoadImages()
        module.file_types.value = LI.FF_STK_MOVIES
        module.images[0].common_text.value = 'C0.stk'
        module.images[0].channels[0].image_name.value = 'MyImage'
        module.add_imagecb()
        module.images[1].common_text.value = 'C1.stk'
        module.images[1].channels[0].image_name.value = 'MyOtherImage'

        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        self.assertEqual(m.image_set_count, 7)

    def test_09_02_02_load_stk(self):
        # Regression test of issue #783 - color STK.
        path = T.testimages_directory()
        maybe_download_tesst_image("C0.stk")
        maybe_download_tesst_image("C1.stk")
        module = LI.LoadImages()
        module.file_types.value = LI.FF_STK_MOVIES
        module.images[0].common_text.value = 'C0.stk'
        module.images[0].channels[0].image_name.value = 'MyImage'
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = path
        module.module_num = 1
        pipeline = P.Pipeline()
        pipeline.add_module(module)
        pipeline.add_listener(self.error_callback)
        image_set_list = I.ImageSetList()
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        workspace = W.Workspace(pipeline, module, m, cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        image = m.get_image("MyImage")
        pixel_data = image.pixel_data
        self.assertEqual(tuple(pixel_data.shape), (800, 800, 3))

    def test_09_03_load_flex(self):
        file_name = 'RLM1 SSN3 300308 008015000.flex'
        maybe_download_tesst_image(file_name)
        flex_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_OTHER_MOVIES
        module.images[0].common_text.value = file_name
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
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        keys, groupings = module.get_groupings(workspace)
        self.assertTrue("URL" in keys)
        self.assertTrue("Series" in keys)
        self.assertEqual(len(groupings), 4)
        for grouping, image_numbers in groupings:
            module.prepare_group(workspace, grouping, image_numbers)
            for image_number in image_numbers:
                image_set = image_set_list.get_image_set(image_number - 1)
                workspace = W.Workspace(pipeline, module, image_set,
                                        cpo.ObjectSet(), m,
                                        image_set_list)
                module.run(workspace)
                for feature, expected in (("Series_Green", int(grouping[LI.C_SERIES])),
                                          ("Metadata_Z", 0),
                                          ("Metadata_T", 0)):
                    value = m.get_current_image_measurement(feature)
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
        file_name = 'DrosophilaEmbryo_GFPHistone.avi'
        maybe_download_tesst_image(file_name)
        avi_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_AVI_MOVIES
        image = module.images[0]
        image.common_text.value = file_name
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
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        self.assertEqual(m.image_set_count, 13)
        module.prepare_group(workspace, (), np.arange(1, 16))
        image_set = image_set_list.get_image_set(0)
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.names)
        image = image_set.get_image('Channel01')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (264, 542, 3))
        self.assertAlmostEqual(np.mean(img1), .07897, 3)
        self.assertTrue('Channel03' in image_set.names)
        self.assertEqual(m.get_current_image_measurement("Frame_Channel03"), 2)
        image = image_set.get_image('Channel03')
        img3 = image.pixel_data
        self.assertEqual(tuple(img3.shape), (264, 542, 3))
        self.assertAlmostEqual(np.mean(img3), .07781, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 0)
        image_set = image_set_list.get_image_set(1)
        m.next_image_set()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.names)
        image = image_set.get_image('Channel01')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (264, 542, 3))
        self.assertAlmostEqual(np.mean(img2), .07860, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 1)
        self.assertEqual(m.get_current_image_measurement("Frame_Channel03"), 7)

    def test_09_05_group_separated_avi_frames(self):
        #
        # Test separated grouping by movie frames
        #
        if LI.FF_AVI_MOVIES not in LI.FF:
            sys.stderr.write("WARNING: AVI movies not supported\n")
            return
        file_name = 'DrosophilaEmbryo_GFPHistone.avi'
        maybe_download_tesst_image(file_name)
        avi_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_AVI_MOVIES
        image = module.images[0]
        image.common_text.value = file_name
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
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m, image_set_list)
        module.prepare_run(workspace)
        self.assertEqual(m.image_set_count, 13)
        module.prepare_group(workspace, (), np.arange(1, 16))
        image_set = image_set_list.get_image_set(0)
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.names)
        image = image_set.get_image('Channel01')
        img1 = image.pixel_data
        self.assertEqual(tuple(img1.shape), (264, 542, 3))
        self.assertAlmostEqual(np.mean(img1), .07897, 3)
        self.assertEqual(m.get_current_image_measurement("Frame_Channel03"), 26)
        image = image_set.get_image('Channel03')
        img3 = image.pixel_data
        self.assertEqual(tuple(img3.shape), (264, 542, 3))
        self.assertAlmostEqual(np.mean(img3), .073312, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 0)
        image_set = image_set_list.get_image_set(1)
        m.next_image_set()
        workspace = W.Workspace(pipeline, module, image_set,
                                cpo.ObjectSet(), m,
                                image_set_list)
        module.run(workspace)
        self.assertTrue('Channel01' in image_set.names)
        self.assertEqual(m.get_current_image_measurement("Frame_Channel01"), 1)
        image = image_set.get_image('Channel01')
        img2 = image.pixel_data
        self.assertEqual(tuple(img2.shape), (264, 542, 3))
        self.assertAlmostEqual(np.mean(img2), .079923, 3)
        t = m.get_current_image_measurement("_".join((measurements.C_METADATA, LI.M_T)))
        self.assertEqual(t, 1)
        self.assertEqual(m.get_current_image_measurement("Frame_Channel03"), 27)

    def test_09_06_load_flex_interleaved(self):
        # needs better test case file
        file_name = 'RLM1 SSN3 300308 008015000.flex'
        maybe_download_tesst_image(file_name)
        flex_path = T.testimages_directory()
        module = LI.LoadImages()
        module.file_types.value = LI.FF_OTHER_MOVIES
        module.images[0].common_text.value = file_name
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
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        keys, groupings = module.get_groupings(workspace)
        self.assertTrue("URL" in keys)
        self.assertTrue("Series" in keys)
        self.assertEqual(len(groupings), 4)
        for group_number, (grouping, image_numbers) in enumerate(groupings):
            module.prepare_group(workspace, grouping, image_numbers)
            for group_index, image_number in enumerate(image_numbers):
                image_set = image_set_list.get_image_set(image_number - 1)
                m.add_image_measurement(cpp.GROUP_INDEX, group_index)
                m.add_image_measurement(cpp.GROUP_NUMBER, group_number)
                workspace = W.Workspace(pipeline, module, image_set,
                                        cpo.ObjectSet(), m,
                                        image_set_list)
                module.run(workspace)
                for feature, expected in (("Series_Green", int(grouping[LI.C_SERIES])),
                                          ("Series_Red", int(grouping[LI.C_SERIES])),
                                          ("Frame_Red", group_index * 2),
                                          ("Frame_Green", group_index * 2 + 1),
                                          ("Metadata_Z", 0),
                                          ("Metadata_T", 0)):
                    value = m.get_current_image_measurement(feature)
                    self.assertEqual(value, expected)

                red_image = image_set.get_image("Red")
                green_image = image_set.get_image("Green")
                self.assertEqual(tuple(red_image.pixel_data.shape),
                                 tuple(green_image.pixel_data.shape))
                m.next_image_set()

    def test_09_07_load_flex_separated(self):
        # Needs better test case file
        flex_path = T.testimages_directory()
        file_name = 'RLM1 SSN3 300308 008015000.flex'
        maybe_download_tesst_image(file_name)
        module = LI.LoadImages()
        module.file_types.value = LI.FF_OTHER_MOVIES
        module.images[0].common_text.value = file_name
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
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        keys, groupings = module.get_groupings(workspace)
        self.assertTrue("URL" in keys)
        self.assertTrue("Series" in keys)
        self.assertEqual(len(groupings), 4)
        for group_number, (grouping, image_numbers) in enumerate(groupings):
            module.prepare_group(workspace, grouping, image_numbers)

            for group_index, image_number in enumerate(image_numbers):
                image_set = image_set_list.get_image_set(image_number - 1)
                workspace = W.Workspace(pipeline, module, image_set,
                                        cpo.ObjectSet(), m,
                                        image_set_list)
                m.add_image_measurement(cpp.GROUP_INDEX, group_index)
                m.add_image_measurement(cpp.GROUP_NUMBER, group_number)
                module.run(workspace)
                for feature, expected in (("Series_Red", int(grouping[LI.C_SERIES])),
                                          ("Series_Green", int(grouping[LI.C_SERIES])),
                                          ("Frame_Red", 0),
                                          ("Frame_Green", 1),
                                          ("Metadata_Z", 0),
                                          ("Metadata_T", 0)):
                    value = m.get_current_image_measurement(feature)
                    self.assertEqual(value, expected)

                red_image = image_set.get_image("Red")
                green_image = image_set.get_image("Green")
                self.assertEqual(tuple(red_image.pixel_data.shape),
                                 tuple(green_image.pixel_data.shape))
                m.next_image_set()

    # def test_10_01_load_unscaled(self):
    #     '''Load a image with and without rescaling'''
    #     make_12_bit_image('ExampleSpecklesImages', '1-162hrh2ax2.tif', (21, 31))
    #     path = os.path.join(example_images_directory(),
    #                         "ExampleSpecklesImages")
    #     module = LI.LoadImages()
    #     module.file_types.value = LI.FF_INDIVIDUAL_IMAGES
    #     module.images[0].common_text.value = '1-162hrh2ax2'
    #     module.images[0].channels[0].image_name.value = 'MyImage'
    #     module.images[0].channels[0].rescale.value = False
    #     module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
    #     module.location.custom_path = path
    #     module.module_num = 1
    #     pipeline = P.Pipeline()
    #     pipeline.add_module(module)
    #     pipeline.add_listener(self.error_callback)
    #     image_set_list = I.ImageSetList()
    #     m = measurements.Measurements()
    #     workspace = W.Workspace(pipeline, module, None, None, m,
    #                             image_set_list)
    #     module.prepare_run(workspace)
    #     module.prepare_group(workspace, (), [1])
    #     image_set = image_set_list.get_image_set(0)
    #     workspace = W.Workspace(pipeline, module, image_set,
    #                             cpo.ObjectSet(), m,
    #                             image_set_list)
    #     module.run(workspace)
    #     scales = m.get_all_measurements(measurements.IMAGE,
    #                                     LI.C_SCALING + "_MyImage")
    #     self.assertEqual(len(scales), 1)
    #     self.assertEqual(scales[0], 4095)
    #     image = image_set.get_image("MyImage")
    #     self.assertTrue(np.all(image.pixel_data <= 1.0 / 16.0))
    #     pixel_data = image.pixel_data
    #     module.images[0].channels[0].rescale.value = True
    #     image_set_list = I.ImageSetList()
    #     workspace = W.Workspace(pipeline, module, None, None, m,
    #                             image_set_list)
    #     module.prepare_run(workspace)
    #     module.prepare_group(workspace, (), [1])
    #     image_set = image_set_list.get_image_set(0)
    #     workspace = W.Workspace(pipeline, module, image_set,
    #                             cpo.ObjectSet(), m,
    #                             image_set_list)
    #     module.run(workspace)
    #     image = image_set.get_image("MyImage")
    #     np.testing.assert_almost_equal(pixel_data * 65535.0 / 4095.0,
    #                                    image.pixel_data)

    def make_objects_workspace(self, image, mode="L", filename="myfile.tif"):
        directory = tempfile.mkdtemp()
        self.directory = directory
        path = os.path.join(directory, filename)
        if mode == "raw":
            fd = open(path, "wb")
            fd.write(image)
            fd.close()
        else:
            write_image(path, image.astype(np.uint8), PT_UINT8)
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
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)
        module.prepare_group(workspace, (), [0])
        workspace = W.Workspace(pipeline, module, image_set_list.get_image_set(0),
                                cpo.ObjectSet(), m,
                                image_set_list)
        return workspace, module

    def test_12_01_load_empty_objects(self):
        workspace, module = self.make_objects_workspace(np.zeros((20, 30), int))
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
        self.assertEqual(m.get_current_image_measurement(LI.I.FF_COUNT % OBJECTS_NAME), 0)

    def test_12_02_load_indexed_objects(self):
        r = np.random.RandomState()
        r.seed(1202)
        image = r.randint(0, 10, size=(20, 30))
        workspace, module = self.make_objects_workspace(image)
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))
        m = workspace.measurements
        assert isinstance(m, measurements.Measurements)
        self.assertEqual(m.get_current_image_measurement(LI.I.FF_COUNT % OBJECTS_NAME), 9)
        i, j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        c = np.bincount(image.ravel())[1:].astype(float)
        x = np.bincount(image.ravel(), j.ravel())[1:].astype(float) / c
        y = np.bincount(image.ravel(), i.ravel())[1:].astype(float) / c
        v = m.get_current_measurement(OBJECTS_NAME, LI.I.M_NUMBER_OBJECT_NUMBER)
        self.assertTrue(np.all(v == np.arange(1, 10)))
        v = m.get_current_measurement(OBJECTS_NAME, LI.I.M_LOCATION_CENTER_X)
        self.assertTrue(np.all(v == x))
        v = m.get_current_measurement(OBJECTS_NAME, LI.I.M_LOCATION_CENTER_Y)
        self.assertTrue(np.all(v == y))

    def test_12_03_load_sparse_objects(self):
        r = np.random.RandomState()
        r.seed(1203)
        image = r.randint(0, 10, size=(20, 30))
        workspace, module = self.make_objects_workspace(image * 10)
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))

    def test_12_04_load_color_objects(self):
        r = np.random.RandomState()
        r.seed(1203)
        image = r.randint(0, 10, size=(20, 30))
        colors = np.array([[0, 0, 0], [1, 4, 2], [1, 5, 0],
                           [2, 0, 0], [3, 0, 0], [4, 0, 0],
                           [5, 0, 0], [6, 0, 0], [7, 0, 0],
                           [8, 0, 0], [9, 0, 0]])
        cimage = colors[image]
        workspace, module = self.make_objects_workspace(cimage, mode="RGB",
                                                        filename="myimage.png")
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))

    def test_12_05_object_outlines(self):
        image = np.zeros((30, 40), int)
        image[10:15, 20:30] = 1
        workspace, module = self.make_objects_workspace(image)
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertTrue(np.all(o.segmented == image))
        expected_outlines = image != 0
        expected_outlines[11:14, 21:29] = False
        image_set = workspace.get_image_set()
        outlines = image_set.get_image(OUTLINES_NAME)
        np.testing.assert_equal(outlines.pixel_data, expected_outlines)

    def test_12_06_overlapped_objects(self):
        workspace, module = self.make_objects_workspace(
                overlapped_objects_data, mode="raw")
        module.run(workspace)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        self.assertEqual(o.count, 2)
        labels_and_indices = o.get_labels()
        for n, mask in enumerate(overlapped_objects_data_masks):
            object_number = n + 1
            for label, idx in labels_and_indices:
                if object_number in idx:
                    np.testing.assert_array_equal(
                            label, mask.astype(label.dtype) * object_number)
                    break
            else:
                assert "Object number %d not found" % object_number

    def test_13_01_batch_images(self):
        module = LI.LoadImages()
        module.match_method.value = LI.MS_REGEXP
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        orig_path = os.path.join(T.example_images_directory(), "ExampleSBSImages")
        module.location.custom_path = orig_path
        target_path = orig_path.replace("ExampleSBSImages", "ExampleTrackObjects")
        url_path = LI.url2pathname(LI.pathname2url(orig_path))

        file_regexp = "^Channel1-[0-9]{2}-[A-P]-[0-9]{2}.tif$"
        module.images[0].common_text.value = file_regexp
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(module)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m, image_set_list)
        module.prepare_run(workspace)

        def fn_alter_path(pathname, **varargs):
            is_path = (pathname == orig_path)
            is_file = re.match(file_regexp, pathname) is not None
            if not (is_path or is_file):
                self.assertTrue(pathname.startswith(url_path))
                file_part = pathname[(len(url_path) + 1):]
                self.assertTrue(re.match(file_regexp, file_part) is not None)
                return os.path.join(target_path, file_part)
            elif is_path:
                return target_path
            else:
                return pathname

        module.prepare_to_create_batch(workspace, fn_alter_path)
        key_names, group_list = pipeline.get_groupings(workspace)
        self.assertEqual(len(group_list), 1)
        group_keys, image_numbers = group_list[0]
        self.assertEqual(len(image_numbers), 96)
        module.prepare_group(workspace, group_keys, image_numbers)
        for image_number in image_numbers:
            path = m.get_measurement(measurements.IMAGE,
                                     LI.C_PATH_NAME + "_" + IMAGE_NAME,
                                     image_set_number=image_number)
            self.assertEqual(path, target_path)
            file_name = m.get_measurement(measurements.IMAGE,
                                          LI.C_FILE_NAME + "_" + IMAGE_NAME,
                                          image_set_number=image_number)
            self.assertTrue(re.match(file_regexp, file_name) is not None)
            url = m.get_measurement(measurements.IMAGE,
                                    LI.C_URL + "_" + IMAGE_NAME,
                                    image_set_number=image_number)
            self.assertEqual(url, LI.pathname2url(
                    os.path.join(path, file_name)))

    def test_13_02_batch_movies(self):
        module = LI.LoadImages()
        module.match_method.value = LI.MS_EXACT_MATCH
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.file_types.value = LI.FF_AVI_MOVIES
        orig_path = T.testimages_directory()
        module.location.custom_path = orig_path
        target_path = os.path.join(orig_path, "Images")
        orig_url = LI.pathname2url(orig_path)
        # Can switch cases in Windows.
        orig_url_path = LI.url2pathname(orig_url)

        file_name = "DrosophilaEmbryo_GFPHistone.avi"
        maybe_download_tesst_image(file_name)
        target_url = LI.pathname2url(os.path.join(target_path, file_name))
        module.images[0].common_text.value = file_name
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.module_num = 1
        image_set_list = I.ImageSetList()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(module)
        m = measurements.Measurements()
        workspace = W.Workspace(pipeline, module, None, None, m, image_set_list)
        module.prepare_run(workspace)

        def fn_alter_path(pathname, **varargs):
            is_fullpath = (os.path.join(orig_path, file_name) == pathname)
            is_path = (orig_path == pathname)
            is_file = (pathname == file_name)
            if not (is_fullpath or is_path or is_file):
                self.assertTrue(
                        pathname.startswith(orig_url_path),
                        """Expected pathname = "%s" to start with "%s".""" %
                        (pathname, orig_url))
                self.assertEqual(pathname[(len(orig_url_path) + 1):], file_name)
                return target_path
            elif is_file:
                return pathname
            if is_fullpath:
                return os.path.join(target_path, file_name)
            else:
                return target_path

        module.prepare_to_create_batch(workspace, fn_alter_path)
        key_names, group_list = pipeline.get_groupings(workspace)
        self.assertEqual(len(group_list), 1)
        group_keys, image_numbers = group_list[0]
        self.assertEqual(len(image_numbers), 65)
        module.prepare_group(workspace, group_keys, image_numbers)
        for image_number in image_numbers:
            self.assertEqual(m.get_measurement(
                    measurements.IMAGE, "PathName_" + IMAGE_NAME,
                    image_set_number=image_number), target_path)
            self.assertEqual(m.get_measurement(
                    measurements.IMAGE, "Metadata_T",
                    image_set_number=image_number), image_number - 1)

    def test_13_03_batch_flex(self):
        module = LI.LoadImages()
        module.match_method.value = LI.MS_EXACT_MATCH
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.file_types.value = LI.FF_OTHER_MOVIES
        orig_path = T.testimages_directory()
        orig_url = LI.pathname2url(orig_path)
        module.location.custom_path = orig_path
        target_path = os.path.join(orig_path, "Images")
        # Can switch cases in Windows.
        orig_url_path = LI.url2pathname(orig_url)

        file_name = "RLM1 SSN3 300308 008015000.flex"
        maybe_download_tesst_image(file_name)
        target_url = LI.pathname2url(os.path.join(orig_path, file_name))
        module.images[0].common_text.value = file_name
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.module_num = 1
        image_set_list = I.ImageSetList()
        m = measurements.Measurements()
        pipeline = P.Pipeline()
        pipeline.add_listener(self.error_callback)
        pipeline.add_module(module)
        workspace = W.Workspace(pipeline, module, None, None, m,
                                image_set_list)
        module.prepare_run(workspace)

        def fn_alter_path(pathname, **varargs):
            is_fullpath = (os.path.join(orig_path, file_name) == pathname)
            is_path = (orig_path == pathname)
            is_file = pathname == file_name
            if not (is_fullpath or is_path or is_file):
                self.assertTrue(
                        pathname.startswith(orig_url_path),
                        """Expected pathname = "%s" to start with "%s".""" %
                        (pathname, orig_url))
                self.assertEqual(pathname[(len(orig_url_path) + 1):], file_name)
                return target_path
            if is_fullpath:
                return os.path.join(target_path, file_name)
            else:
                return target_path

        module.prepare_to_create_batch(workspace, fn_alter_path)
        key_names, group_list = pipeline.get_groupings(workspace)
        self.assertEqual(len(group_list), 4)
        for i, (group_keys, image_numbers) in enumerate(group_list):
            self.assertEqual(len(image_numbers), 1)
            module.prepare_group(workspace, group_keys, image_numbers)
            for image_number in image_numbers:
                self.assertEqual(m.get_measurement(
                        measurements.IMAGE,
                        "PathName_" + IMAGE_NAME, image_number), target_path)
                self.assertEqual(m.get_measurement(
                        measurements.IMAGE, LI.C_SERIES + "_" + IMAGE_NAME,
                        image_number), i)

    # def test_14_01_load_unicode(self):
    #     '''Load an image from a unicode - encoded location'''
    #     self.directory = tempfile.mkdtemp()
    #     directory = os.path.join(self.directory, u"\u2211a")
    #     os.mkdir(directory)
    #     filename = u"\u03b1\u00b2.jpg"
    #     path = os.path.join(directory, filename)
    #     data = base64.b64decode(T.jpg_8_1)
    #     fd = open(path, 'wb')
    #     fd.write(data)
    #     fd.close()
    #     module = LI.LoadImages()
    #     module.module_num = 1
    #     module.match_method.value = LI.MS_EXACT_MATCH
    #     module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
    #     module.location.custom_path = directory
    #     module.images[0].common_text.value = ".jpg"
    #     module.images[0].channels[0].image_name.value = IMAGE_NAME
    #     image_set_list = I.ImageSetList()
    #     pipeline = cpp.Pipeline()
    #
    #     def callback(caller, event):
    #         self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
    #
    #     pipeline.add_listener(callback)
    #     pipeline.add_module(module)
    #     m = measurements.Measurements()
    #     workspace = W.Workspace(pipeline, module, None, None, m, image_set_list)
    #     self.assertTrue(module.prepare_run(workspace))
    #     self.assertEqual(len(m.get_image_numbers()), 1)
    #     key_names, group_list = pipeline.get_groupings(workspace)
    #     self.assertEqual(len(group_list), 1)
    #     group_keys, image_numbers = group_list[0]
    #     self.assertEqual(len(image_numbers), 1)
    #     module.prepare_group(workspace, group_keys, image_numbers)
    #     image_set = image_set_list.get_image_set(image_numbers[0] - 1)
    #     workspace = W.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
    #                             m, image_set_list)
    #     module.run(workspace)
    #     image_provider = image_set.get_image_provider(IMAGE_NAME)
    #     self.assertEqual(image_provider.get_filename(), filename)
    #     pixel_data = image_set.get_image(IMAGE_NAME).pixel_data
    #     self.assertEqual(tuple(pixel_data.shape[:2]), tuple(T.raw_8_1_shape))
    #     file_feature = '_'.join((LI.C_FILE_NAME, IMAGE_NAME))
    #     file_measurement = m.get_current_image_measurement(file_feature)
    #     self.assertEqual(file_measurement, filename)
    #     path_feature = '_'.join((LI.C_PATH_NAME, IMAGE_NAME))
    #     path_measurement = m.get_current_image_measurement(path_feature)
    #     self.assertEqual(os.path.split(directory)[1],
    #                      os.path.split(path_measurement)[1])

    def make_prepare_run_workspace(self, file_names):
        '''Make a workspace and image files for prepare_run

        file_names - a list of file names of files to create in self.directory

        returns tuple of workspace and module
        '''
        self.directory = tempfile.mkdtemp()
        data = base64.b64decode(T.png_8_1)
        for file_name in file_names:
            path = os.path.join(self.directory, file_name)
            fd = open(path, "wb")
            fd.write(data)
            fd.close()

        module = LI.LoadImages()
        module.module_num = 1
        module.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
        module.location.custom_path = self.directory

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, (
                cpp.LoadExceptionEvent, cpp.RunExceptionEvent)))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        m = measurements.Measurements()
        image_set_list = I.ImageSetList()
        return (W.Workspace(pipeline, module, None, None, m, image_set_list),
                module)

    def test_15_01_prepare_run_measurements(self):
        filenames = ["channel1-A01.png", "channel2-A01.png",
                     "channel1-A02.png", "channel2-A02.png"]
        workspace, module = self.make_prepare_run_workspace(filenames)
        self.assertTrue(isinstance(module, LI.LoadImages))
        module.match_method.value = LI.MS_EXACT_MATCH
        module.add_imagecb()
        module.images[0].common_text.value = "channel1-"
        module.images[1].common_text.value = "channel2-"
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.images[1].channels[0].image_name.value = ALT_IMAGE_NAME
        self.assertTrue(module.prepare_run(workspace))

        m = workspace.measurements
        self.assertTrue(isinstance(m, measurements.Measurements))
        for i in range(1, 3):
            for j, image_name in ((1, IMAGE_NAME), (2, ALT_IMAGE_NAME)):
                filename = "channel%d-A0%d.png" % (j, i)
                full_path = os.path.join(self.directory, filename)
                url = "file:" + urllib.pathname2url(full_path)
                for category, expected in (
                        (LI.C_FILE_NAME, filename),
                        (LI.C_PATH_NAME, self.directory),
                        (LI.C_URL, url)):
                    value = m.get_measurement(measurements.IMAGE,
                                              "_".join((category, image_name)),
                                              i)
                    self.assertEqual(value, expected)

    @unittest.expectedFailure  # fly image URLs have moved
    def test_16_01_00_load_from_url(self):
        from bioformats.formatreader import release_image_reader

        module = LI.LoadImages()
        module.module_num = 1
        module.location.dir_choice = LI.URL_FOLDER_NAME
        url_base = "http://www.cellprofiler.org/ExampleFlyImages"
        module.location.custom_path = url_base
        module.match_method.value = LI.MS_EXACT_MATCH
        module.add_imagecb()
        module.images[0].common_text.value = "_D.TIF"
        module.images[0].channels[0].image_name.value = IMAGE_NAME
        module.images[1].common_text.value = "_F.TIF"
        module.images[1].channels[0].image_name.value = ALT_IMAGE_NAME

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, (
                cpp.LoadExceptionEvent, cpp.RunExceptionEvent)))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        m = measurements.Measurements()
        image_set_list = I.ImageSetList()
        workspace = W.Workspace(pipeline, module, None, None, m, image_set_list)
        self.assertTrue(module.prepare_run(workspace))
        image_numbers = m.get_image_numbers()
        self.assertEqual(len(image_numbers), 3)
        names = (("01_POS002_D.TIF", "01_POS002_F.TIF"),
                 ("01_POS076_D.TIF", "01_POS076_F.TIF"),
                 ("01_POS218_D.TIF", "01_POS218_F.TIF"))
        for image_number, (filename, alt_filename) in zip(image_numbers, names):
            url = m.get_measurement(measurements.IMAGE,
                                    LI.C_URL + "_" + IMAGE_NAME,
                                    image_set_number=image_number)
            expected = url_base + "/" + filename
            self.assertEqual(expected, url)
            url = m.get_measurement(measurements.IMAGE,
                                    LI.C_URL + "_" + ALT_IMAGE_NAME,
                                    image_set_number=image_number)
            expected = url_base + "/" + alt_filename
            self.assertEqual(expected, url)
        image_set = image_set_list.get_image_set(0)
        module.run(W.Workspace(pipeline, module, image_set,
                               cpo.ObjectSet(), m, image_set_list))
        image = image_set.get_image(IMAGE_NAME, must_be_grayscale=True)
        self.assertEqual(tuple(image.pixel_data.shape), (1006, 1000))
        #
        # Make sure the file on disk goes away
        #
        provider = image_set.get_image_provider(IMAGE_NAME)
        pathname = provider.get_full_name()
        self.assertTrue(os.path.exists(pathname))
        provider.release_memory()
        release_image_reader(IMAGE_NAME)
        self.assertFalse(os.path.exists(pathname))

        # def test_16_01_01_load_url_mat(self):
        #     #
        #     # Unfortunately, MAT files end up in temporary files using a different
        #     # mechanism than everything else
        #     #
        #     image_provider = LI.LoadImagesImageProviderURL(
        #         IMAGE_NAME,
        #         example_images_url() + "/ExampleSBSImages/Channel1ILLUM.mat?r11710")
        #     pathname = image_provider.get_full_name()
        #     image = image_provider.provide_image(None)
        #     self.assertEqual(tuple(image.pixel_data.shape), (640, 640))
        #     expected_md5 = "f3c4d57ee62fa2fd96e3686179656d82"
        #     md5 = image_provider.get_md5_hash(None)
        #     self.assertEqual(expected_md5, md5)
        #     image_provider.release_memory()
        #     self.assertFalse(os.path.exists(pathname))

        # def test_16_02_load_url_with_groups(self):
        #     module = LI.LoadImages()
        #     module.module_num = 1
        #     module.location.dir_choice = LI.URL_FOLDER_NAME
        #     url_base = "http://www.cellprofiler.org/ExampleFlyImages"
        #     module.location.custom_path = url_base
        #     module.group_by_metadata.value = True
        #     module.metadata_fields.set_value("Column")
        #     module.match_method.value = LI.MS_EXACT_MATCH
        #     module.add_imagecb()
        #     module.images[0].common_text.value = "_D.TIF"
        #     module.images[0].channels[0].image_name.value = IMAGE_NAME
        #     module.images[0].metadata_choice.value = LI.M_FILE_NAME
        #     module.images[0].file_metadata.value = \
        #           "^01_POS(?P<Column>[0-9])(?P<Row>[0-9]{2})_[DF].TIF$"
        #     module.images[1].common_text.value = "_F.TIF"
        #     module.images[1].channels[0].image_name.value = ALT_IMAGE_NAME
        #     module.images[1].metadata_choice.value = LI.M_FILE_NAME
        #     module.images[1].file_metadata.value = \
        #           "^01_POS(?P<Column>[0-9])(?P<Row>[0-9]{2})_[DF].TIF$"
        #
        #     pipeline = cpp.Pipeline()
        #     def callback(caller, event):
        #         self.assertFalse(isinstance(event, (
        #             cpp.LoadExceptionEvent, cpp.RunExceptionEvent)))
        #     pipeline.add_listener(callback)
        #     pipeline.add_module(module)
        #
        #     m = measurements.Measurements()
        #     image_set_list = I.ImageSetList()
        #     workspace = W.Workspace(pipeline, module, None, None, m, image_set_list)
        #     self.assertTrue(module.prepare_run(workspace))
        #     image_numbers = m.get_image_numbers()
        #     self.assertEqual(len(image_numbers), 3)
        #
        #     key_names, group_list = module.get_groupings(workspace)
        #     self.assertEqual(len(key_names), 1)
        #     self.assertEqual(key_names[0], "Column")
        #     self.assertEqual(len(group_list), 2)
        #     self.assertEqual(group_list[0][0]["Column"], "0")
        #     self.assertEqual(len(group_list[0][1]), 2)
        #     self.assertEqual(tuple(group_list[0][1]), tuple(image_numbers[:2]))
        #     self.assertEqual(group_list[1][0]["Column"], "2")
        #     self.assertEqual(len(group_list[1][1]), 1)
        #     self.assertEqual(group_list[1][1][0], image_numbers[-1])

    #     def test_17_01_single_channel(self):
    #         pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
    # Version:3
    # DateRevision:20120830205040
    # ModuleCount:1
    # HasImagePlaneDetails:False
    #
    # LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\'Load the images by matching files in the folder against the unique text pattern for each stain\x3A D.TIF for DAPI, F.TIF for the FITC image, R.TIF for the rhodamine image. The three images together comprise an image set.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    #     File type to be loaded:individual images
    #     File selection method:Text-Exact match
    #     Number of images in each group?:3
    #     Type the text that the excluded images have in common:Do not use
    #     Analyze all subfolders within the selected folder?:None
    #     Input image file location:Default Input Folder\x7C.
    #     Check image sets for unmatched or duplicate files?:No
    #     Group images by metadata?:No
    #     Exclude certain files?:No
    #     Specify metadata fields to group by:
    #     Select subfolders to analyze:
    #     Image count:1
    #     Text that these images have in common (case-sensitive):D.TIF
    #     Position of this image in each group:D.TIF
    #     Extract metadata from where?:None
    #     Regular expression that finds metadata in the file name:None
    #     Type the regular expression that finds metadata in the subfolder path:None
    #     Channel count:1
    #     Group the movie frames?:No
    #     Grouping method:Interleaved
    #     Number of channels per group:2
    #     Load the input as images or objects?:Images
    #     Name this loaded image:OrigBlue
    #     Name this loaded object:Nuclei
    #     Retain outlines of loaded objects?:No
    #     Name the outline image:NucleiOutlines
    #     Channel number:1
    #     Rescale intensities?:Yes
    # """
    #         maybe_download_fly()
    #         directory = os.path.join(example_images_directory(),
    #                                  "ExampleFlyImages")
    #         self.convtester(pipeline_text, directory)

    def test_17_02_three_channels(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120830205040
ModuleCount:1
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\'Load the images by matching files in the folder against the unique text pattern for each stain\x3A D.TIF for DAPI, F.TIF for the FITC image, R.TIF for the rhodamine image. The three images together comprise an image set.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C.
    Check image sets for unmatched or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:3
    Text that these images have in common (case-sensitive):D.TIF
    Position of this image in each group:D.TIF
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:None
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:DAPI
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):F.TIF
    Position of this image in each group:2
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:FITC
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):R.TIF
    Position of this image in each group:3
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:Rhodamine
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        maybe_download_fly()
        directory = os.path.join(example_images_directory(),
                                 "ExampleFlyImages")
        self.convtester(pipeline_text, directory)

    def test_17_03_regexp(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120830205040
ModuleCount:1
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\'Load the images by matching files in the folder against the unique text pattern for each stain\x3A D.TIF for DAPI, F.TIF for the FITC image, R.TIF for the rhodamine image. The three images together comprise an image set.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Regular expressions
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C.
    Check image sets for unmatched or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:3
    Text that these images have in common (case-sensitive):.*?D.TIF
    Position of this image in each group:D.TIF
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:None
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:DAPI
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):.*?F.TIF
    Position of this image in each group:2
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:FITC
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):.*?R.TIF
    Position of this image in each group:3
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:Rhodamine
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        maybe_download_fly()
        directory = os.path.join(example_images_directory(),
                                 "ExampleFlyImages")
        self.convtester(pipeline_text, directory)

    def test_17_04_order_by_metadata(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120830205040
ModuleCount:1
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\'Load the images by matching files in the folder against the unique text pattern for each stain\x3A D.TIF for DAPI, F.TIF for the FITC image, R.TIF for the rhodamine image. The three images together comprise an image set.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C.
    Check image sets for unmatched or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:3
    Text that these images have in common (case-sensitive):D.TIF
    Position of this image in each group:D.TIF
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:^(?P<field>.+?)_\x5BDFR\x5D\\\\.TIF$
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:DAPI
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):F.TIF
    Position of this image in each group:2
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:^(?P<field>.+?)_\x5BDFR\x5D\\\\.TIF$
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:FITC
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):R.TIF
    Position of this image in each group:3
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:^(?P<field>.+?)_\x5BDFR\x5D\\\\.TIF$
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:Rhodamine
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        maybe_download_fly()
        directory = os.path.join(example_images_directory(),
                                 "ExampleFlyImages")
        self.convtester(pipeline_text, directory)

    def test_17_05_directory_metadata(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120830205040
ModuleCount:1
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\'Load the images by matching files in the folder against the unique text pattern for each stain\x3A D.TIF for DAPI, F.TIF for the FITC image, R.TIF for the rhodamine image. The three images together comprise an image set.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C.
    Check image sets for unmatched or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:3
    Text that these images have in common (case-sensitive):D.TIF
    Position of this image in each group:D.TIF
    Extract metadata from where?:Both
    Regular expression that finds metadata in the file name:^(?P<field>.+?)_\x5BDFR\x5D\\\\.TIF$
    Type the regular expression that finds metadata in the subfolder path:Example(?P<fly>\x5B^I\x5D+?)Images
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:DAPI
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):F.TIF
    Position of this image in each group:2
    Extract metadata from where?:Both
    Regular expression that finds metadata in the file name:^(?P<field>.+?)_\x5BDFR\x5D\\\\.TIF$
    Type the regular expression that finds metadata in the subfolder path:Example(?P<fly>\x5B^I\x5D+?)Images
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:FITC
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):R.TIF
    Position of this image in each group:3
    Extract metadata from where?:Both
    Regular expression that finds metadata in the file name:^(?P<field>.+?)_\x5BDFR\x5D\\\\.TIF$
    Type the regular expression that finds metadata in the subfolder path:Example(?P<fly>\x5B^I\x5D+?)Images
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:Rhodamine
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        maybe_download_fly()
        directory = os.path.join(example_images_directory(),
                                 "ExampleFlyImages")
        self.convtester(pipeline_text, directory)

    def test_17_06_objects(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120830205040
ModuleCount:1
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B\'Load the images by matching files in the folder against the unique text pattern for each stain\x3A D.TIF for DAPI, F.TIF for the FITC image, R.TIF for the rhodamine image. The three images together comprise an image set.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C.
    Check image sets for unmatched or duplicate files?:No
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:3
    Text that these images have in common (case-sensitive):D.TIF
    Position of this image in each group:D.TIF
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:None
    Type the regular expression that finds metadata in the subfolder path:None
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:DAPI
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):F.TIF
    Position of this image in each group:2
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Objects
    Name this loaded image:FITC
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):R.TIF
    Position of this image in each group:3
    Extract metadata from where?:None
    Regular expression that finds metadata in the file name:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:Rhodamine
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        maybe_download_fly()
        directory = os.path.join(example_images_directory(),
                                 "ExampleFlyImages")
        self.convtester(pipeline_text, directory)

    def test_17_07_group_by_metadata(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120917145632
ModuleCount:1
HasImagePlaneDetails:False

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:11|show_window:True|notes:\x5B"Load the images by matching files in the folder against the unique text pattern for each stain\x3A \'Channel1-\' for nuclei, \'Channel2-\' for the GFP image. The two images together comprise an image set."\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:None
    Input image file location:Default Input Folder\x7C
    Check image sets for unmatched or duplicate files?:Yes
    Group images by metadata?:Yes
    Exclude certain files?:No
    Specify metadata fields to group by:Column,Row
    Select subfolders to analyze:
    Image count:2
    Text that these images have in common (case-sensitive):Channel1-
    Position of this image in each group:1
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:rawGFP
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
    Text that these images have in common (case-sensitive):Channel2-
    Position of this image in each group:2
    Extract metadata from where?:File name
    Regular expression that finds metadata in the file name:.*-(?P<ImageNumber>\\\\d*)-(?P<Row>.*)-(?P<Column>\\\\d*)
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\\\\\/\x5D(?P<Date>.*)\x5B\\\\\\\\/\x5D(?P<Run>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:2
    Load the input as images or objects?:Images
    Name this loaded image:rawDNA
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:NucleiOutlines
    Channel number:1
    Rescale intensities?:Yes
"""
        directory = os.path.join(example_images_directory(),
                                 "ExampleSBSImages")
        self.convtester(pipeline_text, directory)


'''A two-channel tif containing two overlapped objects'''
overlapped_objects_data = zlib.decompress(base64.b64decode(
        "eJztlU9oI1Ucx1/abau7KO7irq6u8BwQqnYymfTfNiRZ2NRAILsJNMVuQfBl"
        "5iV5dOa9MPPGtJ487UkUBL148uTdkxc9CZ48efIuIp48CZ7qdyaTbbs0DXHL"
        "itAXPplvXn7f937z5pdfarW3yDOEkGuEZGZJBuomyKT6RTCT6hfAbKpj5o/p"
        "5zMzxJgj5MYVQq5mLiW+38D1YzE3MD/SL6Uxa/jwchoTj5upPjwk5JXMXKIf"
        "4u3VVH+Ct1tpzAxYTPUssJHDVVy/wMTWsX0/RV5fQ/wCrsBUAgp8BX4GczCv"
        "AwG+BD+BQ2BfIsQBn4Mfwd/gbaz1HsiT8yUev88eXeuNSo3eFcqsqsBnOiT/"
        "h5E58ZouerLr9PizPNM6xseP81w4zud8x43pHdPX1Wmu/248eRseZT9qw/78"
        "5Db83fzkNvzcwlEbvr4wuQ2/tnCyDccji7n3wWfgB/AXWMS/zy74GHwP/gTG"
        "s4S0wEPwLfgD3LpMyH3wEfgG/Hr5og1PGDMnXtNFT3adHn+WZ1rH+PhxngvH"
        "+ZzvuDG9Y/q6Os31tEfxzr7v0Q94EAolS4adzRmUS0e5QnZLxnarat42aKiZ"
        "dJmnJC8ZUhl3ysXXTZO+ywKJqALVPRFSR/k+l5pCMkkb994xd+7Vqc81c5lm"
        "tO0pZ2+JDnrC6SFWaiYkTEHkCOZRV8AbZwDdZwGDDRlhIZcq3eMBFX5fBchC"
        "P1oxS5seZyGn3BWaOizSQkWhd0AXRYcyTZnnvbmUrNBmzh6N+kiTUxWIroh3"
        "GSbFOyrg1FW4DRqqLEX3o348JWQnaYRIaYmGnCfm+KZatWqVDnibhkLzAu1p"
        "3S9YlvK5iXPMqqBrDcSesBo+b4lOJ0tNs1yEj+JbGZaMNH4wGGRVn0tfOIEK"
        "HdU/SKxbTo/7LLRgsPI52zZza8bQWdgPxQn3YDlx5HM528JBD50mzhSH5HCD"
        "bm/XNktGFMhCFAm3YHdW2RpfXjHzTsc2Vzba6+bGqp0z83lnvbN8e4VvrOLB"
        "Y5NCmKxUV05y8/8iYzq1Iz6+7H7oGuVizWddTuPUhezjkcYfCw08tLtexLMa"
        "R4qgptjnXkg3R0XTCFwelIydB5XdlnG2mW6JD3mlZOSHqoWKH6odzK0O5QPI"
        "3FDuJt+3Dvoo/EhIba9h+0qPScm9xzeqszayesOFk/l9j4dNHiSZxmuUi3XR"
        "7ekm0z2rXLTSJc53rbjgNuOyroog1LhJ3EQiW0dyN5G16mZybXpM8oqKpB6u"
        "GxcN4jx+H7/AkvHYuU9VTEUrXgzpjbI6JT/7zPzsp52fNawriKQKcUWNlsk/"
        "8nHpyw=="))

'''The two objects that were used to generate the above TIF'''
overlapped_objects_data_masks = [
    np.arange(-offi, 20 - offi)[:, np.newaxis] ** 2 +
    np.arange(-offj, 25 - offj)[np.newaxis, :] ** 2 < 64
    for offi, offj in ((10, 10), (10, 15))]

if __name__ == "main":
    unittest.main()
