'''test_images.py - test the Images module
'''

import csv
import os
import tempfile
import unittest
import urllib
from cStringIO import StringIO

import numpy as np

import cellprofiler.measurement as cpmeas
import cellprofiler.modules.images as I
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw


class TestImages(unittest.TestCase):
    def setUp(self):
        # The Images module needs a workspace and the workspace needs
        # an HDF5 file.
        #
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.measurements = cpmeas.Measurements(
                filename=self.temp_filename)
        os.close(self.temp_fd)

    def tearDown(self):
        self.measurements.close()
        os.unlink(self.temp_filename)
        self.assertFalse(os.path.exists(self.temp_filename))

    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120209212234
ModuleCount:1
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    :{"ShowFiltered"\x3A false}
    Filter based on rules:Yes
    Filter:or (directory does startwith "foo") (file does contain "bar")
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, I.Images))
        self.assertEqual(module.filter_choice, I.FILTER_CHOICE_CUSTOM)
        self.assertEqual(module.filter.value, 'or (directory does startwith "foo") (file does contain "bar")')

    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120209212234
ModuleCount:1
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    :{"ShowFiltered"\x3A false}
    Filter choice:%s
    Filter:or (directory does startwith "foo") (file does contain "bar")
"""
        for fc, fctext in ((I.FILTER_CHOICE_CUSTOM, "Custom"),
                           (I.FILTER_CHOICE_IMAGES, "Images only"),
                           (I.FILTER_CHOICE_NONE, "No filtering")):
            pipeline = cpp.Pipeline()

            def callback(caller, event):
                self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

            pipeline.add_listener(callback)
            pipeline.load(StringIO(data % fctext))
            self.assertEqual(len(pipeline.modules()), 1)
            module = pipeline.modules()[0]
            self.assertTrue(isinstance(module, I.Images))
            self.assertEqual(module.filter_choice, fc)
            self.assertEqual(module.filter.value, 'or (directory does startwith "foo") (file does contain "bar")')

    def test_02_04_filter_url(self):
        module = I.Images()
        module.filter_choice.value = I.FILTER_CHOICE_CUSTOM
        for url, filter_value, expected in (
                ("file:/TestImages/NikonTIF.tif",
                 'and (file does startwith "Nikon") (extension does istif)', True),
                ("file:/TestImages/NikonTIF.tif",
                 'or (file doesnot startwith "Nikon") (extension doesnot istif)', False),
                ("file:/TestImages/003002000.flex",
                 'and (directory does endwith "ges") (directory doesnot contain "foo")', True),
                ("file:/TestImages/003002000.flex",
                 'or (directory doesnot endwith "ges") (directory does contain "foo")', False)):
            module.filter.value = filter_value
            self.check(module, url, expected)

    def check(self, module, url, expected):
        '''Check filtering of one URL using the module as configured'''
        pipeline = cpp.Pipeline()
        pipeline.add_urls([url])
        module.module_num = 1
        pipeline.add_module(module)
        m = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, None, None, m, None)
        file_list = pipeline.get_filtered_file_list(workspace)
        if expected:
            self.assertEqual(len(file_list), 1)
            self.assertEqual(file_list[0], url)
        else:
            self.assertEqual(len(file_list), 0)

    def test_02_05_filter_standard(self):
        module = I.Images()
        module.filter_choice.value = I.FILTER_CHOICE_IMAGES
        for url, expected in (
                ("file:/TestImages/NikonTIF.tif", True),
                ("file:/foo/.bar/baz.tif", False),
                ("file:/TestImages/foo.bar", False)):
            self.check(module, url, expected)
