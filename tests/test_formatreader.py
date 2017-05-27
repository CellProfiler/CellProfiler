# Python-bioformats is distributed under the GNU General Public
# License, but this file is licensed under the more permissive BSD
# license.  See the accompanying file LICENSE for details.
#
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.

'''test_formatreader.py - test the Bioformats format reader wrapper

'''

from __future__ import absolute_import, unicode_literals

import numpy as np
import os
import re
import sys
if sys.version_info.major == 2:
    from urllib import urlopen
else:
    from urllib.request import urlopen
import unittest

import javabridge as J
import bioformats.formatreader as F
import bioformats

class TestFormatReader(unittest.TestCase):
    def setUp(self):
        J.start_vm(class_path=bioformats.JARS)
        J.attach()
        bioformats.init_logger()

    def tearDown(self):
        J.detach()

    def test_01_01_make_format_tools_class(self):
        FormatTools = F.make_format_tools_class()
        self.assertEqual(FormatTools.CAN_GROUP, 1)
        self.assertEqual(FormatTools.CANNOT_GROUP, 2)
        self.assertEqual(FormatTools.DOUBLE, 7)
        self.assertEqual(FormatTools.FLOAT, 6)
        self.assertEqual(FormatTools.INT16, 2)
        self.assertEqual(FormatTools.INT8, 0)
        self.assertEqual(FormatTools.MUST_GROUP, 0)
        self.assertEqual(FormatTools.UINT16, 3)
        self.assertEqual(FormatTools.UINT32, 5)
        self.assertEqual(FormatTools.UINT8, 1)

    def test_02_01_make_image_reader(self):
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        ImageReader = F.make_image_reader_class()
        FormatTools = F.make_format_tools_class()
        reader = ImageReader()
        reader.setId(path)
        self.assertEqual(reader.getDimensionOrder(), "XYCZT")
        metadata = J.jdictionary_to_string_dictionary(reader.getMetadata())
        self.assertEqual(int(metadata["ImageWidth"]), reader.getSizeX())
        self.assertEqual(int(metadata["ImageLength"]), reader.getSizeY())
        self.assertEqual(reader.getImageCount(), 1)
        self.assertEqual(reader.getSizeC(), 1)
        self.assertEqual(reader.getSizeT(), 1)
        self.assertEqual(reader.getSizeZ(), 1)
        self.assertEqual(reader.getPixelType(), FormatTools.UINT8)
        self.assertEqual(reader.getRGBChannelCount(), 1)

    def test_03_01_read_tif(self):
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        ImageReader = F.make_image_reader_class()
        FormatTools = F.make_format_tools_class()
        reader = ImageReader()
        reader.setId(path)
        data = reader.openBytes(0)
        data.shape = (reader.getSizeY(), reader.getSizeX())
        #
        # Data as read by cellprofiler.modules.loadimages.load_using_PIL
        #
        expected_0_10_0_10 = np.array(
            [[ 0,  7,  7,  6,  5,  8,  4,  2,  1,  2],
             [ 0,  8,  8,  7,  6, 10,  4,  2,  2,  2],
             [ 0,  9,  9,  7,  8,  8,  2,  1,  3,  2],
             [ 0, 10,  9,  8, 10,  6,  2,  2,  3,  2],
             [ 0, 10, 10, 10,  9,  4,  2,  2,  2,  2],
             [ 0,  9,  9, 10,  8,  3,  2,  4,  2,  2],
             [ 0,  9,  9, 10,  8,  2,  2,  4,  3,  2],
             [ 0,  9,  8,  9,  7,  4,  2,  2,  2,  2],
             [ 0, 10, 11,  9,  9,  4,  2,  2,  2,  2],
             [ 0, 12, 13, 12,  9,  4,  2,  2,  2,  2]], dtype=np.uint8)
        expected_n10_n10 = np.array(
            [[2, 1, 1, 1, 2, 2, 1, 2, 1, 2],
             [1, 2, 2, 2, 2, 1, 1, 1, 2, 1],
             [1, 1, 1, 2, 1, 2, 2, 2, 2, 1],
             [2, 2, 2, 2, 3, 2, 2, 2, 2, 1],
             [1, 2, 2, 1, 1, 1, 1, 1, 2, 2],
             [2, 1, 2, 2, 2, 1, 1, 2, 2, 2],
             [2, 2, 3, 2, 2, 1, 2, 2, 2, 1],
             [3, 3, 1, 2, 2, 2, 2, 3, 2, 2],
             [3, 2, 2, 2, 2, 2, 2, 2, 3, 3],
             [5, 2, 3, 3, 2, 2, 2, 3, 2, 2]], dtype=np.uint8)
        self.assertTrue(np.all(expected_0_10_0_10 == data[:10,:10]))
        self.assertTrue(np.all(expected_n10_n10 == data[-10:,-10:]))

    def test_03_02_load_using_bioformats(self):
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        data = F.load_using_bioformats(path, rescale=False)
        expected_0_10_0_10 = np.array(
            [[ 0,  7,  7,  6,  5,  8,  4,  2,  1,  2],
             [ 0,  8,  8,  7,  6, 10,  4,  2,  2,  2],
             [ 0,  9,  9,  7,  8,  8,  2,  1,  3,  2],
             [ 0, 10,  9,  8, 10,  6,  2,  2,  3,  2],
             [ 0, 10, 10, 10,  9,  4,  2,  2,  2,  2],
             [ 0,  9,  9, 10,  8,  3,  2,  4,  2,  2],
             [ 0,  9,  9, 10,  8,  2,  2,  4,  3,  2],
             [ 0,  9,  8,  9,  7,  4,  2,  2,  2,  2],
             [ 0, 10, 11,  9,  9,  4,  2,  2,  2,  2],
             [ 0, 12, 13, 12,  9,  4,  2,  2,  2,  2]], dtype=np.uint8)
        expected_n10_n10 = np.array(
            [[2, 1, 1, 1, 2, 2, 1, 2, 1, 2],
             [1, 2, 2, 2, 2, 1, 1, 1, 2, 1],
             [1, 1, 1, 2, 1, 2, 2, 2, 2, 1],
             [2, 2, 2, 2, 3, 2, 2, 2, 2, 1],
             [1, 2, 2, 1, 1, 1, 1, 1, 2, 2],
             [2, 1, 2, 2, 2, 1, 1, 2, 2, 2],
             [2, 2, 3, 2, 2, 1, 2, 2, 2, 1],
             [3, 3, 1, 2, 2, 2, 2, 3, 2, 2],
             [3, 2, 2, 2, 2, 2, 2, 2, 3, 3],
             [5, 2, 3, 3, 2, 2, 2, 3, 2, 2]], dtype=np.uint8)
        self.assertTrue(np.all(expected_0_10_0_10 == data[:10,:10]))
        self.assertTrue(np.all(expected_n10_n10 == data[-10:,-10:]))

    def test_03_03_read_subimage_tif(self):
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        with bioformats.ImageReader(path) as f:
            data_0_10_0_10 = f.read(XYWH=(0, 0, 10, 10), rescale=False)

        #
        # Data as read by cellprofiler.modules.loadimages.load_using_PIL
        #
        expected_0_10_0_10 = np.array(
            [[ 0,  7,  7,  6,  5,  8,  4,  2,  1,  2],
             [ 0,  8,  8,  7,  6, 10,  4,  2,  2,  2],
             [ 0,  9,  9,  7,  8,  8,  2,  1,  3,  2],
             [ 0, 10,  9,  8, 10,  6,  2,  2,  3,  2],
             [ 0, 10, 10, 10,  9,  4,  2,  2,  2,  2],
             [ 0,  9,  9, 10,  8,  3,  2,  4,  2,  2],
             [ 0,  9,  9, 10,  8,  2,  2,  4,  3,  2],
             [ 0,  9,  8,  9,  7,  4,  2,  2,  2,  2],
             [ 0, 10, 11,  9,  9,  4,  2,  2,  2,  2],
             [ 0, 12, 13, 12,  9,  4,  2,  2,  2,  2]], dtype=np.uint8)
        expected_n10_n10 = np.array(
            [[2, 1, 1, 1, 2, 2, 1, 2, 1, 2],
             [1, 2, 2, 2, 2, 1, 1, 1, 2, 1],
             [1, 1, 1, 2, 1, 2, 2, 2, 2, 1],
             [2, 2, 2, 2, 3, 2, 2, 2, 2, 1],
             [1, 2, 2, 1, 1, 1, 1, 1, 2, 2],
             [2, 1, 2, 2, 2, 1, 1, 2, 2, 2],
             [2, 2, 3, 2, 2, 1, 2, 2, 2, 1],
             [3, 3, 1, 2, 2, 2, 2, 3, 2, 2],
             [3, 2, 2, 2, 2, 2, 2, 2, 3, 3],
             [5, 2, 3, 3, 2, 2, 2, 3, 2, 2]], dtype=np.uint8)
        self.assertTrue(np.all(expected_0_10_0_10 == data_0_10_0_10))
        # self.assertTrue(np.all(expected_n10_n10 == data[-10:,-10:]))
        
    def test_03_03_load_using_bioformats_url(self):
        url = "https://github.com/CellProfiler/python-bioformats/raw/1.0.5/bioformats/tests/Channel1-01-A-01.tif"
        try:
            fd = urlopen(url)
            if fd.code < 200 or fd.code >= 300:
                raise OSError("Http error %d" % fd.code)
        except OSError as e:
            def bad_url(e=e):
                raise e
            unittest.expectedFailure(bad_url)()
        
        data = F.load_using_bioformats_url(url, rescale=False)
        self.assertSequenceEqual(data.shape, (640, 640))

    def test_04_01_read_omexml_metadata(self):
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        xml = F.get_omexml_metadata(path)
        pattern = r'<\s*Image\s+ID\s*=\s*"Image:0"\s+Name\s*=\s*"Channel1-01-A-01.tif"\s*>'
        self.assertTrue(re.search(pattern, xml))


