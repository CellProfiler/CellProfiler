'''test_formatreader.py - test the Bioformats format reader wrapper

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import numpy as np
import os
import re
import unittest

import cellprofiler.utilities.jutil as J
import bioformats.formatreader as F
from cellprofiler.modules.tests import example_images_directory

class TestFormatReader(unittest.TestCase):
    def setUp(self):
        J.attach()
        
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
        path = os.path.join(example_images_directory(), 'ExampleSBSImages',
                            'Channel1-01-A-01.tif')
        ImageReader = F.make_image_reader_class()
        FormatTools = F.make_format_tools_class()
        reader = ImageReader()
        reader.setId(path)
        self.assertEqual(reader.getDimensionOrder(), "XYCZT")
        self.assertEqual(640, reader.getSizeX())
        self.assertEqual(640, reader.getSizeY())
        self.assertEqual(reader.getImageCount(), 1)
        self.assertEqual(reader.getSizeC(), 1)
        self.assertEqual(reader.getSizeT(), 1)
        self.assertEqual(reader.getSizeZ(), 1)
        self.assertEqual(reader.getPixelType(), FormatTools.UINT8)
        self.assertEqual(reader.getRGBChannelCount(), 1)
        
    def test_03_01_read_tif(self):
        path = os.path.join(example_images_directory(), 'ExampleSBSImages',
                            'Channel1-01-A-01.tif')
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
        path = os.path.join(example_images_directory(), 'ExampleSBSImages',
                            'Channel1-01-A-01.tif')
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
        
    def test_03_03_load_using_bioformats_url(self):
        data = F.load_using_bioformats_url(
            "http://www.cellprofiler.org/linked_files/broad-logo.gif",
            rescale=False)
        self.assertSequenceEqual(data.shape, (38, 150, 3))
        expected_0_10_0_10 = np.array([
            [181, 176, 185, 185, 175, 175, 176, 195, 187, 185],
            [ 25,   7,   7,   7,   2,   2,  13,  13,   0,   1],
            [ 21,   1,   1,   0,   0,   1,   0,   1,   0,   0],
            [ 64,  13,   1,   1,  12,  12,   2,   1,   1,   1],
            [ 22,  56,  26,  13,   1,   1,   6,   0,   0,   0],
            [ 12,  13,  82,  57,   9,  12,   2,   6,   6,   6],
            [ 12,  13,  20,  89,  89,  21,  11,  12,   1,   0],
            [  6,   1,   7,  21,  89, 102,  26,   0,  10,   1],
            [ 26,   0,   0,   1,  20,  84, 151,  58,  12,   1],
            [ 23,   6,   1,   1,   0,   1,  55, 166, 100,  12]], 
                                      dtype=np.uint8)
        self.assertTrue(np.all(expected_0_10_0_10 == data[:10,:10, 0]))
        
    def test_04_01_read_omexml_metadata(self):
        path = os.path.join(example_images_directory(), 'ExampleSBSImages',
                            'Channel1-01-A-01.tif')
        xml = F.get_omexml_metadata(path)
        pattern = r'<\s*Image\s+ID\s*=\s*"Image:0"\s+Name\s*=\s*"Channel1-01-A-01.tif"\s*>'
        self.assertTrue(re.search(pattern, xml))
       
        
