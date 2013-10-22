'''test_imageplus - test imageplus.py

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

__version__ = "$Revision$"

import os
import unittest

import cellprofiler.utilities.jutil as J
import imagej.imageplus as I


class TestImagePlus(unittest.TestCase):
    def setUp(self):
        from cellprofiler.modules.tests import example_images_directory
        self.root_dir = example_images_directory()
        J.attach()
        
    def tearDown(self):
        J.detach()
        
    def test_01_01_load_imageplus(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertTrue(J.to_string(ip.o).startswith("imp"))
        
    def test_01_02_get_id(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        ip.getID()
        
    def test_01_03_lock_and_unlock(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertTrue(ip.lockSilently())
        try:
            self.assertFalse(ip.lockSilently())
        finally:
            ip.unlock()
            
    def test_01_04_get_bit_depth(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getBitDepth(), 8)

    def test_01_05_get_bytes_per_pixel(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getBytesPerPixel(), 1)
        
    def test_01_06_get_channel(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getChannel(), 1)

    def test_01_07_get_current_slice(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getCurrentSlice(), 1)
        
    def test_01_08_get_width(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getWidth(), 640)
        
    def test_01_08_get_height(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getHeight(), 640)
        
    def test_01_09_get_nframes(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getNFrames(), 1)
        
    def test_01_10_get_nslices(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getNSlices(), 1)
        
    def test_01_11_get_n_dimensions(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertEqual(ip.getNDimensions(), 2)
        
    def test_01_12_get_dimensions(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        width, height, nChannels, nSlices, nFrames = ip.getDimensions()
        self.assertEqual(width, ip.getWidth())
        self.assertEqual(height, ip.getHeight())
        self.assertEqual(nChannels, ip.getNChannels())
        self.assertEqual(nSlices, ip.getNSlices())
        self.assertEqual(nFrames, ip.getNFrames())
        
    def test_01_13_get_channel_processor(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        p = ip.getChannelProcessor()
        self.assertTrue(J.to_string(p).startswith("ip"))
        
    def test_01_14_get_processor(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        p = ip.getProcessor()
        self.assertTrue(J.to_string(p).startswith("ip"))  
        
    def test_02_01_show_get_and_hide(self):
        file_name = os.path.join(
            self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        ip.show()
        window = ip.getWindow()
        self.assertTrue(J.to_string(window).startswith("Channel1-01-A-01.tif"))
        ip.hide()