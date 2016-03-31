"""test_imageplus - test imageplus.py"""

__version__ = "$Revision$"

import os
import unittest

import javabridge as J
from bioformats import load_image

import imagej.imageplus as I


class TestImagePlus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cellprofiler.modules.tests \
            import example_images_directory, maybe_download_sbs
        maybe_download_sbs()
        cls.root_dir = example_images_directory()

        J.attach()

    @classmethod
    def tearDownClass(cls):
        J.detach()

    def test_01_01_load_imageplus(self):
        file_name = os.path.join(
                self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        self.assertTrue(ip.getHeight() > 0)

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
        target = load_image(file_name)
        self.assertEqual(ip.getWidth(), target.shape[1])

    def test_01_08_get_height(self):
        file_name = os.path.join(
                self.root_dir, "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        target = load_image(file_name)
        self.assertEqual(ip.getHeight(), target.shape[0])

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
        dims = J.get_env().get_int_array_elements(ip.getDimensions())
        width, height, nChannels, nSlices, nFrames = dims
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
