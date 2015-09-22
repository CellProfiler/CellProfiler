'''test_windowmanager - test ij.WindowManager

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

import javabridge as J
import imagej.imageplus as I
import imagej.windowmanager as W

class TestWindowManager(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from cellprofiler.modules.tests import \
             example_images_directory, maybe_download_example_image
        self.root_dir = example_images_directory()
        self.file_name = maybe_download_example_image(
            ["ExampleSBSImages"], "Channel1-01-A-01.tif")
        J.attach()
    @classmethod
    def tearDownClass(self):
        J.detach()
    
    def test_01_01_set_current_image(self):
        ip = I.load_imageplus(self.file_name)
        W.set_current_image(ip)
        
    def test_01_05_set_temp_current_image(self):
        ip = I.load_imageplus(self.file_name)
        W.set_temp_current_image(ip)
        
    def test_01_06_get_temp_current_image(self):
        ip = I.load_imageplus(self.file_name)
        W.set_temp_current_image(ip)
        ip_other = W.get_temp_current_image()
        self.assertEqual(ip.getID(), ip_other.getID())
        
    def test_01_07_make_unique_name(self):
        self.assertTrue(W.make_unique_name("Foo").startswith("Foo"))
        
    def test_01_08_get_current_image(self):
        ip = I.load_imageplus(self.file_name)
        W.set_current_image(ip)
        ip_out = W.get_current_image()
        self.assertEqual(ip.getID(), ip_out.getID())