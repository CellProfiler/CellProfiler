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

import cellprofiler.utilities.jutil as J
import imagej.imageplus as I
import imagej.windowmanager as W

class TestWindowManager(unittest.TestCase):
    def setUp(self):
        from cellprofiler.modules.tests import example_images_directory
        self.root_dir = example_images_directory()
        J.attach()
        
    def tearDown(self):
        J.detach()
    
    def test_01_01_set_current_image(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_current_image(ip)
        
    def test_01_02_get_id_list(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_current_image(ip)
        id_list = W.get_id_list()
        self.assertTrue(ip.getID() in id_list)
        
    def test_01_03_get_image_by_id(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        ip.show()
        ip_other = W.get_image_by_id(ip.getID())
        self.assertEqual(ip_other.getID(), ip.getID())
        
    def test_01_04_get_image_by_name(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        title = W.make_unique_name("Whatever")
        ip.setTitle(title)
        self.assertEqual(ip.getTitle(), title)
        ip.show()
        ip_other = W.get_image_by_name(title)
        self.assertEqual(ip_other.getID(), ip.getID())
    
    def test_01_05_set_temp_current_image(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_temp_current_image(ip)
        
    def test_01_06_get_temp_current_image(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_temp_current_image(ip)
        ip_other = W.get_temp_current_image()
        self.assertEqual(ip.getID(), ip_other.getID())
        
    def test_01_07_make_unique_name(self):
        self.assertTrue(W.make_unique_name("Foo").startswith("Foo"))
        
    def test_01_08_get_current_image(self):
        file_name = os.path.join(self.root_dir, 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_current_image(ip)
        ip_out = W.get_current_image()
        self.assertEqual(ip.getID(), ip_out.getID())