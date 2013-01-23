'''test_windowmanager - test ij.WindowManager

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
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
from cellprofiler.modules.tests import example_images_directory

class TestWindowManager(unittest.TestCase):
    
    def test_01_01_set_current_image(self):
        file_name = os.path.join(example_images_directory(), 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_current_image(ip)
        
    def test_01_02_get_id_list(self):
        file_name = os.path.join(example_images_directory(), 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_current_image(ip)
        id_list = W.get_id_list()
        self.assertTrue(ip.getID() in id_list)
        
    def test_01_03_get_image_by_id(self):
        file_name = os.path.join(example_images_directory(), 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        ip.show()
        ip_other = W.get_image_by_id(ip.getID())
        self.assertEqual(ip_other.getID(), ip.getID())
        
    def test_01_04_get_image_by_name(self):
        file_name = os.path.join(example_images_directory(), 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        title = W.make_unique_name("Whatever")
        ip.setTitle(title)
        self.assertEqual(ip.getTitle(), title)
        ip.show()
        ip_other = W.get_image_by_name(title)
        self.assertEqual(ip_other.getID(), ip.getID())
    
    def test_01_05_set_temp_current_image(self):
        file_name = os.path.join(example_images_directory(), 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_temp_current_image(ip)
        
    def test_01_06_get_temp_current_image(self):
        file_name = os.path.join(example_images_directory(), 
                                 "ExampleSBSImages", "Channel1-01-A-01.tif")
        ip = I.load_imageplus(file_name)
        W.set_temp_current_image(ip)
        ip_other = W.get_temp_current_image()
        self.assertEqual(ip.getID(), ip_other.getID())
        
    def test_01_07_make_unique_name(self):
        self.assertTrue(W.make_unique_name("Foo").startswith("Foo"))
        
