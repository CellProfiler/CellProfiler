'''test_images.py - test the Images module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import numpy as np
from cStringIO import StringIO
import unittest
import urllib

import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.modules.images as I

class TestImages(unittest.TestCase):
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
        self.assertTrue(module.wants_filter)
        self.assertEqual(module.filter.value, 'or (directory does startwith "foo") (file does contain "bar")')
        
    def make_module(self):
        '''Set up an Images module with a pipeline and some example data
        
        returns an activated Images module
        '''
        data = """"Version":"1","PlaneCount":"127"
"URL","Series","Index","Channel","ChannelName","ColorFormat","SizeC","SizeT","SizeZ","T","Z"
"file:/TestImages/003002000.flex",,,,,"Planar","2","1","1",,
"file:/TestImages/003002000.flex","0","0",,"Exp1Cam2","monochrome",,,,"0","0"
"file:/TestImages/003002000.flex","0","1",,"Exp2Cam3","monochrome",,,,"0","0"
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF",,,,,"monochrome","1","21","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","0",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","1",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","2",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","3",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","4",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","5",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","6",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","7",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","8",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","9",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","10",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","11",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","12",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","13",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","14",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","15",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","16",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","17",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","18",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","19",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF","0","20",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF",,,,,"monochrome","1","21","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","0",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","1",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","2",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","3",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","4",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","5",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","6",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","7",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","8",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","9",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","10",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","11",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","12",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","13",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","14",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","15",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","16",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","17",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","18",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","19",,,"monochrome","1","1","1",,
"file:/TestImages/397_w1447%20laser_s9_t1_good.TIF","0","20",,,"monochrome","1","1","1",,
"file:/TestImages/5channel.tif",,,,,"Planar","5","1","1",,
"file:/TestImages/Control.mov",,,,,,,,,,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi",,,,,"RGB","3","65","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","0",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","1",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","2",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","3",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","4",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","5",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","6",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","7",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","8",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","9",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","10",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","11",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","12",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","13",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","14",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","15",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","16",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","17",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","18",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","19",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","20",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","21",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","22",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","23",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","24",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","25",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","26",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","27",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","28",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","29",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","30",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","31",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","32",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","33",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","34",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","35",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","36",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","37",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","38",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","39",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","40",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","41",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","42",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","43",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","44",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","45",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","46",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","47",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","48",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","49",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","50",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","51",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","52",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","53",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","54",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","55",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","56",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","57",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","58",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","59",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","60",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","61",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","62",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","63",,,"RGB","3","1","1",,
"file:/TestImages/DrosophilaEmbryo_GFPHistone.avi","0","64",,,"RGB","3","1","1",,
"file:/TestImages/IXMtest_P24_s9_w560D948A4-4D16-49D0-9080-7575267498F9.tif",,,,,"monochrome","1","1","1",,
"file:/TestImages/NikonTIF.tif",,,,,"RGB","3","1","1",,
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex",,,,,,,,,,
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","0","0",,"1_Exp1Cam1","monochrome",,,,"0","0"
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","0","1",,"1_Exp1Cam2","monochrome",,,,"0","0"
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","1","0",,"2_Exp1Cam1","monochrome",,,,"0","0"
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","1","1",,"2_Exp1Cam2","monochrome",,,,"0","0"
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","2","0",,"3_Exp1Cam1","monochrome",,,,"0","0"
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","2","1",,"3_Exp1Cam2","monochrome",,,,"0","0"
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","3","0",,"4_Exp1Cam1","monochrome",,,,"0","0"
"file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex","3","1",,"4_Exp1Cam2","monochrome",,,,"0","0"
"file:/TestImages/icd002235_090127090001_a01f00d1.c01",,,,,"monochrome","1","1","1",,
"""
        
        pipeline = cpp.Pipeline()
        ipds = cpp.read_image_plane_details(StringIO(data))
        pipeline.add_image_plane_details(ipds)
        module = I.Images()
        module.module_num = 1
        pipeline.add_module(module)
        module.on_activated(pipeline)
        return module
    
    def test_02_01_activate(self):
        module = self.make_module()
        self.assertIsInstance(module, I.Images)
        #
        # Make sure every IPD is in the file tree
        #
        file_tree = module.file_collection_display.file_tree
        for ipd in module.pipeline.image_plane_details:
            path = module.make_modpath_from_ipd(ipd)
            t = file_tree
            while(len(path) > 0):
                self.assertTrue(t.has_key(path[0]))
                t = t[path[0]]
                path = path[1:]
    
    def test_02_02_find_ipd(self):
        module = self.make_module()
        self.assertIsInstance(module, I.Images)
        exemplar = cpp.ImagePlaneDetails("file:/TestImages/NikonTIF.tif", None, None, None)
        modpath = module.make_modpath_from_ipd(exemplar)
        ipd = module.get_image_plane_details(modpath)
        self.assertIsNotNone(ipd)
        self.assertEqual(ipd.metadata[cpp.ImagePlaneDetails.MD_COLOR_FORMAT],
                         cpp.ImagePlaneDetails.MD_RGB)
        ipd = module.get_image_plane_details(modpath[:-1] + [ "foo.tif" ])
        self.assertIsNone(ipd)
        
    def test_02_03_on_remove(self):
        module = self.make_module()
        self.assertIsInstance(module, I.Images)
        exemplar1 = cpp.ImagePlaneDetails("file:/TestImages/NikonTIF.tif", None, None, None)
        exemplar2 = cpp.ImagePlaneDetails("file:/TestImages/DrosophilaEmbryo_GFPHistone.avi", 0, 35, None)
        exemplar3 = cpp.ImagePlaneDetails("file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex", None, None, None)
        kept = cpp.ImagePlaneDetails("file:/TestImages/DrosophilaEmbryo_GFPHistone.avi", 0, 63, None)
        modpath = module.make_modpath_from_ipd(exemplar1)
        mods = []
        current_list = mods
        for part in modpath[:-1]:
            next_list = []
            current_list.append((part, next_list))
            current_list = next_list
        current_list.append(modpath[-1])
        modpath = module.make_modpath_from_ipd(exemplar2)
        current_list.append((modpath[-2], [modpath[-1]]))
        modpath = module.make_modpath_from_ipd(exemplar3)
        current_list.append(modpath[-1])
        module.on_remove(mods)
        kept_modpath = module.make_modpath_from_ipd(kept)
        self.assertIsNotNone(module.get_image_plane_details(kept_modpath))
        for exemplar in (exemplar1, exemplar2, exemplar3):
            self.assertIsNone(module.get_image_plane_details(
                module.make_modpath_from_ipd(exemplar)))
        t = module.file_collection_display.file_tree
        modpath = module.make_modpath_from_ipd(exemplar1)
        for part in modpath[:-1]:
            self.assertTrue(t.has_key(part))
            t = t[part]
        self.assertFalse(t.has_key(modpath[-1]))
        modpath = module.make_modpath_from_ipd(exemplar2)
        self.assertTrue(t.has_key(modpath[-2]))
        self.assertFalse(t[modpath[-2]].has_key(modpath[-1]))
        self.assertTrue(t[modpath[-2]].has_key(kept_modpath[-1]))
        modpath = module.make_modpath_from_ipd(exemplar3)
        self.assertFalse(t.has_key(modpath[-1]))
        
    def test_02_04_get_path_info(self):
        module = self.make_module()
        exemplar1 = cpp.ImagePlaneDetails("file:/TestImages/NikonTIF.tif", None, None, None)
        exemplar2 = cpp.ImagePlaneDetails("file:/TestImages/DrosophilaEmbryo_GFPHistone.avi", 0, 35, None)
        exemplar3 = cpp.ImagePlaneDetails("file:/TestImages/003002000.flex", None, None, None)
        exemplar4 = cpp.ImagePlaneDetails("file:/TestImages/DrosophilaEmbryo_GFPHistone.avi", None, None, None)
        exemplar5 = cpp.ImagePlaneDetails("file:/TestImages/003002000.flex", 0, 1, None)
        
        for exemplar, expected_node_type in (
            (exemplar1, cps.FileCollectionDisplay.NODE_COLOR_IMAGE),
            (exemplar2, cps.FileCollectionDisplay.NODE_COLOR_IMAGE),
            (exemplar3, cps.FileCollectionDisplay.NODE_COMPOSITE_IMAGE),
            (exemplar4, cps.FileCollectionDisplay.NODE_MOVIE),
            (exemplar5, cps.FileCollectionDisplay.NODE_IMAGE_PLANE)):
            modpath = module.make_modpath_from_ipd(exemplar)
            name, node_type, tooltip, menu = module.get_path_info(modpath)
            self.assertEqual(node_type, expected_node_type)
            
    def test_02_05_on_ipds_added(self):
        module = self.make_module()
        ipds = [ cpp.ImagePlaneDetails("file:/ExampleImages/ExampleSBSImages/Channel1-01-A-01.tif", None, None, None),
                 cpp.ImagePlaneDetails("file:/ExampleImages/ExampleSBSImages/Channel2-01-A-01.tif", None, None, None),
                 cpp.ImagePlaneDetails("file:/ExampleImages/ExampleHT29/AS_09125_050116030001_D03f00d0.tif", 0, 1, None)]
        
        module.pipeline.add_image_plane_details(ipds)
        for ipd in ipds:
            modpath = module.make_modpath_from_ipd(ipd)
            t = module.file_collection_display.file_tree
            for part in modpath:
                self.assertTrue(t.has_key(part))
                t = t[part]
                
    def test_02_06_filter_ipd(self):
        module = self.make_module()
        module.wants_filter.value = True
        for ipd, filter_value, expected in (
            (cpp.ImagePlaneDetails("file:/TestImages/NikonTIF.tif", None, None, None),
             'and (file does startwith "Nikon") (extension does istif)', True),
            (cpp.ImagePlaneDetails("file:/TestImages/NikonTIF.tif", None, None, None),
             'or (file doesnot startwith "Nikon") (extension doesnot istif)', False),
            (cpp.ImagePlaneDetails("file:/TestImages/NikonTIF.tif", None, None, None),
             'and (image does iscolor) (image doesnot ismonochrome)', True),
            (cpp.ImagePlaneDetails("file:/TestImages/NikonTIF.tif", None, None, None),
             'or (image doesnot iscolor) (image does ismonochrome)', False),
            (cpp.ImagePlaneDetails("file:/TestImages/003002000.flex", 0, 1, None),
             'and (directory does endwith "ges") (directory doesnot contain "foo")', True),
            (cpp.ImagePlaneDetails("file:/TestImages/003002000.flex", 0, 1, None),
             'or (directory doesnot endwith "ges") (directory does contain "foo")', False)):
            module.filter.value = filter_value
            self.assertEqual(module.filter_ipd(ipd), expected)
        
        