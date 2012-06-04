'''test_images.py - test the Images module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import csv
import numpy as np
import os
from cStringIO import StringIO
import tempfile
import unittest
import urllib

import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
import cellprofiler.modules.images as I

class TestImages(unittest.TestCase):
    def setUp(self):
        # The Images module needs a workspace and the workspace needs
        # an HDF5 file.
        #
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.measurements = cpmeas.Measurements(
            filename = self.temp_filename)
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
        self.assertTrue(module.wants_filter)
        self.assertEqual(module.filter.value, 'or (directory does startwith "foo") (file does contain "bar")')
        
    def make_module(self):
        '''Set up an Images module with a pipeline and some example data
        
        returns an activated Images module
        '''
        data = """file:/TestImages/003002000.flex
file:/TestImages/397_w1447%20laser_s17_t1_bad.TIF
file:/TestImages/397_w1447%20laser_s9_t1_good.TIF
file:/TestImages/5channel.tif
file:/TestImages/C0.stk
file:/TestImages/C1.stk
file:/TestImages/Control.mov
file:/TestImages/DrosophilaEmbryo_GFPHistone.avi
file:/TestImages/icd002235_090127090001_a01f00d1.c01
file:/TestImages/IXMtest_P24_s9_w560D948A4-4D16-49D0-9080-7575267498F9.tif
file:/TestImages/NikonTIF.tif
file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex
file:/ExampleImages/ExampleSBSImages/1049_FilenamesAndMetadata.csv
file:/ExampleImages/ExampleSBSImages/1049_FilenamesAndMetadata_short.csv
file:/ExampleImages/ExampleSBSImages/1049_Metadata.csv
file:/ExampleImages/ExampleSBSImages/allscales.cp
file:/ExampleImages/ExampleSBSImages/Channel1-01-A-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-02-A-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-03-A-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-04-A-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-05-A-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-06-A-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-07-A-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-08-A-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-09-A-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-10-A-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-11-A-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-12-A-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1-13-B-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-14-B-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-15-B-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-16-B-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-17-B-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-18-B-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-19-B-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-20-B-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-21-B-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-22-B-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-23-B-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-24-B-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1-25-C-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-26-C-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-27-C-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-28-C-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-29-C-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-30-C-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-31-C-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-32-C-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-33-C-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-34-C-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-35-C-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-36-C-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1-37-D-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-38-D-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-39-D-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-40-D-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-41-D-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-42-D-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-43-D-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-44-D-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-45-D-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-46-D-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-47-D-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-48-D-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1-49-E-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-50-E-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-51-E-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-52-E-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-53-E-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-54-E-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-55-E-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-56-E-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-57-E-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-58-E-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-59-E-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-60-E-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1-61-F-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-62-F-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-63-F-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-64-F-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-65-F-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-66-F-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-67-F-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-68-F-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-69-F-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-70-F-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-71-F-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-72-F-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1-73-G-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-74-G-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-75-G-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-76-G-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-77-G-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-78-G-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-79-G-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-80-G-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-81-G-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-82-G-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-83-G-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-84-G-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1-85-H-01.tif
file:/ExampleImages/ExampleSBSImages/Channel1-86-H-02.tif
file:/ExampleImages/ExampleSBSImages/Channel1-87-H-03.tif
file:/ExampleImages/ExampleSBSImages/Channel1-88-H-04.tif
file:/ExampleImages/ExampleSBSImages/Channel1-89-H-05.tif
file:/ExampleImages/ExampleSBSImages/Channel1-90-H-06.tif
file:/ExampleImages/ExampleSBSImages/Channel1-91-H-07.tif
file:/ExampleImages/ExampleSBSImages/Channel1-92-H-08.tif
file:/ExampleImages/ExampleSBSImages/Channel1-93-H-09.tif
file:/ExampleImages/ExampleSBSImages/Channel1-94-H-10.tif
file:/ExampleImages/ExampleSBSImages/Channel1-95-H-11.tif
file:/ExampleImages/ExampleSBSImages/Channel1-96-H-12.tif
file:/ExampleImages/ExampleSBSImages/Channel1ILLUM.mat
file:/ExampleImages/ExampleSBSImages/Channel2-01-A-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-02-A-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-03-A-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-04-A-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-05-A-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-06-A-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-07-A-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-08-A-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-09-A-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-10-A-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-11-A-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-12-A-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2-13-B-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-14-B-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-15-B-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-16-B-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-17-B-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-18-B-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-19-B-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-20-B-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-21-B-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-22-B-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-23-B-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-24-B-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2-25-C-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-26-C-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-27-C-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-28-C-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-29-C-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-30-C-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-31-C-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-32-C-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-33-C-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-34-C-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-35-C-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-36-C-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2-37-D-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-38-D-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-39-D-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-40-D-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-41-D-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-42-D-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-43-D-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-44-D-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-45-D-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-46-D-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-47-D-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-48-D-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2-49-E-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-50-E-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-51-E-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-52-E-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-53-E-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-54-E-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-55-E-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-56-E-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-57-E-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-58-E-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-59-E-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-60-E-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2-61-F-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-62-F-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-63-F-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-64-F-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-65-F-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-66-F-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-67-F-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-68-F-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-69-F-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-70-F-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-71-F-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-72-F-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2-73-G-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-74-G-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-75-G-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-76-G-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-77-G-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-78-G-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-79-G-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-80-G-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-81-G-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-82-G-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-83-G-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-84-G-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2-85-H-01.tif
file:/ExampleImages/ExampleSBSImages/Channel2-86-H-02.tif
file:/ExampleImages/ExampleSBSImages/Channel2-87-H-03.tif
file:/ExampleImages/ExampleSBSImages/Channel2-88-H-04.tif
file:/ExampleImages/ExampleSBSImages/Channel2-89-H-05.tif
file:/ExampleImages/ExampleSBSImages/Channel2-90-H-06.tif
file:/ExampleImages/ExampleSBSImages/Channel2-91-H-07.tif
file:/ExampleImages/ExampleSBSImages/Channel2-92-H-08.tif
file:/ExampleImages/ExampleSBSImages/Channel2-93-H-09.tif
file:/ExampleImages/ExampleSBSImages/Channel2-94-H-10.tif
file:/ExampleImages/ExampleSBSImages/Channel2-95-H-11.tif
file:/ExampleImages/ExampleSBSImages/Channel2-96-H-12.tif
file:/ExampleImages/ExampleSBSImages/Channel2ILLUM.mat
file:/ExampleImages/ExampleSBSImages/CreateBatchFile.cp
file:/ExampleImages/ExampleSBSImages/ExampleSBS.cp
file:/ExampleImages/ExampleSBSImages/ExampleSBSIllumination.cp
file:/ExampleImages/ExampleSBSImages/foo.jpg
file:/ExampleImages/ExampleSBSImages/foo.tif
file:/ExampleImages/ExampleSBSImages/LoadDataSBS.cp
file:/ExampleImages/ExampleSBSImages/Nuclei_A01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_A12.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_B12.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_C12.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_D12.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_E12.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_F12.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_G12.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H01.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H02.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H03.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H04.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H05.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H06.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H07.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H08.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H09.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H10.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H11.tiff
file:/ExampleImages/ExampleSBSImages/Nuclei_H12.tiff"""
        pipeline = cpp.Pipeline()
        module = I.Images()
        module.module_num = 1
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, None, None,
                                  self.measurements, None)
        file_list = workspace.file_list
        self.urls = [d.strip() for d in data.split("\n")]
        file_list.add_files_to_filelist(self.urls)
        module.on_activated(workspace)
        return module
    
    def in_filetree(self, url, module):
        '''Return true if the URL is in the file collection display file tree
        
        '''
        file_tree = module.file_collection_display.file_tree
        path = module.url_to_modpath(url)
        for part in path:
            if part not in file_tree:
                return False
            file_tree = file_tree[part]
        return True
    
    def in_filelist(self, url, module):
        '''Return true if the url is in the module workspace file list'''
        file_list = module.workspace.file_list
        return file_list.get_type(url) != file_list.TYPE_NONE
        
    def test_02_01_activate(self):
        module = self.make_module()
        self.assertIsInstance(module, I.Images)
        #
        # Make sure every IPD is in the file tree
        #
        for url in self.urls:
            self.assertTrue(self.in_filetree(url, module))
            self.assertTrue(self.in_filelist(url, module))
    
    def test_02_02_on_remove(self):
        module = self.make_module()
        self.assertIsInstance(module, I.Images)
        url1 = "file:/TestImages/NikonTIF.tif"
        url2 = "file:/TestImages/DrosophilaEmbryo_GFPHistone.avi"
        url3 = "file:/TestImages/RLM1%20SSN3%20300308%20008015000.flex"
        kept = "file:/TestImages/Control.mov"
        mods = []
        for url in (url1, url2, url3):
            module.add_modpath_to_modlist(module.url_to_modpath(url), mods)
        module.on_remove(mods)
        self.assertTrue(self.in_filetree(kept, module))
        self.assertTrue(self.in_filelist(kept, module))
        for url in (url1, url2, url3):
            self.assertFalse(self.in_filetree(url, module))
            self.assertFalse(self.in_filelist(url, module))
        
    def test_02_03_get_path_info(self):
        module = self.make_module()
        
        for url, expected_node_type in (
            ("file:/TestImages/NikonTIF.tif", 
             cps.FileCollectionDisplay.NODE_MONOCHROME_IMAGE),
            ("file:/TestImages/DrosophilaEmbryo_GFPHistone.avi", 
             cps.FileCollectionDisplay.NODE_MOVIE),
            ("file:/ExampleImages/ExampleSBSImages/Channel2-96-H-12.tif",
             cps.FileCollectionDisplay.NODE_MONOCHROME_IMAGE),
            ("file:/ExampleImages/ExampleSBSImages",
             cps.FileCollectionDisplay.NODE_DIRECTORY),
            ("file:/ExampleImages/ExampleSBSImages/1049_Metadata.csv",
             cps.FileCollectionDisplay.NODE_CSV),
            ("file:/ExampleImages/ExampleSBSImages/ExampleSBS.cp",
             cps.FileCollectionDisplay.NODE_FILE)):

            modpath = module.url_to_modpath(url)
            name, node_type, tooltip, menu = module.get_path_info(modpath)
            self.assertEqual(node_type, expected_node_type)
                
    def test_02_04_filter_url(self):
        module = self.make_module()
        module.wants_filter.value = True
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
            self.assertEqual(module.filter_url(url), expected)
            
    def test_02_05_apply_filter(self):
        module = self.make_module()
        module.wants_filter.value = True
        module.filter.value = ("and (directory does endwith \"SBSImages\") "
                               "(extension does istif) "
                               "(file doesnot contain \"9\")")
        module.apply_filter()
        for url in self.urls:
            good = True
            if not url.startswith("file:/ExampleImages/ExampleSBSImages"):
                good = False
            if not (url.lower().endswith(".tif") or
                    url.lower().endswith(".tiff")):
                good = False
            if url.find("9") != -1:
                good = False
            file_tree = module.file_collection_display.file_tree
            path = module.url_to_modpath(url)
            for part in path[:-1]:
                self.assertIn(part, file_tree)
                file_tree = file_tree[part]
            self.assertEqual(good, file_tree[path[-1]])
                    
        
        