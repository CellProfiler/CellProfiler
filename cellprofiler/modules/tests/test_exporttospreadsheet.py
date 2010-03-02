'''test_exporttoexcel.py - test the ExportToExcel module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import base64
import csv
import os
import numpy as np
from StringIO import StringIO
import tempfile
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.modules.exporttospreadsheet as E 

class TestExportToSpreadsheet(unittest.TestCase):

    def setUp(self):
        cpprefs.set_headless()
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        for file_name in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, file_name)
            if os.path.isdir(path):
                for ffiillee_nnaammee in os.listdir(path):
                    os.remove(os.path.join(path, ffiillee_nnaammee))
                os.rmdir(path)
            else:
                os.remove(path)
        os.rmdir(self.output_dir)
        self.output_dir = None
        
    def test_000_01_load_mat_pipe(self):
        '''Load a matlab pipeline'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUgjJKFXwTy5RMDBUMDS1MrKwMjJSMDIwsFQgGTAwevryMzAwvGRk'
                'YKiY83baSb/DBgIOy6W9BVN3FzBouus5cE+XEwzo9LqgFKVosWvlbKmlhVMt'
                'ND6w1gg+OtGv8FHzhmqm8PYJJkkvFZMjr1kYnzdOPu3HyJAXxvDsx44D1VJ2'
                'qs9facaLVzEbu/GcPeUoZnb3Z/i99RM9A7yPX/fYvpHV+eOz1W7tZvK/DvuL'
                'b8l9vaOtqi5Zz3T39UPLQt9X9L177v0oYa+jnCHLUePdh74fUbpS8Dd2n411'
                'zHSjzLtM799/Xmy1ZIlVx66cH89a/wr2vNks/Grbq83xpdXC1jnbHylM1bQ9'
                'cmXdgynOFpadyl+y+DMtkmS8J+2/4L/rhMGGug3ROq7n1PULW+q8RJuNjm77'
                'sEIv8aeOh8zC676LZD8v/jy5UGVOy+SCY5ZHBfy0DngejPPZf1jy7H3p804V'
                '2UL+L45kTK6+tG3iOoFC1miDc7yrg60O+Dr3+j3/f/Wfy9rLi6ZMv2AqbBuh'
                'dWljv7DpPJ+G6Y3RcVP3r2nfs29jP9O/TxHn6v6f+/29dtYuv6r0zNd8q32M'
                'V979bxonkX6p/sSDD3wGXPaXdsx/Mv2J+eufJz/+vf6DKf6+/Kdase93rZav'
                'iP1qaxBocVrI/YXocn/fa0tE3Y9a9dxfXqz4cXll9Pnj1dO9DK+fTl32/fSn'
                'b3v+blP9My9h19+bP4/dm7zjm3zsyo+3AAVEF7c=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, E.ExportToExcel))
        self.assertEqual(len(module.object_groups), 1)
        og = module.object_groups[0]
        self.assertEqual(og.name, "Image")
        self.assertEqual(og.file_name, "Image.csv")
    
    def test_000_02_load_v1(self):
        '''Load a version 1 pipeline'''
        data = ('eJztWnFv0zgUd7ZuusHpNMRJ8A+S/2R3a5SMTQcTGi103FVHu4pVIISAc1N3'
                'NXLjKnG2FoTEx+Ej3EfhY9xHuDhzmtTKmrZr1+4ulqzkvfj33s/Pz3ZipVKs'
                'vyg+hXu6ASvFer5FKIY1iniLOZ192GUu6W3DZw5GHDchs/dhve3BI4tDaEJz'
                'b//B7r5pwh3DeASmKFq58pO45gFY9y8/+HVFPlqTsharQj7GnBP7xF0DOXBX'
                '6r/79RVyCGpQ/ApRD7uRi1Bftlus3u8OHlVY06O4ijrxxn6pep0GdtyjVgiU'
                'j2ukh+kx+YSVLoTNXuJT4hJmS7y0r2oHfhlX/B632dlzx6ej2Bfx+f5zFB9N'
                'ic+qX+/F9KL9HyBqn0uI561Y+00pE7tJTknTQxSSDjoZsBP2jBR7q0P2VkGp'
                'WpwKV0eNAFdIwW0q/EWt4x7PH/aQn5kdxK22sPMwxc66YkfIVc+imEzX7zd+'
                '1MbBaUM4DTyQcU7ju6bwFbJpbO8aYDy+K0P4FVBlYKx431D8CrnEoM049FyZ'
                'sPOM10W8rwsubT7eUeIr5BJuIY9yWBaTEZaIgy3OnP7C47yu4MIS4jbkdZy8'
                'uqn0W8hH3PXg75Q1EB3YmWZeBHED881LFWfqxlLmV9J6b+hGULZNeSOfjzNu'
                'G4q9jTDeuuWexuzMKw65IVxO9MWc9Xo/yzxPjJfNse0S3p9xvBbBO81O0v5x'
                'HvcoYeb5nqHGzTQuF7dJ93dzStxvCbhZ8lTnUZXZeFbr5SQ8v6b4+xMM54+Q'
                '399/UnssPlTwgf7r1gchvcaUvmRnB2+L+dq7rVDzjFGvYx+8NfKP3n02t3e+'
                'nDc+Jj4yUG4NeBRSeEzzHjRJHNop/h8q/oUs+vIGI0d2cPfLVl6oKszmbanb'
                'kboS6keaRYzzvPI44zlfnkbCe80y8Bxn/VoGnrPet7P8VPNzbyE8Cyk8k75v'
                '6mcMWhS5rjxhWQTvab4XXmNy0hZngKfiwMu2cMzessU9aZ9+zhx84jDPbi6O'
                '939l/qnvqXuX9Pft9mTnm1eZN8FhqEic7uX9X+V8ZY2P2OIBcUjsJu7OgEeG'
                'y3AZLsIVwOh5mbT/x+bltetvhsvy4DrhCmD0uGyC4XERNdrvz7fN69TfDJfl'
                'T4bLcMuOK4DR82pZv8MyXIbLcBnu/47rahFOPa9TzzFF+79ifpLW+1/A8Hov'
                'ZAtT2nWY+B/V0TvBT5OuThlqnv+dqL/wb8uxHxUDXil+CoqfwkV+SBPbnLT6'
                'Xcf35nHWQZxYellqa762GGrH6Z+h+DUu8ot7XeZwznDPf6wfBlKdHQpJHa+N'
                'BD/xuK/40q17P44cZ3V8o3H/58k0/nI5LfAX/w/nZgouF+MkisD/DSbLr/sj'
                '2od9vMr2k8ZN0zTwLyPBmvk=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, E.ExportToExcel))
        self.assertEqual(len(module.object_groups), 2)
        og = module.object_groups[0]
        self.assertEqual(og.name, "Image")
        self.assertEqual(og.file_name, "Image.csv")
        og = module.object_groups[1]
        self.assertEqual(og.name, "Nuclei")
        self.assertEqual(og.file_name, "Nuclei.csv")
        
    def test_000_03_load_v2(self):
        '''Load a version 2 pipeline'''
        data = ('eJztVtFOwjAU7SYQCMT46GMffdBlmJAILzoVExIHRBaibw7oYGZbSdch+hV+'
                'Hp/go59gixtsy2QCMUZjk6Y97Tk7997cpFMV7Vo5hxVJhqqiHRmmhWDb0qmB'
                'iV2DDj2EFwTpFA0gdmpQG3mw1adQLsNypVau1uQKPJblKthsCA11ly2veQBy'
                'bGULEP2rrI+F0OS4gyg1naGbBRmw75/P2OzqxNR7FurqlofcpUVw3nAMrD2N'
                'F1cqHngWaup2mMxG07N7iLgtIxD6121ziqyO+YxiKQS0GzQxXRM7vt7/fvx0'
                '4YtpzLczwo9XhIUT+z6vz0xc1kdIqE8pdM75MljyMwn8nRB/jyFN722ku2M5'
                'fEUnRnQiaOLN4vxuv98S52d+Jym6bET3gRu2PuQ9t028ab55EPXl+FLRFKnv'
                'Trbt71yMH4yAX/jX/RndGVjdZ0UQ7TOOce8B9emQYG/8Y3G/gNX9LYBof9+n'
                '5CnH8uS4jyxrTDB/v4lkzx8ZV0LTMSaUYjRl11J9jjRc5yieTyHBJxyXyHal'
                'lDrE81/W5e10Ez8xwa+Yosv4fxBcd7tm3Q9W8EECf918+P4doGKH5Q==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, E.ExportToExcel))
        self.assertTrue(module.prepend_output_filename)
        self.assertFalse(module.add_metadata)
        self.assertTrue(module.add_indexes)
        self.assertFalse(module.excel_limits)
        self.assertFalse(module.pick_columns)
        self.assertTrue(module.wants_aggregate_means)
        self.assertFalse(module.wants_aggregate_medians)
        self.assertTrue(module.wants_aggregate_std)
        
    def test_000_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8948

ExportToSpreadsheet:[module_num:1|svn_version:\'8947\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select or enter the column delimiter:Tab
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Add image/object numbers to output?:Yes
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:Yes
    Calculate the per-image standard deviation values for object measurements?:No
    Where do you want to save the files?:Custom folder with metadata
    Folder name\x3A:./\\<?Plate>
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    Name the data file (not including the output filename, if prepending was requested above):PFX_Image.csv
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    Name the data file (not including the output filename, if prepending was requested above):Nuclei.csv
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module,E.ExportToExcel))
        self.assertEqual(module.delimiter_char, "\t")
        self.assertTrue(module.prepend_output_filename)
        self.assertFalse(module.add_metadata)
        self.assertTrue(module.add_indexes)
        self.assertFalse(module.excel_limits)
        self.assertTrue(module.pick_columns)
        self.assertFalse(module.wants_aggregate_means)
        self.assertTrue(module.wants_aggregate_medians)
        self.assertFalse(module.wants_aggregate_std)
        self.assertEqual(module.directory_choice, E.DIR_CUSTOM_WITH_METADATA)
        self.assertEqual(module.custom_directory,r"./\<?Plate>")
        self.assertEqual(len(module.object_groups), 2)
        for group, object_name, file_name in zip(module.object_groups,
                                                 ("Image", "Nuclei"),
                                                 ("PFX_Image.csv", "Nuclei.csv")):
            self.assertEqual(group.name, object_name)
            self.assertEqual(group.file_name, file_name)
            self.assertFalse(group.wants_automatic_file_name)
            
    def test_000_05_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9152

ExportToSpreadsheet:[module_num:1|svn_version:\'9144\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:No
    Add image metadata columns to your object data file?:No
    No longer used, always saved:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Where do you want to save the files?:Default output folder
    Folder name\x3A:.
    Export all measurements?:No
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name\x3A:Image.csv
    Use the object name for the file name?:Yes
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name\x3A:Nuclei.csv
    Use the object name for the file name?:Yes
    Data to export:PropCells
    Combine these object measurements with those of the previous object?:No
    File name\x3A:PropCells.csv
    Use the object name for the file name?:Yes
    Data to export:DistanceCells
    Combine these object measurements with those of the previous object?:No
    File name\x3A:DistanceCells.csv
    Use the object name for the file name?:Yes
    Data to export:DistCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name\x3A:DistCytoplasm.csv
    Use the object name for the file name?:Yes
    Data to export:PropCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name\x3A:PropCytoplasm.csv
    Use the object name for the file name?:Yes
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module,E.ExportToExcel))
        self.assertEqual(module.delimiter, E.DELIMITER_COMMA)
        self.assertFalse(module.prepend_output_filename)
        self.assertFalse(module.add_metadata)
        self.assertFalse(module.excel_limits)
        self.assertFalse(module.pick_columns)
        self.assertFalse(module.wants_aggregate_means)
        self.assertFalse(module.wants_aggregate_medians)
        self.assertFalse(module.wants_aggregate_std)
        self.assertEqual(module.directory_choice, E.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertFalse(module.wants_everything)
        for group, object_name in zip(module.object_groups,
                                      ("Image","Nuclei","PropCells",
                                       "DistanceCells","DistCytoplasm",
                                       "PropCytoplasm")):
            self.assertEqual(group.name, object_name)
            self.assertEqual(group.file_name, "%s.csv" % object_name)
            self.assertFalse(group.previous_file)
            self.assertTrue(group.wants_automatic_file_name)
            
    def test_00_00_no_measurements(self):
        '''Test an image set with objects but no measurements'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_measurement("my_object","my_measurement",np.zeros((0,)))
        m.add_image_measurement("Count_my_object", 0)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_object")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        fd = open(path,"r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),3)
            self.assertEqual(header[2],"my_measurement")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_01_01_experiment_measurement(self):
        '''Test writing one experiment measurement'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.EXPERIMENT
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_experiment_measurement("my_measurement", "Hello, world")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            row = reader.next()
            self.assertEqual(len(row),2)
            self.assertEqual(row[0],"my_measurement")
            self.assertEqual(row[1],"Hello, world")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
            
    def test_01_02_two_experiment_measurements(self):
        '''Test writing two experiment measurements'''
        path = os.path.join(self.output_dir, "%s.csv" % cpmeas.EXPERIMENT)
        cpprefs.set_default_output_directory(self.output_dir)
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.EXPERIMENT
        module.object_groups[0].file_name.value = "badfile"
        module.object_groups[0].wants_automatic_file_name.value = True
        m = cpmeas.Measurements()
        m.add_experiment_measurement("my_measurement", "Hello, world")
        m.add_experiment_measurement("my_other_measurement","Goodbye")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            row = reader.next()
            self.assertEqual(len(row),2)
            self.assertEqual(row[0],"my_measurement")
            self.assertEqual(row[1],"Hello, world")
            row = reader.next()
            self.assertEqual(len(row),2)
            self.assertEqual(row[0],"my_other_measurement")
            self.assertEqual(row[1],"Goodbye")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
        
    def test_01_03_experiment_measurements_output_file(self):
        '''Test prepend_output_filename'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = True
        module.wants_everything.value = False
        module.directory_choice.value = E.DIR_CUSTOM
        module.object_groups[0].name.value = cpmeas.EXPERIMENT
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_experiment_measurement("my_measurement", "Hello, world")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        file_name = cpprefs.get_output_file_name()[:-4]+"_my_file.csv"
        path = os.path.join(self.output_dir,file_name)
        self.assertTrue(os.path.isfile(path),"Could not find file %s"%path)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            row = reader.next()
            self.assertEqual(len(row),2)
            self.assertEqual(row[0],"my_measurement")
            self.assertEqual(row[1],"Hello, world")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
        
    def test_02_01_image_measurement(self):
        '''Test writing an image measurement'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_image_measurement("my_measurement", "Hello, world")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),2)
            self.assertEqual(header[0], 'ImageNumber')
            self.assertEqual(header[1], "my_measurement")
            row = reader.next()
            self.assertEqual(row[0],"1")
            self.assertEqual(row[1],"Hello, world")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()

    def test_02_02_three_by_two_image_measurements(self):
        '''Test writing three image measurements over two image sets'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        image_sets = [image_set_list.get_image_set(i)
                      for i in range(2)]
        for i in range(2):
            if i:
                m.next_image_set()
            for j in range(3):
                m.add_image_measurement("measurement_%d"%(j), "%d:%d"%(i,j))
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_sets[i],
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),4)
            self.assertEqual(header[0],"ImageNumber")
            for i in range(3):
                self.assertEqual(header[i+1],"measurement_%d"%(i))
            for i in range(2):
                row = reader.next()
                self.assertEqual(len(row),4)
                for j in range(3):
                    self.assertEqual(row[j+1],"%d:%d"%(i,j))
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_03_01_object_measurement(self):
        '''Test getting a single object measurement'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(1,))
        m.add_measurement("my_object", "my_measurement", mvalues)
        m.add_image_measurement("Count_my_object",1)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),3)
            self.assertEqual(header[0],"ImageNumber")
            self.assertEqual(header[1],"ObjectNumber")
            self.assertEqual(header[2],"my_measurement")
            row = reader.next()
            self.assertEqual(len(row),3)
            self.assertAlmostEqual(float(row[2]),mvalues[0],4)
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()

    def test_03_02_three_by_two_object_measurements(self):
        '''Test getting three measurements from two objects'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(2,3))
        for i in range(3):
            m.add_measurement("my_object", "measurement_%d"%(i), mvalues[:,i])
        m.add_image_measurement("Count_my_object",2)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),5)
            self.assertEqual(header[0],"ImageNumber")
            self.assertEqual(header[1],"ObjectNumber")
            for i in range(3):
                self.assertEqual(header[i+2],"measurement_%d"%(i))
            for i in range(2):
                row = reader.next()
                self.assertEqual(len(row),5)
                self.assertEqual(int(row[0]),1)
                self.assertEqual(int(row[1]),i+1)
                for j in range(3):
                    self.assertAlmostEqual(float(row[j+2]),mvalues[i,j])
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_03_03_get_measurements_from_two_objects(self):
        '''Get three measurements from four cells and two objects'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.add_object_group()
        module.object_groups[0].name.value = "object_0"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.object_groups[1].previous_file.value = True
        module.object_groups[1].name.value = "object_1"
        module.object_groups[1].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        # cell, measurement, object
        mvalues = np.random.uniform(size=(4,3,2))
        for oidx in range(2):
            for i in range(3):
                m.add_measurement("object_%d"%(oidx),
                                  "measurement_%d"%(i), mvalues[:,i,oidx])
        m.add_image_measurement("Count_object_0",4)
        m.add_image_measurement("Count_object_1",4)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "object_0")
        object_set.add_objects(cpo.Objects(), "object_1")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),8)
            for oidx in range(2):
                for i in range(3):
                    self.assertEqual(header[i+oidx*3+2],"object_%d"%(oidx))
            header = reader.next()
            self.assertEqual(len(header),8)
            self.assertEqual(header[0],"ImageNumber")
            self.assertEqual(header[1],"ObjectNumber")
            for oidx in range(2):
                for i in range(3):
                    self.assertEqual(header[i+oidx*3+2],"measurement_%d"%(i))

            for i in range(4):
                row = reader.next()
                self.assertEqual(len(row),8)
                self.assertEqual(int(row[0]), 1)
                self.assertEqual(int(row[1]), i+1)
                for j in range(3):
                    for k in range(2):
                        self.assertAlmostEqual(float(row[k*3+j+2]),mvalues[i,j,k])
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_04_01_object_with_metadata(self):
        '''Test writing objects with 2 pairs of 2 image sets w same metadata'''
        # +++backslash+++ here because Windows and join don't do well
        # if you have the raw backslash
        path = os.path.join(self.output_dir, "+++backslash+++g<tag>.csv")
        path = path.replace("\\","\\\\")
        path = path.replace("+++backslash+++","\\")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_object"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        for index,measurement,metadata in zip(range(4),mvalues,('foo','bar','bar','foo')):
            image_set = image_set_list.get_image_set(index)
            m.add_measurement("my_object", "my_measurement", np.array([measurement]))
            m.add_image_measurement("Metadata_tag", metadata)
            m.add_image_measurement("Count_my_object", 1)
            if index < 3:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        for i in range(4):
            module.post_run(workspace)
        for file_name,value_indexes in (("foo.csv",(0,3)),
                                        ("bar.csv",(1,2))):
            path = os.path.join(self.output_dir, file_name)
            fd = open(path,"r")
            try:
                reader = csv.reader(fd, delimiter=module.delimiter_char)
                header = reader.next()
                self.assertEqual(len(header),3)
                self.assertEqual(header[0],"ImageNumber")
                self.assertEqual(header[1],"ObjectNumber")
                self.assertEqual(header[2],"my_measurement")
                for value_index in value_indexes:
                    row = reader.next()
                    self.assertEqual(len(row),3)
                    self.assertEqual(int(row[0]), value_index+1)
                    self.assertEqual(int(row[1]), 1)
                    self.assertAlmostEqual(float(row[2]),
                                           mvalues[value_index],4)
                self.assertRaises(StopIteration,reader.next)
            finally:
                fd.close()
        
    def test_04_02_image_with_metadata(self):
        '''Test writing image data with 2 pairs of 2 image sets w same metadata'''
        path = os.path.join(self.output_dir, "+++backslash+++g<tag>.csv")
        path = path.replace("\\","\\\\")
        path = path.replace("+++backslash+++","\\")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        for index,measurement,metadata in zip(range(4),mvalues,('foo','bar','bar','foo')):
            image_set = image_set_list.get_image_set(index)
            m.add_image_measurement("my_measurement", measurement)
            m.add_image_measurement("Metadata_tag", metadata)
            if index < 3:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        for i in range(4):
            module.post_run(workspace)
        for file_name,value_indexes in (("foo.csv",(0,3)),
                                        ("bar.csv",(1,2))):
            path = os.path.join(self.output_dir, file_name)
            fd = open(path,"r")
            try:
                reader = csv.reader(fd, delimiter=module.delimiter_char)
                header = reader.next()
                self.assertEqual(len(header),3)
                d = {}
                self.assertTrue("ImageNumber" in header)
                self.assertTrue("my_measurement" in header)
                self.assertTrue("Metadata_tag" in header)
                for caption, index in zip(header,range(3)):
                    d[caption] = index
                for value_index in value_indexes:
                    row = reader.next()
                    self.assertEqual(len(row),3)
                    self.assertAlmostEqual(float(row[d["my_measurement"]]),
                                           mvalues[value_index],4)
                self.assertRaises(StopIteration,reader.next)
            finally:
                fd.close()
        
    def test_04_03_image_with_path_metadata(self):
        '''Test writing image data with 2 pairs of 2 image sets w same metadata'''
        path = os.path.join(self.output_dir, "+++backslash+++g<tag>")
        path = path.replace("\\","\\\\")
        path = path.replace("+++backslash+++","\\")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.directory_choice.value = E.DIR_CUSTOM_WITH_METADATA
        module.custom_directory.value = path
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = "output.csv"
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        for index,measurement,metadata in zip(range(4),mvalues,('foo','bar','bar','foo')):
            image_set = image_set_list.get_image_set(index)
            m.add_image_measurement("my_measurement", measurement)
            m.add_image_measurement("Metadata_tag", metadata)
            if index < 3:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        for i in range(4):
            module.post_run(workspace)
        for path_name,value_indexes in (("foo",(0,3)),
                                        ("bar",(1,2))):
            path = os.path.join(self.output_dir, path_name, "output.csv")
            fd = open(path,"r")
            try:
                reader = csv.reader(fd, delimiter=module.delimiter_char)
                header = reader.next()
                self.assertEqual(len(header),3)
                d = {}
                self.assertTrue("ImageNumber" in header)
                self.assertTrue("my_measurement" in header)
                self.assertTrue("Metadata_tag" in header)
                for caption, index in zip(header,range(3)):
                    d[caption] = index
                for value_index in value_indexes:
                    row = reader.next()
                    self.assertEqual(len(row),3)
                    self.assertAlmostEqual(float(row[d["my_measurement"]]),
                                           mvalues[value_index],4)
                self.assertRaises(StopIteration,reader.next)
            finally:
                fd.close()
                
    def test_04_04_image_measurement_custom_directory(self):
        '''Test writing an image measurement'''
        path = os.path.join(self.output_dir, "my_dir", "my_file.csv")
        cpprefs.set_headless()
        cpprefs.set_default_output_directory(self.output_dir)
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.directory_choice.value = E.DIR_CUSTOM
        module.custom_directory.value = "./my_dir"
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = "my_file.csv"
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_image_measurement("my_measurement", "Hello, world")
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),2)
            self.assertEqual(header[0], 'ImageNumber')
            self.assertEqual(header[1], "my_measurement")
            row = reader.next()
            self.assertEqual(row[0],"1")
            self.assertEqual(row[1],"Hello, world")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
            
    def test_05_01_aggregate_image_columns(self):
        """Test output of aggregate object data for images"""
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = True
        module.wants_aggregate_medians.value = True
        module.wants_aggregate_std.value = True
        m = cpmeas.Measurements()
        m.add_image_measurement("Count_my_objects", 6)
        np.random.seed(0)
        data = np.random.uniform(size=(6,))
        m.add_measurement("my_objects","my_measurement",data)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        fd = open(path,"r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),len(cpmeas.AGG_NAMES)+2)
            d = {}
            for index, caption in enumerate(header):
                d[caption]=index
            
            row = reader.next()
            self.assertEqual(row[d["Count_my_objects"]],"6")
            for agg in cpmeas.AGG_NAMES:
                value = (np.mean(data) if agg == cpmeas.AGG_MEAN
                         else np.std(data) if agg == cpmeas.AGG_STD_DEV
                         else np.median(data) if agg == cpmeas.AGG_MEDIAN
                         else np.NAN)
                self.assertAlmostEqual(float(row[d["%s_my_objects_my_measurement"%agg]]), value)
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_05_02_no_aggregate_image_columns(self):
        """Test output of aggregate object data for images"""
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = False
        module.wants_aggregate_medians.value = False
        module.wants_aggregate_std.value = False
        m = cpmeas.Measurements()
        m.add_image_measurement("Count_my_objects", 6)
        np.random.seed(0)
        data = np.random.uniform(size=(6,))
        m.add_measurement("my_objects","my_measurement",data)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),2)
            d = {}
            for index, caption in enumerate(header):
                d[caption]=index
            row = reader.next()
            self.assertEqual(row[d["Count_my_objects"]],"6")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_06_01_image_index_columns(self):
        '''Test presence of index column if add_indexes is on'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.add_indexes.value = True
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        data = ("The reverse side also has a reverse side. (Japanese proverb)",
                "When I was younger, I could remember anything, whether it had happened or not. (Mark Twain)",
                "A thing worth having is a thing worth cheating for. (W.C. Fields)"
                )
        for i in range(len(data)):
            image_set = image_set_list.get_image_set(i)
            m.add_image_measurement("quotation",data[i])
            if i < len(data)-1:
                m.next_image_set()
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),2)
            self.assertEqual(header[0],E.IMAGE_NUMBER)
            self.assertEqual(header[1],"quotation")
            for i in range(len(data)):
                row = reader.next()
                self.assertEqual(int(row[0]),i+1)
                self.assertEqual(row[1],data[i])
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
        
    def test_06_02_object_index_columns(self):
        '''Test presence of image and object index columns'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_objects"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_indexes.value = True
        m = cpmeas.Measurements()
        np.random.seed(0)
        # Three images with four objects each
        mvalues = np.random.uniform(size=(3,4))
        for image_idx in range(mvalues.shape[0]):
            if image_idx:
                m.next_image_set()
            m.add_image_measurement("Count_my_objects", mvalues.shape[1])
            m.add_measurement("my_objects", "my_measurement", mvalues[image_idx,:])
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),3)
            self.assertEqual(header[0],E.IMAGE_NUMBER)
            self.assertEqual(header[1],E.OBJECT_NUMBER)
            self.assertEqual(header[2],"my_measurement")
            for image_idx in range(mvalues.shape[0]):
                for object_idx in range(mvalues.shape[1]):
                    row = reader.next()
                    self.assertEqual(len(row),3)
                    self.assertEqual(int(row[0]),image_idx+1)
                    self.assertEqual(int(row[1]),object_idx+1)
                    self.assertAlmostEqual(float(row[2]),
                                           mvalues[image_idx,object_idx],4)
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_06_03_object_metadata_columns(self):
        '''Test addition of image metadata columns to an object metadata file'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_objects"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_metadata.value = True
        m = cpmeas.Measurements()
        np.random.seed(0)
        # Three images with four objects each
        mvalues = np.random.uniform(size=(3,4))
        for image_idx in range(mvalues.shape[0]):
            if image_idx:
                m.next_image_set()
            m.add_image_measurement("Count_my_objects", mvalues.shape[1])
            m.add_image_measurement("Metadata_Plate", "P-X9TRG")
            m.add_image_measurement("Metadata_Well", "C0%d"%(image_idx+1))
            m.add_measurement("my_objects", "my_measurement", mvalues[image_idx,:])
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),5)
            d = {}
            for index, column in enumerate(header):
                d[column]=index
            self.assertTrue(d.has_key("Metadata_Plate"))
            self.assertTrue(d.has_key("Metadata_Well"))
            self.assertTrue(d.has_key("my_measurement"))
            for image_idx in range(mvalues.shape[0]):
                for object_idx in range(mvalues.shape[1]):
                    row = reader.next()
                    self.assertEqual(len(row),5)
                    self.assertEqual(row[d["Metadata_Plate"]], "P-X9TRG")
                    self.assertEqual(row[d["Metadata_Well"]], "C0%d"%(image_idx+1))
                    self.assertAlmostEqual(float(row[d["my_measurement"]]),
                                           mvalues[image_idx,object_idx],4)
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_07_01_missing_measurements(self):
        '''Make sure ExportToExcel can continue when measurements are missing
        
        Regression test of IMG-361
        Take measurements for 3 image sets, some measurements missing
        from the middle one.
        '''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = "my_objects"
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_metadata.value = True
        m = cpmeas.Measurements()
        np.random.seed(0)
        # Three images with four objects each
        mvalues = np.random.uniform(size=(3,4))
        for image_idx in range(mvalues.shape[0]):
            if image_idx:
                m.next_image_set()
            m.add_image_measurement("Count_my_objects", mvalues.shape[1])
            if image_idx != 1:
                m.add_image_measurement("my_measurement", 100)
                m.add_measurement("my_objects", "my_measurement", mvalues[image_idx,:])
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        object_set.add_objects(cpo.Objects(), "my_objects")
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  image_set,
                                  object_set,
                                  m,
                                  image_set_list)
        module.post_run(workspace)
        try:
            fd = open(path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),3)
            d = {}
            for index, column in enumerate(header):
                d[column]=index
            self.assertTrue(d.has_key("my_measurement"))
            for image_idx in range(3):
                for object_idx in range(mvalues.shape[1]):
                    row = reader.next()
                    self.assertEqual(len(row),3)
                    if image_idx == 1:
                        self.assertEqual(row[d["my_measurement"]],str(np.NAN))
                    else:
                        self.assertAlmostEqual(float(row[d["my_measurement"]]),
                                               mvalues[image_idx,object_idx],4)
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
        
