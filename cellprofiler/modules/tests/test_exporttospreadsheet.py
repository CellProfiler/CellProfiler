'''test_exporttoexcel.py - test the ExportToExcel module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

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
from cellprofiler.modules.tests import example_images_directory

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
        self.assertFalse(module.excel_limits)
        self.assertTrue(module.pick_columns)
        self.assertFalse(module.wants_aggregate_means)
        self.assertTrue(module.wants_aggregate_medians)
        self.assertFalse(module.wants_aggregate_std)
        self.assertEqual(module.directory.dir_choice, 
                         E.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(module.directory.custom_path, r"./\<?Plate>")
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
        self.assertEqual(module.directory.dir_choice, 
                         E.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertFalse(module.wants_everything)
        for group, object_name in zip(module.object_groups,
                                      ("Image","Nuclei","PropCells",
                                       "DistanceCells","DistCytoplasm",
                                       "PropCytoplasm")):
            self.assertEqual(group.name, object_name)
            self.assertEqual(group.file_name, "%s.csv" % object_name)
            self.assertFalse(group.previous_file)
            self.assertTrue(group.wants_automatic_file_name)
    
    def test_000_06_load_v5(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9434

ExportToSpreadsheet:[module_num:1|svn_version:\'9434\'|variable_revision_number:5|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Tab
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    No longer used, always saved:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:Yes
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder
    Folder name://iodine/imaging_analysis/People/Lee
    Export all measurements?:No
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name:Image.csv
    Use the object name for the file name?:No
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Nuclei.csv
    Use the object name for the file name?:No
    Data to export:PropCells
    Combine these object measurements with those of the previous object?:No
    File name:PropCells.csv
    Use the object name for the file name?:No
    Data to export:DistanceCells
    Combine these object measurements with those of the previous object?:No
    File name:DistanceCells.csv
    Use the object name for the file name?:No
    Data to export:DistCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name:DistCytoplasm.csv
    Use the object name for the file name?:No
'''
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module,E.ExportToExcel))
        self.assertEqual(module.delimiter, E.DELIMITER_TAB)
        self.assertTrue(module.prepend_output_filename)
        self.assertEqual(module.directory.dir_choice, 
                         E.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.directory.custom_path, 
                         "//iodine/imaging_analysis/People/Lee")
        self.assertFalse(module.add_metadata)
        self.assertFalse(module.excel_limits)
        self.assertTrue(module.pick_columns)
        self.assertTrue(all([module.columns.get_measurement_object(x) == "Image"
                             for x in module.columns.selections]))
        self.assertEqual(len(module.columns.selections), 7)
        features = set([module.columns.get_measurement_feature(x)
                             for x in module.columns.selections])
        for feature in (
            "FileName_rawGFP", "FileName_IllumGFP", "FileName_IllumDNA",
            "FileName_rawDNA", "Metadata_SBS_Doses", "Metadata_Well",
            "Metadata_Controls"):
            self.assertTrue(feature in features)
        self.assertFalse(module.wants_aggregate_means)
        self.assertTrue(module.wants_aggregate_medians)
        self.assertFalse(module.wants_aggregate_std)
        self.assertFalse(module.wants_everything)
        self.assertEqual(len(module.object_groups), 5)
        for i, (object_name, file_name) in enumerate((
            ( "Image", "Image.csv"),
            ( "Nuclei", "Nuclei.csv"),
            ( "PropCells", "PropCells.csv"),
            ( "DistanceCells", "DistanceCells.csv"),
            ( "DistCytoplasm", "DistCytoplasm.csv"))):
            group = module.object_groups[i]
            self.assertFalse(group.previous_file)
            self.assertFalse(group.wants_automatic_file_name)
            self.assertEqual(group.name, object_name)
            self.assertEqual(group.file_name, file_name)
        
    def test_000_07_load_v6(self):
        data = r'''CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9434

ExportToSpreadsheet:[module_num:1|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Tab
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:Yes
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:No
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name:Image.csv
    Use the object name for the file name?:No
    Data to export:Nuclei
    Combine these object measurements with those of the previous object?:No
    File name:Nuclei.csv
    Use the object name for the file name?:No
    Data to export:PropCells
    Combine these object measurements with those of the previous object?:No
    File name:PropCells.csv
    Use the object name for the file name?:No
    Data to export:DistanceCells
    Combine these object measurements with those of the previous object?:No
    File name:DistanceCells.csv
    Use the object name for the file name?:No
    Data to export:DistCytoplasm
    Combine these object measurements with those of the previous object?:No
    File name:DistCytoplasm.csv
    Use the object name for the file name?:No

ExportToSpreadsheet:[module_num:2|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Default Input Folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes

ExportToSpreadsheet:[module_num:3|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Default Input Folder sub-folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes

ExportToSpreadsheet:[module_num:4|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Default Output Folder sub-folder\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes

ExportToSpreadsheet:[module_num:5|svn_version:\'9434\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:Yes
    Limit output to a size that is allowed in Excel?:Yes
    Select the columns of measurements to export?:No
    Calculate the per-image mean values for object measurements?:Yes
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:Yes
    Output file location:Elsewhere...\x7C//iodine/imaging_analysis/People/Lee
    Export all measurements?:Yes
    Press button to select measurements to export:Image\x7CFileName_rawGFP,Image\x7CFileName_IllumGFP,Image\x7CFileName_IllumDNA,Image\x7CFileName_rawDNA,Image\x7CMetadata_SBS_Doses,Image\x7CMetadata_Well,Image\x7CMetadata_Controls
    Data to export:Image
    Combine these object measurements with those of the previous object?:Yes
    File name:Image.csv
    Use the object name for the file name?:Yes
'''
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module,E.ExportToExcel))
        self.assertEqual(module.delimiter, E.DELIMITER_TAB)
        self.assertTrue(module.prepend_output_filename)
        self.assertEqual(module.directory.dir_choice, 
                         E.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.directory.custom_path, 
                         "//iodine/imaging_analysis/People/Lee")
        self.assertFalse(module.add_metadata)
        self.assertFalse(module.excel_limits)
        self.assertTrue(module.pick_columns)
        self.assertTrue(all([module.columns.get_measurement_object(x) == "Image"
                             for x in module.columns.selections]))
        self.assertEqual(len(module.columns.selections), 7)
        features = set([module.columns.get_measurement_feature(x)
                             for x in module.columns.selections])
        for feature in (
            "FileName_rawGFP", "FileName_IllumGFP", "FileName_IllumDNA",
            "FileName_rawDNA", "Metadata_SBS_Doses", "Metadata_Well",
            "Metadata_Controls"):
            self.assertTrue(feature in features)
        self.assertFalse(module.wants_aggregate_means)
        self.assertTrue(module.wants_aggregate_medians)
        self.assertFalse(module.wants_aggregate_std)
        self.assertFalse(module.wants_everything)
        self.assertEqual(len(module.object_groups), 5)
        for i, (object_name, file_name) in enumerate((
            ( "Image", "Image.csv"),
            ( "Nuclei", "Nuclei.csv"),
            ( "PropCells", "PropCells.csv"),
            ( "DistanceCells", "DistanceCells.csv"),
            ( "DistCytoplasm", "DistCytoplasm.csv"))):
            group = module.object_groups[i]
            self.assertFalse(group.previous_file)
            self.assertFalse(group.wants_automatic_file_name)
            self.assertEqual(group.name, object_name)
            self.assertEqual(group.file_name, file_name)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module,E.ExportToExcel))
        self.assertEqual(module.delimiter, E.DELIMITER_COMMA)
        self.assertTrue(module.prepend_output_filename)
        self.assertEqual(module.directory.dir_choice, 
                         E.DEFAULT_INPUT_FOLDER_NAME)
        self.assertEqual(module.directory.custom_path, 
                         "//iodine/imaging_analysis/People/Lee")
        self.assertTrue(module.add_metadata)
        self.assertTrue(module.excel_limits)
        self.assertFalse(module.pick_columns)
        self.assertTrue(module.wants_aggregate_means)
        self.assertFalse(module.wants_aggregate_medians)
        self.assertTrue(module.wants_aggregate_std)
        self.assertTrue(module.wants_everything)
        group = module.object_groups[0]
        self.assertTrue(group.previous_file)
        self.assertTrue(group.wants_automatic_file_name)
        
        for module, dir_choice in zip(pipeline.modules()[2:],
                                      (E.DEFAULT_INPUT_SUBFOLDER_NAME,
                                       E.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                                       E.ABSOLUTE_FOLDER_NAME)):
            self.assertTrue(isinstance(module,E.ExportToExcel))
            self.assertEqual(module.directory.dir_choice, dir_choice)

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
        fd = open(path,"r")
        try:
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
        file_name = "my_file.csv"
        path = os.path.join(self.output_dir, file_name)
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = True
        module.wants_everything.value = False
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = self.output_dir
        module.object_groups[0].name.value = cpmeas.EXPERIMENT
        module.object_groups[0].file_name.value = file_name
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        m.add_experiment_measurement("my_measurement", "Hello, world")
        m.add_experiment_measurement("Exit_Status", "Complete")
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
        fd = open(path,"r")
        try:
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            row = reader.next()
            self.assertEqual(len(row),2)
            self.assertEqual(row[0],"my_measurement")
            self.assertEqual(row[1],"Hello, world")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
            
    def test_01_04_img_887_no_experiment_file(self):
        '''Regression test of IMG-887: spirious experiment file
        
        ExportToSpreadsheet shouldn't generate an experiment file if
        the only measurements are Exit_Status or Complete.
        '''
        np.random.seed(14887)
        module = E.ExportToSpreadsheet()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = self.output_dir
        module.wants_everything.value = True
        m = cpmeas.Measurements()
        m.add_experiment_measurement("Exit_Status", "Complete")
        image_measurements = np.random.uniform(size=4)
        m.add_all_measurements(cpmeas.IMAGE, "my_measurement", image_measurements)
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
        path = os.path.join(self.output_dir, "Experiment.csv")
        self.assertFalse(os.path.exists(path))
        path = os.path.join(self.output_dir, "Image.csv")
        self.assertTrue(os.path.exists(path))
        
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
        fd = open(path,"r")
        try:
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
        fd = open(path,"r")
        try:
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
        module.directory.dir_choice = E.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = path
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = "output.csv"
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        np.random.seed(0)
        mvalues = np.random.uniform(size=(4,))
        image_set_list = cpi.ImageSetList()
        metadata_values = ('foo','bar','bar','foo')
        for index, (measurement, metadata) in \
            enumerate(zip(mvalues,metadata_values)):
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
        module.directory.dir_choice = E.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.directory.custom_path = "./my_dir"
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
            
    def test_04_05_unicode_image_metadata(self):
        '''Write image measurements containing unicode characters'''
        path = os.path.join(self.output_dir, "my_dir", "my_file.csv")
        cpprefs.set_headless()
        cpprefs.set_default_output_directory(self.output_dir)
        module = E.ExportToSpreadsheet()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.directory.dir_choice = E.DEFAULT_OUTPUT_SUBFOLDER_NAME
        module.directory.custom_path = "./my_dir"
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = "my_file.csv"
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        metadata_value = u"\u2211(Hello, world)"
        m.add_image_measurement("my_measurement", metadata_value)
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
            self.assertEqual(unicode(row[1], 'utf8'), metadata_value)
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
            
    def test_05_03_aggregate_and_filtered(self):
        '''Regression test of IMG-987
        
        A bug in ExportToSpreadsheet caused it to fail to write any
        aggregate object measurements if measurements were filtered by
        pick_columns.
        '''
        image_path = os.path.join(self.output_dir, "my_image_file.csv")
        object_path = os.path.join(self.output_dir, "my_object_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = image_path
        module.object_groups[0].wants_automatic_file_name.value = False
        module.add_object_group()
        module.object_groups[1].name.value = "my_objects"
        module.object_groups[1].file_name.value = object_path
        module.object_groups[1].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = True
        module.wants_aggregate_medians.value = False
        module.wants_aggregate_std.value = False
        module.pick_columns.value = True
        columns = [module.columns.make_measurement_choice(ob, feature)
                   for ob, feature in (
                       (cpmeas.IMAGE, "ImageNumber"),
                       (cpmeas.IMAGE, "Count_my_objects"),
                       (cpmeas.IMAGE, "first_measurement"),
                       ("my_objects", "my_measurement"),
                       ("my_objects", "ImageNumber"),
                       ("my_objects", "Number_Object_Number")
                   )]
        module.columns.value = module.columns.get_value_string(columns)
        
        m = cpmeas.Measurements()
        np.random.seed(0)
        data = np.random.uniform(size=(6,))
        m.add_image_measurement("Count_my_objects", 6)
        m.add_image_measurement("first_measurement", np.sum(data))
        m.add_image_measurement("another_measurement", 43.2)
        m.add_measurement("my_objects","Number_Object_Number", np.arange(1,7))
        m.add_measurement("my_objects","my_measurement",data)
        m.add_measurement("my_objects","my_filtered_measurement", 
                          np.random.uniform(size=(6,)))
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
            fd = open(image_path,"r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),4)
            expected_image_columns = (
                "ImageNumber", "Count_my_objects", "first_measurement",
                "Mean_my_objects_my_measurement")
            d = {}
            for index, caption in enumerate(header):
                self.assertTrue(caption in expected_image_columns)
                d[caption]=index
            row = reader.next()
            self.assertEqual(row[d["ImageNumber"]], "1")
            self.assertEqual(row[d["Count_my_objects"]],"6")
            self.assertAlmostEqual(float(row[d["first_measurement"]]), np.sum(data))
            self.assertAlmostEqual(float(row[d["Mean_my_objects_my_measurement"]]), 
                                   np.mean(data))
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
        try:
            fd = open(object_path, "r")
            reader = csv.reader(fd, delimiter=module.delimiter_char)
            header = reader.next()
            self.assertEqual(len(header),4)
            expected_object_columns = (
                "ImageNumber", "ObjectNumber", "Number_Object_Number", 
                "my_measurement")
            d = {}
            for index, caption in enumerate(header):
                self.assertTrue(caption in expected_object_columns)
                d[caption]=index
            for index, row in enumerate(reader):
                self.assertEqual(row[d["ImageNumber"]],  "1")
                self.assertEqual(int(row[d["ObjectNumber"]]), index+1)
                # all object values get written as floats
                self.assertEqual(int(float(row[d["Number_Object_Number"]])), index+1)
                self.assertAlmostEqual(float(row[d["my_measurement"]]),
                                       data[index])
        finally:
            fd.close()
                
    def test_06_01_image_index_columns(self):
        '''Test presence of index column'''
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
        
    def make_pipeline(self, csv_text):
        import cellprofiler.modules.loaddata as L
        
        handle, name = tempfile.mkstemp("csv")
        fd = os.fdopen(handle, 'w')
        fd.write(csv_text)
        fd.close()
        csv_path, csv_file = os.path.split(name) 
        module = L.LoadText()
        module.csv_directory.dir_choice = L.ABSOLUTE_FOLDER_NAME
        module.csv_directory.custom_path = csv_path
        module.csv_file_name.value = csv_file
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        def error_callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(error_callback)
        return pipeline, module, name
    
    def add_gct_settings(self,output_csv_filename):
        module = E.ExportToSpreadsheet()
        module.module_num = 2
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = cpmeas.IMAGE
        module.object_groups[0].file_name.value = output_csv_filename
        module.object_groups[0].wants_automatic_file_name.value = False
        module.wants_aggregate_means.value = False
        module.wants_aggregate_medians.value = False
        module.wants_aggregate_std.value = False
        module.wants_genepattern_file.value = True
        return module
    
    def test_08_01_basic_gct_check(self):
    # LoadData with data
        input_dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        metadata_name = "Metadata_Bar"
        info = ('Image_FileName_Foo','Image_PathName_Foo',metadata_name,input_dir,input_dir)
        csv_text = '''"%s","%s","%s"
"Channel1-01-A-01.tif","%s","Hi"
"Channel1-02-A-02.tif","%s","Hello"
'''%info
        pipeline, module, input_filename = self.make_pipeline(csv_text)
        
        output_csv_filename = os.path.join(tempfile.mkdtemp(), "my_file.csv")
            
        # ExportToSpreadsheet
        module = self.add_gct_settings(output_csv_filename)
        module.how_to_specify_gene_name.value = "Image filename"
        module.use_which_image_for_gene_name.value = "Foo"
        pipeline.add_module(module)
        
        try:
            m = pipeline.run()
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            p,n = os.path.splitext(output_csv_filename)
            output_gct_filename = p + '.gct'
            fd = open(output_gct_filename,"r")
            reader = csv.reader(fd, delimiter="\t")
            row = reader.next()
            self.assertEqual(len(row),1)
            self.assertEqual(row[0],"#1.2")
            row = reader.next()
            self.assertEqual(len(row),2)
            self.assertEqual(row[0],"2")
            self.assertEqual(row[1],"1")
            row = reader.next()
            self.assertEqual(len(row),3)
            self.assertEqual(row[0].lower(),"name")
            self.assertEqual(row[1].lower(),"description")
            self.assertEqual(row[2],metadata_name)
            row = reader.next()
            self.assertEqual(row[1],input_dir)
        finally:
            try:
                os.remove(input_filename)
                os.remove(output_csv_filename)
            except:
                print("Failed to clean up files")
            
    def test_08_02_make_gct_file_with_filename(self):
            
        # LoadData with data
        input_dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        metadata_name = "Metadata_Bar"
        info = ('Image_FileName_Foo','Image_PathName_Foo',metadata_name,input_dir,input_dir)
        csv_text = '''"%s","%s","%s"
"Channel1-01-A-01.tif","%s","Hi"
"Channel1-02-A-02.tif","%s","Hello"
'''%info
        pipeline, module, input_filename = self.make_pipeline(csv_text)
        
        output_csv_filename = os.path.join(tempfile.mkdtemp(), "my_file.csv")
            
        # ExportToSpreadsheet
        module = self.add_gct_settings(output_csv_filename)
        module.how_to_specify_gene_name.value = "Image filename"
        module.use_which_image_for_gene_name.value = "Foo"
        pipeline.add_module(module)
        
        try:
            m = pipeline.run()
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            p,n = os.path.splitext(output_csv_filename)
            output_gct_filename = p + '.gct'
            fd = open(output_gct_filename,"r")
            reader = csv.reader(fd, delimiter="\t")
            row = reader.next()
            row = reader.next()
            row = reader.next()
            row = reader.next()
            self.assertEqual(row[0],"Channel1-01-A-01.tif")
            row = reader.next()
            self.assertEqual(row[0],"Channel1-02-A-02.tif")
            fd.close()
        finally:
            os.remove(input_filename)
            os.remove(output_csv_filename)
              
    def test_08_03_make_gct_file_with_metadata(self):
            
        # LoadData with data
        input_dir = os.path.join(example_images_directory(), "ExampleSBSImages")
        metadata_name = "Metadata_Bar"
        info = ('Image_FileName_Foo','Image_PathName_Foo',metadata_name,input_dir,input_dir)
        csv_text = '''"%s","%s","%s"
"Channel1-01-A-01.tif","%s","Hi"
"Channel1-02-A-02.tif","%s","Hello"
'''%info
        pipeline, module, input_filename = self.make_pipeline(csv_text)
        
        output_csv_filename = os.path.join(tempfile.mkdtemp(), "my_file.csv")
            
        # ExportToSpreadsheet
        module = self.add_gct_settings(output_csv_filename)
        module.how_to_specify_gene_name.value = "Metadata"
        module.gene_name_column.value = "Metadata_Bar"
        pipeline.add_module(module)
        
        try:
            m = pipeline.run()
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            p,n = os.path.splitext(output_csv_filename)
            output_gct_filename = p + '.gct'
            fd = open(output_gct_filename,"r")
            reader = csv.reader(fd, delimiter="\t")
            row = reader.next()
            row = reader.next()
            row = reader.next()
            row = reader.next()
            self.assertEqual(row[0],"Hi")
            row = reader.next()
            self.assertEqual(row[0],"Hello")
            fd.close()
        finally:
            os.remove(input_filename)
            os.remove(output_csv_filename)

    def test_09_01_relationships_file(self):
        r = np.random.RandomState()
        r.seed(91)
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToSpreadsheet()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.wants_everything.value = False
        module.object_groups[0].name.value = E.OBJECT_RELATIONSHIPS
        module.object_groups[0].file_name.value = path
        module.object_groups[0].wants_automatic_file_name.value = False
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        for i in range(0,10):
            image_set = image_set_list.get_image_set(i)
            m.add_image_measurement(cpp.IMAGE_NUMBER, i+1)
            m.add_image_measurement(cpp.GROUP_NUMBER, 0)
            m.add_image_measurement(cpp.GROUP_INDEX, i)
            if i < 9:
                m.next_image_set()
        my_relationship = "BlahBlah"
        my_object_name1 = "ABC"
        my_object_name2 = "DEF"
        my_group_indexes1 = r.randint(1,10, size=10)
        my_object_numbers1 = r.randint(1,10, size=10)
        my_group_indexes2 = r.randint(1,10, size=10)
        my_object_numbers2 = r.randint(1,10, size=10)
        m.add_relate_measurement(1, my_relationship, 
                                 my_object_name1, my_object_name2,
                                 my_group_indexes1, my_object_numbers1, 
                                 my_group_indexes2, my_object_numbers2)
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(), m,
                                  image_set_list)
        fd = None
        try:
            module.post_run(workspace)
            fd = open(path, "rb")
            rdr = csv.reader(fd)
            header = rdr.next()
            for heading, expected in zip(
                header, ["Module", "Module Number", "Relationship",
                         "First Object Name", "First Image Number", 
                         "First Object Number", "Second Object Name",
                         "Second Image Number", "Second Object Number"]):
                self.assertEqual(heading, expected)
            for i in range(len(my_group_indexes1)):
                (module_name, module_number, relationship, 
                 object_name_1, image_number_1, object_number_1, 
                 object_name_2, image_number_2, object_number_2) = rdr.next()
                self.assertEqual(module_name, module.module_name)
                self.assertEqual(int(module_number), module.module_num)
                self.assertEqual(relationship, my_relationship)
                self.assertEqual(object_name_1, my_object_name1)
                self.assertEqual(int(image_number_1), my_group_indexes1[i]+1)
                self.assertEqual(int(object_number_1), my_object_numbers1[i])
                self.assertEqual(object_name_2, my_object_name2)
                self.assertEqual(int(image_number_2), my_group_indexes2[i]+1)
                self.assertEqual(int(object_number_2), my_object_numbers2[i])
        finally:
            try:
                if fd is not None:
                    fd.close()
                os.remove(path)
            except:
                pass

