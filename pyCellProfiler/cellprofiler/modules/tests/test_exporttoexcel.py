'''test_exporttoexcel.py - test the ExportToExcel module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import csv
import os
import numpy as np
import tempfile
import unittest

import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.modules.exporttoexcel as E 

class TestExportToExcel(unittest.TestCase):

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        for file_name in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file_name))
        os.rmdir(self.output_dir)
        self.output_dir = None

    def test_00_00_no_measurements(self):
        '''Test an image set with objects but no measurements'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = "my_object"
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
            self.assertEqual(len(header),1)
            self.assertEqual(header[0],"my_measurement")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_01_01_experiment_measurement(self):
        '''Test writing one experiment measurement'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.EXPERIMENT
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.EXPERIMENT
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.EXPERIMENT
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
        file_name = cpprefs.get_output_file_name()[:-4]+"my_file.csv"
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
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.IMAGE
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
            self.assertEqual(len(header),1)
            self.assertEqual(header[0],"my_measurement")
            row = reader.next()
            self.assertEqual(row[0],"Hello, world")
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()

    def test_02_02_three_by_two_image_measurements(self):
        '''Test writing three image measurements over two image sets'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.IMAGE
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
            self.assertEqual(len(header),3)
            for i in range(3):
                self.assertEqual(header[i],"measurement_%d"%(i))
            for i in range(2):
                row = reader.next()
                self.assertEqual(len(row),3)
                for j in range(3):
                    self.assertEqual(row[j],"%d:%d"%(i,j))
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_03_01_object_measurement(self):
        '''Test getting a single object measurement'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = "my_object"
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
            self.assertEqual(len(header),1)
            self.assertEqual(header[0],"my_measurement")
            row = reader.next()
            self.assertEqual(len(row),1)
            self.assertAlmostEqual(float(row[0]),mvalues[0],4)
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()

    def test_03_02_three_by_two_object_measurements(self):
        '''Test getting three measurements from two objects'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = "my_object"
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
            self.assertEqual(len(header),3)
            for i in range(3):
                self.assertEqual(header[i],"measurement_%d"%(i))
            for i in range(2):
                row = reader.next()
                self.assertEqual(len(row),3)
                for j in range(3):
                    self.assertAlmostEqual(float(row[j]),mvalues[i,j])
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_03_03_get_measurements_from_two_objects(self):
        '''Get three measurements from four cells and two objects'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.add_object_group()
        module.object_groups[0][E.OG_OBJECT_NAME].value = "object_0"
        module.object_groups[0][E.OG_FILE_NAME].value = path
        module.object_groups[1][E.OG_PREVIOUS_FILE].value = True
        module.object_groups[1][E.OG_OBJECT_NAME].value = "object_1"
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
            self.assertEqual(len(header),6)
            for oidx in range(2):
                for i in range(3):
                    self.assertEqual(header[i+oidx*3],"object_%d"%(oidx))
            header = reader.next()
            self.assertEqual(len(header),6)
            for oidx in range(2):
                for i in range(3):
                    self.assertEqual(header[i+oidx*3],"measurement_%d"%(i))

            for i in range(4):
                row = reader.next()
                self.assertEqual(len(row),6)
                for j in range(3):
                    for k in range(2):
                        self.assertAlmostEqual(float(row[k*3+j]),mvalues[i,j,k])
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
    
    def test_04_01_object_with_metadata(self):
        '''Test writing objects with 2 pairs of 2 image sets w same metadata'''
        # +++backslash+++ here because Windows and join don't do well
        # if you have the raw backslash
        path = os.path.join(self.output_dir, "+++backslash+++g<tag>.csv")
        path = path.replace("+++backslash+++","\\")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = "my_object"
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
                self.assertEqual(len(header),1)
                self.assertEqual(header[0],"my_measurement")
                for value_index in value_indexes:
                    row = reader.next()
                    self.assertEqual(len(row),1)
                    self.assertAlmostEqual(float(row[0]),
                                           mvalues[value_index],4)
                self.assertRaises(StopIteration,reader.next)
            finally:
                fd.close()
        
    def test_04_02_image_with_metadata(self):
        '''Test writing image data with 2 pairs of 2 image sets w same metadata'''
        path = os.path.join(self.output_dir, "+++backslash+++g<tag>.csv")
        path = path.replace("+++backslash+++","\\")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.IMAGE
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
                self.assertEqual(len(header),2)
                d = {}
                self.assertTrue("my_measurement" in header)
                self.assertTrue("Metadata_tag" in header)
                for caption, index in zip(header,range(2)):
                    d[caption] = index
                for value_index in value_indexes:
                    row = reader.next()
                    self.assertEqual(len(row),2)
                    self.assertAlmostEqual(float(row[d["my_measurement"]]),
                                           mvalues[value_index],4)
                self.assertRaises(StopIteration,reader.next)
            finally:
                fd.close()
        
    def test_05_01_aggregate_image_columns(self):
        """Test output of aggregate object data for images"""
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.IMAGE
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
            self.assertEqual(len(header),len(cpmeas.AGG_NAMES)+1)
            d = {}
            for index, caption in zip(range(len(header)),header):
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
    
    def test_06_01_image_index_columns(self):
        '''Test presence of index column if add_indexes is on'''
        path = os.path.join(self.output_dir, "my_file.csv")
        module = E.ExportToExcel()
        module.module_num = 1
        module.prepend_output_filename.value = False
        module.add_indexes.value = True
        module.object_groups[0][E.OG_OBJECT_NAME].value = cpmeas.IMAGE
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
        module.object_groups[0][E.OG_OBJECT_NAME].value = "my_objects"
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
        module.object_groups[0][E.OG_OBJECT_NAME].value = "my_objects"
        module.object_groups[0][E.OG_FILE_NAME].value = path
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
            self.assertEqual(len(header),3)
            d = {}
            for index, column in zip(range(3),header):
                d[column]=index
            self.assertTrue(d.has_key("Metadata_Plate"))
            self.assertTrue(d.has_key("Metadata_Well"))
            self.assertTrue(d.has_key("my_measurement"))
            for image_idx in range(mvalues.shape[0]):
                for object_idx in range(mvalues.shape[1]):
                    row = reader.next()
                    self.assertEqual(len(row),3)
                    self.assertEqual(row[d["Metadata_Plate"]], "P-X9TRG")
                    self.assertEqual(row[d["Metadata_Well"]], "C0%d"%(image_idx+1))
                    self.assertAlmostEqual(float(row[d["my_measurement"]]),
                                           mvalues[image_idx,object_idx],4)
            self.assertRaises(StopIteration,reader.next)
        finally:
            fd.close()
        