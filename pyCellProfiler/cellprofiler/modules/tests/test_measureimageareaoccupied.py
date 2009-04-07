"""test_measureimagearea.py - test the MeasureImageArea module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revison: 1 $"

import numpy as np
import unittest

import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmm
import cellprofiler.objects as cpo

import cellprofiler.modules.measureimageareaoccupied as mia

OBJECTS_NAME = "MyObjects"
class TestMeasureImageArea(unittest.TestCase):
    def make_workspace(self, labels, parent_image = None):
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        objects.parent_image = parent_image
        object_set.add_objects(objects, OBJECTS_NAME)
        
        pipeline = cpp.Pipeline()
        module = mia.MeasureImageAreaOccupied()
        module.module_num = 1
        module.object_name.value = OBJECTS_NAME
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(pipeline, module, 
                                  image_set_list.get_image_set(0),
                                  object_set,
                                  cpmm.Measurements(),
                                  image_set_list)
        return workspace
        
    def test_00_00_zeros(self):
        workspace = self.make_workspace(np.zeros((10,10),int))
        module = workspace.module
        module.run(workspace)
        m = workspace.measurements
        def mn(x):
            return "AreaOccupied_%s_%s"%(x, module.object_name.value)
        
        self.assertEqual(m.get_current_measurement("Image",mn("AreaOccupied"))[0], 0)
        self.assertEqual(m.get_current_measurement("Image",mn("TotalArea"))[0],100)
    
    def test_01_01_one_object(self):
        labels = np.zeros((10,10),int)
        labels[2:7,3:8] = 1
        area_occupied = np.sum(labels)
        workspace = self.make_workspace(labels)
        module = workspace.module
        module.run(workspace)
        m = workspace.measurements
        def mn(x):
            return "AreaOccupied_%s_%s"%(x, module.object_name.value)
        
        self.assertEqual(m.get_current_measurement("Image",mn("AreaOccupied"))[0], area_occupied)
        self.assertEqual(m.get_current_measurement("Image",mn("TotalArea"))[0],100)
    
    def test_01_02_object_with_cropping(self):
        labels = np.zeros((10,10),int)
        labels[0:7,3:8] = 1
        mask = np.zeros((10,10),bool)
        mask[1:9,1:9] = True
        image = cpi.Image(np.zeros((10,10)),mask=mask)
        area_occupied = np.sum(labels[mask])
        total_area = np.sum(mask)
        workspace = self.make_workspace(labels, image)
        module = workspace.module
        module.run(workspace)
        m = workspace.measurements
        def mn(x):
            return "AreaOccupied_%s_%s"%(x, module.object_name.value)
        
        self.assertEqual(m.get_current_measurement("Image",mn("AreaOccupied"))[0], area_occupied)
        self.assertEqual(m.get_current_measurement("Image",mn("TotalArea"))[0],total_area)
        
