"""test_measureimagearea.py - test the MeasureImageArea module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revison: 1 $"

import base64
import numpy as np
import StringIO
import unittest
import zlib

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
        
        columns = module.get_measurement_columns(workspace.pipeline)
        features = m.get_feature_names(cpmm.IMAGE)
        self.assertEqual(len(columns), len(features))
        for column in columns:
            self.assertTrue(column[1] in features)
    
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
        
    def test_02_01_get_measurement_columns(self):
        module = mia.MeasureImageAreaOccupied()
        module.object_name.value = OBJECTS_NAME
        columns = module.get_measurement_columns(None)
        expected = ((cpmm.IMAGE, "AreaOccupied_AreaOccupied_%s"%OBJECTS_NAME,
                     cpmm.COLTYPE_FLOAT),
                    (cpmm.IMAGE, "AreaOccupied_TotalArea_%s"%OBJECTS_NAME,
                     cpmm.COLTYPE_FLOAT))
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf==ef for cf,ef in zip(column,ex)])
                                 for ex in expected]))

    def test_03_01_load_v1(self):
        '''Load a pipeline with MeasureImageAreaOccupied revision_number 1'''
        data = ('eJztWlFv2zYQlmMnWFpgyIphK9AXPjZdLEheg6VBkdqNl81b7RiN0aAouo2R'
                '6JgDRRoSlcYbCvRxP2mPe9zP2U+YKFOxzDiVbCe2UkiAIN+RH7+74+lE0WrW'
                'Oi9qz8G2boBmrVPuYoJAm0DeZa6zCyjfAvsughzZgNFdcBxcf/IJMHaAaeya'
                '27uVbVAxjCfabEeh0fw8uPz7SNPWgutnwbkim1alXIidQj5CnGN66q1qJe2+'
                '1P8TnK+gi+EJQa8g8ZE3ooj0DdplnUH/oqnJbJ+gFnTinYOj5TsnyPUOuxFQ'
                'NrfxOSJH+A+kuBB1e4nOsIcZlXg5vqq94GVc4RVx2PhyFIeCEodicD6I6UX/'
                'H7VR/9KEuH0R678hZUxtfIZtHxKAHXh6YYUYz0gYrzg2XlGrt2ohbicBt6bY'
                'sRbG2SIID3mrCfgNBS/ODjrn5e/PocWBA7nVuw47kvxfGcOvaC2WDlcYwxW0'
                'b2W8k+xdVewVsmlsPTbmwB9xiKmWLu53FLyQ6wxQxoHvyRthlrx5HWSdiltT'
                'cNER4da19HyzzlMa3DR2Jt2fX2vj8RVyHXWhTzhoiJsT1LGLLM7cwdLjPA2u'
                'mmDnXcVvIR9yzwc/EHYCydz81xGnNDhTN5Zi5yx139CN8Ngy5Y8r7MjCfVga'
                'w5WE7eY8dibVyfjzdkPK+z1IKSKVNPm8ruCF3KAcUQ/zwRXxusk8iZ7LWbdb'
                'fS6aKXFqXpnGzdqp5mOLUTSLf99N4JvGzg8JfD9r4/Mp5F8ePms/FQt6tKd/'
                's/mrkI4RIS/Zu703tXL77Wak2WfEd+jeG6P85O2f5lbl/bDzEQ6QoXIzdbyu'
                '+znUS+DbUfwWsrD9NYKudOjx+82yUDUZ5T2pq0hdHQ5GmptaFy2oXl2qk4uY'
                'n6R4Taoz+wPO+gR6TmycrK0j1fu3klE7Z61Pqp3GnOuZDwl8WalPaeKV5fq0'
                '6PeYZa7Ds2inoW/fmveqzjsGrKDOenKnZxl2z/K+cozwaU9sO56JDTZqodh4'
                'WYv7pPXAAXPRqct8as/P37833b7gIv0MNxGFo/3040zKU3byO7L4aKBF51uM'
                'H2Bqo35svGXWo+ucv0n7qKP5G7p9m/zNcTkux+W4rOGqMVxeh3Pcp4pLWmfd'
                '08bzXMjM5wRTdGmhdZv8zutCjsti/qR9P7st/ua4HJfjlof7qzDCqftO6r6o'
                '6P9bjGdSfXqkjdcnIVuIkL7LxHeHru6EH8d5OmHQHn6dpr8IfjZiH6qF+2EJ'
                'PFWFp3oVD7YR5bg76LsBm8+ZAzm29IbUtgNtLdIK3vME3gOF9+AqXgdBz3dR'
                '6CJ0EWSW5fcxsvXmsCF0uBY0HMqGy/O4PoE/Ph8rgfTVg+JH51+d91E+/Pds'
                'Fr5isXDp/8u7CbhSzCZxCPzf2nR59/Aj/SMfF9X/f/JWyJ0=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, mia.MeasureImageAreaOccupied))
        self.assertEqual(module.object_name.value, "Nuclei")
