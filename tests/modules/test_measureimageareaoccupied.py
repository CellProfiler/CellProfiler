"""test_measureimagearea.py - test the MeasureImageArea module
"""

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmm
import cellprofiler.object as cpo
from centrosome.outline import outline
import cellprofiler.modules.measureimageareaoccupied as mia

OBJECTS_NAME = "MyObjects"


class TestMeasureImageArea(unittest.TestCase):
    def make_workspace(self, labels, parent_image=None):
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        objects.parent_image = parent_image
        object_set.add_objects(objects, OBJECTS_NAME)

        pipeline = cpp.Pipeline()
        module = mia.MeasureImageAreaOccupied()
        module.module_num = 1
        module.operands[0].operand_objects.value = OBJECTS_NAME
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(pipeline, module,
                                  image_set_list.get_image_set(0),
                                  object_set,
                                  cpmm.Measurements(),
                                  image_set_list)
        return workspace

    def test_00_00_zeros(self):
        workspace = self.make_workspace(np.zeros((10, 10), int))
        module = workspace.module
        module.operands[0].operand_choice.value = "Objects"
        module.run(workspace)
        m = workspace.measurements

        def mn(x):
            return "AreaOccupied_%s_%s" % (x, module.operands[0].operand_objects.value)

        self.assertEqual(m.get_current_measurement("Image", mn("AreaOccupied")), 0)
        self.assertEqual(m.get_current_measurement("Image", mn("TotalArea")), 100)

        columns = module.get_measurement_columns(workspace.pipeline)
        features = m.get_feature_names(cpmm.IMAGE)
        self.assertEqual(len(columns), len(features))
        for column in columns:
            self.assertTrue(column[1] in features)

    def test_01_01_one_object(self):
        labels = np.zeros((10, 10), int)
        labels[2:7, 3:8] = 1
        area_occupied = np.sum(labels)
        workspace = self.make_workspace(labels)
        module = workspace.module
        module.operands[0].operand_choice.value = "Objects"
        module.run(workspace)
        m = workspace.measurements

        def mn(x):
            return "AreaOccupied_%s_%s" % (x, module.operands[0].operand_objects.value)

        self.assertEqual(m.get_current_measurement("Image", mn("AreaOccupied")), area_occupied)
        self.assertEqual(m.get_current_measurement("Image", mn("TotalArea")), 100)

    def test_01_02_object_with_cropping(self):
        labels = np.zeros((10, 10), int)
        labels[0:7, 3:8] = 1
        mask = np.zeros((10, 10), bool)
        mask[1:9, 1:9] = True
        image = cpi.Image(np.zeros((10, 10)), mask=mask)
        area_occupied = np.sum(labels[mask])
        perimeter = np.sum(outline(np.logical_and(labels, mask)))
        total_area = np.sum(mask)
        workspace = self.make_workspace(labels, image)
        module = workspace.module
        module.operands[0].operand_choice.value = "Objects"
        module.run(workspace)
        m = workspace.measurements

        def mn(x):
            return "AreaOccupied_%s_%s" % (x, module.operands[0].operand_objects.value)

        self.assertEqual(m.get_current_measurement("Image", mn("AreaOccupied")), area_occupied)
        self.assertEqual(m.get_current_measurement("Image", mn("Perimeter")), perimeter)
        self.assertEqual(m.get_current_measurement("Image", mn("TotalArea")), total_area)

    def test_02_01_get_measurement_columns(self):
        module = mia.MeasureImageAreaOccupied()
        module.operands[0].operand_objects.value = OBJECTS_NAME
        module.operands[0].operand_choice.value = "Objects"
        columns = module.get_measurement_columns(None)
        expected = ((cpmm.IMAGE, "AreaOccupied_AreaOccupied_%s" % OBJECTS_NAME,
                     cpmm.COLTYPE_FLOAT),
                    (cpmm.IMAGE, "AreaOccupied_Perimeter_%s" % OBJECTS_NAME,
                     cpmm.COLTYPE_FLOAT),
                    (cpmm.IMAGE, "AreaOccupied_TotalArea_%s" % OBJECTS_NAME,
                     cpmm.COLTYPE_FLOAT))
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))
