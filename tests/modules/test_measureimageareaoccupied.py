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

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, mia.MeasureImageAreaOccupied))
        self.assertEqual(module.operands[0].operand_objects.value, "Nuclei")

    def test_03_02_load_v2(self):
        data = ('eJztW92P2kYQN4Q79RopujylqhrJj7n2QOYalMupvUCg16KGDwWUKIrSds9e'
                'YCuzi+z13dEqUh/7p/Yxj33sLtjYbM3ZGJuPxJYsmPH+9rczO7szGLtR6b6o'
                'PJdLBUVuVLr5HtKh3NYB7RFjeCZjeixXDQgo1GSCz+TuwJJbKpWLJbmonJWe'
                'nBUV+URRnkrRjky9cY99nH4vSfvs8zN2Zu1Le7ac8Zxc7kBKEe6be1JO+sLW'
                'f2DnK2AgcKnDV0C3oOlSOPo67pHueDS71CCapcMmGHobs6NpDS+hYbZ6DtC+'
                '3EY3UO+gP6BggtPsJbxCJiLYxtv9i9oZL6ECb2dAri8MNhyh/+eAqoMOZTMw'
                'r+d+G33l+i0j+C3HzocePW//k+S2z/n4+b6n/aEtI6yhK6RZQJfREPRno+b9'
                'KQH93Znr745Ua1YmuNMA3L4wDi43LVWHSIoFXw7AHwp4fnbhDc3/cANY8A/5'
                'lKzDjj0Bz+Uq1HUzpP+zc/is1CThcJk5XEb6Voo+3qJy/FiRwvn9roDnctsg'
                'I9AHlC2iiT7qONgSQjjkOD4X8FyuERkTKlumvQ6jxP8btnrC8B8I/FyujikZ'
                '6cAcSi7/tsbPIlyQ3X7z36KmJf+ok0ugh/b7Iv6g/e+BwM/lGuwBS6dynW9+'
                'cg0ZUKXEGK80/8viigUllrj38u4LeOdw8Af2Z5zzFiX/KAVlchwX7S+ecUW1'
                'Z93zFoTLzeFy3ObiJuxLcn7CrNOikuy8Rt3PwsxPSFwpLvviXL9++aaOKcQm'
                'omNPP1HiowNVgjVgjFsW1RFeWD8uY0/QPuithw9tuToAGEP9JE5/LLvOFJ/9'
                'IMl17dTbcazDMPHdJBgmaZ9YlxYj4p6ExIXZL+K0L4w/46wz/fJ295rIKqsz'
                'TXulrmLvXwH8Pwv8XP7l0bP2d/wGBDwvfHP0K5des5L1Jbk+f1vJt98dOZoq'
                '0a0hPn+r5J+++7N4fPJ+2riDGHKiPJqNI8gPYev9KPvfa4j6A3775IrfKMCq'
                '8zN+Fb8OAsZxKoyDy9w3byAwbIc9fn+U56oGwXRg605sXQ2MXU2S8efn9wti'
                'wL5BLKyt7qcV8kSo/B6mTtzW/O73e3JX7F12vz+JaF+c+Xob8kSS+XoX88Im'
                '8//HuN9vcp0kWUcrhdLGx5n0/Zo4675147alLtu2eU66vtr2dfux7UtiXVPa'
                '0Dj//tLFZQSc3/+O64zvyZ+UPMBH4fvx2w/J5e9QpW5Hcfazzn3Owy8jrMFR'
                'gv3t4jrbRDzu6riD+kk6/lPcdsbFrtib4lJcilsdV/bgwj4/5e4b07Jhl+z9'
                '1PwbxJ/mhRSXBK4sbTbuU9yniStLadyluDTvpbgUl+JSXIpLcbuE+zfj4jIC'
                'jsve5154+988PH55/mtP+0NbVqGujwzC3480CsPJS3xmQSdAm74VV3jBvtY9'
                'L8hxnlEAT1ngKS/iQRrEFPXGI4OxWZQMAUVqoW5r20xbcbScdxDA6/e8yK28'
                'pvMQ94xz9lg357sJ4LsQ+C4W8Q0hMC0DTlwKDAiIqlojBLVCY3ph4uAKu9Cy'
                'L/w/bg58+L3zn2XSg4f392+LN0majzM3/j48i8KXy2Wz96T55xHvBuBy0nzc'
                'c/w/0nJx/uiW9o6N29x+WT9n2LGqn1ye3GxM0/63s/1/UENheQ==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, mia.MeasureImageAreaOccupied))
        self.assertEqual(len(module.operands), 2)
        self.assertEqual(module.operands[0].operand_objects.value, "Nuclei")

# self.assertEqual(module.operands[1].operand_objects.value, "Cells")
