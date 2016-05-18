'''test_editobjectsmanually - test the EditObjectsManually module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.object as cpo
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.editobjectsmanually as E

INPUT_OBJECTS_NAME = "inputobjects"
OUTPUT_OBJECTS_NAME = "outputobjects"


class TestEditObjectsManually(unittest.TestCase):
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUgjJKFXwSsxTMDJUMDS2MjGzAjKMDAwNFEgGDIyevvwMDAwLmBgY'
                'KuY8DT+ff9lApOz2Fime0NSrKnwepyceNgtqFHBXM1sglL08d9aRi753zp3Y'
                'uvrIzFtzLT4cL3h0ZqJ9746z/MdfzzKx3+UZsWnN8z3f956Pn5O+nJPhx0W2'
                'BQ97o+PWN7mu/l3m+urQPoc5yVZFrxyY9f/83B+3v8msjblpnULhROZDYmfK'
                'X06IPaf2cHFw2k6Wrw8iyr5FnZfln/D2S7vjgv8/ePwFdx7YznEk+kFZX96G'
                'Zsmvj2trSs3OnN4xabvmte2yLwMCS1ZU7tU9y/2fX+HT9Wkvrk36E1233yW4'
                'rsCoVe368QmzrzMd9ziw8mhB1oMd5Sbn/9sprlv4JP/MV1POQy9/2Ac+av37'
                'USqoy+O5w6WFOXK+qnE35hs8POE2rSd4Wl+p6v4N1sf+RTMvefw/pFP8m+Fl'
                'zce7Xi/wOL6o4NrVh7O/6y65mMh95DP3b5mLP7dv9fx3VcJP9MsNpXcbX0SY'
                '9KvMyWV1v1H7bsev9ENsE1osZl9bLnR4ZcM55o8B9ydvvN/5MaJcNj/oWOZG'
                'C16Jdrfnit+Ob7K6t1QmkP3zzCPqEvtVHtiGf2jpXdmav+Gp06LHhjKty7Ml'
                '/2y61q5j0xdzbk+z1GzJihW/nxm0y62uSDkofbPg2YwJCpzq+XtPS80/Yv+2'
                'fMI092t/A034Q/l/SYl9kvv0SuDI/0th99ufWk3Yvu5s97ETae9UbM9tKu7n'
                '3ac82d2h8vWsluXVpny+Xke8o95v+t/psZ01RirOmqtvw8zTnMte8G00+vBO'
                'ujuqPmjef9X7ybKnb/mvPfX3q9fa2ZGvr4SzrV1XctBz87xYoI9juQP72jcb'
                'S/c/3WzLGxd9UPhbZ2tlfVz0nGCPvdkSM1v/PCleMHGB99XNZbei1v3+u+vn'
                'q/Oryv/ZT35kcv/53Xvn997XF6/83/D13v/Vov8cb0/5b3+mX74BAJGShuQ=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, E.EditObjectsManually))
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.filtered_objects, "FilteredNuclei")
        self.assertFalse(module.wants_outlines)
        self.assertEqual(module.renumber_choice, E.R_RENUMBER)

    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9120

EditObjectsManually:[module_num:1|svn_version:\'1\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the input objects:Nuclei
    Name the objects left after editing:EditedNuclei
    Do you want to save outlines of the edited objects?:Yes
    What do you want to call the outlines?:EditedNucleiOutlines
    Do you want to renumber the objects created by this module or retain the original numbering?:Renumber
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, E.EditObjectsManually))
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.filtered_objects, "EditedNuclei")
        self.assertTrue(module.wants_outlines)
        self.assertEqual(module.outlines_name, "EditedNucleiOutlines")
        self.assertEqual(module.renumber_choice, E.R_RENUMBER)
        self.assertFalse(module.wants_image_display)

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9120

EditObjectsManually:[module_num:1|svn_version:\'10039\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the objects to be edited:Nuclei
    Name the edited objects:EditedNuclei
    Retain outlines of the edited objects?:No
    Name the outline image:EditedObjectOutlines
    Numbering of the edited objects:Retain
    Display a guiding image?:Yes
    Image name\x3A:DNA
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, E.EditObjectsManually))
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.filtered_objects, "EditedNuclei")
        self.assertFalse(module.wants_outlines)
        self.assertEqual(module.renumber_choice, E.R_RETAIN)
        self.assertTrue(module.wants_image_display)
        self.assertEqual(module.image_name, "DNA")
        self.assertFalse(module.allow_overlap)

    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9120

EditObjectsManually:[module_num:1|svn_version:\'10039\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Select the objects to be edited:Nuclei
    Name the edited objects:EditedNuclei
    Retain outlines of the edited objects?:No
    Name the outline image:EditedObjectOutlines
    Numbering of the edited objects:Retain
    Display a guiding image?:Yes
    Image name\x3A:DNA
    Allow overlapping objects:Yes
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, E.EditObjectsManually))
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.filtered_objects, "EditedNuclei")
        self.assertFalse(module.wants_outlines)
        self.assertEqual(module.renumber_choice, E.R_RETAIN)
        self.assertTrue(module.wants_image_display)
        self.assertEqual(module.image_name, "DNA")
        self.assertTrue(module.allow_overlap)

    def test_02_02_measurements(self):
        module = E.EditObjectsManually()
        module.object_name.value = INPUT_OBJECTS_NAME
        module.filtered_objects.value = OUTPUT_OBJECTS_NAME

        columns = module.get_measurement_columns(None)
        expected_columns = [
            (cpmeas.IMAGE, E.I.FF_COUNT % OUTPUT_OBJECTS_NAME, cpmeas.COLTYPE_INTEGER),
            (OUTPUT_OBJECTS_NAME, E.I.M_NUMBER_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER),
            (OUTPUT_OBJECTS_NAME, E.I.M_LOCATION_CENTER_X, cpmeas.COLTYPE_FLOAT),
            (OUTPUT_OBJECTS_NAME, E.I.M_LOCATION_CENTER_Y, cpmeas.COLTYPE_FLOAT),
            (OUTPUT_OBJECTS_NAME, E.I.FF_PARENT % INPUT_OBJECTS_NAME, cpmeas.COLTYPE_INTEGER),
            (INPUT_OBJECTS_NAME, E.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS_NAME, cpmeas.COLTYPE_INTEGER)]

        for column in columns:
            self.assertTrue(any([all([column[i] == expected[i] for i in range(3)])
                                 for expected in expected_columns]),
                            "Unexpected column: %s, %s, %s" % column)
            # Make sure no duplicates
            self.assertEqual(len(['x' for c in columns
                                  if all([column[i] == c[i]
                                          for i in range(3)])]), 1)
        for expected in expected_columns:
            self.assertTrue(any([all([column[i] == expected[i] for i in range(3)])
                                 for column in columns]),
                            "Missing column: %s, %s, %s" % expected)

        #
        # Check the measurement features
        #
        d = {cpmeas.IMAGE: {E.I.C_COUNT: [OUTPUT_OBJECTS_NAME],
                            "Foo": []},
             INPUT_OBJECTS_NAME: {E.I.C_CHILDREN: ["%s_Count" % OUTPUT_OBJECTS_NAME],
                                  "Foo": []},
             OUTPUT_OBJECTS_NAME: {
                 E.I.C_LOCATION: [E.I.FTR_CENTER_X, E.I.FTR_CENTER_Y],
                 E.I.C_PARENT: [INPUT_OBJECTS_NAME],
                 E.I.C_NUMBER: [E.I.FTR_OBJECT_NUMBER],
                 "Foo": []},
             "Foo": {}
             }

        for object_name, category_d in d.iteritems():
            #
            # Check get_categories for the object
            #
            categories = module.get_categories(None, object_name)
            self.assertEqual(len(categories), len([k for k in category_d.keys()
                                                   if k != "Foo"]))
            for category in categories:
                self.assertTrue(category_d.has_key(category))
            for category in category_d.keys():
                if category != "Foo":
                    self.assertTrue(category in categories)

            for category, expected_features in category_d.iteritems():
                #
                # check get_measurements for each category
                #
                features = module.get_measurements(None, object_name,
                                                   category)
                self.assertEqual(len(features), len(expected_features))
                for feature in features:
                    self.assertTrue(feature in expected_features,
                                    "Unexpected feature: %s" % feature)
                for feature in expected_features:
                    self.assertTrue(feature in features,
                                    "Missing feature: %s" % feature)
