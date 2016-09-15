'''test_maskobjects.py - test the MaskObjects module
'''

import base64
import os
import unittest
import zlib
from StringIO import StringIO

import PIL.Image as PILImage
import numpy as np
import scipy.ndimage
from matplotlib.image import pil_to_array

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.maskobjects as M

INPUT_OBJECTS = 'inputobjects'
OUTPUT_OBJECTS = 'outputobjects'
MASKING_OBJECTS = 'maskingobjects'
MASKING_IMAGE = 'maskingobjects'
OUTPUT_OUTLINES = 'outputoutlines'


class TestMaskObjects(unittest.TestCase):
    def test_01_01_load_matlab_exclude(self):
        '''Load a pipeline with an Exclude module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUvDNz1NwS01SMDBUMLCwMjGzAjKMDAwNFEgGDIyevvwMDAwKzAwM'
                'FXOeRtzNu2UgYiYSsTss1N10pfjUWadjc2I32b/d9FtT9hLbirVab1cZdXrb'
                'Su72msNXtpf9RvcX2W8Z3YqND5YVFzC1nj1hZXXs/vNiP7+apHwmhh5bBq3G'
                'GN4qQZ5oyyvcG/W+CB+xXCa7nscrQV3mv+wX4xdJN9Re8AjJVyZcWfTRam/C'
                '8r+rL8bHz3gsfKXh7o8nC0z/5EnvjPusxNRWv72Af6LwH+bGk2K+1m7Xbxzu'
                'afsxIV5Odu3pHVvepV24H3a8UMOjUKDI2kzvmXzDxz9BTne2JX3efqiuq+rP'
                'zUT+W3deOOXF810L/bDGy8Ypf9eOz3YcpVLW11T+RHXYFwkuKvwvaiPa6f6C'
                '974ax/Pm6x8Xp+61+77m8/trMn81vCsW1ju0VGhe35+/R2t+yaP0mf66l69s'
                'tz+w38hxl82NKR+c1icemtg9J/NTZuyjyP6PVySke9LPnhZ/uOOBoaiNtrCN'
                '9oPsDe6pB18+OWi4qHlvkDl/n2D7jFfHdTPjg37Gz3mg2cR2ui345t+ATdEv'
                'trpdfs5yecWaF59qUthS73ywnWJ/ftHT+Vr1Xz7ElvU+k/6/pHbJSdv7Fkpr'
                'eD3f2Ww6c9r/arGezweRhqmC5fF1J35KTVnnY3GzP/PxRyPVdSaN2ozF97ds'
                'epCym1/EpnKvvufvurlFLrZnfgffmDzNfdp8kT39KnELk189yH7zY4LEDm7r'
                'xZcPG/2wXm01I+W89fzNDxKa3/bUqX70eT39z7y3z3POr/lz6eHbHRve35yw'
                'QOL8ntd737Tr9Z/6X1lv7vs6fN79qd1Ppv/Z86J4h0DltGv/9pzn/PTr2Kv7'
                '7/sflapm35UVWdh3+s/qn59Vapdv+mae8tkwcJ7NxCsrbxvaKa94+CKrraZb'
                'zFzU/9LfJxbybe90g/98fuat/Vz2+aoj6iVme2p3PD1h9OFu+JzNp1w3P6+f'
                'enra/f1c5fXqIgee/N2RF/v/zILgftmQyz++RjwSrLrrOOVxd93po3d/3EwV'
                'q1n1ex+70PwL/NVmH2WnWO9bVKF3NGB9Xn64lsN7G6m7zS/XRN+7oSF28O2f'
                'gIetsUr299785akz322++3Dhwc3GqXF2jOXn+2/tN/rPr2W2VQgAxRjE4w==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 5)

        module = pipeline.modules()[-2]
        self.assertTrue(isinstance(module, M.MaskObjects))
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.mask_choice.value, M.MC_OBJECTS)
        self.assertEqual(module.masking_objects.value, "Cells")
        self.assertEqual(module.remaining_objects.value, "FilteredNuclei")
        self.assertEqual(module.retain_or_renumber.value, M.R_RENUMBER)
        self.assertEqual(module.overlap_choice.value, M.P_MASK)
        self.assertTrue(module.wants_outlines)
        self.assertEqual(module.outlines_name.value, "FNOutlines")

        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, M.MaskObjects))
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.mask_choice.value, M.MC_OBJECTS)
        self.assertEqual(module.masking_objects.value, "Cells")
        self.assertEqual(module.remaining_objects.value, "FilteredFoo")
        self.assertEqual(module.retain_or_renumber.value, M.R_RETAIN)
        self.assertEqual(module.overlap_choice.value, M.P_REMOVE)
        self.assertFalse(module.wants_outlines)

    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9193

MaskObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Objects to be masked\x3A:Nuclei
    Name the masked objects\x3A:MaskedNuclei
    Mask using other objects or binary image?:Objects
    Masking objects\x3A:Wells
    Masking image\x3A:None
    How do you want to handle objects that are partially masked?:Keep overlapping region
    How much of the object must overlap?:0.5
    Retain original numbering or renumber objects?:Renumber
    Save outlines for the resulting objects?:No
    Outlines name\x3A:MaskedOutlines

MaskObjects:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Objects to be masked\x3A:Cells
    Name the masked objects\x3A:MaskedCells
    Mask using other objects or binary image?:Image
    Masking objects\x3A:None
    Masking image\x3A:WellBoundary
    How do you want to handle objects that are partially masked?:Keep
    How much of the object must overlap?:0.5
    Retain original numbering or renumber objects?:Retain
    Save outlines for the resulting objects?:Yes
    Outlines name\x3A:MaskedCellOutlines

MaskObjects:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Objects to be masked\x3A:Cytoplasm
    Name the masked objects\x3A:MaskedCytoplasm
    Mask using other objects or binary image?:Objects
    Masking objects\x3A:Cells
    Masking image\x3A:None
    How do you want to handle objects that are partially masked?:Remove
    How much of the object must overlap?:0.5
    Retain original numbering or renumber objects?:Renumber
    Save outlines for the resulting objects?:No
    Outlines name\x3A:MaskedOutlines

MaskObjects:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Objects to be masked\x3A:Speckles
    Name the masked objects\x3A:MaskedSpeckles
    Mask using other objects or binary image?:Objects
    Masking objects\x3A:Cells
    Masking image\x3A:None
    How do you want to handle objects that are partially masked?:Remove depending on overlap
    How much of the object must overlap?:0.3
    Retain original numbering or renumber objects?:Renumber
    Save outlines for the resulting objects?:No
    Outlines name\x3A:MaskedOutlines
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 4)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, M.MaskObjects))
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.mask_choice.value, M.MC_OBJECTS)
        self.assertEqual(module.masking_objects.value, "Wells")
        self.assertEqual(module.remaining_objects.value, "MaskedNuclei")
        self.assertEqual(module.retain_or_renumber.value, M.R_RENUMBER)
        self.assertEqual(module.overlap_choice.value, M.P_MASK)
        self.assertFalse(module.wants_outlines)
        self.assertFalse(module.wants_inverted_mask)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, M.MaskObjects))
        self.assertEqual(module.object_name.value, "Cells")
        self.assertEqual(module.mask_choice.value, M.MC_IMAGE)
        self.assertEqual(module.masking_image.value, "WellBoundary")
        self.assertEqual(module.remaining_objects.value, "MaskedCells")
        self.assertEqual(module.retain_or_renumber.value, M.R_RETAIN)
        self.assertEqual(module.overlap_choice.value, M.P_KEEP)
        self.assertTrue(module.wants_outlines)
        self.assertEqual(module.outlines_name.value, "MaskedCellOutlines")
        self.assertFalse(module.wants_inverted_mask)

        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, M.MaskObjects))
        self.assertEqual(module.object_name.value, "Cytoplasm")
        self.assertEqual(module.mask_choice.value, M.MC_OBJECTS)
        self.assertEqual(module.masking_objects.value, "Cells")
        self.assertEqual(module.remaining_objects.value, "MaskedCytoplasm")
        self.assertEqual(module.retain_or_renumber.value, M.R_RENUMBER)
        self.assertEqual(module.overlap_choice.value, M.P_REMOVE)
        self.assertFalse(module.wants_outlines)
        self.assertFalse(module.wants_inverted_mask)

        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, M.MaskObjects))
        self.assertEqual(module.object_name.value, "Speckles")
        self.assertEqual(module.mask_choice.value, M.MC_OBJECTS)
        self.assertEqual(module.masking_objects.value, "Cells")
        self.assertEqual(module.remaining_objects.value, "MaskedSpeckles")
        self.assertEqual(module.retain_or_renumber.value, M.R_RENUMBER)
        self.assertEqual(module.overlap_choice.value, M.P_REMOVE_PERCENTAGE)
        self.assertAlmostEqual(module.overlap_fraction.value, .3)
        self.assertFalse(module.wants_outlines)
        self.assertFalse(module.wants_inverted_mask)

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9193

MaskObjects:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Objects to be masked\x3A:Nuclei
    Name the masked objects\x3A:MaskedNuclei
    Mask using other objects or binary image?:Objects
    Masking objects\x3A:Wells
    Masking image\x3A:None
    How do you want to handle objects that are partially masked?:Keep overlapping region
    How much of the object must overlap?:0.5
    Retain original numbering or renumber objects?:Renumber
    Save outlines for the resulting objects?:No
    Outlines name\x3A:MaskedOutlines
    Invert the mask?:Yes
    """
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)

        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, M.MaskObjects))
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.mask_choice.value, M.MC_OBJECTS)
        self.assertEqual(module.masking_objects.value, "Wells")
        self.assertEqual(module.remaining_objects.value, "MaskedNuclei")
        self.assertEqual(module.retain_or_renumber.value, M.R_RENUMBER)
        self.assertEqual(module.overlap_choice.value, M.P_MASK)
        self.assertFalse(module.wants_outlines)
        self.assertTrue(module.wants_inverted_mask)

    def make_workspace(self, labels, overlap_choice, masking_objects=None,
                       masking_image=None, renumber=True,
                       wants_outlines=False):
        module = M.MaskObjects()
        module.module_num = 1
        module.object_name.value = INPUT_OBJECTS
        module.remaining_objects.value = OUTPUT_OBJECTS
        module.mask_choice.value = (M.MC_OBJECTS if masking_objects is not None
                                    else M.MC_IMAGE)
        module.masking_image.value = MASKING_IMAGE
        module.masking_objects.value = MASKING_OBJECTS
        module.retain_or_renumber.value = (M.R_RENUMBER if renumber
                                           else M.R_RETAIN)
        module.overlap_choice.value = overlap_choice
        module.wants_outlines.value = wants_outlines
        module.outlines_name.value = OUTPUT_OUTLINES

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        object_set = cpo.ObjectSet()
        io = cpo.Objects()
        io.segmented = labels
        object_set.add_objects(io, INPUT_OBJECTS)

        if masking_objects is not None:
            oo = cpo.Objects()
            oo.segmented = masking_objects
            object_set.add_objects(oo, MASKING_OBJECTS)

        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        if masking_image is not None:
            mi = cpi.Image(masking_image)
            image_set.add(MASKING_IMAGE, mi)

        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  cpmeas.Measurements(), image_set_list)
        return workspace, module

    def test_02_01_measurement_columns(self):
        '''Test get_measurement_columns'''
        workspace, module = self.make_workspace(np.zeros((20, 10), int),
                                                M.P_MASK,
                                                np.zeros((20, 10), int))
        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 6)
        for expected in (
                (cpmeas.IMAGE, M.I.FF_COUNT % OUTPUT_OBJECTS, cpmeas.COLTYPE_INTEGER),
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_X, cpmeas.COLTYPE_FLOAT),
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_Y, cpmeas.COLTYPE_FLOAT),
                (OUTPUT_OBJECTS, M.I.FF_PARENT % INPUT_OBJECTS, cpmeas.COLTYPE_INTEGER),
                (OUTPUT_OBJECTS, M.I.M_NUMBER_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER),
                (INPUT_OBJECTS, M.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS, cpmeas.COLTYPE_INTEGER)):
            self.assertTrue(any([all([c in e for c, e in zip(column, expected)])
                                 for column in columns]))

    def test_02_02_measurement_categories(self):
        workspace, module = self.make_workspace(np.zeros((20, 10), int),
                                                M.MC_OBJECTS,
                                                np.zeros((20, 10), int))
        categories = module.get_categories(workspace.pipeline, "Foo")
        self.assertEqual(len(categories), 0)

        categories = module.get_categories(workspace.pipeline, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], M.I.C_COUNT)

        categories = module.get_categories(workspace.pipeline, OUTPUT_OBJECTS)
        self.assertEqual(len(categories), 3)
        for category, expected in zip(sorted(categories),
                                      (M.I.C_LOCATION, M.I.C_NUMBER,
                                       M.I.C_PARENT)):
            self.assertEqual(category, expected)

        categories = module.get_categories(workspace.pipeline, INPUT_OBJECTS)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], M.I.C_CHILDREN)

    def test_02_03_measurements(self):
        workspace, module = self.make_workspace(np.zeros((20, 10), int),
                                                M.P_MASK,
                                                np.zeros((20, 10), int))
        ftr_count = (M.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS).split('_', 1)[1]
        d = {"Foo": {},
             cpmeas.IMAGE: {"Foo": [], M.I.C_COUNT: [OUTPUT_OBJECTS]},
             OUTPUT_OBJECTS: {"Foo": [],
                              M.I.C_LOCATION: [M.I.FTR_CENTER_X,
                                               M.I.FTR_CENTER_Y],
                              M.I.C_PARENT: [INPUT_OBJECTS],
                              M.I.C_NUMBER: [M.I.FTR_OBJECT_NUMBER]},
             INPUT_OBJECTS: {"Foo": [],
                             M.I.C_CHILDREN: [ftr_count]}
             }
        for object_name in d.keys():
            od = d[object_name]
            for category in od.keys():
                features = module.get_measurements(workspace.pipeline,
                                                   object_name,
                                                   category)
                expected = od[category]
                self.assertEqual(len(features), len(expected))
                for feature, e in zip(sorted(features), sorted(expected)):
                    self.assertEqual(feature, e)

    def test_03_01_mask_nothing(self):
        workspace, module = self.make_workspace(np.zeros((20, 10), int),
                                                M.MC_OBJECTS,
                                                np.zeros((20, 10), int))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        value = m.get_current_image_measurement(M.I.FF_COUNT % OUTPUT_OBJECTS)
        self.assertEqual(value, 0)
        for object_name, feature in (
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_X),
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_Y),
                (OUTPUT_OBJECTS, M.I.M_NUMBER_OBJECT_NUMBER),
                (OUTPUT_OBJECTS, M.I.FF_PARENT % INPUT_OBJECTS),
                (INPUT_OBJECTS, M.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS)):
            data = m.get_current_measurement(object_name, feature)
            self.assertEqual(len(data), 0)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == 0))

    def test_03_02_mask_with_objects(self):
        labels = np.zeros((20, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 2
        mask = np.zeros((20, 10), int)
        mask[3:17, 2:6] = 1
        expected = labels.copy()
        expected[mask == 0] = 0
        workspace, module = self.make_workspace(labels, M.P_MASK, mask)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

        expected_x = np.array([4, 4])
        expected_y = np.array([5, 14])
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        value = m.get_current_image_measurement(M.I.FF_COUNT % OUTPUT_OBJECTS)
        self.assertEqual(value, 2)

        for object_name, feature, expected in (
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_X, expected_x),
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_Y, expected_y),
                (OUTPUT_OBJECTS, M.I.M_NUMBER_OBJECT_NUMBER, np.array([1, 2])),
                (OUTPUT_OBJECTS, M.I.FF_PARENT % INPUT_OBJECTS, np.array([1, 2])),
                (INPUT_OBJECTS, M.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS, np.array([1, 1]))):
            data = m.get_current_measurement(object_name, feature)
            self.assertEqual(len(data), len(expected))
            for value, e in zip(data, expected):
                self.assertEqual(value, e)

    def test_03_03_mask_with_image(self):
        labels = np.zeros((20, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 2
        mask = np.zeros((20, 10), bool)
        mask[3:17, 2:6] = True
        expected = labels.copy()
        expected[~ mask] = 0
        workspace, module = self.make_workspace(labels, M.P_MASK,
                                                masking_image=mask)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

        expected_x = np.array([4, 4])
        expected_y = np.array([5, 14])
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        value = m.get_current_image_measurement(M.I.FF_COUNT % OUTPUT_OBJECTS)
        self.assertEqual(value, 2)

        for object_name, feature, expected in (
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_X, expected_x),
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_Y, expected_y),
                (OUTPUT_OBJECTS, M.I.M_NUMBER_OBJECT_NUMBER, np.array([1, 2])),
                (OUTPUT_OBJECTS, M.I.FF_PARENT % INPUT_OBJECTS, np.array([1, 2])),
                (INPUT_OBJECTS, M.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS, np.array([1, 1]))):
            data = m.get_current_measurement(object_name, feature)
            self.assertEqual(len(data), len(expected))
            for value, e in zip(data, expected):
                self.assertEqual(value, e)

    def test_03_04_mask_renumber(self):
        labels = np.zeros((30, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 3
        labels[22:28, 3:7] = 2
        mask = np.zeros((30, 10), bool)
        mask[3:17, 2:6] = True
        expected = labels.copy()
        expected[~ mask] = 0
        expected[expected == 3] = 2
        workspace, module = self.make_workspace(labels, M.P_MASK,
                                                masking_image=mask)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

        expected_x = np.array([4, 4])
        expected_y = np.array([5, 14])
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        value = m.get_current_image_measurement(M.I.FF_COUNT % OUTPUT_OBJECTS)
        self.assertEqual(value, 2)

        for object_name, feature, expected in (
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_X, expected_x),
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_Y, expected_y),
                (OUTPUT_OBJECTS, M.I.M_NUMBER_OBJECT_NUMBER, np.array([1, 2])),
                (OUTPUT_OBJECTS, M.I.FF_PARENT % INPUT_OBJECTS, np.array([1, 3])),
                (INPUT_OBJECTS, M.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS, np.array([1, 0, 1]))):
            data = m.get_current_measurement(object_name, feature)
            self.assertEqual(len(data), len(expected))
            for value, e in zip(data, expected):
                self.assertEqual(value, e)

    def test_03_05_mask_retain(self):
        labels = np.zeros((30, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 3
        labels[22:28, 3:7] = 2
        mask = np.zeros((30, 10), bool)
        mask[3:17, 2:6] = True
        expected = labels.copy()
        expected[~ mask] = 0
        workspace, module = self.make_workspace(labels, M.P_MASK,
                                                masking_image=mask,
                                                renumber=False)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

        expected_x = np.array([4, None, 4])
        expected_y = np.array([5, None, 14])
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        value = m.get_current_image_measurement(M.I.FF_COUNT % OUTPUT_OBJECTS)
        self.assertEqual(value, 3)

        for object_name, feature, expected in (
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_X, expected_x),
                (OUTPUT_OBJECTS, M.I.M_LOCATION_CENTER_Y, expected_y),
                (OUTPUT_OBJECTS, M.I.M_NUMBER_OBJECT_NUMBER, np.array([1, 2, 3])),
                (OUTPUT_OBJECTS, M.I.FF_PARENT % INPUT_OBJECTS, np.array([1, 2, 3])),
                (INPUT_OBJECTS, M.I.FF_CHILDREN_COUNT % OUTPUT_OBJECTS, np.array([1, 0, 1]))):
            data = m.get_current_measurement(object_name, feature)
            self.assertEqual(len(data), len(expected))
            for value, e in zip(data, expected):
                if e is None:
                    self.assertTrue(np.isnan(value))
                else:
                    self.assertEqual(value, e)

    def test_03_06_mask_outlines(self):
        labels = np.zeros((20, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 2
        mask = np.zeros((20, 10), bool)
        mask[3:17, 2:6] = True
        expected = np.zeros((20, 10), bool)
        expected[3:8, 3:6] = True
        expected[4:7, 4] = False
        expected[12:17, 3:6] = True
        expected[13:16, 4] = False
        workspace, module = self.make_workspace(labels, M.P_MASK,
                                                masking_image=mask,
                                                wants_outlines=True)
        module.run(workspace)
        outlines = workspace.image_set.get_image(OUTPUT_OUTLINES)
        self.assertTrue(np.all(outlines.pixel_data == expected))

    def test_03_07_mask_invert(self):
        labels = np.zeros((20, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 2
        #
        # Make a mask that covers only object # 1 and that is missing
        # one pixel of that object. Invert it in anticipation of invert op
        #
        mask = labels == 1
        mask[2, 3] = False
        mask = ~ mask
        expected = labels
        expected[labels != 1] = 0
        expected[2, 3] = 0
        workspace, module = self.make_workspace(labels, M.P_MASK,
                                                masking_image=mask)
        self.assertTrue(isinstance(module, M.MaskObjects))
        module.wants_inverted_mask.value = True
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_04_01_keep(self):
        labels = np.zeros((30, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 3
        labels[22:28, 3:7] = 2
        mask = np.zeros((30, 10), bool)
        mask[3:17, 2:6] = True
        expected = labels.copy()
        expected[expected == 2] = 0
        expected[expected == 3] = 2
        workspace, module = self.make_workspace(labels, M.P_KEEP,
                                                masking_image=mask)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_04_02_remove(self):
        labels = np.zeros((20, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 2
        mask = np.zeros((20, 10), bool)
        mask[2:17, 2:7] = True
        expected = labels.copy()
        expected[labels == 2] = 0
        workspace, module = self.make_workspace(labels, M.P_REMOVE,
                                                masking_image=mask)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_04_03_remove_percent(self):
        labels = np.zeros((20, 10), int)
        labels[2:8, 3:6] = 1
        labels[12:18, 3:7] = 2
        mask = np.zeros((20, 10), bool)
        mask[3:17, 2:6] = True
        # loses 3 of 18 from object 1 = .1666
        # loses 9 of 24 from object 2 = .375
        expected = labels.copy()
        expected[labels == 2] = 0
        workspace, module = self.make_workspace(labels, M.P_REMOVE_PERCENTAGE,
                                                masking_image=mask)
        module.overlap_fraction.value = .75
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_05_01_different_object_sizes(self):
        labels = np.zeros((30, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 2
        mask = np.zeros((20, 20), int)
        mask[3:17, 2:6] = 1
        expected = labels.copy()
        expected[:20, :][mask[:, :10] == 0] = 0
        workspace, module = self.make_workspace(labels, M.P_MASK, mask)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_05_02_different_image_sizes(self):
        labels = np.zeros((30, 10), int)
        labels[2:8, 3:7] = 1
        labels[12:18, 3:7] = 2
        mask = np.zeros((20, 20), bool)
        mask[3:17, 2:6] = 1
        expected = labels.copy()
        expected[:20, :][mask[:, :10] == 0] = 0
        workspace, module = self.make_workspace(labels, M.P_MASK,
                                                masking_image=mask)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_OBJECTS)
        self.assertTrue(np.all(objects.segmented == expected))
