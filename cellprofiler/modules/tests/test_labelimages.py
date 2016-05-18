'''test_labelimages.py - test the labelimages module
'''

import base64
import os
import unittest
import zlib
from StringIO import StringIO

import numpy as np
import scipy.ndimage

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.labelimages as L


class TestLabelImages(unittest.TestCase):
    def test_01_00_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUvDNz1PwTaxUMDRXMDS1Mja3MjFQMDIwNFAgGTAwevryMzAwPGdk'
                'YKiY8zbstN8hB5G9uq0PlF0idxps9Bbf55Ac03TQRchl0bEjTxmS5y4p8bBW'
                'fSD+w1Pv7i/2uj7jW4dKFn17mH04N+/E3HvF5+Za9wcxNUxPbWD6r1t0/eHd'
                'T45rPPUeeZ+RKNTR52dbw/rY/876id4B149dP/FtI3sw9/HZ53T3zuJZvv5c'
                'xv3thdK8T1LycusvHF1XXifN1Hxd1ujYN9WdxwW2Grxl6u9wXyH8gLmutGSV'
                '98xF0Y/2hc/46hj8S+FX0LRjrnW8aT+PsddGT/wUzrT/UPK2WiM32zQ+gdd3'
                'VR9GBlW43NkoeFjXZKHd7Kvyh5NdKngyrlj7Nlp3HXwqWisYzrd9FXtjnsy+'
                'FQ9ca94JvZ6RP1Wl9LnSD5kr31d1/tq/4kXXDkffdqPL35t6p7jEd3/ICGhN'
                'eT0/RP5Zjf2h6WWy3+t3zrsssS+01f/IkZeTBV/+PmF2kiHycrTWWZuvHOv5'
                'cn/ttXu75M3lJiN9wWi9WRW2Kcmd6pm2r37ufxj783Tdil/n9LoevC+Krzd/'
                'UlezLDqgYGGrReyhWvH47pnv3e+v3X42Y+cV+Vk13Y8e7vlX+fd/6H/lmrp7'
                'Vz7P+xqXH3gl30u84Nr/TN0rR7e5iJjnb62ba5+qfHtv7MkV069feszzTybP'
                'Zj33HeYl0fPP6n9yPj3rP1fMtVVHANI9Gb4=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, L.LabelImages))
        self.assertEqual(module.row_count.value, 16)
        self.assertEqual(module.column_count.value, 24)
        self.assertEqual(module.site_count.value, 2)
        self.assertEqual(module.order, L.O_COLUMN)

    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9973

LabelImages:[module_num:1|svn_version:\'9970\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    # sites / well\x3A:3
    # of columns\x3A:48
    # of rows\x3A:32
    Order\x3A:Column

LabelImages:[module_num:2|svn_version:\'9970\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    # sites / well\x3A:1
    # of columns\x3A:12
    # of rows\x3A:8
    Order\x3A:Row
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, L.LabelImages))
        self.assertEqual(module.site_count, 3)
        self.assertEqual(module.row_count, 32)
        self.assertEqual(module.column_count, 48)
        self.assertEqual(module.order, L.O_COLUMN)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, L.LabelImages))
        self.assertEqual(module.site_count, 1)
        self.assertEqual(module.row_count, 8)
        self.assertEqual(module.column_count, 12)
        self.assertEqual(module.order, L.O_ROW)

    def make_workspace(self, image_set_count):
        image_set_list = cpi.ImageSetList()
        for i in range(image_set_count):
            image_set = image_set_list.get_image_set(i)
        module = L.LabelImages()
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        module.module_num = 1
        pipeline.add_module(module)

        workspace = cpw.Workspace(pipeline, module,
                                  image_set_list.get_image_set(0),
                                  cpo.ObjectSet(), cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module

    def test_02_01_label_plate_by_row(self):
        '''Label one complete plate'''
        nsites = 6
        nimagesets = 96 * nsites
        workspace, module = self.make_workspace(nimagesets)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        self.assertTrue(isinstance(module, L.LabelImages))
        module.row_count.value = 8
        module.column_count.value = 12
        module.order.value = L.O_ROW
        module.site_count.value = nsites
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_SITE)
        rows = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_ROW)
        columns = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_COLUMN)
        plates = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_PLATE)
        wells = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_WELL)
        for i in range(nimagesets):
            self.assertEqual(sites[i], (i % 6) + 1)
            this_row = 'ABCDEFGH'[int(i / 6 / 12)]
            this_column = (int(i / 6) % 12) + 1
            self.assertEqual(rows[i], this_row)
            self.assertEqual(columns[i], this_column)
            self.assertEqual(wells[i], '%s%02d' % (this_row, this_column))
            self.assertEqual(plates[i], 1)

    def test_02_02_label_plate_by_column(self):
        '''Label one complete plate'''
        nsites = 6
        nimagesets = 96 * nsites
        workspace, module = self.make_workspace(nimagesets)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        self.assertTrue(isinstance(module, L.LabelImages))
        module.row_count.value = 8
        module.column_count.value = 12
        module.order.value = L.O_COLUMN
        module.site_count.value = nsites
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_SITE)
        rows = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_ROW)
        columns = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_COLUMN)
        plates = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_PLATE)
        wells = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_WELL)
        for i in range(nimagesets):
            self.assertEqual(sites[i], (i % 6) + 1)
            this_row = 'ABCDEFGH'[int(i / 6) % 8]
            this_column = int(i / 6 / 8) + 1
            self.assertEqual(rows[i], this_row)
            self.assertEqual(columns[i], this_column)
            self.assertEqual(wells[i], '%s%02d' % (this_row, this_column))
            self.assertEqual(plates[i], 1)

    def test_02_03_label_many_plates(self):
        nsites = 1
        nplates = 6
        nimagesets = 96 * nsites * nplates
        workspace, module = self.make_workspace(nimagesets)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        self.assertTrue(isinstance(module, L.LabelImages))
        module.row_count.value = 8
        module.column_count.value = 12
        module.order.value = L.O_ROW
        module.site_count.value = nsites
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_SITE)
        rows = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_ROW)
        columns = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_COLUMN)
        plates = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_PLATE)
        wells = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_WELL)
        for i in range(nimagesets):
            self.assertEqual(sites[i], 1)
            this_row = 'ABCDEFGH'[int(i / 12) % 8]
            this_column = (i % 12) + 1
            self.assertEqual(rows[i], this_row)
            self.assertEqual(columns[i], this_column)
            self.assertEqual(wells[i], '%s%02d' % (this_row, this_column))
            self.assertEqual(plates[i], int(i / 8 / 12) + 1)

    def test_02_04_multichar_row_names(self):
        nimagesets = 1000
        workspace, module = self.make_workspace(nimagesets)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cpmeas.Measurements))
        self.assertTrue(isinstance(module, L.LabelImages))
        module.row_count.value = 1000
        module.column_count.value = 1
        module.order.value = L.O_ROW
        module.site_count.value = 1
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_SITE)
        rows = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_ROW)
        columns = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_COLUMN)
        plates = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_PLATE)
        wells = measurements.get_all_measurements(cpmeas.IMAGE, cpmeas.M_WELL)
        abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(nimagesets):
            self.assertEqual(sites[i], 1)
            this_row = (abc[int(i / 26 / 26)] +
                        abc[int(i / 26) % 26] +
                        abc[i % 26])
            self.assertEqual(rows[i], this_row)
