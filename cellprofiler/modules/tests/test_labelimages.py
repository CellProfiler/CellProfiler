import StringIO
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.labelimages
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace

cellprofiler.preferences.set_headless()


class TestLabelImages(unittest.TestCase):
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.labelimages.LabelImages))
        self.assertEqual(module.site_count, 3)
        self.assertEqual(module.row_count, 32)
        self.assertEqual(module.column_count, 48)
        self.assertEqual(module.order, cellprofiler.modules.labelimages.O_COLUMN)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.labelimages.LabelImages))
        self.assertEqual(module.site_count, 1)
        self.assertEqual(module.row_count, 8)
        self.assertEqual(module.column_count, 12)
        self.assertEqual(module.order, cellprofiler.modules.labelimages.O_ROW)

    def make_workspace(self, image_set_count):
        image_set_list = cellprofiler.image.ImageSetList()
        for i in range(image_set_count):
            image_set = image_set_list.get_image_set(i)
        module = cellprofiler.modules.labelimages.LabelImages()
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        module.module_num = 1
        pipeline.add_module(module)

        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     cellprofiler.region.Set(), cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def test_02_01_label_plate_by_row(self):
        '''Label one complete plate'''
        nsites = 6
        nimagesets = 96 * nsites
        workspace, module = self.make_workspace(nimagesets)
        measurements = workspace.measurements
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        self.assertTrue(isinstance(module, cellprofiler.modules.labelimages.LabelImages))
        module.row_count.value = 8
        module.column_count.value = 12
        module.order.value = cellprofiler.modules.labelimages.O_ROW
        module.site_count.value = nsites
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_SITE)
        rows = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_ROW)
        columns = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_COLUMN)
        plates = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_PLATE)
        wells = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_WELL)
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
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        self.assertTrue(isinstance(module, cellprofiler.modules.labelimages.LabelImages))
        module.row_count.value = 8
        module.column_count.value = 12
        module.order.value = cellprofiler.modules.labelimages.O_COLUMN
        module.site_count.value = nsites
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_SITE)
        rows = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_ROW)
        columns = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_COLUMN)
        plates = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_PLATE)
        wells = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_WELL)
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
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        self.assertTrue(isinstance(module, cellprofiler.modules.labelimages.LabelImages))
        module.row_count.value = 8
        module.column_count.value = 12
        module.order.value = cellprofiler.modules.labelimages.O_ROW
        module.site_count.value = nsites
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_SITE)
        rows = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_ROW)
        columns = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_COLUMN)
        plates = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_PLATE)
        wells = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_WELL)
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
        self.assertTrue(isinstance(measurements, cellprofiler.measurement.Measurements))
        self.assertTrue(isinstance(module, cellprofiler.modules.labelimages.LabelImages))
        module.row_count.value = 1000
        module.column_count.value = 1
        module.order.value = cellprofiler.modules.labelimages.O_ROW
        module.site_count.value = 1
        for i in range(nimagesets):
            if i != 0:
                measurements.next_image_set()
            module.run(workspace)
        sites = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_SITE)
        rows = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_ROW)
        columns = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_COLUMN)
        plates = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_PLATE)
        wells = measurements.get_all_measurements(cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_WELL)
        abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(nimagesets):
            self.assertEqual(sites[i], 1)
            this_row = (abc[int(i / 26 / 26)] +
                        abc[int(i / 26) % 26] +
                        abc[i % 26])
            self.assertEqual(rows[i], this_row)
