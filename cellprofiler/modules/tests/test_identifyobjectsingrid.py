import StringIO
import base64
import unittest
import zlib

import cellprofiler.grid
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.definegrid
import cellprofiler.modules.identifyobjectsingrid
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import centrosome.outline
import numpy

cellprofiler.preferences.set_headless()

OUTPUT_OBJECTS_NAME = 'outputobjects'
GRID_NAME = 'mygrid'
GUIDING_OBJECTS_NAME = 'inputobjects'
OUTLINES_NAME = 'myoutlines'


class TestIdentifyObjectsInGrid(unittest.TestCase):
    def test_01_02_load_v1(self):
        data = ('eJztW1tv2zYUplInTdZhS4EBGzYU4OMwpIKcxMOSPUxOsrYGYjtYjBZ7myLR'
                'DgeZNCTKjfeL95ifMNKWLJmTo4vlREYkgLAPxe8cngsPSUlsN3uXzTPYUDXY'
                'bvbe9rGN4JVtsD51hqeQsAN47iCDIQtScgrblMAOHcPDI1jXThva6eHP8FDT'
                'TkC+S2m1v+I/978CsMN/d3nZ8m9t+7QSKYK+RoxhMnC3QQ1859ff8/LRcLBx'
                'Y6OPhu0hNxQR1LdIn/Ymo/mtNrU8G3WMYbQxvzre8AY5brcfAP3bV/gO2df4'
                'HySpEDT7A42xiynx8T5/uXYulzJJ7vUt/fzO4d2R+J8ZzLy9ZtwDi/XCbh9+'
                'CO2mSHar8fImUj9tD8L2tRg7v4603/dpTCw8xpZn2BAPjcG814LfLwn8diR+'
                'gm5P3jvYSofflfCC7jp4cMZdXIR8PQG/L+FF6aE79vb3O8NkcChcI/hoCXyU'
                'BT4KOElpv21JvqA/Idt2Qbr+fyHhBf0OE8MOmPh8svb/KCVuawG3BepH+fWu'
                'awfHGkgXx99KeEGfY8fkqa1jMM/hsXxJTYPxgZnbjhcUEsqg56LQjkl89iQ+'
                'gu7REeQhfctAyCfJri8W+LwAf/JRmVd+02OUBzI2M8iX/dqh6fz6UpIv6HNq'
                'e0PiriQ3K04EUt44ukB9w7MZbIlkCC+wg0xGnUmqftQW+NVAl5hoFX+vmj+y'
                'jnsVrJD3DcLnkHTj5JWEF3SXuR58b9MbwSRl/5fZbV3xXbS96yB/nm9Puh6z'
                'MUEZ8vwyvZPkx/nrnFLH4nNNsM7J4y/toP4/3I6EC64AtwdW13fVdUnWPKCp'
                'Wr0oPbPgcs3DjYPDYP7XE/Bx802LMERczCaP0P+4PGTV1REZrHc8PrXeq8bn'
                'Mr0zr/e0p/VvnvXI8Unj4JhHeB47NWLsVGS+kvNGh5Jc/gz2H0XMn2XQb9k+'
                'owh5ZdAv7zpmnf5bp13EDn2T9Ft3Pi6LnnpCP7+U9BR0k0zglYPGmHpunPxU'
                '60BVe/K4LdJOefY3Zda3iH37JutXlvH5WP5TG0/fz3WPx95nCk3bcN0l+9cy'
                '6/shQd+45/yfkHgIiSw4Fi80iIki/DZFbz1B79jn4dRBA4d6xNo8fZ97Xip6'
                '/3X3fYhTJFzc+7R1xenXYDFOBT3ga2IRpyOIiYVGueN9+hJvyih9f+KeZ8z7'
                'U4A9HjPP0Zu/kcmillwfv00cZxWuwpUNp4OHx2Xc+i0yLjdO3+dm3yT5+5J8'
                'UcJ5bJZ2N8neZfNv2nXCpuhb4cqB00E1ritchVuG00E1PsqI08Hz8kuSvtX6'
                '4HngdFDFQYWrcBWuwlW4CvfYuH+VEKdIOEFHv5MS7f+KyImbr3+KtN/3aRPZ'
                '9sih4pyfow6nh9Fc1aaGNTvdpV7yv63IQa9C5ViojwkSr47Ui+nf2VmsQM4o'
                'QY4uydGXycEWIgz3JyOHaxUccFFbfu0Vr50fe1mL3NnDUReTqaqB3O6stkXi'
                'vm/bi5Eb9fcWp15/82b3ofgCYDGuwni7/y2PvJpSUwQu+n3bqwRcDSzG+TSu'
                'Qba4/vGB9oGOZW6f1c4Kv1a1UyinNu/TjH852/8HMyEGeg==')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid))
        self.assertEqual(module.grid_name, "MyGrid")
        self.assertEqual(module.output_objects_name, "FinalWells")
        self.assertEqual(module.shape_choice, cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_NATURAL)
        self.assertEqual(module.guiding_object_name, "Wells")
        self.assertEqual(module.diameter_choice, cellprofiler.modules.identifyobjectsingrid.AM_AUTOMATIC)
        self.assertTrue(module.wants_outlines)
        self.assertEqual(module.outlines_name, "MyOutlines")

    def make_workspace(self, gridding, labels=None):
        module = cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid()
        module.module_num = 1
        module.grid_name.value = GRID_NAME
        module.output_objects_name.value = OUTPUT_OBJECTS_NAME
        module.guiding_object_name.value = GUIDING_OBJECTS_NAME
        module.outlines_name.value = OUTLINES_NAME
        image_set_list = cellprofiler.image.ImageSetList()
        object_set = cellprofiler.region.Set()
        if labels is not None:
            my_objects = cellprofiler.region.Region()
            my_objects.segmented = labels
            object_set.add_objects(my_objects, GUIDING_OBJECTS_NAME)
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     object_set, cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        workspace.set_grid(GRID_NAME, gridding)
        return workspace, module

    def make_rectangular_grid(self, gridding):
        self.assertTrue(isinstance(gridding, cellprofiler.grid.Grid))
        i0 = gridding.y_location_of_lowest_y_spot
        j0 = gridding.x_location_of_lowest_x_spot
        di = gridding.y_spacing
        dj = gridding.x_spacing
        ni = gridding.rows
        nj = gridding.columns
        i, j = numpy.mgrid[0:(i0 + di * (ni + 1)), 0:(j0 + di * (nj + 1))]
        i = numpy.round((i - i0).astype(float) / di).astype(int)
        j = numpy.round((j - j0).astype(float) / dj).astype(int)
        mask = ((i >= 0) & (j >= 0) & (i < ni) & (j < nj))
        grid = numpy.zeros((gridding.image_height, gridding.image_width), int)
        g = grid[:i.shape[0], :i.shape[1]]
        g[mask[:g.shape[0], :g.shape[1]]] = gridding.spot_table[i[mask], j[mask]]
        return grid

    def test_02_01_forced_location(self):
        d = cellprofiler.modules.definegrid.DefineGrid()
        d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
        #
        # Grid with x spacing = 10, y spacing = 20
        #
        diameter = 6
        gridding = d.build_grid_info(15, 25, 1, 1, 25, 45, 2, 2)
        expected = self.make_rectangular_grid(gridding)
        i, j = numpy.mgrid[0:expected.shape[0], 0:expected.shape[1]]
        ispot, jspot = numpy.mgrid[0:gridding.rows, 0:gridding.columns]
        y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        y_locations[gridding.spot_table.flatten()] = \
            gridding.y_locations[ispot.flatten()]
        x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        x_locations[gridding.spot_table.flatten()] = \
            gridding.x_locations[jspot.flatten()]
        idist = (i - y_locations[expected])
        jdist = (j - x_locations[expected])
        expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
        workspace, module = self.make_workspace(gridding)
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid))
        module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
        module.diameter.value = diameter
        module.shape_choice.value = cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_FORCED
        module.wants_outlines.value = True
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
        self.assertTrue(numpy.all(labels == expected[0:labels.shape[0], 0:labels.shape[1]]))
        #
        # Check measurements
        #
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_X')
        self.assertTrue(numpy.all(xm == x_locations[1:]))
        ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_Y')
        self.assertTrue(numpy.all(ym == y_locations[1:]))
        count = m.get_current_image_measurement('Count_%s' % OUTPUT_OBJECTS_NAME)
        self.assertEqual(count, gridding.rows * gridding.columns)

        columns = module.get_measurement_columns(workspace.pipeline)
        self.assertEqual(len(columns), 4)
        count_feature = 'Count_%s' % OUTPUT_OBJECTS_NAME
        self.assertTrue(all([column[0] == ("Image" if column[1] == count_feature
                                           else OUTPUT_OBJECTS_NAME)
                             for column in columns]))
        self.assertTrue(
                all([column[1] in ('Location_Center_X', 'Location_Center_Y', count_feature, 'Number_Object_Number')
                     for column in columns]))
        #
        # Check the outlines
        #
        outlines = workspace.image_set.get_image(OUTLINES_NAME)
        outlines = outlines.pixel_data
        expected_outlines = centrosome.outline.outline(expected)
        self.assertTrue(numpy.all(outlines == (expected_outlines[0:outlines.shape[0], 0:outlines.shape[1]] > 0)))
        #
        # Check the measurements
        #
        categories = list(module.get_categories(None, OUTPUT_OBJECTS_NAME))
        self.assertEqual(len(categories), 2)
        categories.sort()
        self.assertEqual(categories[0], "Location")
        self.assertEqual(categories[1], "Number")
        categories = module.get_categories(None, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], "Count")
        measurements = module.get_measurements(None, cellprofiler.measurement.IMAGE, "Count")
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0], OUTPUT_OBJECTS_NAME)
        measurements = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Location")
        self.assertEqual(len(measurements), 2)
        self.assertTrue(all(m in ('Center_X', 'Center_Y') for m in measurements))
        self.assertTrue('Center_X' in measurements)
        self.assertTrue('Center_Y' in measurements)
        measurements = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Number")
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0], "Object_Number")

    def test_02_02_forced_location_auto(self):
        #
        # Automatic diameter
        #
        d = cellprofiler.modules.definegrid.DefineGrid()
        d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
        diameter = 7
        gridding = d.build_grid_info(15, 25, 1, 1, 25, 45, 2, 2)
        expected = self.make_rectangular_grid(gridding)
        i, j = numpy.mgrid[0:expected.shape[0], 0:expected.shape[1]]
        ispot, jspot = numpy.mgrid[0:gridding.rows, 0:gridding.columns]
        y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        y_locations[gridding.spot_table.flatten()] = \
            gridding.y_locations[ispot.flatten()]
        x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        x_locations[gridding.spot_table.flatten()] = \
            gridding.x_locations[jspot.flatten()]
        idist = (i - y_locations[expected])
        jdist = (j - x_locations[expected])
        expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
        #
        # Make a fuzzy mask to account for the diameter being +/- 1
        #
        mask = numpy.ones(expected.shape, bool)
        mask[numpy.abs(numpy.sqrt(idist ** 2 + jdist ** 2) - float(diameter + 1) / 2) <= 1] = False
        #
        # Make a labels matrix that's like the expected one, but
        # is relabeled randomly
        #
        guide_labels = expected.copy()
        numpy.random.seed(0)
        p = numpy.random.permutation(numpy.arange(expected.max() + 1))
        p[p == 0] = p[0]
        p[0] = 0
        guide_labels = p[guide_labels]
        workspace, module = self.make_workspace(gridding, guide_labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid))
        module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_AUTOMATIC
        module.shape_choice.value = cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_FORCED
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
        self.assertTrue(numpy.all(labels[mask] ==
                               expected[0:labels.shape[0], 0:labels.shape[1]][mask]))

    def test_03_01_natural_circle(self):
        d = cellprofiler.modules.definegrid.DefineGrid()
        d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
        #
        # Grid with x spacing = 10, y spacing = 20
        #
        diameter = 6
        gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
        expected = self.make_rectangular_grid(gridding)
        i, j = numpy.mgrid[0:expected.shape[0], 0:expected.shape[1]]
        ispot, jspot = numpy.mgrid[0:gridding.rows, 0:gridding.columns]
        y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        y_locations[gridding.spot_table.flatten()] = \
            gridding.y_locations[ispot.flatten()]
        x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        x_locations[gridding.spot_table.flatten()] = \
            gridding.x_locations[jspot.flatten()]
        #
        # Perturb the X and Y locations and diameters randomly
        #
        numpy.random.seed(0)
        x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
        y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
        random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
        idist = (i - y_locations[expected])
        jdist = (j - x_locations[expected])
        guide_labels = expected.copy()
        expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
        guide_labels[idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2] = 0
        workspace, module = self.make_workspace(gridding, guide_labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid))
        module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
        module.diameter.value = diameter
        module.shape_choice.value = cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_NATURAL
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
        self.assertTrue(numpy.all(labels == expected[0:labels.shape[0], 0:labels.shape[1]]))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_X')
        self.assertTrue(numpy.all(xm == x_locations[1:]))
        ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_Y')
        self.assertTrue(numpy.all(ym == y_locations[1:]))
        count = m.get_current_image_measurement('Count_%s' % OUTPUT_OBJECTS_NAME)
        self.assertEqual(count, gridding.rows * gridding.columns)

    def test_03_02_natural_circle_edges(self):
        #
        # Put objects near the edges of the circle and make sure
        # they are filtered out
        #
        d = cellprofiler.modules.definegrid.DefineGrid()
        d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
        #
        # Grid with x spacing = 10, y spacing = 20
        #
        diameter = 6
        gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
        expected = self.make_rectangular_grid(gridding)
        i, j = numpy.mgrid[0:expected.shape[0], 0:expected.shape[1]]
        ispot, jspot = numpy.mgrid[0:gridding.rows, 0:gridding.columns]
        y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        y_locations[gridding.spot_table.flatten()] = \
            gridding.y_locations[ispot.flatten()]
        x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        x_locations[gridding.spot_table.flatten()] = \
            gridding.x_locations[jspot.flatten()]
        #
        # save some bad places - at the corners of the grids
        #
        bad_x_locations = (x_locations - gridding.x_spacing / 2).astype(int)
        bad_y_locations = (y_locations - gridding.y_spacing / 2).astype(int)
        #
        # Perturb the X and Y locations and diameters randomly
        #
        numpy.random.seed(0)
        x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
        y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
        random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
        idist = (i - y_locations[expected])
        jdist = (j - x_locations[expected])
        guide_labels = expected.copy()
        expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
        guide_labels[idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2] = 0
        #
        # Add objects in bad places
        #
        for i_off in (-1, 0, 1):
            for j_off in (-1, 0, 1):
                guide_labels[bad_y_locations + i_off, bad_x_locations + j_off] = numpy.arange(
                        len(bad_y_locations)) + gridding.rows * gridding.columns + 1
        #
        # run the module
        #
        workspace, module = self.make_workspace(gridding, guide_labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid))
        module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
        module.diameter.value = diameter
        module.shape_choice.value = cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_NATURAL
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
        self.assertTrue(numpy.all(labels == expected[0:labels.shape[0], 0:labels.shape[1]]))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_X')
        self.assertTrue(numpy.all(xm == x_locations[1:]))
        ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_Y')
        self.assertTrue(numpy.all(ym == y_locations[1:]))
        count = m.get_current_image_measurement('Count_%s' % OUTPUT_OBJECTS_NAME)
        self.assertEqual(count, gridding.rows * gridding.columns)

    def test_03_03_img_891(self):
        """Regression test of img-891, last spot filtered out"""
        d = cellprofiler.modules.definegrid.DefineGrid()
        d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
        #
        # Grid with x spacing = 10, y spacing = 20
        #
        diameter = 6
        gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
        expected = self.make_rectangular_grid(gridding)
        i, j = numpy.mgrid[0:expected.shape[0], 0:expected.shape[1]]
        ispot, jspot = numpy.mgrid[0:gridding.rows, 0:gridding.columns]
        y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        y_locations[gridding.spot_table.flatten()] = \
            gridding.y_locations[ispot.flatten()]
        x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        x_locations[gridding.spot_table.flatten()] = \
            gridding.x_locations[jspot.flatten()]
        #
        # Perturb the X and Y locations and diameters randomly
        #
        numpy.random.seed(0)
        x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
        y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
        random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
        idist = (i - y_locations[expected])
        jdist = (j - x_locations[expected])
        guide_labels = expected.copy()
        expected[idist ** 2 + jdist ** 2 > (float(diameter + 1) / 2) ** 2] = 0
        guide_labels[idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2] = 0
        #
        # Erase the last one... this triggered the bug
        #
        expected[expected == numpy.max(guide_labels)] = 0
        guide_labels[guide_labels == numpy.max(guide_labels)] = 0
        workspace, module = self.make_workspace(gridding, guide_labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid))
        module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
        module.diameter.value = diameter
        module.shape_choice.value = cellprofiler.modules.identifyobjectsingrid.SHAPE_CIRCLE_NATURAL
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
        self.assertTrue(numpy.all(labels == expected[0:labels.shape[0], 0:labels.shape[1]]))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_X')
        self.assertEqual(len(xm), 96)
        self.assertTrue(numpy.all(xm[:-1] == x_locations[1:-1]))
        self.assertTrue(numpy.isnan(xm[-1]))
        ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_Y')
        self.assertTrue(numpy.all(ym[:-1] == y_locations[1:-1]))
        count = m.get_current_image_measurement('Count_%s' % OUTPUT_OBJECTS_NAME)
        self.assertEqual(count, gridding.rows * gridding.columns)

    def test_04_01_natural(self):
        # Use natural objects.
        #
        # Put objects near the edges of the circle and make sure
        # they are filtered out.
        #
        # Randomly distribute points in objects to keep between
        # two different groups of objects.
        #
        d = cellprofiler.modules.definegrid.DefineGrid()
        d.ordering.value = cellprofiler.modules.definegrid.NUM_BY_COLUMNS
        #
        # Grid with x spacing = 10, y spacing = 20
        #
        diameter = 6
        gridding = d.build_grid_info(15, 25, 1, 1, 32, 45, 2, 2)
        guide_labels = self.make_rectangular_grid(gridding)
        i, j = numpy.mgrid[0:guide_labels.shape[0], 0:guide_labels.shape[1]]
        ispot, jspot = numpy.mgrid[0:gridding.rows, 0:gridding.columns]
        y_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        y_locations[gridding.spot_table.flatten()] = \
            gridding.y_locations[ispot.flatten()]
        x_locations = numpy.zeros(numpy.max(gridding.spot_table) + 1, int)
        x_locations[gridding.spot_table.flatten()] = \
            gridding.x_locations[jspot.flatten()]
        #
        # save some bad places - at the corners of the grids
        #
        bad_x_locations = (x_locations - gridding.x_spacing / 2).astype(int)
        bad_y_locations = (y_locations - gridding.y_spacing / 2).astype(int)
        #
        # Perturb the X and Y locations and diameters randomly
        #
        numpy.random.seed(0)
        x_locations += (numpy.random.uniform(size=x_locations.shape[0]) * 3 - 1).astype(int)
        y_locations += (numpy.random.uniform(size=y_locations.shape[0]) * 3 - 1).astype(int)
        random_diameters = numpy.random.uniform(size=y_locations.shape[0] + 1) * 4 * 3
        idist = (i - y_locations[guide_labels])
        jdist = (j - x_locations[guide_labels])
        guide_labels[idist ** 2 + jdist ** 2 > ((random_diameters[guide_labels] + 1) / 2) ** 2] = 0
        expected = guide_labels.copy()
        guide_labels[guide_labels != 0] += gridding.rows * gridding.columns * (
            numpy.random.uniform(size=numpy.sum(guide_labels != 0)) > .5)
        #
        # Take 1/2 of the points and assign them to a second class of objects.
        # All of the objects should be merged.
        #
        #
        # Add objects in bad places
        #
        for i_off in (-1, 0, 1):
            for j_off in (-1, 0, 1):
                guide_labels[bad_y_locations + i_off, bad_x_locations + j_off] = numpy.arange(
                        len(bad_y_locations)) + gridding.rows * gridding.columns * 2 + 1
        #
        # Scramble the label numbers
        #
        p = numpy.random.permutation(numpy.arange(numpy.max(guide_labels) + 1))
        p[p == 0] = p[0]
        p[0] = 0
        guide_labels = p[guide_labels]
        #
        # run the module
        #
        workspace, module = self.make_workspace(gridding, guide_labels)
        self.assertTrue(isinstance(module, cellprofiler.modules.identifyobjectsingrid.IdentifyObjectsInGrid))
        module.diameter_choice.value = cellprofiler.modules.identifyobjectsingrid.AM_MANUAL
        module.diameter.value = diameter
        module.shape_choice.value = cellprofiler.modules.identifyobjectsingrid.SHAPE_NATURAL
        module.run(workspace)
        labels = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME).segmented
        self.assertTrue(numpy.all(labels == expected[0:labels.shape[0], 0:labels.shape[1]]))
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        xm = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_X')
        self.assertTrue(numpy.all(xm == x_locations[1:]))
        ym = m.get_current_measurement(OUTPUT_OBJECTS_NAME, 'Location_Center_Y')
        self.assertTrue(numpy.all(ym == y_locations[1:]))
        count = m.get_current_image_measurement('Count_%s' % OUTPUT_OBJECTS_NAME)
        self.assertEqual(count, gridding.rows * gridding.columns)
