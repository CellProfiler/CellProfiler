'''test_definegrid - test the DefineGrid module
'''

import base64
import unittest
import zlib

import numpy as np
from six.moves import StringIO

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.grid as cpg
import cellprofiler.image as cpi
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.definegrid as D

GRID_NAME = "grid"
INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"
OBJECTS_NAME = "objects"


class TestDefineGrid(unittest.TestCase):
    def test_01_02_load_v1(self):
        data = ('eJztWN1P2lAUvyi6ocumi4l7vA97UCOMj+mALAoOnSTCiBC3RZ2r9AJ3Kb2k'
                'vfVjy9735/q4x/VAS8tdY6FoposlTTm353d/5/zuR09bytf28lt4LRbHpXwt'
                '2qAKwRVF4g2mtbNY5av4nUYkTmTM1CyutQxcZmc4kcGJVDaVyr5ew8l4PIOC'
                'HaFi6al5uVpEaNq8PjbPCevWlGWHXCfYVcI5VZv6FAqjF1b7lXkeSBqVThVy'
                'ICkG0R0Ku72oNljtstO/VWKyoZCy1HY7m0fZaJ8STf/QsIHW7Qq9IEqVfidC'
                'CrbbPjmjOmWqhbf6F1v7vIwLvNUWO9/RzHCE/rckXm9VuTkCg+2gW3zG0S0k'
                '6AY6LrjawX8XOf5hD53nXf5zlk1VmZ5R2ZAUTNtSsx91l9+nv/BAf2H0XqMy'
                '4HI+uDkhDjhr5IJHty+kOsdtkGQY/tBAPyGUHjJuEZcaEjcxgJtAiWQP55fv'
                'jJAv2AWGVcaxoRNHb79+ngj9gL3FOGdtrNFmi9v9jJpHmQUb7312rg8z7xaF'
                'uMEukIZkKBwXYdLhAtVInTPtMrCe21K9heuXdcWlpzufaaEf+7D7iSAHl/bh'
                'nxb4wS5JqrmGhtNxcgA/iT6bq24YHZ8LvGDvUIUTjchVplD5I1EU3Tv/UeaB'
                'X/5TQhxgl1hvKo/Dm/PBRQResGHP6c4h5J33TY67V96p5Oq6vX/4xT8r4MHO'
                'rJ+bg9ZR7AdAkP0rMSR/EP1G4Q+K89P9kRA32Mnk2moinQ60b5WZSsZ5vrR8'
                'cG+EeMH+srRZeQuFF9mIrSyfgAWrdeMwH60cH8ajmeMfyZ/LJzrcqFLTq9u2'
                'PNZz6ZMP7qUQJ9ixlcOjo1fHEEbBCrbfsG+oYIPbWPtc0PUZtB4Jypfz4bvt'
                '/SjoProbGa1uvC19ngn6gN009WlqzOhgs/IknRvT5wH3gLtruBy6fn141QPs'
                '9JtZCXdXyH+Xr1fd3n3pdNK9V/nelk5e78eOTr2N8z7l+4C7G7gc+rfzzo//'
                'ru4Pv104sZ4S6zDw/+qT54qQJ9h1eAvUGHyf1WLt7kdEPaYwSe59lYvtmX+L'
                'rg90N8ojkwZVCZRmsUL3L1Sx3rpFPHjc+U+Yv/mF6/UWdXb0v9oMwhcO/c03'
                '64MLW4oB7hcabXyXrvG3cxvHf9T8Q6Hx83B4wv2YEHK+14/q/wcChSof')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, D.DefineGrid))
        self.assertEqual(module.grid_image, "Grid")
        self.assertEqual(module.grid_rows, 8)
        self.assertEqual(module.grid_columns, 12)
        self.assertEqual(module.origin, D.NUM_BOTTOM_RIGHT)
        self.assertEqual(module.ordering, D.NUM_BY_ROWS)
        self.assertEqual(module.each_or_once, D.EO_EACH)
        self.assertEqual(module.auto_or_manual, D.AM_MANUAL)
        self.assertEqual(module.object_name, "FilteredSolidWells")
        self.assertEqual(module.failed_grid_choice, D.FAIL_NO)
        self.assertEqual(module.manual_choice, D.MAN_MOUSE)
        self.assertEqual(module.display_image_name, "GridImage")
        self.assertEqual(module.manual_image, "GridImage")
        self.assertEqual(module.save_image_name, "Grid")

    def make_workspace(self, image, labels):
        module = D.DefineGrid()
        module.module_num = 1
        module.grid_image.value = GRID_NAME
        module.manual_image.value = INPUT_IMAGE_NAME
        module.display_image_name.value = INPUT_IMAGE_NAME
        module.object_name.value = OBJECTS_NAME
        module.save_image_name.value = OUTPUT_IMAGE_NAME
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(INPUT_IMAGE_NAME, cpi.Image(image))
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, measurements,
                                  image_set_list)
        return workspace, module

    def test_02_01_grid_automatic(self):
        image = np.zeros((50, 100))
        labels = np.zeros((50, 100), int)
        ii, jj = np.mgrid[0:50, 0:100]
        #
        # Make two circles at 10,11 and 40, 92
        #
        first_x, first_y = (11, 10)
        second_x, second_y = (92, 40)
        rows = 4
        columns = 10
        spacing_y = 10
        spacing_x = 9
        for i in range(rows):
            for j in range(columns):
                center_i = first_y + spacing_y * i
                center_j = first_x + spacing_x * j
                labels[(ii - center_i) ** 2 + (jj - center_j) ** 2 <= 9] = i * columns + j + 1
        workspace, module = self.make_workspace(image, labels)
        self.assertTrue(isinstance(module, D.DefineGrid))
        self.assertTrue(isinstance(workspace, cpw.Workspace))
        module.grid_rows.value = rows
        module.grid_columns.value = columns
        module.ordering.value = D.NUM_BY_COLUMNS
        module.auto_or_manual.value = D.AM_AUTOMATIC
        module.wants_image.value = True
        module.run(workspace)
        gridding = workspace.get_grid(GRID_NAME)
        self.assertTrue(isinstance(gridding, cpg.Grid))
        self.assertEqual(gridding.rows, rows)
        self.assertEqual(gridding.columns, columns)
        self.assertEqual(gridding.x_spacing, spacing_x)
        self.assertEqual(gridding.y_spacing, spacing_y)
        self.assertEqual(gridding.x_location_of_lowest_x_spot, first_x)
        self.assertEqual(gridding.y_location_of_lowest_y_spot, first_y)
        self.assertTrue(np.all(gridding.x_locations == first_x + np.arange(columns) * spacing_x))
        self.assertTrue(np.all(gridding.y_locations == first_y + np.arange(rows) * spacing_y))
        spot_table = np.arange(rows * columns) + 1
        spot_table.shape = (rows, columns)
        self.assertTrue(np.all(gridding.spot_table == spot_table))

        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, value in ((D.F_COLUMNS, columns),
                               (D.F_ROWS, rows),
                               (D.F_X_LOCATION_OF_LOWEST_X_SPOT, first_x),
                               (D.F_Y_LOCATION_OF_LOWEST_Y_SPOT, first_y),
                               (D.F_X_SPACING, spacing_x),
                               (D.F_Y_SPACING, spacing_y)):
            measurement = '_'.join((D.M_CATEGORY, GRID_NAME, feature))
            self.assertTrue(m.has_feature(cpmeas.IMAGE, measurement))
            self.assertEqual(m.get_current_image_measurement(measurement), value)

        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(image is not None)

    def test_02_02_fail(self):
        image = np.zeros((50, 100))
        labels = np.zeros((50, 100), int)
        labels[20:40, 51:62] = 1
        workspace, module = self.make_workspace(image, labels)
        self.assertTrue(isinstance(module, D.DefineGrid))
        self.assertTrue(isinstance(workspace, cpw.Workspace))
        module.ordering.value = D.NUM_BY_COLUMNS
        module.auto_or_manual.value = D.AM_AUTOMATIC
        module.wants_image.value = True
        self.assertRaises(RuntimeError, module.run, workspace)

    def test_03_01_coordinates_plus_savedimagesize(self):
        image = np.zeros((50, 100))
        labels = np.zeros((50, 100), int)
        first_x, first_y = (11, 10)
        second_x, second_y = (92, 40)
        rows = 4
        columns = 10
        spacing_y = 10
        spacing_x = 9
        workspace, module = self.make_workspace(image, labels)
        self.assertTrue(isinstance(module, D.DefineGrid))
        self.assertTrue(isinstance(workspace, cpw.Workspace))
        module.grid_rows.value = rows
        module.grid_columns.value = columns
        module.ordering.value = D.NUM_BY_COLUMNS
        module.auto_or_manual.value = D.AM_MANUAL
        module.manual_choice.value = D.MAN_COORDINATES
        module.first_spot_coordinates.value = "%d,%d" % (first_x, first_y)
        module.second_spot_coordinates.value = "%d,%d" % (second_x, second_y)
        module.first_spot_col.value = 1
        module.first_spot_row.value = 1
        module.second_spot_col.value = columns
        module.second_spot_row.value = rows
        module.grid_rows.value = rows
        module.grid_columns.value = columns
        module.wants_image.value = True
        module.run(workspace)
        gridding = workspace.get_grid(GRID_NAME)
        self.assertTrue(isinstance(gridding, cpg.Grid))
        self.assertEqual(gridding.rows, rows)
        self.assertEqual(gridding.columns, columns)
        self.assertEqual(gridding.x_spacing, spacing_x)
        self.assertEqual(gridding.y_spacing, spacing_y)
        self.assertEqual(gridding.x_location_of_lowest_x_spot, first_x)
        self.assertEqual(gridding.y_location_of_lowest_y_spot, first_y)
        self.assertTrue(np.all(gridding.x_locations == first_x + np.arange(columns) * spacing_x))
        self.assertTrue(np.all(gridding.y_locations == first_y + np.arange(rows) * spacing_y))
        spot_table = np.arange(rows * columns) + 1
        spot_table.shape = (rows, columns)
        self.assertTrue(np.all(gridding.spot_table == spot_table))

        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, value in ((D.F_COLUMNS, columns),
                               (D.F_ROWS, rows),
                               (D.F_X_LOCATION_OF_LOWEST_X_SPOT, first_x),
                               (D.F_Y_LOCATION_OF_LOWEST_Y_SPOT, first_y),
                               (D.F_X_SPACING, spacing_x),
                               (D.F_Y_SPACING, spacing_y)):
            measurement = '_'.join((D.M_CATEGORY, GRID_NAME, feature))
            self.assertTrue(m.has_feature(cpmeas.IMAGE, measurement))
            self.assertEqual(m.get_current_image_measurement(measurement), value)

        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(image is not None)
        shape = image.pixel_data.shape
        self.assertEqual(shape[0], 50)
        self.assertEqual(shape[1], 100)
