import StringIO
import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.tile
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"


def input_image_name(index):
    return INPUT_IMAGE_NAME + str(index + 1)


class TestTile(unittest.TestCase):
    def test_01_03_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9098

Tile:[module_num:1|svn_version:\'9034\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the input image:ResizedColorImage
    Name the output image:TiledImage
    Tile within cycles or across cycles?:Across cycles
    Number of rows in final tiled image\x3A:2
    Number of columns in final tiled image\x3A:12
    Begin tiling in this corner of the final image\x3A:top left
    Begin tiling across a row, or down a column?:row
    Tile in meander mode?:No
    Automatically calculate # of rows?:Yes
    Automatically calculate # of columns?:No
    Select an additional image\x3A:Cytoplasm
    Select an additional image\x3A:ColorImage
    Select an additional image\x3A:DNA
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertEqual(module.input_image, "ResizedColorImage")
        self.assertEqual(module.output_image, "TiledImage")
        self.assertEqual(module.tile_method, cellprofiler.modules.tile.T_ACROSS_CYCLES)
        self.assertEqual(module.rows, 2)
        self.assertEqual(module.columns, 12)
        self.assertTrue(module.wants_automatic_rows)
        self.assertFalse(module.wants_automatic_columns)
        self.assertEqual(module.place_first, cellprofiler.modules.tile.P_TOP_LEFT)
        self.assertEqual(module.tile_style, cellprofiler.modules.tile.S_ROW)
        self.assertFalse(module.meander)
        self.assertEqual(len(module.additional_images), 3)
        for g, expected in zip(module.additional_images,
                               ("Cytoplasm", "ColorImage", "DNA")):
            self.assertEqual(g.input_image_name, expected)

    def make_tile_workspace(self, images):
        module = cellprofiler.modules.tile.Tile()
        module.module_num = 1
        module.tile_method.value = cellprofiler.modules.tile.T_ACROSS_CYCLES
        module.input_image.value = INPUT_IMAGE_NAME
        module.output_image.value = OUTPUT_IMAGE_NAME

        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cellprofiler.image.ImageSetList()
        for i, image in enumerate(images):
            image_set = image_set_list.get_image_set(i)
            image_set.add(INPUT_IMAGE_NAME, cellprofiler.image.Image(image))

        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def test_02_01_manual_rows_and_columns(self):
        numpy.random.seed(0)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_ROW

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = int(i / 16)
            jj = i % 16
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_02_automatic_rows(self):
        numpy.random.seed(1)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = True
        module.rows.value = 8
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_ROW

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = int(i / 16)
            jj = i % 16
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_03_automatic_columns(self):
        numpy.random.seed(2)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = True
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 365
        module.tile_style.value = cellprofiler.modules.tile.S_ROW

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = int(i / 16)
            jj = i % 16
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_04_automatic_rows_and_columns(self):
        numpy.random.seed(3)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = True
        module.wants_automatic_rows.value = True
        module.rows.value = 365
        module.columns.value = 24
        module.tile_style.value = cellprofiler.modules.tile.S_ROW

        module.prepare_group(workspace, (), numpy.arange(1, 97))
        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 9 * 20)
        self.assertEqual(pixel_data.shape[1], 11 * 10)
        for i, image in enumerate(images):
            ii = int(i / 11)
            jj = i % 11
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_05_color(self):
        numpy.random.seed(4)
        images = [numpy.random.uniform(size=(20, 10, 3)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_ROW

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = int(i / 16)
            jj = i % 16
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10), :] ==
                                      image))

    def test_02_06_columns_first(self):
        numpy.random.seed(5)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_COL

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        module.post_group(workspace, None)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = i % 6
            jj = int(i / 6)
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_07_top_right(self):
        numpy.random.seed(0)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_ROW
        module.place_first.value = cellprofiler.modules.tile.P_TOP_RIGHT

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        module.post_group(workspace, None)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = int(i / 16)
            jj = 15 - (i % 16)
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_08_bottom_left(self):
        numpy.random.seed(8)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_ROW
        module.place_first.value = cellprofiler.modules.tile.P_BOTTOM_LEFT

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        module.post_group(workspace, None)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = 5 - int(i / 16)
            jj = i % 16
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_09_bottom_right(self):
        numpy.random.seed(9)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_ROW
        module.place_first.value = cellprofiler.modules.tile.P_BOTTOM_RIGHT

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(96):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        module.post_group(workspace, None)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images):
            ii = 5 - int(i / 16)
            jj = 15 - (i % 16)
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def test_02_10_different_sizes(self):
        numpy.random.seed(10)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32),
                  numpy.random.uniform(size=(10, 20)).astype(numpy.float32),
                  numpy.random.uniform(size=(40, 5)).astype(numpy.float32),
                  numpy.random.uniform(size=(40, 20)).astype(numpy.float32)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 1
        module.columns.value = 4
        module.tile_style.value = cellprofiler.modules.tile.S_ROW
        module.prepare_group(workspace, (), numpy.arange(1, 4))

        for i in range(4):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        module.post_group(workspace, None)
        pixel_data = workspace.image_set.get_image(OUTPUT_IMAGE_NAME).pixel_data
        self.assertEqual(pixel_data.shape[0], 20)
        self.assertEqual(pixel_data.shape[1], 40)
        self.assertTrue(numpy.all(pixel_data[:, :10] == images[0]))
        self.assertTrue(numpy.all(pixel_data[:10, 10:20] == images[1][:, :10]))
        self.assertTrue(numpy.all(pixel_data[10:, 10:20] == 0))
        self.assertTrue(numpy.all(pixel_data[:, 20:25] == images[2][:20, :]))
        self.assertTrue(numpy.all(pixel_data[:, 25:30] == 0))
        self.assertTrue(numpy.all(pixel_data[:, 30:] == images[3][:20, :10]))

    def test_02_11_filtered(self):
        numpy.random.seed(9)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for i in range(96)]
        workspace, module = self.make_tile_workspace(images)
        self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
        self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))
        module.wants_automatic_columns.value = False
        module.wants_automatic_rows.value = False
        module.rows.value = 6
        module.columns.value = 16
        module.tile_style.value = cellprofiler.modules.tile.S_ROW
        module.place_first.value = cellprofiler.modules.tile.P_BOTTOM_RIGHT

        module.prepare_group(workspace, (), numpy.arange(1, 97))

        for i in range(95):
            workspace.set_image_set_for_testing_only(i)
            module.run(workspace)
        workspace.set_image_set_for_testing_only(95)
        module.post_group(workspace, None)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 6 * 20)
        self.assertEqual(pixel_data.shape[1], 16 * 10)
        for i, image in enumerate(images[:-1]):
            ii = 5 - int(i / 16)
            jj = 15 - (i % 16)
            iii = ii * 20
            jjj = jj * 10
            self.assertTrue(numpy.all(pixel_data[iii:(iii + 20), jjj:(jjj + 10)] ==
                                      image))

    def make_place_workspace(self, images):
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = cellprofiler.modules.tile.Tile()
        module.module_num = 1
        module.tile_method.value = cellprofiler.modules.tile.T_WITHIN_CYCLES
        module.output_image.value = OUTPUT_IMAGE_NAME
        module.wants_automatic_rows.value = False
        module.wants_automatic_columns.value = True
        module.rows.value = 1
        for i, image in enumerate(images):
            image_name = input_image_name(i)
            if i == 0:
                module.input_image.value = image_name
            else:
                if len(module.additional_images) <= i:
                    module.add_image()
                module.additional_images[i - 1].input_image_name.value = image_name
            image_set.add(image_name, cellprofiler.image.Image(image))

        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set,
                                                     cellprofiler.region.Set(),
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace, module

    def test_03_01_some_images(self):
        numpy.random.seed(31)
        for i in range(1, 5):
            images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32) for ii in range(i)]
            workspace, module = self.make_place_workspace(images)
            self.assertTrue(isinstance(module, cellprofiler.modules.tile.Tile))
            self.assertTrue(isinstance(workspace, cellprofiler.workspace.Workspace))

            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            pixel_data = image.pixel_data
            for j, p in enumerate(images):
                jj = 10 * j
                self.assertTrue(numpy.all(pixel_data[:, jj:(jj + 10)] == p))

    def test_03_02_mix_color_bw(self):
        numpy.random.seed(32)
        for color in range(3):
            images = [numpy.random.uniform(size=(20, 10, 3) if i == color else (20, 10)).astype(numpy.float32)
                      for i in range(3)]
            workspace, module = self.make_place_workspace(images)
            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            pixel_data = image.pixel_data

            for j, p in enumerate(images):
                jj = 10 * j
                if j == color:
                    self.assertTrue(numpy.all(pixel_data[:, jj:(jj + 10), :] == p))
                else:
                    for k in range(3):
                        self.assertTrue(numpy.all(pixel_data[:, jj:(jj + 10), k] == p))

    def test_03_03_different_sizes(self):
        numpy.random.seed(33)
        images = [numpy.random.uniform(size=(20, 10)).astype(numpy.float32),
                  numpy.random.uniform(size=(10, 20)).astype(numpy.float32),
                  numpy.random.uniform(size=(40, 5)).astype(numpy.float32),
                  numpy.random.uniform(size=(40, 20)).astype(numpy.float32)]
        workspace, module = self.make_place_workspace(images)
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertEqual(pixel_data.shape[0], 40)
        self.assertEqual(pixel_data.shape[1], 80)
        mask = numpy.ones(pixel_data.shape, bool)
        self.assertTrue(numpy.all(pixel_data[:20, :10] == images[0]))
        mask[:20, :10] = False
        self.assertTrue(numpy.all(pixel_data[:10, 20:40] == images[1]))
        mask[:10, 20:40] = False
        self.assertTrue(numpy.all(pixel_data[:, 40:45] == images[2]))
        mask[:, 40:45] = False
        self.assertTrue(numpy.all(pixel_data[:, 60:] == images[3]))
        mask[:, 60:] = False
        self.assertTrue(numpy.all(pixel_data[mask] == 0))
