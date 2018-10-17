import unittest

import numpy
import six

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.measureimageareaoccupied
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()


OBJECTS_NAME = "MyObjects"


class TestMeasureImageArea(unittest.TestCase):
    def make_workspace(self, labels, parent_image=None):
        object_set = cellprofiler.object.ObjectSet()
        objects = cellprofiler.object.Objects()
        objects.segmented = labels
        objects.parent_image = parent_image
        object_set.add_objects(objects, OBJECTS_NAME)

        pipeline = cellprofiler.pipeline.Pipeline()
        module = cellprofiler.modules.measureimageareaoccupied.MeasureImageAreaOccupied()
        module.module_num = 1
        module.operands[0].operand_objects.value = OBJECTS_NAME
        pipeline.add_module(module)
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     image_set_list.get_image_set(0),
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        return workspace

    def test_00_00_zeros(self):
        workspace = self.make_workspace(numpy.zeros((10, 10), int))
        module = workspace.module
        module.operands[0].operand_choice.value = "Objects"
        module.run(workspace)
        m = workspace.measurements

        def mn(x):
            return "AreaOccupied_%s_%s" % (x, module.operands[0].operand_objects.value)

        self.assertEqual(m.get_current_measurement("Image", mn("AreaOccupied")), 0.0)
        self.assertEqual(m.get_current_measurement("Image", mn("TotalArea")), 100)

        columns = module.get_measurement_columns(workspace.pipeline)
        features = m.get_feature_names(cellprofiler.measurement.IMAGE)
        self.assertEqual(len(columns), len(features))
        for column in columns:
            self.assertTrue(column[1] in features)

    def test_01_01_one_object(self):
        labels = numpy.zeros((10, 10), int)
        labels[2:7, 3:8] = 1
        area_occupied = numpy.sum(labels)
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
        labels = numpy.zeros((10, 10), int)
        labels[0:7, 3:8] = 1
        mask = numpy.zeros((10, 10), bool)
        mask[1:9, 1:9] = True
        image = cellprofiler.image.Image(numpy.zeros((10, 10)), mask=mask)
        area_occupied = [30]
        perimeter = [18]
        total_area = 64
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
        module = cellprofiler.modules.measureimageareaoccupied.MeasureImageAreaOccupied()
        module.operands[0].operand_objects.value = OBJECTS_NAME
        module.operands[0].operand_choice.value = "Objects"
        columns = module.get_measurement_columns(cellprofiler.pipeline.Pipeline())
        expected = ((cellprofiler.measurement.IMAGE, "AreaOccupied_AreaOccupied_%s" % OBJECTS_NAME,
                     cellprofiler.measurement.COLTYPE_FLOAT),
                    (cellprofiler.measurement.IMAGE, "AreaOccupied_Perimeter_%s" % OBJECTS_NAME,
                     cellprofiler.measurement.COLTYPE_FLOAT),
                    (cellprofiler.measurement.IMAGE, "AreaOccupied_TotalArea_%s" % OBJECTS_NAME,
                     cellprofiler.measurement.COLTYPE_FLOAT))
        self.assertEqual(len(columns), len(expected))
        for column in columns:
            self.assertTrue(any([all([cf == ef for cf, ef in zip(column, ex)])
                                 for ex in expected]))

    def test_objects_volume(self):
        labels = numpy.zeros((5, 10, 10), dtype=numpy.uint8)
        labels[:2, :2, :2] = 1
        labels[3:, 8:, 8:] = 2

        expected_area = 16
        expected_perimeter = 16
        expected_total_area = 500

        workspace = self.make_workspace(labels)
        workspace.pipeline.set_volumetric(True)

        module = workspace.module
        module.operands[0].operand_choice.value = "Objects"

        module.run(workspace)

        def mn(x):
            return "AreaOccupied_%s_%s" % (x, module.operands[0].operand_objects.value)

        numpy.testing.assert_array_equal(
            workspace.measurements.get_current_measurement("Image", mn("VolumeOccupied")),
            expected_area
        )

        numpy.testing.assert_array_almost_equal(
            workspace.measurements.get_current_measurement("Image", mn("SurfaceArea")),
            expected_perimeter,
            decimal=0
        )

        numpy.testing.assert_array_equal(
            workspace.measurements.get_current_measurement("Image", mn("TotalVolume")),
            expected_total_area
        )

    def test_image_volume(self):
        pixel_data = numpy.zeros((5, 10, 10), dtype=numpy.bool)
        pixel_data[:2, :2, :2] = True
        pixel_data[3:, 8:, 8:] = True

        image = cellprofiler.image.Image(pixel_data, dimensions=3)

        expected_area = [16]
        expected_perimeter = [16]
        expected_total_area = 500

        workspace = self.make_workspace(numpy.zeros_like(pixel_data), parent_image=image)
        workspace.pipeline.set_volumetric(True)
        workspace.image_set.add("MyBinaryImage", image)

        module = workspace.module
        module.operands[0].operand_choice.value = "Binary Image"
        module.operands[0].binary_name.value = "MyBinaryImage"

        module.run(workspace)

        def mn(x):
            return "AreaOccupied_%s_%s" % (x, module.operands[0].binary_name.value)

        numpy.testing.assert_array_equal(
            workspace.measurements.get_current_measurement("Image", mn("VolumeOccupied")),
            expected_area
        )

        numpy.testing.assert_array_almost_equal(
            workspace.measurements.get_current_measurement("Image", mn("SurfaceArea")),
            expected_perimeter,
            decimal=0
        )

        numpy.testing.assert_array_equal(
            workspace.measurements.get_current_measurement("Image", mn("TotalVolume")),
            expected_total_area
        )

    def test_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:300
GitHash:
ModuleCount:1
HasImagePlaneDetails:False

MeasureImageAreaOccupied:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Hidden:3
    Measure the area occupied in a binary image, or in objects?:Binary Image
    Select objects to measure:None
    Retain a binary image of the object regions?:No
    Name the output binary image:Stain
    Select a binary image to measure:DNA
    Measure the area occupied in a binary image, or in objects?:Objects
    Select objects to measure:Cells
    Retain a binary image of the object regions?:Yes
    Name the output binary image:Stain
    Select a binary image to measure:None
    Measure the area occupied in a binary image, or in objects?:Objects
    Select objects to measure:Nuclei
    Retain a binary image of the object regions?:No
    Name the output binary image:Stain
    Select a binary image to measure:None
"""


        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(callback)
        pipeline.load(six.StringIO(data))

        module = pipeline.modules()[0]

        assert module.count.value == 3

        assert module.operands[0].operand_choice == "Binary Image"

        assert module.operands[1].operand_choice == "Objects"
        assert module.operands[1].operand_objects == "Cells"

        assert module.operands[2].operand_choice == "Objects"
        assert module.operands[2].operand_objects == "Nuclei"
