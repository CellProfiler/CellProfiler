import cStringIO
import os
import tempfile
import unittest

import cellh5
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.exporttocellh5
import cellprofiler.pipeline
import cellprofiler.region
import cellprofiler.setting
import cellprofiler.workspace
import numpy

IMAGE_NAME = "imagename"
OBJECTS_NAME = "objectsname"
FEATURE1 = "feature1"
FEATURE2 = "feature2"


class TestExportToCellH5(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)

    def test_01_01_load_v1(self):
        pipeline_text = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150220145847
GitHash:bc550fe
ModuleCount:15
HasImagePlaneDetails:False

ExportToCellH5:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Hidden:3
    Hidden:4
    Output file location:Default Output Folder\x7C
    Output file name:myfile.ch5
    Overwrite existing data without warning?:Yes
    Repack after analysis:Yes
    Plate metadata:None
    Well metadata:None
    Site metadata:Site
    Choose measurements?:No
    Measurements to export:
    Objects name:Nuclei
    Objects name:Cells
    Objects name:Cytoplasm
    Image name:CropBlue
    Image name:CropGreen
    Image name:CropRed
    Image name:RGBImage
"""
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.loadtxt(cStringIO.StringIO(pipeline_text))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        assert isinstance(module, cellprofiler.modules.exporttocellh5.ExportToCellH5)
        self.assertEqual(module.directory.dir_choice,
                         cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)
        self.assertEqual(module.file_name, "myfile.ch5")
        self.assertTrue(module.overwrite_ok)
        self.assertEqual(module.plate_metadata, module.IGNORE_METADATA)
        self.assertEqual(module.well_metadata, module.IGNORE_METADATA)
        self.assertEqual(module.site_metadata, "Site")
        self.assertFalse(module.wants_to_choose_measurements)
        self.assertEqual(len(module.objects_to_export), 3)
        for object_name, group in zip(("Nuclei", "Cells", "Cytoplasm"),
                                      module.objects_to_export):
            self.assertEqual(group.objects_name, object_name)
        self.assertEqual(len(module.images_to_export), 4)
        for image_name, group in zip(
                ("CropBlue", "CropGreen", "CropRed", "RGBImage"),
                module.images_to_export):
            self.assertEqual(group.image_name, image_name)

    def get_plate_name(self, idx):
        return "Plate%d" % idx

    def get_well_name(self, idx):
        return "%s%02d" % (chr(ord('A') + int(idx / 12)), idx % 12)

    def get_site_name(self, idx):
        return str(idx + 1)

    def prepare_workspace(self, sites=None):
        """Create a module and workspace for testing

        returns module, workspace
        """
        if sites is None:
            sites = [(self.get_plate_name(0),
                      self.get_well_name(0),
                      self.get_site_name(0))]
        m = cellprofiler.measurement.Measurements()
        for idx, (plate, well, site) in enumerate(sites):
            m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_PLATE, idx + 1] = plate
            m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_WELL, idx + 1] = well
            m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_SITE, idx + 1] = site
        module = cellprofiler.modules.exporttocellh5.ExportToCellH5()
        module.plate_metadata.value = cellprofiler.measurement.FTR_PLATE
        module.well_metadata.value = cellprofiler.measurement.FTR_WELL
        module.site_metadata.value = cellprofiler.measurement.FTR_SITE
        module.directory.dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
        module.directory.custom_path = self.temp_dir
        module.module_num = 1
        pipeline = MyPipeline()
        pipeline.add_module(module)
        pipeline.extra_measurement_columns += [
            (
            cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_PLATE, cellprofiler.measurement.COLTYPE_VARCHAR),
            (cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_WELL, cellprofiler.measurement.COLTYPE_VARCHAR),
            (cellprofiler.measurement.IMAGE, cellprofiler.measurement.M_SITE, cellprofiler.measurement.COLTYPE_VARCHAR)]
        return module, cellprofiler.workspace.Workspace(
            pipeline, module, m, cellprofiler.region.ObjectSet(), m, None)

    def test_02_01_export_image(self):
        module, workspace = self.prepare_workspace()
        assert isinstance(module, cellprofiler.modules.exporttocellh5.ExportToCellH5)
        r = numpy.random.RandomState()
        r.seed(201)
        image = cellprofiler.image.Image(r.uniform(size=(20, 35)), scale=255)
        workspace.image_set.add(IMAGE_NAME, image)
        module.add_image()
        module.images_to_export[0].image_name.value = IMAGE_NAME
        module.run(workspace)
        with cellh5.ch5open(
                os.path.join(self.temp_dir, module.file_name.value), "r") as ch5:
            well = self.get_well_name(0)
            site = self.get_site_name(0)
            self.assertTrue(ch5.has_position(well, site))
            pos = ch5.get_position(well, site)
            image_defs = ch5.image_definition
            image_channels = image_defs[cellh5.CH5Const.RAW_IMAGE]
            self.assertEqual(len(image_channels), 1)
            self.assertEqual(image_channels[0, "channel_name"], IMAGE_NAME)
            image_out = pos.get_image(0, 0)
            numpy.testing.assert_array_equal(
                (image.pixel_data * 255).astype(int), image_out)

    def test_02_02_export_rgb_image(self):
        module, workspace = self.prepare_workspace()
        assert isinstance(module, cellprofiler.modules.exporttocellh5.ExportToCellH5)
        r = numpy.random.RandomState()
        r.seed(202)
        image = cellprofiler.image.Image(r.uniform(size=(20, 35, 3)), scale=255)
        workspace.image_set.add(IMAGE_NAME, image)
        module.add_image()
        module.images_to_export[0].image_name.value = IMAGE_NAME
        module.run(workspace)
        with cellh5.ch5open(
                os.path.join(self.temp_dir, module.file_name.value), "r") as ch5:
            well = self.get_well_name(0)
            site = self.get_site_name(0)
            self.assertTrue(ch5.has_position(well, site))
            pos = ch5.get_position(well, site)
            image_defs = ch5.image_definition
            image_channels = image_defs[cellh5.CH5Const.RAW_IMAGE]
            self.assertEqual(len(image_channels), 3)
            for i, image_channel in enumerate(image_channels):
                self.assertEqual(
                    image_channel["channel_name"],
                    "_".join((IMAGE_NAME, cellprofiler.modules.exporttocellh5.COLORS[i][0])))
                image_out = pos.get_image(0, i)
                numpy.testing.assert_array_equal(
                    (image.pixel_data[:, :, i] * 255).astype(int), image_out)

    def test_02_03_export_int16_image(self):
        module, workspace = self.prepare_workspace()
        assert isinstance(module, cellprofiler.modules.exporttocellh5.ExportToCellH5)
        r = numpy.random.RandomState()
        r.seed(203)
        image = cellprofiler.image.Image(r.uniform(size=(20, 35)), scale=4095)
        workspace.image_set.add(IMAGE_NAME, image)
        module.add_image()
        module.images_to_export[0].image_name.value = IMAGE_NAME
        module.run(workspace)
        with cellh5.ch5open(
                os.path.join(self.temp_dir, module.file_name.value), "r") as ch5:
            well = self.get_well_name(0)
            site = self.get_site_name(0)
            self.assertTrue(ch5.has_position(well, site))
            pos = ch5.get_position(well, site)
            image_defs = ch5.image_definition
            image_channels = image_defs[cellh5.CH5Const.RAW_IMAGE]
            self.assertEqual(len(image_channels), 1)
            self.assertEqual(image_channels[0, "channel_name"], IMAGE_NAME)
            image_out = pos.get_image(0, 0)
            self.assertTrue(numpy.issubdtype(image_out.dtype, numpy.uint16))
            numpy.testing.assert_array_equal(
                (image.pixel_data * 4095).astype(int), image_out)

    def test_03_01_export_objects(self):
        module, workspace = self.prepare_workspace()
        assert isinstance(module, cellprofiler.modules.exporttocellh5.ExportToCellH5)
        labels = numpy.zeros((21, 17), int)
        labels[1:4, 2:5] = 1
        labels[11:14, 12:15] = 2
        centers = numpy.array([[2, 3], [12, 13]])
        minima = numpy.array([[1, 2], [11, 12]])
        maxima = numpy.array([[3, 4], [13, 14]])
        objects = cellprofiler.region.Objects()
        objects.segmented = labels
        workspace.object_set.add_objects(objects, OBJECTS_NAME)
        module.add_objects()
        module.objects_to_export[0].objects_name.value = OBJECTS_NAME
        module.run(workspace)
        with cellh5.ch5open(
                os.path.join(self.temp_dir, module.file_name.value), "r") as ch5:
            well = self.get_well_name(0)
            site = self.get_site_name(0)
            self.assertTrue(ch5.has_position(well, site))
            pos = ch5.get_position(well, site)
            image_defs = ch5.image_definition
            object_channels = image_defs[cellh5.CH5Const.REGION]
            self.assertEqual(len(object_channels), 1)
            self.assertEqual(
                object_channels[0, "region_name"], OBJECTS_NAME)
            self.assertEqual(object_channels[0, "channel_idx"], 0)
            centers_out = pos.get_center([0, 1], OBJECTS_NAME)
            numpy.testing.assert_array_equal(centers_out[:]["x"], centers[:, 1])
            numpy.testing.assert_array_equal(centers_out[:]["y"], centers[:, 0])
            self.assertEqual(len(pos.get_object_table(OBJECTS_NAME)), 2)
            labels_out = \
                pos[cellh5.CH5Const.IMAGE][cellh5.CH5Const.REGION][0, 0, 0]
            numpy.testing.assert_array_equal(labels, labels_out)

    def test_04_01_all_features(self):
        r = numpy.random.RandomState()
        r.seed(401)
        module, workspace = self.prepare_workspace()
        assert isinstance(module, cellprofiler.modules.exporttocellh5.ExportToCellH5)
        labels = numpy.zeros((21, 17), int)
        labels[1:4, 2:5] = 1
        labels[11:14, 12:15] = 2
        centers = numpy.array([[2, 3], [12, 13]])
        minima = numpy.array([[1, 2], [11, 12]])
        maxima = numpy.array([[3, 4], [13, 14]])
        objects = cellprofiler.region.Objects()
        objects.segmented = labels
        workspace.object_set.add_objects(objects, OBJECTS_NAME)
        module.add_objects()
        module.objects_to_export[0].objects_name.value = OBJECTS_NAME
        m = workspace.measurements
        m[OBJECTS_NAME, FEATURE1] = r.uniform(size=2)
        m[OBJECTS_NAME, FEATURE2] = r.uniform(size=2)
        workspace.pipeline.extra_measurement_columns += [
            (OBJECTS_NAME, FEATURE1, cellprofiler.measurement.COLTYPE_FLOAT),
            (OBJECTS_NAME, FEATURE2, cellprofiler.measurement.COLTYPE_FLOAT)]
        module.run(workspace)
        with cellh5.ch5open(
                os.path.join(self.temp_dir, module.file_name.value), "r") as ch5:
            well = self.get_well_name(0)
            site = self.get_site_name(0)
            self.assertTrue(ch5.has_position(well, site))
            pos = ch5.get_position(well, site)
            defs = pos.object_feature_def(OBJECTS_NAME)
            self.assertTrue(all([x in defs for x in (FEATURE1, FEATURE2)]))
            feature1_idx, feature2_idx = [
                defs.index(ftr) for ftr in (FEATURE1, FEATURE2)]
            ftr_out = pos.get_object_features(OBJECTS_NAME)
            feature1 = ftr_out[:, feature1_idx]
            numpy.testing.assert_almost_equal(feature1, m[OBJECTS_NAME, FEATURE1])
            feature2 = ftr_out[:, feature2_idx]
            numpy.testing.assert_almost_equal(feature2, m[OBJECTS_NAME, FEATURE2])

    def test_04_02_some_features(self):
        r = numpy.random.RandomState()
        r.seed(402)
        module, workspace = self.prepare_workspace()
        assert isinstance(module, cellprofiler.modules.exporttocellh5.ExportToCellH5)
        labels = numpy.zeros((21, 17), int)
        labels[1:4, 2:5] = 1
        labels[11:14, 12:15] = 2
        centers = numpy.array([[2, 3], [12, 13]])
        minima = numpy.array([[1, 2], [11, 12]])
        maxima = numpy.array([[3, 4], [13, 14]])
        objects = cellprofiler.region.Objects()
        objects.segmented = labels
        workspace.object_set.add_objects(objects, OBJECTS_NAME)
        module.add_objects()
        module.objects_to_export[0].objects_name.value = OBJECTS_NAME
        module.wants_to_choose_measurements.value = True
        module.measurements.set_value(
            [module.measurements.make_measurement_choice(
                OBJECTS_NAME, FEATURE1)])
        m = workspace.measurements
        m[OBJECTS_NAME, FEATURE1] = r.uniform(size=2)
        m[OBJECTS_NAME, FEATURE2] = r.uniform(size=2)
        workspace.pipeline.extra_measurement_columns += [
            (OBJECTS_NAME, FEATURE1, cellprofiler.measurement.COLTYPE_FLOAT),
            (OBJECTS_NAME, FEATURE2, cellprofiler.measurement.COLTYPE_FLOAT)]
        module.run(workspace)
        with cellh5.ch5open(
                os.path.join(self.temp_dir, module.file_name.value), "r") as ch5:
            well = self.get_well_name(0)
            site = self.get_site_name(0)
            self.assertTrue(ch5.has_position(well, site))
            pos = ch5.get_position(well, site)
            defs = pos.object_feature_def(OBJECTS_NAME)
            self.assertTrue(FEATURE1 in defs)
            self.assertFalse(FEATURE2 in defs)
            feature1 = pos.get_object_features(OBJECTS_NAME) \
                [:, defs.index(FEATURE1)]
            numpy.testing.assert_almost_equal(feature1, m[OBJECTS_NAME, FEATURE1])


class MyPipeline(cellprofiler.pipeline.Pipeline):
    """Fake pipeline class for mock injecting measurement columns"""

    def __init__(self):
        cellprofiler.pipeline.Pipeline.__init__(self)
        self.extra_measurement_columns = []

    def get_measurement_columns(self, terminating_module=None):
        return super(MyPipeline, self).get_measurement_columns(
            terminating_module) + self.extra_measurement_columns
