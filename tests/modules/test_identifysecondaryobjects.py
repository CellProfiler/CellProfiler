import StringIO
import base64
import unittest
import zlib

import centrosome.threshold
import numpy

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.applythreshold
import cellprofiler.modules.identify
import cellprofiler.modules.identifysecondaryobjects
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()


INPUT_OBJECTS_NAME = "input_objects"
OUTPUT_OBJECTS_NAME = "output_objects"
NEW_OBJECTS_NAME = "new_objects"
IMAGE_NAME = "image"
THRESHOLD_IMAGE_NAME = "threshold"


class TestIdentifySecondaryObjects(unittest.TestCase):
    def test_01_09_load_v9(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130226215424
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    :
    Filter based on rules:No
    Filter:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Extract metadata?:No
    Extraction method count:1
    Extraction method:Automatic
    Source:From file name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Assignment method:Assign all images
    Load as:Grayscale image
    Image name:DNA
    :\x5B\x5D
    Assign channels by:Order
    Assignments count:1
    Match this rule:or (file does contain "")
    Image name:DNA
    Objects name:Cell
    Load as:Grayscale image

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

IdentifySecondaryObjects:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:9|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input objects:ChocolateChips
    Name the objects to be identified:Cookies
    Select the method to identify the secondary objects:Propagation
    Select the input image:BakingSheet
    Number of pixels by which to expand the primary objects:11
    Regularization factor:0.125
    Name the outline image:CookieEdges
    Retain outlines of the identified secondary objects?:No
    Discard secondary objects touching the border of the image?:Yes
    Discard the associated primary objects?:No
    Name the new primary objects:FilteredChocolateChips
    Retain outlines of the new primary objects?:No
    Name the new primary object outlines:FilteredChocolateChipOutlines
    Fill holes in identified objects?:Yes
    Threshold setting version:1
    Threshold strategy:Automatic
    Threshold method:Otsu
    Smoothing for threshold:Automatic
    Threshold smoothing scale:1.5
    Threshold correction factor:.95
    Lower and upper bounds on threshold:0.01,.95
    Approximate fraction of image covered by objects?:0.02
    Manual threshold:0.3
    Select the measurement to threshold with:Count_Cookies
    Select binary image:CookieMask
    Masking objects:CookieMonsters
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Method to calculate adaptive window size:Image size
    Size of adaptive window:9
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects))
        self.assertEqual(module.x_name, "ChocolateChips")
        self.assertEqual(module.y_name, "Cookies")
        self.assertEqual(module.image_name, "BakingSheet")
        self.assertEqual(module.method, cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION)
        self.assertEqual(module.distance_to_dilate, 11)
        self.assertEqual(module.regularization_factor, .125)
        self.assertEqual(module.outlines_name, "CookieEdges")
        self.assertFalse(module.use_outlines)
        self.assertTrue(module.wants_discard_edge)
        self.assertFalse(module.wants_discard_primary)
        self.assertEqual(module.new_primary_objects_name, "FilteredChocolateChips")
        self.assertFalse(module.wants_primary_outlines)
        self.assertEqual(module.new_primary_outlines_name, "FilteredChocolateChipOutlines")
        self.assertTrue(module.fill_holes)
        self.assertEqual(module.apply_threshold.threshold_scope, cellprofiler.modules.identify.TS_GLOBAL)
        self.assertEqual(module.apply_threshold.global_operation.value, cellprofiler.modules.applythreshold.TM_LI)
        self.assertEqual(module.apply_threshold.threshold_smoothing_scale.value, 1.3488)
        self.assertEqual(module.apply_threshold.threshold_correction_factor, 1)
        self.assertEqual(module.apply_threshold.threshold_range.min, 0.0)
        self.assertEqual(module.apply_threshold.threshold_range.max, 1.0)
        self.assertEqual(module.apply_threshold.manual_threshold, .3)
        self.assertEqual(module.apply_threshold.thresholding_measurement, "Count_Cookies")
        self.assertEqual(module.apply_threshold.two_class_otsu, cellprofiler.modules.identify.O_TWO_CLASS)
        self.assertEqual(module.apply_threshold.assign_middle_to_foreground, cellprofiler.modules.identify.O_FOREGROUND)
        self.assertEqual(module.apply_threshold.adaptive_window_size, 9)

    def make_workspace(self, image, segmented, unedited_segmented=None,
                       small_removed_segmented=None):
        p = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        p.add_listener(callback)
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(image)
        objects = cellprofiler.object.Objects()
        if unedited_segmented is not None:
            objects.unedited_segmented = unedited_segmented
        if small_removed_segmented is not None:
            objects.small_removed_segmented = small_removed_segmented
        objects.segmented = segmented
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.use_outlines.value = False
        module.outlines_name.value = "my_outlines"
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        return workspace, module

    def test_02_01_zeros_propagation(self):
        workspace, module = self.make_workspace(numpy.zeros((10, 10)),
                                                numpy.zeros((10, 10), int))
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 0)
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name in (cellprofiler.measurement.IMAGE, OUTPUT_OBJECTS_NAME, INPUT_OBJECTS_NAME):
            ocolumns = [x for x in columns if x[0] == object_name]
            features = m.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))
        self.assertTrue("my_outlines" not in workspace.get_outline_names())

    def test_02_02_one_object_propagation(self):
        img = numpy.zeros((10, 10))
        img[2:7, 2:7] = .5
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .25
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        expected = numpy.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(numpy.all(objects_out.segmented == expected))
        child_counts = m.get_current_measurement(INPUT_OBJECTS_NAME,
                                                 "Children_%s_Count" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 1)
        parents = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Parent_%s" % INPUT_OBJECTS_NAME)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0], 1)

    def test_02_03_two_objects_propagation_image(self):
        img = numpy.zeros((10, 20))
        img[2:7, 2:7] = .3
        img[2:7, 7:17] = .5
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.regularization_factor.value = 0  # propagate by image
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .2
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        expected[2:7, 7:17] = 2
        mask = numpy.ones((10, 10), bool)
        mask[:, 7:9] = False
        self.assertTrue(numpy.all(objects_out.segmented[:10, :10][mask] == expected[mask]))

    def test_02_04_two_objects_propagation_distance(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 20))
        img[2:7, 2:7] = .3
        img[2:7, 7:17] = .5
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.regularization_factor.value = 1000  # propagate by distance
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .2
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 20), int)
        expected[2:7, 2:10] = 1
        expected[2:7, 10:17] = 2
        mask = numpy.ones((10, 20), bool)
        mask[:, 9:11] = False
        self.assertTrue(numpy.all(objects_out.segmented[mask] == expected[mask]))

    def test_02_05_propagation_wrong_size(self):
        '''Regression test of img-961: different image / object sizes'''
        img = numpy.zeros((10, 20))
        img[2:7, 2:7] = .5
        labels = numpy.zeros((20, 10), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        expected = numpy.zeros((10, 20), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(numpy.all(objects_out.segmented == expected))
        child_counts = m.get_current_measurement(INPUT_OBJECTS_NAME, "Children_%s_Count" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 1)
        parents = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Parent_%s" % INPUT_OBJECTS_NAME)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0], 1)

    def test_03_01_zeros_watershed_gradient(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(numpy.zeros((10, 10)))
        objects = cellprofiler.object.Objects()
        objects.unedited_segmented = numpy.zeros((10, 10), int)
        objects.small_removed_segmented = numpy.zeros((10, 10), int)
        objects.segmented = numpy.zeros((10, 10), int)
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 0)

    def test_03_02_one_object_watershed_gradient(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(numpy.all(objects_out.segmented == expected))
        self.assertTrue("Location_Center_X" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
        self.assertEqual(numpy.product(values.shape), 1)
        self.assertEqual(values[0], 4)
        self.assertTrue("Location_Center_Y" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
        self.assertEqual(numpy.product(values.shape), 1)
        self.assertEqual(values[0], 4)

    def test_03_03_two_objects_watershed_gradient(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 20))
        # There should be a gradient at :,7 which should act
        # as the watershed barrier
        img[2:7, 2:7] = .3
        img[2:7, 7:17] = .5
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .2
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 20), int)
        expected[2:7, 2:7] = 1
        expected[2:7, 7:17] = 2
        mask = numpy.ones((10, 20), bool)
        mask[:, 7:9] = False
        self.assertTrue(numpy.all(objects_out.segmented[mask] == expected[mask]))

    def test_03_04_watershed_gradient_wrong_size(self):
        img = numpy.zeros((20, 10))
        img[2:7, 2:7] = .5
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((20, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(numpy.all(objects_out.segmented == expected))
        self.assertTrue("Location_Center_X" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
        self.assertEqual(numpy.product(values.shape), 1)
        self.assertEqual(values[0], 4)
        self.assertTrue("Location_Center_Y" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
        self.assertEqual(numpy.product(values.shape), 1)
        self.assertEqual(values[0], 4)

    def test_04_01_zeros_watershed_image(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(numpy.zeros((10, 10)))
        objects = cellprofiler.object.Objects()
        objects.unedited_segmented = numpy.zeros((10, 10), int)
        objects.small_removed_segmented = numpy.zeros((10, 10), int)
        objects.segmented = numpy.zeros((10, 10), int)
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 0)

    def test_04_02_one_object_watershed_image(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(numpy.all(objects_out.segmented == expected))

    def test_04_03_two_objects_watershed_image(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 20))
        # There should be a saddle at 7 which should serve
        # as the watershed barrier
        x, y = numpy.mgrid[0:10, 0:20]
        img[2:7, 2:7] = .05 * (7 - y[2:7, 2:7])
        img[2:7, 7:17] = .05 * (y[2:7, 7:17] - 6)
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .01
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 20), int)
        expected[2:7, 2:7] = 1
        expected[2:7, 7:17] = 2
        mask = numpy.ones((10, 20), bool)
        mask[:, 7] = False
        self.assertTrue(numpy.all(objects_out.segmented[mask] == expected[mask]))

    def test_04_04_watershed_image_wrong_size(self):
        img = numpy.zeros((20, 10))
        img[2:7, 2:7] = .5
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((20, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(numpy.all(objects_out.segmented == expected))

    def test_05_01_zeros_distance_n(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(numpy.zeros((10, 10)))
        objects = cellprofiler.object.Objects()
        objects.unedited_segmented = numpy.zeros((10, 10), int)
        objects.small_removed_segmented = numpy.zeros((10, 10), int)
        objects.segmented = numpy.zeros((10, 10), int)
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_N
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 0)

    def test_05_02_one_object_distance_n(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 10))
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_N
        module.distance_to_dilate.value = 1
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        for x in (2, 6):
            for y in (2, 6):
                expected[x, y] = 0
        self.assertTrue(numpy.all(objects_out.segmented == expected))

    def test_05_03_two_objects_distance_n(self):
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 20))
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_N
        module.distance_to_dilate.value = 100
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((10, 20), int)
        expected[:, :10] = 1
        expected[:, 10:] = 2
        self.assertTrue(numpy.all(objects_out.segmented == expected))

    def test_05_04_distance_n_wrong_size(self):
        img = numpy.zeros((20, 10))
        labels = numpy.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_N
        module.distance_to_dilate.value = 1
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = numpy.zeros((20, 10), int)
        expected[2:7, 2:7] = 1
        for x in (2, 6):
            for y in (2, 6):
                expected[x, y] = 0
        self.assertTrue(numpy.all(objects_out.segmented == expected))

    def test_06_01_save_outlines(self):
        '''Test the "save_outlines" feature'''
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.use_outlines.value = True
        module.outlines_name.value = "my_outlines"
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        outlines_out = workspace.image_set.get_image("my_outlines",
                                                     must_be_binary=True).pixel_data
        expected = numpy.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        outlines = expected == 1
        outlines[3:6, 3:6] = False
        self.assertTrue(numpy.all(objects_out.segmented == expected))
        self.assertTrue(numpy.all(outlines == outlines_out))

    def test_06_02_save_primary_outlines(self):
        '''Test saving new primary outlines'''
        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        img = numpy.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cellprofiler.image.Image(img)
        objects = cellprofiler.object.Objects()
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.use_outlines.value = True
        module.outlines_name.value = "my_outlines"
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.wants_primary_outlines.value = True
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME
        module.new_primary_outlines_name.value = "newprimaryoutlines"
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        count_feature = "Count_%s" % OUTPUT_OBJECTS_NAME
        self.assertTrue(count_feature in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", count_feature)
        self.assertEqual(numpy.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        outlines_out = workspace.image_set.get_image("newprimaryoutlines",
                                                     must_be_binary=True)
        expected = numpy.zeros((10, 10), bool)
        expected[3:6, 3:6] = True
        expected[4, 4] = False
        self.assertTrue(numpy.all(outlines_out.pixel_data == expected))

    def test_07_01_measurements_no_new_primary(self):
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        for discard_edge in (True, False):
            module.wants_discard_edge.value = discard_edge
            module.wants_discard_primary.value = False
            module.x_name.value = INPUT_OBJECTS_NAME
            module.y_name.value = OUTPUT_OBJECTS_NAME
            module.new_primary_objects_name.value = NEW_OBJECTS_NAME

            categories = module.get_categories(None, cellprofiler.measurement.IMAGE)
            self.assertEqual(len(categories), 2)
            self.assertTrue(all([any([x == y for x in categories])
                                 for y in ("Count", "Threshold")]))
            categories = module.get_categories(None, OUTPUT_OBJECTS_NAME)
            self.assertEqual(len(categories), 3)
            self.assertTrue(all([any([x == y for x in categories])
                                 for y in ("Location", "Parent", "Number")]))
            categories = module.get_categories(None, INPUT_OBJECTS_NAME)
            self.assertEqual(len(categories), 1)
            self.assertEqual(categories[0], "Children")

            categories = module.get_categories(None, NEW_OBJECTS_NAME)
            self.assertEqual(len(categories), 0)

            features = module.get_measurements(None, cellprofiler.measurement.IMAGE, "Count")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], OUTPUT_OBJECTS_NAME)

            features = module.get_measurements(None, cellprofiler.measurement.IMAGE, "Threshold")
            threshold_features = ("OrigThreshold", "FinalThreshold",
                                  "WeightedVariance", "SumOfEntropies")
            self.assertEqual(len(features), 4)
            self.assertTrue(all([any([x == y for x in features])
                                 for y in threshold_features]))
            for threshold_feature in threshold_features:
                objects = module.get_measurement_objects(None, cellprofiler.measurement.IMAGE,
                                                         "Threshold",
                                                         threshold_feature)
                self.assertEqual(len(objects), 1)
                self.assertEqual(objects[0], OUTPUT_OBJECTS_NAME)

            features = module.get_measurements(None, INPUT_OBJECTS_NAME, "Children")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS_NAME)

            features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Parent")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], INPUT_OBJECTS_NAME)

            features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Location")
            self.assertEqual(len(features), 3)
            self.assertTrue(all([any([x == y for x in features])
                                 for y in ("Center_X", "Center_Y", "Center_Z")]))
            features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Number")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], "Object_Number")

            columns = module.get_measurement_columns(None)
            expected_columns = [(cellprofiler.measurement.IMAGE,
                                 "Threshold_%s_%s" % (f, OUTPUT_OBJECTS_NAME),
                                 cellprofiler.measurement.COLTYPE_FLOAT)
                                for f in threshold_features]
            expected_columns += [(cellprofiler.measurement.IMAGE, "Count_%s" % OUTPUT_OBJECTS_NAME,
                                  cellprofiler.measurement.COLTYPE_INTEGER),
                                 (INPUT_OBJECTS_NAME,
                                  "Children_%s_Count" % OUTPUT_OBJECTS_NAME,
                                  cellprofiler.measurement.COLTYPE_INTEGER),
                                 (OUTPUT_OBJECTS_NAME, "Location_Center_X", cellprofiler.measurement.COLTYPE_FLOAT),
                                 (OUTPUT_OBJECTS_NAME, "Location_Center_Y", cellprofiler.measurement.COLTYPE_FLOAT),
                                 (OUTPUT_OBJECTS_NAME, "Location_Center_Z", cellprofiler.measurement.COLTYPE_FLOAT),
                                 (OUTPUT_OBJECTS_NAME, "Number_Object_Number", cellprofiler.measurement.COLTYPE_INTEGER),
                                 (OUTPUT_OBJECTS_NAME,
                                  "Parent_%s" % INPUT_OBJECTS_NAME,
                                  cellprofiler.measurement.COLTYPE_INTEGER)]
            self.assertEqual(len(columns), len(expected_columns))
            for column in expected_columns:
                self.assertTrue(any([all([fa == fb
                                          for fa, fb
                                          in zip(column, expected_column)])
                                     for expected_column in expected_columns]))

    def test_07_02_measurements_new_primary(self):
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME

        categories = module.get_categories(None, cellprofiler.measurement.IMAGE)
        self.assertEqual(len(categories), 2)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Count", "Threshold")]))
        categories = module.get_categories(None, OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(categories), 3)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Location", "Parent", "Number")]))
        categories = module.get_categories(None, INPUT_OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], "Children")

        categories = module.get_categories(None, NEW_OBJECTS_NAME)
        self.assertEqual(len(categories), 4)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Location", "Parent", "Children", "Number")]))

        features = module.get_measurements(None, cellprofiler.measurement.IMAGE, "Count")
        self.assertEqual(len(features), 2)
        self.assertTrue(OUTPUT_OBJECTS_NAME in features)
        self.assertTrue(NEW_OBJECTS_NAME in features)

        features = module.get_measurements(None, cellprofiler.measurement.IMAGE, "Threshold")
        threshold_features = ("OrigThreshold", "FinalThreshold",
                              "WeightedVariance", "SumOfEntropies")
        self.assertEqual(len(features), 4)
        self.assertTrue(all([any([x == y for x in features])
                             for y in threshold_features]))
        for threshold_feature in threshold_features:
            objects = module.get_measurement_objects(None, cellprofiler.measurement.IMAGE,
                                                     "Threshold",
                                                     threshold_feature)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0], OUTPUT_OBJECTS_NAME)

        features = module.get_measurements(None, INPUT_OBJECTS_NAME, "Children")
        self.assertEqual(len(features), 2)
        self.assertTrue(all([any([x == y for x in features])
                             for y in (cellprofiler.measurement.FF_COUNT % OUTPUT_OBJECTS_NAME,
                                       cellprofiler.measurement.FF_COUNT % NEW_OBJECTS_NAME)]))

        features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Parent")
        self.assertEqual(len(features), 2)
        self.assertTrue(all([any([x == y for x in features])
                             for y in (INPUT_OBJECTS_NAME, NEW_OBJECTS_NAME)]))

        for oname in (OUTPUT_OBJECTS_NAME, NEW_OBJECTS_NAME):
            features = module.get_measurements(None, oname, "Location")
            self.assertEqual(len(features), 3)
            self.assertTrue(all([any([x == y for x in features])
                                 for y in ("Center_X", "Center_Y", "Center_Z")]))

        columns = module.get_measurement_columns(None)
        expected_columns = [(cellprofiler.measurement.IMAGE,
                             "Threshold_%s_%s" % (f, OUTPUT_OBJECTS_NAME),
                             cellprofiler.measurement.COLTYPE_FLOAT)
                            for f in threshold_features]
        for oname in (NEW_OBJECTS_NAME, OUTPUT_OBJECTS_NAME):
            expected_columns += [(cellprofiler.measurement.IMAGE, cellprofiler.measurement.FF_COUNT % oname, cellprofiler.measurement.COLTYPE_INTEGER),
                                 (INPUT_OBJECTS_NAME, "Children_%s_Count" % oname, cellprofiler.measurement.COLTYPE_INTEGER),
                                 (oname, "Location_Center_X", cellprofiler.measurement.COLTYPE_FLOAT),
                                 (oname, "Location_Center_Y", cellprofiler.measurement.COLTYPE_FLOAT),
                                 (oname, "Location_Center_Z", cellprofiler.measurement.COLTYPE_FLOAT),
                                 (oname, "Number_Object_Number", cellprofiler.measurement.COLTYPE_INTEGER),
                                 (oname, "Parent_Primary", cellprofiler.measurement.COLTYPE_INTEGER)]
        expected_columns += [(NEW_OBJECTS_NAME,
                              "Children_%s_Count" % OUTPUT_OBJECTS_NAME,
                              cellprofiler.measurement.COLTYPE_INTEGER),
                             (OUTPUT_OBJECTS_NAME,
                              "Parent_%s" % NEW_OBJECTS_NAME,
                              cellprofiler.measurement.COLTYPE_INTEGER)]
        self.assertEqual(len(columns), len(expected_columns))
        for column in expected_columns:
            self.assertTrue(any([all([fa == fb
                                      for fa, fb
                                      in zip(column, expected_column)])
                                 for expected_column in expected_columns]))

    def test_08_01_filter_edge(self):
        labels = numpy.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]])
        image = numpy.array([[0, 0, .5, 0, 0],
                             [0, .5, .5, .5, 0],
                             [0, .5, .5, .5, 0],
                             [0, .5, .5, .5, 0],
                             [0, 0, 0, 0, 0]])
        expected_unedited = numpy.array([[0, 0, 1, 0, 0],
                                         [0, 1, 1, 1, 0],
                                         [0, 1, 1, 1, 0],
                                         [0, 1, 1, 1, 0],
                                         [0, 0, 0, 0, 0]])

        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(image)
        objects = cellprofiler.object.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        self.assertTrue(numpy.all(object_out.segmented == 0))
        self.assertTrue(numpy.all(object_out.unedited_segmented == expected_unedited))

        object_out = workspace.object_set.get_objects(NEW_OBJECTS_NAME)
        self.assertTrue(numpy.all(object_out.segmented == 0))
        self.assertTrue(numpy.all(object_out.unedited_segmented == labels))

    def test_08_02_filter_unedited(self):
        labels = numpy.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0]])
        labels_unedited = numpy.array([[0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 2, 0, 0],
                                       [0, 0, 0, 0, 0]])
        image = numpy.array([[0, 0, .5, 0, 0],
                             [0, .5, .5, .5, 0],
                             [0, .5, .5, .5, 0],
                             [0, .5, .5, .5, 0],
                             [0, 0, 0, 0, 0]])
        expected = numpy.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0]])
        expected_unedited = numpy.array([[0, 0, 1, 0, 0],
                                         [0, 1, 1, 1, 0],
                                         [0, 2, 2, 2, 0],
                                         [0, 2, 2, 2, 0],
                                         [0, 0, 0, 0, 0]])

        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(image)
        objects = cellprofiler.object.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.unedited_segmented = labels_unedited
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
        module.apply_threshold.global_operation.value = centrosome.threshold.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        self.assertTrue(numpy.all(object_out.segmented == expected))
        self.assertTrue(numpy.all(object_out.unedited_segmented == expected_unedited))

        object_out = workspace.object_set.get_objects(NEW_OBJECTS_NAME)
        self.assertTrue(numpy.all(object_out.segmented == labels))
        self.assertTrue(numpy.all(object_out.unedited_segmented == labels_unedited))

    def test_08_03_small(self):
        '''Regression test of IMG-791

        A small object in the seed mask should not attract any of the
        secondary object.
        '''
        labels = numpy.array([[0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])

        labels_unedited = numpy.array([[0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 2, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0]])

        image = numpy.array([[0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0]], float)
        expected = image.astype(int)

        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(image)
        objects = cellprofiler.object.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.unedited_segmented = labels_unedited
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .5
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        self.assertTrue(numpy.all(object_out.segmented == expected))

    def test_08_04_small_touching(self):
        '''Test of logic added for IMG-791

        A small object in the seed mask touching the edge should attract
        some of the secondary object
        '''
        labels = numpy.array([[0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])

        labels_unedited = numpy.array([[0, 0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 2, 0, 0, 0]])

        image = numpy.array([[0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0]], float)

        p = cellprofiler.pipeline.Pipeline()
        o_s = cellprofiler.object.ObjectSet()
        i_l = cellprofiler.image.ImageSetList()
        image = cellprofiler.image.Image(image)
        objects = cellprofiler.object.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.unedited_segmented = labels_unedited
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cellprofiler.measurement.Measurements()
        module = cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects()
        module.x_name.value = INPUT_OBJECTS_NAME
        module.y_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .5
        module.module_num = 1
        p.add_module(module)
        workspace = cellprofiler.workspace.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        i, j = numpy.argwhere(labels_unedited == 2)[0]
        self.assertTrue(numpy.all(object_out.segmented[i - 1:, j - 1:j + 2] == 0))
        self.assertEqual(len(numpy.unique(object_out.unedited_segmented)), 3)
        self.assertEqual(len(numpy.unique(object_out.unedited_segmented[i - 1:, j - 1:j + 2])), 1)

    def test_10_01_holes_no_holes(self):
        for wants_fill_holes in (True, False):
            for method in (cellprofiler.modules.identifysecondaryobjects.M_DISTANCE_B,
                           cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION,
                           cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_G,
                           cellprofiler.modules.identifysecondaryobjects.M_WATERSHED_I):
                labels = numpy.zeros((20, 10), int)
                labels[5, 5] = 1
                labels[15, 5] = 2
                threshold = numpy.zeros((20, 10), bool)
                threshold[1:7, 4:7] = True
                threshold[2, 5] = False
                threshold[14:17, 4:7] = True
                expected = numpy.zeros((20, 10), int)
                expected[1:7, 4:7] = 1
                expected[14:17, 4:7] = 2
                if not wants_fill_holes:
                    expected[2, 5] = 0
                workspace, module = self.make_workspace(threshold * 0.5, labels)
                self.assertTrue(isinstance(module, cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects))
                module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
                module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
                module.apply_threshold.manual_threshold.value = 0.5
                module.method.value = method
                module.fill_holes.value = wants_fill_holes
                module.distance_to_dilate.value = 10000
                image_set = workspace.image_set
                self.assertTrue(isinstance(image_set, cellprofiler.image.ImageSet))

                module.run(workspace)
                object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
                labels_out = object_out.segmented
                indexes = workspace.measurements.get_current_measurement(
                        OUTPUT_OBJECTS_NAME, "Parent_" + INPUT_OBJECTS_NAME)
                self.assertEqual(len(indexes), 2)
                indexes = numpy.hstack(([0], indexes))
                self.assertTrue(numpy.all(indexes[labels_out] == expected))

    def test_11_00_relationships_zero(self):
        workspace, module = self.make_workspace(
                numpy.zeros((10, 10)), numpy.zeros((10, 10), int))
        self.assertTrue(isinstance(module, cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        result = m.get_relationships(
                module.module_num, cellprofiler.modules.identifysecondaryobjects.R_PARENT,
                module.x_name.value, module.y_name.value)
        self.assertEqual(len(result), 0)

    def test_11_01_relationships_one(self):
        img = numpy.zeros((10, 10))
        img[2:7, 2:7] = .5
        labels = numpy.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
        module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
        module.apply_threshold.manual_threshold.value = .25
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
        result = m.get_relationships(
                module.module_num, cellprofiler.modules.identifysecondaryobjects.R_PARENT,
                module.x_name.value, module.y_name.value)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[cellprofiler.measurement.R_FIRST_IMAGE_NUMBER][0], 1)
        self.assertEqual(result[cellprofiler.measurement.R_SECOND_IMAGE_NUMBER][0], 1)
        self.assertEqual(result[cellprofiler.measurement.R_FIRST_OBJECT_NUMBER][0], 1)
        self.assertEqual(result[cellprofiler.measurement.R_SECOND_OBJECT_NUMBER][0], 1)

    def test_11_02_relationships_missing(self):
        for missing in range(1, 4):
            img = numpy.zeros((10, 30))
            labels = numpy.zeros((10, 30), int)
            for i in range(3):
                object_number = i + 1
                center_j = i * 10 + 4
                labels[3:6, (center_j - 1):(center_j + 2)] = object_number
                if object_number != missing:
                    img[2:7, (center_j - 2):(center_j + 3)] = .5
                else:
                    img[0:7, (center_j - 2):(center_j + 3)] = .5
            workspace, module = self.make_workspace(img, labels)
            self.assertTrue(isinstance(module, cellprofiler.modules.identifysecondaryobjects.IdentifySecondaryObjects))
            module.method.value = cellprofiler.modules.identifysecondaryobjects.M_PROPAGATION
            module.apply_threshold.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
            module.apply_threshold.global_operation.value = cellprofiler.modules.applythreshold.TM_MANUAL
            module.wants_discard_edge.value = True
            module.wants_discard_primary.value = False
            module.apply_threshold.manual_threshold.value = .25
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
            result = m.get_relationships(
                    module.module_num, cellprofiler.modules.identifysecondaryobjects.R_PARENT,
                    module.x_name.value, module.y_name.value)
            self.assertEqual(len(result), 2)
            for i in range(2):
                object_number = i + 1
                if object_number >= missing:
                    object_number += 1
                self.assertEqual(result[cellprofiler.measurement.R_FIRST_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cellprofiler.measurement.R_SECOND_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cellprofiler.measurement.R_FIRST_OBJECT_NUMBER][i],
                                 object_number)
                self.assertEqual(result[cellprofiler.measurement.R_SECOND_OBJECT_NUMBER][i],
                                 object_number)
