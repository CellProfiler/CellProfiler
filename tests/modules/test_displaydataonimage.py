''' test_displaydataonimage - test the DisplayDataOnImage module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.grid as cpg
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.displaydataonimage as D
from centrosome.cpmorphology import centers_of_labels

INPUT_IMAGE_NAME = 'inputimage'
OUTPUT_IMAGE_NAME = 'outputimage'
OBJECTS_NAME = 'objects'
MEASUREMENT_NAME = 'measurement'


class TestDisplayDataOnImage(unittest.TestCase):
    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130719180707
ModuleCount:1
HasImagePlaneDetails:False

DisplayDataOnImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Display object or image measurements?:Object
    Select the input objects:Nuclei
    Measurement to display:AreaShape_Zernike_0_0
    Select the image on which to display the measurements:DNA
    Text color:green
    Name the output image that has the measurements displayed:Zernike
    Font size (points):10
    Number of decimals:2
    Image elements to save:Axes
    Annotation offset (in pixels):5
    Display mode:Color
    Color map:jet
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, D.DisplayDataOnImage))
        self.assertEqual(module.objects_or_image, D.OI_OBJECTS)
        self.assertEqual(module.measurement, "AreaShape_Zernike_0_0")
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.text_color, "green")
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.display_image, "Zernike")
        self.assertEqual(module.font_size, 10)
        self.assertEqual(module.decimals, 2)
        self.assertEqual(module.saved_image_contents, D.E_AXES)
        self.assertEqual(module.offset, 5)
        self.assertEqual(module.color_or_text, D.CT_COLOR)
        self.assertEqual(module.colormap, "jet")
        self.assertTrue(module.wants_image)

    def test_01_04_load_v5(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130719180707
ModuleCount:1
HasImagePlaneDetails:False

DisplayDataOnImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:5|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Display object or image measurements?:Object
    Select the input objects:Nuclei
    Measurement to display:AreaShape_Zernike_0_0
    Select the image on which to display the measurements:DNA
    Text color:green
    Name the output image that has the measurements displayed:Zernike
    Font size (points):10
    Number of decimals:2
    Image elements to save:Axes
    Annotation offset (in pixels):5
    Display mode:Color
    Color map:jet
    Display background image:No
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, D.DisplayDataOnImage))
        self.assertEqual(module.objects_or_image, D.OI_OBJECTS)
        self.assertEqual(module.measurement, "AreaShape_Zernike_0_0")
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.text_color, "green")
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(module.display_image, "Zernike")
        self.assertEqual(module.font_size, 10)
        self.assertEqual(module.decimals, 2)
        self.assertEqual(module.saved_image_contents, D.E_AXES)
        self.assertEqual(module.offset, 5)
        self.assertEqual(module.color_or_text, D.CT_COLOR)
        self.assertEqual(module.colormap, "jet")
        self.assertFalse(module.wants_image)
        self.assertEqual(module.color_map_scale_choice,
                         D.CMS_USE_MEASUREMENT_RANGE)
        self.assertEqual(module.color_map_scale.min, 0)
        self.assertEqual(module.color_map_scale.max, 1)

    def test_01_06_load_v6(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20141125133416
GitHash:389a5b5
ModuleCount:2
HasImagePlaneDetails:False

DisplayDataOnImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Display object or image measurements?:Object
    Select the input objects:Nuclei
    Measurement to display:Texture_AngularSecondMoment_CropBlue_3_0
    Select the image on which to display the measurements:RGBImage
    Text color:red
    Name the output image that has the measurements displayed:Whatever
    Font size (points):11
    Number of decimals:3
    Image elements to save:Image
    Annotation offset (in pixels):1
    Display mode:Color
    Color map:jet
    Display background image?:Yes
    Color map scale:Manual
    Color map range:0.05,1.5

DisplayDataOnImage:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:6|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Display object or image measurements?:Object
    Select the input objects:Nuclei
    Measurement to display:Texture_AngularSecondMoment_CropBlue_3_0
    Select the image on which to display the measurements:RGBImage
    Text color:red
    Name the output image that has the measurements displayed:DisplayImage
    Font size (points):12
    Number of decimals:4
    Image elements to save:Image
    Annotation offset (in pixels):1
    Display mode:Color
    Color map:Default
    Display background image?:Yes
    Color map scale:Use this image\'s measurement range
    Color map range:0.05,1.5
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, D.DisplayDataOnImage))
        self.assertEqual(module.objects_or_image, D.OI_OBJECTS)
        self.assertEqual(module.objects_name, "Nuclei")
        self.assertEqual(
                module.measurement, "Texture_AngularSecondMoment_CropBlue_3_0")
        self.assertEqual(module.image_name, "RGBImage")
        self.assertEqual(module.display_image, "Whatever")
        self.assertEqual(module.font_size, 11)
        self.assertEqual(module.decimals, 3)
        self.assertEqual(module.saved_image_contents, D.E_IMAGE)
        self.assertEqual(module.offset, 1)
        self.assertEqual(module.color_or_text, D.CT_COLOR)
        self.assertEqual(module.colormap, "jet")
        self.assertTrue(module.wants_image)
        self.assertEqual(module.color_map_scale_choice,
                         D.CMS_MANUAL)
        self.assertEqual(module.color_map_scale.min, 0.05)
        self.assertEqual(module.color_map_scale.max, 1.5)
        module = pipeline.modules()[1]
        self.assertEqual(module.color_map_scale_choice,
                         D.CMS_USE_MEASUREMENT_RANGE)

    def make_workspace(self, measurement, labels=None, image=None):
        object_set = cpo.ObjectSet()
        module = D.DisplayDataOnImage()
        module.module_num = 1
        module.image_name.value = INPUT_IMAGE_NAME
        module.display_image.value = OUTPUT_IMAGE_NAME
        module.objects_name.value = OBJECTS_NAME
        m = cpmeas.Measurements()

        if labels is None:
            module.objects_or_image.value = D.OI_IMAGE
            m.add_image_measurement(MEASUREMENT_NAME, measurement)
            if image is None:
                image = np.zeros((50, 120))
        else:
            module.objects_or_image.value = D.OI_OBJECTS
            o = cpo.Objects()
            o.segmented = labels
            object_set.add_objects(o, OBJECTS_NAME)
            m.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME, np.array(measurement))
            y, x = centers_of_labels(labels)
            m.add_measurement(OBJECTS_NAME, "Location_Center_X", x)
            m.add_measurement(OBJECTS_NAME, "Location_Center_Y", y)
            if image is None:
                image = np.zeros(labels.shape)
        module.measurement.value = MEASUREMENT_NAME

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(INPUT_IMAGE_NAME, cpi.Image(image))

        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  m, image_set_list)
        return workspace, module

    def test_02_01_display_image(self):
        for display in (D.E_AXES, D.E_FIGURE, D.E_IMAGE):
            workspace, module = self.make_workspace(0)
            module.saved_image_contents.value = display
            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_02_display_objects(self):
        labels = np.zeros((50, 120), int)
        labels[10:20, 20:27] = 1
        labels[30:35, 35:50] = 2
        labels[5:18, 44:100] = 3
        for display in (D.E_AXES, D.E_FIGURE, D.E_IMAGE):
            workspace, module = self.make_workspace([0, 1, 2], labels)
            module.saved_image_contents.value = display
            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_03_display_no_objects(self):
        workspace, module = self.make_workspace([], np.zeros((50, 120)))
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_04_display_nan_objects(self):
        labels = np.zeros((50, 120), int)
        labels[10:20, 20:27] = 1
        labels[30:35, 35:50] = 2
        labels[5:18, 44:100] = 3
        for measurements in (np.array([1.0, np.nan, 5.0]), np.array([np.nan] * 3)):
            workspace, module = self.make_workspace(measurements, labels)
            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_05_display_objects_wrong_size(self):
        labels = np.zeros((50, 120), int)
        labels[10:20, 20:27] = 1
        labels[30:35, 35:50] = 2
        labels[5:18, 44:100] = 3
        input_image = np.random.uniform(size=(60, 110))
        for display in (D.E_AXES, D.E_FIGURE, D.E_IMAGE):
            workspace, module = self.make_workspace([0, 1, 2], labels, input_image)
            module.saved_image_contents.value = display
            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_06_display_colors(self):
        labels = np.zeros((50, 120), int)
        labels[10:20, 20:27] = 1
        labels[30:35, 35:50] = 2
        labels[5:18, 44:100] = 3
        workspace, module = self.make_workspace([1.1, 2.2, 3.3], labels)
        assert isinstance(module, D.DisplayDataOnImage)
        module.color_or_text.value = D.CT_COLOR
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_07_display_colors_missing_measurement(self):
        #
        # Regression test of issue 1084
        #
        labels = np.zeros((50, 120), int)
        labels[10:20, 20:27] = 1
        labels[30:35, 35:50] = 2
        labels[5:18, 44:100] = 3
        workspace, module = self.make_workspace([1.1, 2.2], labels)
        assert isinstance(module, D.DisplayDataOnImage)
        module.color_or_text.value = D.CT_COLOR
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_08_display_colors_nan_measurement(self):
        #
        # Regression test of issue 1084
        #
        labels = np.zeros((50, 120), int)
        labels[10:20, 20:27] = 1
        labels[30:35, 35:50] = 2
        labels[5:18, 44:100] = 3
        workspace, module = self.make_workspace([1.1, np.nan, 2.2], labels)
        assert isinstance(module, D.DisplayDataOnImage)
        module.color_or_text.value = D.CT_COLOR
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    def test_02_09_display_colors_manual(self):
        #
        # Just run the code path for manual color map scale
        #
        labels = np.zeros((50, 120), int)
        labels[10:20, 20:27] = 1
        labels[30:35, 35:50] = 2
        labels[5:18, 44:100] = 3
        workspace, module = self.make_workspace([1.1, 2.2, 3.3], labels)
        assert isinstance(module, D.DisplayDataOnImage)
        module.color_or_text.value = D.CT_COLOR
        module.color_map_scale_choice.value = D.CMS_MANUAL
        module.color_map_scale.min = 2.0
        module.color_map_scale.max = 3.0
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
