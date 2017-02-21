import StringIO
import base64
import unittest
import zlib

import centrosome.filter
import centrosome.otsu
import centrosome.threshold
import numpy

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.applythreshold
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()


INPUT_IMAGE_NAME = 'inputimage'
OUTPUT_IMAGE_NAME = 'outputimage'


class TestApplyThreshold(unittest.TestCase):
    def make_workspace(self, image, mask=None):
        '''Make a workspace for testing ApplyThreshold'''
        module = cellprofiler.modules.applythreshold.ApplyThreshold()
        module.x_name.value = INPUT_IMAGE_NAME
        module.y_name.value = OUTPUT_IMAGE_NAME
        pipeline = cellprofiler.pipeline.Pipeline()
        object_set = cellprofiler.object.ObjectSet()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        image_set.add(INPUT_IMAGE_NAME,
                      cellprofiler.image.Image(image) if mask is None
                      else cellprofiler.image.Image(image, mask))
        return workspace, module

    def test_01_00_write_a_test_for_the_new_variable_revision_please(self):
        self.assertEqual(cellprofiler.modules.applythreshold.ApplyThreshold.variable_revision_number, 9)

    def test_01_07_load_v7(self):
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
            Extract metadata?:Yes
            Extraction method count:1
            Extraction method:Manual
            Source:From file name
            Regular expression:Channel(?P<Wavelength>\x5B12\x5D)-\x5B0-9\x5D{2}-(?P<WellRow>\x5BA-Z\x5D)-(?P<WellColumn>\x5B0-9\x5D{2}).tif
            Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
            Filter images:All images
            :or (file does contain "")
            Metadata file location\x3A:
            Match file and image metadata:\x5B\x5D
            Case insensitive matching:No

        NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
            Assignment method:Assign images matching rules
            Load as:Grayscale image
            Image name:DNA
            :\x5B\x5D
            Assign channels by:Order
            Assignments count:2
            Match this rule:or (metadata does Wavelength "1")
            Image name:GFP
            Objects name:Cell
            Load as:Grayscale image
            Match this rule:or (metadata does Wavelength "2")
            Image name:DNA
            Objects name:Nucleus
            Load as:Grayscale image

        Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
            Do you want to group your images?:No
            grouping metadata count:1
            Metadata category:None

        ApplyThreshold:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
            Select the input image:RainbowPony
            Name the output image:GrayscalePony
            Select the output image type:Grayscale
            Set pixels below or above the threshold to zero?:Below threshold
            Subtract the threshold value from the remaining pixel intensities?:Yes
            Number of pixels by which to expand the thresholding around those excluded bright pixels:2.0
            Threshold setting version:1
            Threshold strategy:Adaptive
            Threshold method:MCT
            Smoothing for threshold:Automatic
            Threshold smoothing scale:1.5
            Threshold correction factor:1.1
            Lower and upper bounds on threshold:0.07,0.99
            Approximate fraction of image covered by objects?:0.02
            Manual threshold:0.1
            Select the measurement to threshold with:Pony_Perimeter
            Select binary image:Pony_yes_or_no
            Masking objects:PonyMask
            Two-class or three-class thresholding?:Two classes
            Minimize the weighted variance or the entropy?:Weighted variance
            Assign pixels in the middle intensity class to the foreground or the background?:Foreground
            Method to calculate adaptive window size:Image size
            Size of adaptive window:13
"""
        fd = StringIO.StringIO(data)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.loadtxt(fd)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.applythreshold.ApplyThreshold))
        self.assertEqual(module.x_name.value, "RainbowPony")
        self.assertEqual(module.y_name.value, "GrayscalePony")
        self.assertEqual(module.threshold_scope.value, cellprofiler.modules.applythreshold.TS_ADAPTIVE)
        self.assertEqual(module.threshold_method.value, centrosome.threshold.TM_MCT)
        self.assertEqual(module.threshold_smoothing_scale.value, 1.3488)
        self.assertEqual(module.threshold_correction_factor.value, 1.1)
        self.assertEqual(module.threshold_range.min, .07)
        self.assertEqual(module.threshold_range.max, .99)
        self.assertEqual(module.manual_threshold.value, 0.1)
        self.assertEqual(module.thresholding_measurement.value, "Pony_Perimeter")
        self.assertEqual(module.two_class_otsu.value, cellprofiler.modules.applythreshold.O_TWO_CLASS)
        self.assertEqual(module.assign_middle_to_foreground.value, cellprofiler.modules.applythreshold.O_FOREGROUND)
        self.assertEqual(module.adaptive_window_size.value, 13)

    def test_01_08_load_v8(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:300
GitHash:
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'To begin creating your project, use the Images module to compile a list of files and/or folders that you want to analyze. You can also specify a set of rules to include only the desired files in your selected folders.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "\x5B\\\\\\\\\\\\\\\\/\x5D\\\\\\\\.")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\'The Metadata module optionally allows you to extract information describing your images (i.e, metadata) which will be stored along with your measurements. This information can be contained in the file name and/or location, or in an external file.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:
    Match file and image metadata:\x5B\x5D
    Use case insensitive matching?:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:7|show_window:False|notes:\x5B\'The NamesAndTypes module allows you to assign a meaningful name to each image by which other modules will refer to it.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:All images
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:\x5B\x5D
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:1
    Single images count:0
    Maximum intensity:255.0
    Volumetric:No
    x:1.0
    y:1.0
    z:1.0
    Select the rule criteria:and (file does contain "")
    Name to assign these images:DNA
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedOutlines
    Maximum intensity:255.0

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'The Groups module optionally allows you to split your list of images into image subsets (groups) which will be processed independently of each other. Examples of groupings include screening batches, microtiter plates, time-lapse movies, etc.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

ApplyThreshold:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:8|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:DNA
    Name the output image:ThreshBlue
    Threshold setting version:3
    Threshold strategy:Global
    Thresholding method:MCT
    Threshold smoothing scale:0
    Threshold correction factor:1.0
    Lower and upper bounds on threshold:0.0,1.0
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Two-class or three-class thresholding?:Two classes
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Size of adaptive window:50
    Use default parameters?:Default
    Lower outlier fraction:0.05
    Upper outlier fraction:0.05
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2.0
        """

        fd = StringIO.StringIO(data)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.loadtxt(fd)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.applythreshold.ApplyThreshold))
        self.assertEqual(module.x_name, "DNA")
        self.assertEqual(module.y_name, "ThreshBlue")
        self.assertEqual(module.threshold_scope, cellprofiler.modules.applythreshold.TS_GLOBAL)
        self.assertEqual(module.threshold_method, centrosome.threshold.TM_MCT)
        self.assertEqual(module.threshold_smoothing_scale, 0)
        self.assertEqual(module.threshold_correction_factor, 1.0)
        self.assertEqual(module.threshold_range.min, 0.0)
        self.assertEqual(module.threshold_range.max, 1.0)
        self.assertEqual(module.manual_threshold, 0.0)
        self.assertEqual(module.thresholding_measurement, "None")
        self.assertEqual(module.two_class_otsu, cellprofiler.modules.applythreshold.O_TWO_CLASS)
        self.assertEqual(module.assign_middle_to_foreground, cellprofiler.modules.applythreshold.O_FOREGROUND)
        self.assertEqual(module.adaptive_window_size, 50)

    def test_01_09_load_v9(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:300
GitHash:
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'To begin creating your project, use the Images module to compile a list of files and/or folders that you want to analyze. You can also specify a set of rules to include only the desired files in your selected folders.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "\x5B\\\\\\\\\\\\\\\\/\x5D\\\\\\\\.")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\'The Metadata module optionally allows you to extract information describing your images (i.e, metadata) which will be stored along with your measurements. This information can be contained in the file name and/or location, or in an external file.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:
    Match file and image metadata:\x5B\x5D
    Use case insensitive matching?:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:7|show_window:False|notes:\x5B\'The NamesAndTypes module allows you to assign a meaningful name to each image by which other modules will refer to it.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:All images
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:\x5B\x5D
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:1
    Single images count:0
    Maximum intensity:255.0
    Volumetric:No
    x:1.0
    y:1.0
    z:1.0
    Select the rule criteria:and (file does contain "")
    Name to assign these images:DNA
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedOutlines
    Maximum intensity:255.0

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'The Groups module optionally allows you to split your list of images into image subsets (groups) which will be processed independently of each other. Examples of groupings include screening batches, microtiter plates, time-lapse movies, etc.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

ApplyThreshold:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:9|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:DNA
    Name the output image:ApplyThreshold
    Threshold strategy:Global
    Thresholding method:MCT
    Threshold smoothing scale:0.0
    Threshold correction factor:1.0
    Lower and upper bounds on threshold:0.0,1.0
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Two-class or three-class thresholding?:Two classes
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Size of adaptive window:50
    Lower outlier fraction:0.05
    Upper outlier fraction:0.05
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2.0
        """

        fd = StringIO.StringIO(data)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.loadtxt(fd)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.applythreshold.ApplyThreshold))
        self.assertEqual(module.x_name, "DNA")
        self.assertEqual(module.y_name, "ApplyThreshold")
        self.assertEqual(module.threshold_scope, cellprofiler.modules.applythreshold.TS_GLOBAL)
        self.assertEqual(module.threshold_method, centrosome.threshold.TM_MCT)
        self.assertEqual(module.threshold_smoothing_scale, 0)
        self.assertEqual(module.threshold_correction_factor, 1.0)
        self.assertEqual(module.threshold_range.min, 0.0)
        self.assertEqual(module.threshold_range.max, 1.0)
        self.assertEqual(module.manual_threshold, 0.0)
        self.assertEqual(module.thresholding_measurement, "None")
        self.assertEqual(module.two_class_otsu, cellprofiler.modules.applythreshold.O_TWO_CLASS)
        self.assertEqual(module.assign_middle_to_foreground, cellprofiler.modules.applythreshold.O_FOREGROUND)
        self.assertEqual(module.adaptive_window_size, 50)

    def test_04_01_binary_manual(self):
        '''Test a binary threshold with manual threshold value'''
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20))
        expected = image > .5
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = centrosome.threshold.TM_MANUAL
        module.manual_threshold.value = .5
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_04_02_binary_global(self):
        '''Test a binary threshold with Otsu global method'''
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20))
        threshold = centrosome.threshold.get_otsu_threshold(image)
        expected = image > threshold
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.threshold_method.value = centrosome.threshold.TM_OTSU
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_04_03_binary_correction(self):
        '''Test a binary threshold with a correction factor'''
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20))
        threshold = centrosome.threshold.get_otsu_threshold(image) * .5
        expected = image > threshold
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.threshold_method.value = centrosome.threshold.TM_OTSU
        module.threshold_correction_factor.value = .5
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_04_04_low_bounds(self):
        '''Test a binary threshold with a low bound'''

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20))
        image[(image > .4) & (image < .6)] = .5
        expected = image > .7
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.threshold_method.value = centrosome.threshold.TM_OTSU
        module.threshold_range.min = .7
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_04_05_high_bounds(self):
        '''Test a binary threshold with a high bound'''

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(40, 40))
        expected = image > .1
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.threshold_method.value = centrosome.threshold.TM_OTSU
        module.threshold_range.max = .1
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_04_07_threshold_from_measurement(self):
        '''Test a binary threshold from previous measurements'''
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20))
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = centrosome.threshold.TM_MANUAL
        module.manual_threshold.value = .5
        module.run(workspace)

        module2 = cellprofiler.modules.applythreshold.ApplyThreshold()
        module2.x_name.value = OUTPUT_IMAGE_NAME
        module2.y_name.value = OUTPUT_IMAGE_NAME + 'new'
        module2.threshold_scope.value = centrosome.threshold.TM_MEASUREMENT
        module2.thresholding_measurement.value = 'Threshold_FinalThreshold_' + OUTPUT_IMAGE_NAME
        module2.run(workspace)

    def test_05_01_otsu_wv(self):
        '''Test the weighted variance version of Otsu'''
        numpy.random.seed(0)
        image = numpy.hstack((numpy.random.exponential(1.5, size=600),
                              numpy.random.poisson(15, size=300)))
        image.shape = (30, 30)
        image = centrosome.filter.stretch(image)
        limage, d = centrosome.threshold.log_transform(image)
        threshold = centrosome.otsu.otsu(limage)
        threshold = centrosome.threshold.inverse_log_transform(threshold, d)
        expected = image > threshold
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.threshold_method.value = centrosome.threshold.TM_OTSU
        module.two_class_otsu.value = cellprofiler.modules.applythreshold.O_TWO_CLASS
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_05_03_otsu3_wv_low(self):
        '''Test the three-class otsu, weighted variance middle = background'''
        numpy.random.seed(0)
        image = numpy.hstack((numpy.random.exponential(1.5, size=300),
                              numpy.random.poisson(15, size=300),
                              numpy.random.poisson(30, size=300))).astype(numpy.float32)
        image.shape = (30, 30)
        image = centrosome.filter.stretch(image)
        limage, d = centrosome.threshold.log_transform(image)
        t1, t2 = centrosome.otsu.otsu3(limage)
        threshold = centrosome.threshold.inverse_log_transform(t2, d)
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.threshold_method.value = centrosome.threshold.TM_OTSU
        module.two_class_otsu.value = cellprofiler.modules.applythreshold.O_THREE_CLASS
        module.assign_middle_to_foreground.value = cellprofiler.modules.applythreshold.O_BACKGROUND
        module.run(workspace)
        m = workspace.measurements
        m_threshold = m[cellprofiler.measurement.IMAGE, cellprofiler.modules.applythreshold.FF_ORIG_THRESHOLD % module.get_measurement_objects_name()]
        self.assertAlmostEqual(m_threshold, threshold)

    def test_05_04_otsu3_wv_high(self):
        '''Test the three-class otsu, weighted variance middle = foreground'''
        numpy.random.seed(0)
        image = numpy.hstack((numpy.random.exponential(1.5, size=300),
                              numpy.random.poisson(15, size=300),
                              numpy.random.poisson(30, size=300)))
        image.shape = (30, 30)
        image = centrosome.filter.stretch(image)
        limage, d = centrosome.threshold.log_transform(image)
        t1, t2 = centrosome.otsu.otsu3(limage)
        threshold = centrosome.threshold.inverse_log_transform(t1, d)
        workspace, module = self.make_workspace(image)
        module.threshold_scope.value = cellprofiler.modules.applythreshold.TS_GLOBAL
        module.threshold_method.value = centrosome.threshold.TM_OTSU
        module.two_class_otsu.value = cellprofiler.modules.applythreshold.O_THREE_CLASS
        module.assign_middle_to_foreground.value = cellprofiler.modules.applythreshold.O_FOREGROUND
        module.run(workspace)
        m = workspace.measurements
        m_threshold = m[cellprofiler.measurement.IMAGE, cellprofiler.modules.applythreshold.FF_ORIG_THRESHOLD % module.get_measurement_objects_name()]
        self.assertAlmostEqual(m_threshold, threshold)

    def test_06_01_adaptive_otsu_small(self):
        """Test the function, get_threshold, using Otsu adaptive / small

        Use a small image (125 x 125) to break the image into four
        pieces, check that the threshold is different in each block
        and that there are four blocks broken at the 75 boundary
        """
        numpy.random.seed(0)
        image = numpy.zeros((120, 110))
        for i0, i1 in ((0, 60), (60, 120)):
            for j0, j1 in ((0, 55), (55, 110)):
                dmin = float(i0 * 2 + j0) / 500.0
                dmult = 1.0 - dmin
                # use the sine here to get a bimodal distribution of values
                r = numpy.random.uniform(0, numpy.pi * 2, (60, 55))
                rsin = (numpy.sin(r) + 1) / 2
                image[i0:i1, j0:j1] = dmin + rsin * dmult
        workspace, x = self.make_workspace(image)
        x.threshold_scope.value = centrosome.threshold.TM_ADAPTIVE
        x.threshold_method.value = centrosome.threshold.TM_OTSU
        threshold, global_threshold = x.get_threshold(
            image,
            numpy.ones_like(image, bool),
            workspace
        )
        self.assertTrue(threshold[0, 0] != threshold[0, 109])
        self.assertTrue(threshold[0, 0] != threshold[119, 0])
        self.assertTrue(threshold[0, 0] != threshold[119, 109])

    def test_06_01_adaptive_otsu_small(self):
        """Test the function, get_threshold, using Otsu adaptive / small

        Use a small image (125 x 125) to break the image into four
        pieces, check that the threshold is different in each block
        and that there are four blocks broken at the 75 boundary
        """
        numpy.random.seed(0)
        image = numpy.zeros((120, 110))
        for i0, i1 in ((0, 60), (60, 120)):
            for j0, j1 in ((0, 55), (55, 110)):
                dmin = float(i0 * 2 + j0) / 500.0
                dmult = 1.0 - dmin
                # use the sine here to get a bimodal distribution of values
                r = numpy.random.uniform(0, numpy.pi * 2, (60, 55))
                rsin = (numpy.sin(r) + 1) / 2
                image[i0:i1, j0:j1] = dmin + rsin * dmult
        workspace, x = self.make_workspace(image)
        x.threshold_scope.value = centrosome.threshold.TM_ADAPTIVE
        x.threshold_method.value = centrosome.threshold.TM_OTSU
        threshold, global_threshold = x.get_threshold(
            image,
            numpy.ones_like(image, bool),
            workspace
        )
        self.assertTrue(threshold[0, 0] != threshold[0, 109])
        self.assertTrue(threshold[0, 0] != threshold[119, 0])
        self.assertTrue(threshold[0, 0] != threshold[119, 109])

    def test_07_01_small_images(self):
        """Test mixture of gaussians thresholding with few pixels

        Run MOG to see if it blows up, given 0-10 pixels"""
        r = numpy.random.RandomState()
        r.seed(91)
        image = r.uniform(size=(9, 11))
        ii, jj = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]
        ii, jj = ii.flatten(), jj.flatten()

        for threshold_method in (centrosome.threshold.TM_MCT,
                                 centrosome.threshold.TM_OTSU,
                                 centrosome.threshold.TM_ROBUST_BACKGROUND):
            for i in range(11):
                mask = numpy.zeros(image.shape, bool)
                if i:
                    p = r.permutation(numpy.prod(image.shape))[:i]
                    mask[ii[p], jj[p]] = True
                workspace, x = self.make_workspace(image, mask)
                x.threshold_method.value = threshold_method
                x.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
                l, g = x.get_threshold(image, mask, workspace)
                v = image[mask]
                image = r.uniform(size=(9, 11))
                image[mask] = v
                l1, g1 = x.get_threshold(image, mask, workspace)
                self.assertAlmostEqual(l1, l)

    def test_08_01_test_manual_background(self):
        """Test manual background"""
        workspace, x = self.make_workspace(numpy.zeros((10, 10)))
        x = cellprofiler.modules.applythreshold.ApplyThreshold()
        x.threshold_scope.value = centrosome.threshold.TM_MANUAL
        x.manual_threshold.value = .5
        local_threshold, threshold = x.get_threshold(
            numpy.zeros((10, 10)),
            numpy.ones((10, 10), bool),
            workspace
        )
        self.assertTrue(threshold == .5)
        self.assertTrue(threshold == .5)