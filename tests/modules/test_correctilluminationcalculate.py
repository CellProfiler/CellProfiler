import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.correctilluminationcalculate
import cellprofiler.modules.injectimage
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.setting
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

INPUT_IMAGE_NAME = "MyImage"
OUTPUT_IMAGE_NAME = "MyResult"
AVERAGE_IMAGE_NAME = "Ave"
DILATED_IMAGE_NAME = "Dilate"


class TestCorrectImage_Calculate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Backwards compatibility for Python 2.6 unittest"""
        if not hasattr(cls, "assertIn"):
            cls.assertIn = lambda self, x, y: self.assertTrue(x in y)
        if not hasattr(cls, "assertNotIn"):
            cls.assertNotIn = lambda self, x, y: self.assertFalse(x in y)

    def error_callback(self, calller, event):
        if isinstance(event, cellprofiler.pipeline.RunExceptionEvent):
            self.fail(event.error.message)

    def make_workspaces(self, images_and_masks):
        """Make a workspace for each image set provided

        images_and_masks - a collection of two-tuples: image+mask

        returns a list of workspaces + the module
        """
        image_set_list = cellprofiler.image.ImageSetList()
        workspaces = []
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 1
        module.image_name.value = INPUT_IMAGE_NAME
        module.illumination_image_name.value = OUTPUT_IMAGE_NAME
        module.average_image_name.value = AVERAGE_IMAGE_NAME
        module.dilated_image_name.value = DILATED_IMAGE_NAME
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        measurements = cellprofiler.measurement.Measurements()

        for i, (image, mask) in enumerate(images_and_masks):
            image_set = image_set_list.get_image_set(i)
            if mask is None:
                image = cellprofiler.image.Image(image)
            else:
                image = cellprofiler.image.Image(image, mask)
            image_set.add(INPUT_IMAGE_NAME, image)
            workspace = cellprofiler.workspace.Workspace(
                    pipeline, module, image_set, cellprofiler.region.Set(),
                    measurements, image_set_list)
            workspaces.append(workspace)
        return workspaces, module

    def test_00_00_zeros(self):
        """Test all combinations of options with an image of all zeros"""
        for image in (numpy.zeros((10, 10)), numpy.zeros((10, 10, 3))):
            pipeline = cellprofiler.pipeline.Pipeline()
            pipeline.add_listener(self.error_callback)
            inj_module = cellprofiler.modules.injectimage.InjectImage("MyImage", image)
            inj_module.module_num = 1
            pipeline.add_module(inj_module)
            module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
            module.module_num = 2
            pipeline.add_module(module)
            module.image_name.value = "MyImage"
            module.illumination_image_name.value = "OutputImage"
            module.save_average_image.value = True
            module.save_dilated_image.value = True

            for ea in (cellprofiler.modules.correctilluminationcalculate.EA_EACH, cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS, cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST):
                module.each_or_all.value = ea
                for intensity_choice in (cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND, cellprofiler.modules.correctilluminationcalculate.IC_REGULAR):
                    module.intensity_choice.value = intensity_choice
                    for dilate_objects in (True, False):
                        module.dilate_objects.value = dilate_objects
                        for rescale_option in (cellprofiler.setting.YES, cellprofiler.setting.NO, cellprofiler.modules.correctilluminationcalculate.RE_MEDIAN):
                            module.rescale_option.value = rescale_option
                            for smoothing_method \
                                    in (cellprofiler.modules.correctilluminationcalculate.SM_NONE, cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL,
                                        cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER, cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER,
                                        cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE, cellprofiler.modules.correctilluminationcalculate.SM_SPLINES,
                                        cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL):
                                module.smoothing_method.value = smoothing_method
                                for ow in (cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC, cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY,
                                           cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE):
                                    module.automatic_object_width.value = ow
                                    measurements = cellprofiler.measurement.Measurements()
                                    image_set_list = cellprofiler.image.ImageSetList()
                                    workspace = cellprofiler.workspace.Workspace(
                                            pipeline, None, None, None,
                                            measurements, image_set_list)
                                    pipeline.prepare_run(workspace)
                                    inj_module.prepare_group(workspace, {}, [1])
                                    module.prepare_group(workspace, {}, [1])
                                    image_set = image_set_list.get_image_set(0)
                                    object_set = cellprofiler.region.Set()
                                    workspace = cellprofiler.workspace.Workspace(pipeline,
                                                                                 inj_module,
                                                                                 image_set,
                                                                                 object_set,
                                                                                 measurements,
                                                                                 image_set_list)
                                    inj_module.run(workspace)
                                    module.run(workspace)
                                    image = image_set.get_image("OutputImage")
                                    self.assertTrue(image is not None)
                                    self.assertTrue(numpy.all(image.pixel_data == 0),
                                                    """Failure case:
                intensity_choice = %(intensity_choice)s
                dilate_objects = %(dilate_objects)s
                rescale_option = %(rescale_option)s
                smoothing_method = %(smoothing_method)s
                automatic_object_width = %(ow)s""" % locals())

    def test_01_01_ones_image(self):
        """The illumination correction of an image of all ones should be uniform

        """
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        for image in (numpy.ones((10, 10)), numpy.ones((10, 10, 3))):
            inj_module = cellprofiler.modules.injectimage.InjectImage("MyImage", image)
            inj_module.module_num = 1
            pipeline.add_module(inj_module)
            module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
            module.module_num = 2
            pipeline.add_module(module)
            module.image_name.value = "MyImage"
            module.illumination_image_name.value = "OutputImage"
            module.rescale_option.value = cellprofiler.setting.YES

            for ea in (cellprofiler.modules.correctilluminationcalculate.EA_EACH, cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS, cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST):
                module.each_or_all.value = ea
                for intensity_choice in (cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND, cellprofiler.modules.correctilluminationcalculate.IC_REGULAR):
                    module.intensity_choice.value = intensity_choice
                    for dilate_objects in (True, False):
                        module.dilate_objects.value = dilate_objects
                        for smoothing_method \
                                in (cellprofiler.modules.correctilluminationcalculate.SM_NONE, cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL,
                                    cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER, cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER,
                                    cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE, cellprofiler.modules.correctilluminationcalculate.SM_SPLINES,
                                    cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL):
                            module.smoothing_method.value = smoothing_method
                            for ow in (cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC, cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY,
                                       cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE):
                                module.automatic_object_width.value = ow
                                measurements = cellprofiler.measurement.Measurements()
                                image_set_list = cellprofiler.image.ImageSetList()
                                workspace = cellprofiler.workspace.Workspace(
                                        pipeline, None, None, None,
                                        measurements, image_set_list)
                                pipeline.prepare_run(workspace)
                                inj_module.prepare_group(workspace, {}, [1])
                                module.prepare_group(workspace, {}, [1])
                                image_set = image_set_list.get_image_set(0)
                                object_set = cellprofiler.region.Set()
                                workspace = cellprofiler.workspace.Workspace(pipeline,
                                                                             inj_module,
                                                                             image_set,
                                                                             object_set,
                                                                             measurements,
                                                                             image_set_list)
                                inj_module.run(workspace)
                                module.run(workspace)
                                image = image_set.get_image("OutputImage")
                                self.assertTrue(image is not None)
                                self.assertTrue(numpy.all(numpy.std(image.pixel_data) < .00001),
                                                """Failure case:
            each_or_all            = %(ea)s
            intensity_choice       = %(intensity_choice)s
            dilate_objects         = %(dilate_objects)s
            smoothing_method       = %(smoothing_method)s
            automatic_object_width = %(ow)s""" % locals())

    def test_01_02_masked_image(self):
        """A masked image should be insensitive to points outside the mask"""
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        numpy.random.seed(12)
        for image in (numpy.random.uniform(size=(10, 10)),
                      numpy.random.uniform(size=(10, 10, 3))):
            mask = numpy.zeros((10, 10), bool)
            mask[2:7, 3:8] = True
            image[mask] = 1
            inj_module = cellprofiler.modules.injectimage.InjectImage("MyImage", image, mask)
            inj_module.module_num = 1
            pipeline.add_module(inj_module)
            module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
            module.module_num = 2
            pipeline.add_module(module)
            module.image_name.value = "MyImage"
            module.illumination_image_name.value = "OutputImage"
            module.rescale_option.value = cellprofiler.setting.YES
            module.dilate_objects.value = False

            for ea in (cellprofiler.modules.correctilluminationcalculate.EA_EACH, cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS, cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST):
                module.each_or_all.value = ea
                for intensity_choice in (cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND, cellprofiler.modules.correctilluminationcalculate.IC_REGULAR):
                    module.intensity_choice.value = intensity_choice
                    for smoothing_method \
                            in (cellprofiler.modules.correctilluminationcalculate.SM_NONE, cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL,
                                cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER, cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER,
                                cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE, cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL):
                        module.smoothing_method.value = smoothing_method
                        for ow in (cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC, cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY,
                                   cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE):
                            module.automatic_object_width.value = ow
                            measurements = cellprofiler.measurement.Measurements()
                            image_set_list = cellprofiler.image.ImageSetList()
                            workspace = cellprofiler.workspace.Workspace(
                                    pipeline, None, None, None,
                                    measurements, image_set_list)
                            pipeline.prepare_run(workspace)
                            inj_module.prepare_group(workspace, {}, [1])
                            module.prepare_group(workspace, {}, [1])
                            image_set = image_set_list.get_image_set(0)
                            object_set = cellprofiler.region.Set()
                            workspace = cellprofiler.workspace.Workspace(pipeline,
                                                                         inj_module,
                                                                         image_set,
                                                                         object_set,
                                                                         measurements,
                                                                         image_set_list)
                            inj_module.run(workspace)
                            module.run(workspace)
                            image = image_set.get_image("OutputImage")
                            self.assertTrue(image is not None)
                            self.assertTrue(numpy.all(abs(image.pixel_data[mask] - 1 < .00001)),
                                            """Failure case:
            each_or_all            = %(ea)s
            intensity_choice       = %(intensity_choice)s
            smoothing_method       = %(smoothing_method)s
            automatic_object_width = %(ow)s""" % locals())

    def test_01_03_filtered(self):
        """Regression test of issue #310

        post_group should add the composite image to the image set
        if CorrectIllumination_Calculate didn't run because the image
        set was filtered.
        """
        r = numpy.random.RandomState()
        r.seed(13)
        i0 = r.uniform(size=(11, 13))
        i1 = r.uniform(size=(11, 13))
        i2 = r.uniform(size=(11, 13))
        workspaces, module = self.make_workspaces((
            (i0, None),
            (i1, None),
            (i2, None)))
        module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE
        module.save_average_image.value = True
        module.save_dilated_image.value = True

        module.prepare_group(workspaces[0], None, [1, 2, 3])
        assert isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate)
        for workspace in workspaces[:-1]:
            assert isinstance(workspace, cellprofiler.workspace.Workspace)
            module.run(workspace)
        image_set = workspaces[-1].image_set
        self.assertNotIn(OUTPUT_IMAGE_NAME, image_set.get_names())
        self.assertNotIn(DILATED_IMAGE_NAME, image_set.get_names())
        self.assertNotIn(AVERAGE_IMAGE_NAME, image_set.get_names())
        module.post_group(workspaces[-1], None)
        self.assertIn(OUTPUT_IMAGE_NAME, image_set.get_names())
        self.assertIn(DILATED_IMAGE_NAME, image_set.get_names())
        self.assertIn(AVERAGE_IMAGE_NAME, image_set.get_names())

    def test_01_04_not_filtered(self):
        """Regression test of issue #310, negative case

        post_group should not add the composite image to the image set
        if CorrectIllumination_Calculate did run.
        """
        r = numpy.random.RandomState()
        r.seed(13)
        i0 = r.uniform(size=(11, 13))
        i1 = r.uniform(size=(11, 13))
        i2 = r.uniform(size=(11, 13))
        workspaces, module = self.make_workspaces((
            (i0, None),
            (i1, None),
            (i2, None)))
        module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE
        module.save_average_image.value = True
        module.save_dilated_image.value = True

        module.prepare_group(workspaces[0], None, [1, 2, 3])
        assert isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate)
        for workspace in workspaces:
            assert isinstance(workspace, cellprofiler.workspace.Workspace)
            module.run(workspace)
        image_set = workspaces[-1].image_set
        self.assertIn(OUTPUT_IMAGE_NAME, image_set.get_names())
        self.assertIn(DILATED_IMAGE_NAME, image_set.get_names())
        self.assertIn(AVERAGE_IMAGE_NAME, image_set.get_names())
        module.post_group(workspaces[-1], None)
        #
        # Make sure it appears only once
        #
        for image_name in (
                OUTPUT_IMAGE_NAME, DILATED_IMAGE_NAME, AVERAGE_IMAGE_NAME):
            self.assertEqual(len(filter(lambda x: x == image_name,
                                        image_set.get_names())), 1)

    def test_02_02_Background(self):
        """Test an image with four distinct backgrounds"""

        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        image = numpy.ones((40, 40))
        image[10, 10] = .25
        image[10, 30] = .5
        image[30, 10] = .75
        image[30, 30] = .9
        inj_module = cellprofiler.modules.injectimage.InjectImage("MyImage", image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.block_size.value = 20
        module.rescale_option.value = cellprofiler.setting.NO
        module.dilate_objects.value = False
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_NONE
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                     measurements, image_set_list)
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     inj_module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(numpy.all(image.pixel_data[:20, :20] == .25))
        self.assertTrue(numpy.all(image.pixel_data[:20, 20:] == .5))
        self.assertTrue(numpy.all(image.pixel_data[20:, :20] == .75))
        self.assertTrue(numpy.all(image.pixel_data[20:, 20:] == .9))

    def test_03_00_no_smoothing(self):
        """Make sure that no smoothing takes place if smoothing is turned off"""
        input_image = numpy.random.uniform(size=(10, 10))
        image_name = "InputImage"
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = cellprofiler.modules.injectimage.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_NONE
        module.rescale_option.value = cellprofiler.setting.NO
        module.dilate_objects.value = False
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                     measurements, image_set_list)
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     inj_module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(numpy.all(numpy.abs(image.pixel_data - input_image) < .001),
                        "Failed to fit polynomial to %s" % image_name)

    def test_03_01_FitPolynomial(self):
        """Test fitting a polynomial to different gradients"""

        y, x = (numpy.mgrid[0:20, 0:20]).astype(float) / 20.0
        image_x = x
        image_y = y
        image_x2 = x ** 2
        image_y2 = y ** 2
        image_xy = x * y
        for input_image, image_name in ((image_x, "XImage"),
                                        (image_y, "YImage"),
                                        (image_x2, "X2Image"),
                                        (image_y2, "Y2Image"),
                                        (image_xy, "XYImage")):
            pipeline = cellprofiler.pipeline.Pipeline()
            pipeline.add_listener(self.error_callback)
            inj_module = cellprofiler.modules.injectimage.InjectImage(image_name, input_image)
            inj_module.module_num = 1
            pipeline.add_module(inj_module)
            module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
            module.module_num = 2
            pipeline.add_module(module)
            module.image_name.value = image_name
            module.illumination_image_name.value = "OutputImage"
            module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
            module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
            module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_FIT_POLYNOMIAL
            module.rescale_option.value = cellprofiler.setting.NO
            module.dilate_objects.value = False
            measurements = cellprofiler.measurement.Measurements()
            image_set_list = cellprofiler.image.ImageSetList()
            workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                         measurements, image_set_list)
            pipeline.prepare_run(workspace)
            inj_module.prepare_group(workspace, {}, [1])
            module.prepare_group(workspace, {}, [1])
            image_set = image_set_list.get_image_set(0)
            object_set = cellprofiler.region.Set()
            workspace = cellprofiler.workspace.Workspace(pipeline,
                                                         inj_module,
                                                         image_set,
                                                         object_set,
                                                         measurements,
                                                         image_set_list)
            inj_module.run(workspace)
            module.run(workspace)
            image = image_set.get_image("OutputImage")
            self.assertTrue(numpy.all(numpy.abs(image.pixel_data - input_image) < .001),
                            "Failed to fit polynomial to %s" % image_name)

    def test_03_02_gaussian_filter(self):
        """Test gaussian filtering a gaussian of a point"""
        input_image = numpy.zeros((101, 101))
        input_image[50, 50] = 1
        image_name = "InputImage"
        i, j = numpy.mgrid[-50:51, -50:51]
        expected_image = numpy.e ** (- (i ** 2 + j ** 2) / (2 * (10.0 / 2.35) ** 2))
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = cellprofiler.modules.injectimage.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER
        module.automatic_object_width.value = cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cellprofiler.setting.NO
        module.dilate_objects.value = False
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                     measurements, image_set_list)
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     inj_module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        ipd = image.pixel_data[40:61, 40:61]
        expected_image = expected_image[40:61, 40:61]
        self.assertTrue(numpy.all(numpy.abs(ipd / ipd.mean() -
                                            expected_image / expected_image.mean()) <
                                  .001))

    def test_03_03_median_filter(self):
        """Test median filtering of a point"""
        input_image = numpy.zeros((101, 101))
        input_image[50, 50] = 1
        image_name = "InputImage"
        expected_image = numpy.zeros((101, 101))
        filter_distance = int(.5 + 10 / 2.35)
        expected_image[-filter_distance:filter_distance + 1,
        -filter_distance:filter_distance + 1] = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = cellprofiler.modules.injectimage.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER
        module.automatic_object_width.value = cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cellprofiler.setting.NO
        module.dilate_objects.value = False
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                     measurements, image_set_list)
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     inj_module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(numpy.all(image.pixel_data == expected_image))

    def test_03_04_smooth_to_average(self):
        """Test smoothing to an average value"""
        numpy.random.seed(0)
        input_image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image_name = "InputImage"
        expected_image = numpy.ones((10, 10)) * input_image.mean()
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = cellprofiler.modules.injectimage.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE
        module.automatic_object_width.value = cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cellprofiler.setting.NO
        module.dilate_objects.value = False
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                     measurements, image_set_list)
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     inj_module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        numpy.testing.assert_almost_equal(image.pixel_data, expected_image)

    def test_03_05_splines(self):
        for automatic, bg_mode, spline_points, threshold, convergence, offset, hi, lo, succeed in (
                (True, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 5, 2, .001, 0, True, False, True),
                (True, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 5, 2, .001, .7, False, True, True),
                (True, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 5, 2, .001, .5, True, True, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 5, 2, .001, 0, True, False, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 5, 2, .001, .7, False, True, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 5, 2, .001, .5, True, True, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_BRIGHT, 5, 2, .001, .7, False, True, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_DARK, 5, 2, .001, 0, True, False, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_GRAY, 5, 2, .001, .5, True, True, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 7, 2, .001, 0, True, False, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, 4, 2, .001, 0, True, False, True),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_DARK, 5, 2, .001, .7, False, True, False),
                (False, cellprofiler.modules.correctilluminationcalculate.MODE_BRIGHT, 5, 2, .001, 0, True, False, False)
        ):

            #
            # Make an image with a random background
            #
            numpy.random.seed(35)
            image = numpy.random.uniform(size=(21, 31)) * .05 + offset
            if hi:
                #
                # Add some "foreground" pixels
                #
                fg = numpy.random.permutation(400)[:100]
                image[fg % image.shape[0], (fg / image.shape[0]).astype(int)] *= 10
            if lo:
                #
                # Add some "background" pixels
                #
                bg = numpy.random.permutation(400)[:100]
                image[bg % image.shape[0], (bg / image.shape[0]).astype(int)] -= offset

            #
            # Make a background function
            #
            ii, jj = numpy.mgrid[-10:11, -15:16]
            bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
            bg *= .2
            image += bg

            workspaces, module = self.make_workspaces(((image, None),))
            self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
            module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
            module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
            module.rescale_option.value = cellprofiler.setting.NO
            module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
            module.automatic_splines.value = automatic
            module.spline_bg_mode.value = bg_mode
            module.spline_convergence.value = convergence
            module.spline_threshold.value = threshold
            module.spline_points.value = spline_points
            module.spline_rescale.value = 1
            module.prepare_group(workspaces[0], {}, [1])
            module.run(workspaces[0])
            img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
            pixel_data = img.pixel_data
            diff = pixel_data - numpy.min(pixel_data) - bg
            if succeed:
                self.assertTrue(numpy.all(diff < .05))
            else:
                self.assertFalse(numpy.all(diff < .05))

    def test_03_06_splines_scaled(self):
        #
        # Make an image with a random background
        #
        numpy.random.seed(36)
        image = numpy.random.uniform(size=(101, 131)) * .05
        #
        # Add some "foreground" pixels
        #
        fg = numpy.random.permutation(numpy.prod(image.shape))[:200]
        image[fg % image.shape[0], (fg / image.shape[0]).astype(int)] *= 15
        #
        # Make a background function
        #
        ii, jj = numpy.mgrid[-50:51, -65:66]
        bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
        bg *= .2
        image += bg

        workspaces, module = self.make_workspaces(((image, None),))
        self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
        module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.rescale_option.value = cellprofiler.setting.NO
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
        module.automatic_splines.value = False
        module.spline_rescale.value = 2
        module.prepare_group(workspaces[0], {}, [1])
        module.run(workspaces[0])
        img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = img.pixel_data
        diff = pixel_data - numpy.min(pixel_data) - bg
        numpy.all(diff < .05)

    def test_03_07_splines_masked(self):
        #
        # Make an image with a random background
        #
        numpy.random.seed(37)
        image = numpy.random.uniform(size=(21, 31)) * .05
        #
        # Mask 1/2 of the pixels
        #
        mask = numpy.random.uniform(size=(21, 31)) < .5
        #
        # Make a background function
        #
        ii, jj = numpy.mgrid[-10:11, -15:16]
        bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
        bg *= .2
        image += bg
        #
        # Offset the background within the mask
        #
        image[~mask] += bg[~mask]

        workspaces, module = self.make_workspaces(((image, mask),))
        self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
        module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.rescale_option.value = cellprofiler.setting.NO
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
        module.automatic_splines.value = True
        module.prepare_group(workspaces[0], {}, [1])
        module.run(workspaces[0])
        img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = img.pixel_data
        diff = pixel_data - numpy.min(pixel_data) - bg
        self.assertTrue(numpy.all(diff < .05))
        #
        # Make sure test fails w/o mask
        #
        workspaces, module = self.make_workspaces(((image, None),))
        self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
        module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.rescale_option.value = cellprofiler.setting.NO
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
        module.automatic_splines.value = True
        module.prepare_group(workspaces[0], {}, [1])
        module.run(workspaces[0])
        img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = img.pixel_data
        diff = pixel_data - numpy.min(pixel_data) - bg
        self.assertFalse(numpy.all(diff < .05))

    def test_03_07_splines_cropped(self):
        #
        # Make an image with a random background
        #
        numpy.random.seed(37)
        image = numpy.random.uniform(size=(21, 31)) * .05
        #
        # Mask 1/2 of the pixels
        #
        mask = numpy.zeros(image.shape, bool)
        mask[4:-4, 6:-6] = True
        #
        # Make a background function
        #
        ii, jj = numpy.mgrid[-10:11, -15:16]
        bg = ((ii.astype(float) / 10) ** 2) * ((jj.astype(float) / 15) ** 2)
        bg *= .2
        image += bg
        #
        # Offset the background within the mask
        #
        image[~mask] += bg[~mask]

        workspaces, module = self.make_workspaces(((image, mask),))
        self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
        module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.rescale_option.value = cellprofiler.setting.NO
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
        module.automatic_splines.value = True
        module.prepare_group(workspaces[0], {}, [1])
        module.run(workspaces[0])
        img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = img.pixel_data
        diff = pixel_data - numpy.min(pixel_data) - bg
        self.assertTrue(numpy.all(diff < .05))
        #
        # Make sure test fails w/o mask
        #
        workspaces, module = self.make_workspaces(((image, None),))
        self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND
        module.each_or_all.value = cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.rescale_option.value = cellprofiler.setting.NO
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_SPLINES
        module.automatic_splines.value = True
        module.prepare_group(workspaces[0], {}, [1])
        module.run(workspaces[0])
        img = workspaces[0].image_set.get_image(OUTPUT_IMAGE_NAME)
        pixel_data = img.pixel_data
        diff = pixel_data - numpy.min(pixel_data) - bg
        self.assertFalse(numpy.all(diff < .05))

    def test_04_01_intermediate_images(self):
        """Make sure the average and dilated image flags work"""
        for average_flag, dilated_flag in ((False, False),
                                           (False, True),
                                           (True, False),
                                           (True, True)):
            pipeline = cellprofiler.pipeline.Pipeline()
            pipeline.add_listener(self.error_callback)
            inj_module = cellprofiler.modules.injectimage.InjectImage("InputImage", numpy.zeros((10, 10)))
            inj_module.module_num = 1
            pipeline.add_module(inj_module)
            module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
            module.module_num = 2
            pipeline.add_module(module)
            module.image_name.value = "InputImage"
            module.illumination_image_name.value = "OutputImage"
            module.save_average_image.value = average_flag
            module.average_image_name.value = "AverageImage"
            module.save_dilated_image.value = dilated_flag
            module.dilated_image_name.value = "DilatedImage"
            measurements = cellprofiler.measurement.Measurements()
            image_set_list = cellprofiler.image.ImageSetList()
            workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                         measurements, image_set_list)
            pipeline.prepare_run(workspace)
            inj_module.prepare_group(workspace, {}, [1])
            module.prepare_group(workspace, {}, [1])
            image_set = image_set_list.get_image_set(0)
            object_set = cellprofiler.region.Set()
            workspace = cellprofiler.workspace.Workspace(pipeline,
                                                         inj_module,
                                                         image_set,
                                                         object_set,
                                                         measurements,
                                                         image_set_list)
            inj_module.run(workspace)
            module.run(workspace)
            if average_flag:
                img = image_set.get_image("AverageImage")
            else:
                self.assertRaises(AssertionError,
                                  image_set.get_image,
                                  "AverageImage")
            if dilated_flag:
                img = image_set.get_image("DilatedImage")
            else:
                self.assertRaises(AssertionError,
                                  image_set.get_image,
                                  "DilatedImage")

    def test_05_01_rescale(self):
        """Test basic rescaling of an image with two values"""
        input_image = numpy.ones((10, 10))
        input_image[0:5, :] *= .5
        image_name = "InputImage"
        expected_image = input_image * 2
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = cellprofiler.modules.injectimage.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_NONE
        module.automatic_object_width.value = cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cellprofiler.setting.YES
        module.dilate_objects.value = False
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                     measurements, image_set_list)
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     inj_module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(numpy.all(image.pixel_data == expected_image))

    def test_05_02_rescale_outlier(self):
        """Test rescaling with one low outlier"""
        input_image = numpy.ones((10, 10))
        input_image[0:5, :] *= .5
        input_image[0, 0] = .1
        image_name = "InputImage"
        expected_image = input_image * 2
        expected_image[0, 0] = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = cellprofiler.modules.injectimage.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = cellprofiler.modules.correctilluminationcalculate.IC_REGULAR
        module.each_or_all.value == cellprofiler.modules.correctilluminationcalculate.EA_EACH
        module.smoothing_method.value = cellprofiler.modules.correctilluminationcalculate.SM_NONE
        module.automatic_object_width.value = cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cellprofiler.setting.YES
        module.dilate_objects.value = False
        measurements = cellprofiler.measurement.Measurements()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline, None, None, None,
                                                     measurements, image_set_list)
        pipeline.prepare_run(workspace)
        inj_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cellprofiler.region.Set()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     inj_module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(numpy.all(image.pixel_data == expected_image))

    def test_06_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9411

LoadImages:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    What type of files are you loading?:individual images
    How do you want to load these files?:Text-Exact match
    How many images are there in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:No
    Image location:Default Image Folder
    Enter the full path to the images:.
    Do you want to check image sets for missing or duplicate files?:No
    Do you want to group image sets by metadata?:No
    Do you want to exclude certain files?:No
    What metadata fields do you want to group by?:
    Type the text that these images have in common (case-sensitive):D.TIF
    What do you want to call this image in CellProfiler?:Image1
    What is the position of this image in each group?:D.TIF
    Do you want to extract metadata from the file name, the subfolder path or both?:None
    Type the regular expression that finds metadata in the file name\x3A:None
    Type the regular expression that finds metadata in the subfolder path\x3A:None

CorrectIlluminationCalculate:[module_num:2|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the input image:Image1
    Name the output image:Illum1
    Select how the illumination function is calculated:Regular
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:Yes
    Calculate function for each image individually, or based on all images?:All
    Smoothing method:No smoothing
    Method to calculate smoothing filter size:Automatic
    Approximate object size:10
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:Yes
    Name the averaged image:Illum1Average
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:Yes
    Name the dilated image:Illum1Dilated

CorrectIlluminationCalculate:[module_num:3|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the input image:Image2
    Name the output image:Illum2
    Select how the illumination function is calculated:Background
    Dilate objects in the final averaged image?:Yes
    Dilation radius:2
    Block size:65
    Rescale the illumination function?:No
    Calculate function for each image individually, or based on all images?:All\x3A First cycle
    Smoothing method:Median Filter
    Method to calculate smoothing filter size:Manually
    Approximate object size:15
    Smoothing filter size:20
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:Yes
    Name the averaged image:Illum2Avg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:Yes
    Name the dilated image:Illum2Dilated

CorrectIlluminationCalculate:[module_num:4|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the input image:Image3
    Name the output image:Illum3
    Select how the illumination function is calculated:Regular
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:Median
    Calculate function for each image individually, or based on all images?:All\x3A Across cycles
    Smoothing method:Median Filter
    Method to calculate smoothing filter size:Automatic
    Approximate object size:10
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:Illum3Avg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:Yes
    Name the dilated image:Illum3Dilated

CorrectIlluminationCalculate:[module_num:5|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the input image:Image4
    Name the output image:Illum4
    Select how the illumination function is calculated:Regular
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:Median
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Gaussian Filter
    Method to calculate smoothing filter size:Object size
    Approximate object size:15
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:Illum4Avg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:Yes
    Name the dilated image:Illum4Dilated

CorrectIlluminationCalculate:[module_num:6|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select the input image:Image5
    Name the output image:Illum5
    Select how the illumination function is calculated:Regular
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:Median
    Calculate function for each image individually, or based on all images?:All
    Smoothing method:Smooth to Average
    Method to calculate smoothing filter size:Object size
    Approximate object size:15
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:Illum5Avg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:No
    Name the dilated image:Illum5Dilated
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 6)
        for i, (image_name, illumination_image_name, intensity_choice,
                dilate_objects, object_dilation_radius, block_size,
                rescale_option, each_or_all, smoothing_method,
                automatic_object_width, object_width, size_of_smoothing_filter,
                save_average_image, average_image_name, save_dilated_image,
                dilated_image_name) in enumerate((
                ("Image1", "Illum1", cellprofiler.modules.correctilluminationcalculate.IC_REGULAR, False, 1, 60, cellprofiler.setting.YES,
                 cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST, cellprofiler.modules.correctilluminationcalculate.SM_NONE, cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC, 10, 10, True,
                 "Illum1Average", True, "Illum1Dilated"),
                ("Image2", "Illum2", cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND, True, 2, 65, cellprofiler.setting.NO,
                 cellprofiler.modules.correctilluminationcalculate.EA_ALL_FIRST, cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER, cellprofiler.modules.correctilluminationcalculate.FI_MANUALLY, 15, 20,
                 True, "Illum2Avg", True, "Illum2Dilated"),
                ("Image3", "Illum3", cellprofiler.modules.correctilluminationcalculate.IC_REGULAR, False, 1, 60,
                 cellprofiler.modules.correctilluminationcalculate.RE_MEDIAN, cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS, cellprofiler.modules.correctilluminationcalculate.SM_MEDIAN_FILTER,
                 cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC, 10, 10, False, "Illum3Avg", True,
                 "Illum3Dilated"),
                ("Image4", "Illum4", cellprofiler.modules.correctilluminationcalculate.IC_REGULAR, cellprofiler.setting.NO, 1, 60,
                 cellprofiler.modules.correctilluminationcalculate.RE_MEDIAN, cellprofiler.modules.correctilluminationcalculate.EA_EACH, cellprofiler.modules.correctilluminationcalculate.SM_GAUSSIAN_FILTER,
                 cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE, 15, 10, False, "Illum4Avg", True,
                 "Illum4Dilated"),
                ("Image5", "Illum5", cellprofiler.modules.correctilluminationcalculate.IC_REGULAR, cellprofiler.setting.NO, 1, 60,
                 cellprofiler.modules.correctilluminationcalculate.RE_MEDIAN, cellprofiler.modules.correctilluminationcalculate.EA_ALL_ACROSS, cellprofiler.modules.correctilluminationcalculate.SM_TO_AVERAGE,
                 cellprofiler.modules.correctilluminationcalculate.FI_OBJECT_SIZE, 15, 10, False, "Illum5Avg",
                 False, "Illum5Dilated"))):
            module = pipeline.modules()[i + 1]
            self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
            self.assertEqual(module.image_name, image_name)
            self.assertEqual(module.illumination_image_name, illumination_image_name)
            self.assertEqual(module.intensity_choice, intensity_choice)
            self.assertEqual(module.dilate_objects, dilate_objects)
            self.assertEqual(module.object_dilation_radius, object_dilation_radius)
            self.assertEqual(module.block_size, block_size)
            self.assertEqual(module.rescale_option, rescale_option)
            self.assertEqual(module.each_or_all, each_or_all)
            self.assertEqual(module.smoothing_method, smoothing_method)
            self.assertEqual(module.automatic_object_width, automatic_object_width)
            self.assertEqual(module.object_width, object_width)
            self.assertEqual(module.size_of_smoothing_filter, size_of_smoothing_filter)
            self.assertEqual(module.save_average_image, save_average_image)
            self.assertEqual(module.average_image_name, average_image_name)
            self.assertEqual(module.save_dilated_image, save_dilated_image)
            self.assertEqual(module.dilated_image_name, dilated_image_name)

    def test_06_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10125

CorrectIlluminationCalculate:[module_num:1|svn_version:\'10063\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:Masked
    Name the output image:Illum
    Select how the illumination function is calculated:Background
    Dilate objects in the final averaged image?:No
    Dilation radius:2
    Block size:55
    Rescale the illumination function?:No
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Splines
    Method to calculate smoothing filter size:Automatic
    Approximate object size:11
    Smoothing filter size:12
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:IllumAverage
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:No
    Name the dilated image:IllumDilated
    Automatically calculate spline parameters?:No
    Background mode:bright
    # of spline points:4
    Background threshold:2
    Image resampling factor:2
    Max # of iterations:40
    Convergence:0.001

CorrectIlluminationCalculate:[module_num:2|svn_version:\'10063\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:Masked
    Name the output image:Illum
    Select how the illumination function is calculated:Background
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:No
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Splines
    Method to calculate smoothing filter size:Automatic
    Approximate object size:10
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:IllumBlueAvg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:No
    Name the dilated image:IllumBlueDilated
    Automatically calculate spline parameters?:Yes
    Background mode:auto
    # of spline points:3
    Background threshold:2
    Image resampling factor:2
    Max # of iterations:40
    Convergence:0.001

CorrectIlluminationCalculate:[module_num:3|svn_version:\'10063\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:Masked
    Name the output image:Illum
    Select how the illumination function is calculated:Background
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:No
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Splines
    Method to calculate smoothing filter size:Automatic
    Approximate object size:10
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:IllumBlueAvg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:No
    Name the dilated image:IllumBlueDilated
    Automatically calculate spline parameters?:Yes
    Background mode:dark
    # of spline points:3
    Background threshold:2
    Image resampling factor:2
    Max # of iterations:40
    Convergence:0.001

CorrectIlluminationCalculate:[module_num:4|svn_version:\'10063\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:Masked
    Name the output image:Illum
    Select how the illumination function is calculated:Background
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:No
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Splines
    Method to calculate smoothing filter size:Automatic
    Approximate object size:10
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:IllumBlueAvg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:No
    Name the dilated image:IllumBlueDilated
    Automatically calculate spline parameters?:No
    Background mode:gray
    # of spline points:3
    Background threshold:2
    Image resampling factor:2
    Max # of iterations:40
    Convergence:0.001

CorrectIlluminationCalculate:[module_num:5|svn_version:\'10063\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:Masked
    Name the output image:Illum
    Select how the illumination function is calculated:Background
    Dilate objects in the final averaged image?:No
    Dilation radius:1
    Block size:60
    Rescale the illumination function?:No
    Calculate function for each image individually, or based on all images?:Each
    Smoothing method:Convex Hull
    Method to calculate smoothing filter size:Automatic
    Approximate object size:10
    Smoothing filter size:10
    Retain the averaged image for use later in the pipeline (for example, in SaveImages)?:No
    Name the averaged image:IllumBlueAvg
    Retain the dilated image for use later in the pipeline (for example, in SaveImages)?:No
    Name the dilated image:IllumBlueDilated
    Automatically calculate spline parameters?:No
    Background mode:gray
    # of spline points:3
    Background threshold:2
    Image resampling factor:2
    Max # of iterations:40
    Convergence:0.001
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
        self.assertEqual(module.image_name, "Masked")
        self.assertEqual(module.illumination_image_name, "Illum")
        self.assertEqual(module.intensity_choice, cellprofiler.modules.correctilluminationcalculate.IC_BACKGROUND)
        self.assertFalse(module.dilate_objects)
        self.assertEqual(module.object_dilation_radius, 2)
        self.assertEqual(module.block_size, 55)
        self.assertEqual(module.rescale_option, cellprofiler.setting.NO)
        self.assertEqual(module.each_or_all, cellprofiler.modules.correctilluminationcalculate.EA_EACH)
        self.assertEqual(module.smoothing_method, cellprofiler.modules.correctilluminationcalculate.SM_SPLINES)
        self.assertEqual(module.automatic_object_width, cellprofiler.modules.correctilluminationcalculate.FI_AUTOMATIC)
        self.assertEqual(module.object_width, 11)
        self.assertEqual(module.size_of_smoothing_filter, 12)
        self.assertFalse(module.save_average_image)
        self.assertEqual(module.average_image_name, "IllumAverage")
        self.assertFalse(module.save_dilated_image)
        self.assertEqual(module.dilated_image_name, "IllumDilated")
        self.assertFalse(module.automatic_splines)
        self.assertEqual(module.spline_bg_mode, cellprofiler.modules.correctilluminationcalculate.MODE_BRIGHT)
        self.assertEqual(module.spline_points, 4)
        self.assertEqual(module.spline_threshold, 2)
        self.assertEqual(module.spline_rescale, 2)
        self.assertEqual(module.spline_maximum_iterations, 40)
        self.assertAlmostEqual(module.spline_convergence.value, 0.001)

        self.assertTrue(pipeline.modules()[1].automatic_splines)

        for module, spline_bg_mode in zip(pipeline.modules()[1:4], (
                cellprofiler.modules.correctilluminationcalculate.MODE_AUTO, cellprofiler.modules.correctilluminationcalculate.MODE_DARK, cellprofiler.modules.correctilluminationcalculate.MODE_GRAY)):
            self.assertTrue(isinstance(module, cellprofiler.modules.correctilluminationcalculate.CorrectIlluminationCalculate))
            self.assertEqual(module.spline_bg_mode, spline_bg_mode)

        module = pipeline.modules()[4]
        self.assertEqual(module.smoothing_method, cellprofiler.modules.correctilluminationcalculate.SM_CONVEX_HULL)
