"""test_correctillumination_calculate.py - test the CorrectIllumination_Calculate module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np
import unittest
import sys

import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm
import cellprofiler.modules.injectimage as inj
import cellprofiler.modules.correctillumination_calculate as calc

class TestCorrectImage_Calculate(unittest.TestCase):
    def error_callback(self, calller, event):
        if isinstance(event, cpp.RunExceptionEvent):
            self.fail(event.error.message)

    def test_00_00_zeros(self):
        """Test all combinations of options with an image of all zeros"""
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage("MyImage",np.zeros((10,10)))
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.save_average_image.value = True
        module.save_dilated_image.value = True
        
        for ea in (calc.EA_EACH, calc.EA_ALL):
            module.each_or_all.value = ea
            for intensity_choice in (calc.IC_BACKGROUND, calc.IC_REGULAR):
                module.intensity_choice = intensity_choice
                for dilate_objects in (True, False):
                    module.dilate_objects.value = dilate_objects
                    for rescale_option in (cps.YES, cps.NO, calc.RE_MEDIAN):
                        module.rescale_option.value = rescale_option
                        for smoothing_method \
                         in (calc.SM_NONE, calc.SM_FIT_POLYNOMIAL, 
                             calc.SM_GAUSSIAN_FILTER, calc.SM_MEDIAN_FILTER, 
                             calc.SM_TO_AVERAGE):
                            module.smoothing_method.value = smoothing_method
                            for ow in (calc.FI_AUTOMATIC, calc.FI_MANUALLY, 
                                       calc.FI_OBJECT_SIZE):
                                module.automatic_object_width.value = ow
                                image_set_list = pipeline.prepare_run(None)
                                image_set = image_set_list.get_image_set(0)
                                object_set = cpo.ObjectSet()
                                measurements = cpm.Measurements()
                                workspace = cpw.Workspace(pipeline,
                                                          inj_module,
                                                          image_set,
                                                          object_set,
                                                          measurements,
                                                          image_set_list)
                                inj_module.run(workspace)
                                module.run(workspace)
                                image = image_set.get_image("OutputImage")
                                self.assertTrue(image != None)
                                self.assertTrue(np.all(image.pixel_data == 0),
                                                """Failure case:
            intensity_choice = %(intensity_choice)s
            dilate_objects = %(dilate_objects)s
            rescale_option = %(rescale_option)s
            smoothing_method = %(smoothing_method)s
            automatic_object_width = %(ow)s"""%locals())

    def test_01_01_ones_image(self):
        """The illumination correction of an image of all ones should be uniform
        
        """
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage("MyImage",np.ones((10,10)))
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.rescale_option.value = cps.YES
        
        for ea in (calc.EA_EACH, calc.EA_ALL):
            module.each_or_all.value = ea
            for intensity_choice in (calc.IC_BACKGROUND, calc.IC_REGULAR):
                module.intensity_choice = intensity_choice
                for dilate_objects in (True, False):
                    module.dilate_objects.value = dilate_objects
                    for smoothing_method \
                     in (calc.SM_NONE, calc.SM_FIT_POLYNOMIAL, 
                         calc.SM_GAUSSIAN_FILTER, calc.SM_MEDIAN_FILTER, 
                         calc.SM_TO_AVERAGE):
                        module.smoothing_method.value = smoothing_method
                        for ow in (calc.FI_AUTOMATIC, calc.FI_MANUALLY, 
                                   calc.FI_OBJECT_SIZE):
                            module.automatic_object_width.value = ow
                            image_set_list = pipeline.prepare_run(None)
                            image_set = image_set_list.get_image_set(0)
                            object_set = cpo.ObjectSet()
                            measurements = cpm.Measurements()
                            workspace = cpw.Workspace(pipeline,
                                                      inj_module,
                                                      image_set,
                                                      object_set,
                                                      measurements,
                                                      image_set_list)
                            inj_module.run(workspace)
                            module.run(workspace)
                            image = image_set.get_image("OutputImage")
                            self.assertTrue(image != None)
                            self.assertTrue(np.all(abs(image.pixel_data - 1 < .00001)),
                                                """Failure case:
            each_or_all            = %(ea)s
            intensity_choice       = %(intensity_choice)s
            dilate_objects         = %(dilate_objects)s
            smoothing_method       = %(smoothing_method)s
            automatic_object_width = %(ow)s"""%locals())
        
    def test_01_02_masked_image(self):
        """A masked image should be insensitive to points outside the mask"""
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        image = np.random.uniform(size=(10,10))
        mask  = np.zeros((10,10),bool)
        mask[2:7,3:8] = True
        image[mask] = 1
        inj_module = inj.InjectImage("MyImage", image, mask)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.rescale_option.value = cps.YES
        module.dilate_objects.value = False
        sys.stderr.write("TO_DO: Median filter does not respect masks\n")
        
        for ea in (calc.EA_EACH, calc.EA_ALL):
            module.each_or_all.value = ea
            for intensity_choice in (calc.IC_BACKGROUND, calc.IC_REGULAR):
                module.intensity_choice = intensity_choice
                for smoothing_method \
                 in (calc.SM_NONE, calc.SM_FIT_POLYNOMIAL, 
                     calc.SM_GAUSSIAN_FILTER, # calc.SM_MEDIAN_FILTER, 
                     calc.SM_TO_AVERAGE):
                    module.smoothing_method.value = smoothing_method
                    for ow in (calc.FI_AUTOMATIC, calc.FI_MANUALLY, 
                               calc.FI_OBJECT_SIZE):
                        module.automatic_object_width.value = ow
                        image_set_list = pipeline.prepare_run(None)
                        image_set = image_set_list.get_image_set(0)
                        object_set = cpo.ObjectSet()
                        measurements = cpm.Measurements()
                        workspace = cpw.Workspace(pipeline,
                                                  inj_module,
                                                  image_set,
                                                  object_set,
                                                  measurements,
                                                  image_set_list)
                        inj_module.run(workspace)
                        module.run(workspace)
                        image = image_set.get_image("OutputImage")
                        self.assertTrue(image != None)
                        self.assertTrue(np.all(abs(image.pixel_data[mask] - 1 < .00001)),
                                            """Failure case:
            each_or_all            = %(ea)s
            intensity_choice       = %(intensity_choice)s
            smoothing_method       = %(smoothing_method)s
            automatic_object_width = %(ow)s"""%locals())
    
    def test_02_02_Background(self):
        """Test an image with four distinct backgrounds"""
        
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        image = np.ones((40,40))
        image[10,10] = .25
        image[10,30] = .5
        image[30,10] = .75
        image[30,30] = .9
        inj_module = inj.InjectImage("MyImage", image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = "MyImage"
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = calc.IC_BACKGROUND
        module.each_or_all.value == calc.EA_EACH
        module.block_size.value = 20
        module.rescale_option.value = cps.NO
        module.dilate_objects.value = False
        module.smoothing_method.value = calc.SM_NONE
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  inj_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(np.all(image.pixel_data[:20,:20] == .25))
        self.assertTrue(np.all(image.pixel_data[:20,20:] == .5))
        self.assertTrue(np.all(image.pixel_data[20:,:20] == .75))
        self.assertTrue(np.all(image.pixel_data[20:,20:] == .9))

    def test_03_00_no_smoothing(self):
        """Make sure that no smoothing takes place if smoothing is turned off"""
        input_image = np.random.uniform(size=(10,10))
        image_name = "InputImage"
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = calc.IC_REGULAR
        module.each_or_all.value == calc.EA_EACH
        module.smoothing_method.value = calc.SM_NONE
        module.rescale_option.value = cps.NO
        module.dilate_objects.value = False
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  inj_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(np.all(np.abs(image.pixel_data-input_image) < .001),
                        "Failed to fit polynomial to %s"%(image_name))
    
    def test_03_01_FitPolynomial(self):
        """Test fitting a polynomial to different gradients"""
        
        y,x = (np.mgrid[0:20,0:20]).astype(float)/20.0
        image_x = x
        image_y = y
        image_x2 = x**2
        image_y2 = y**2
        image_xy = x*y
        for input_image, image_name in ((image_x, "XImage"),
                                        (image_y, "YImage"),
                                        (image_x2, "X2Image"),
                                        (image_y2, "Y2Image"),
                                        (image_xy, "XYImage")):
            pipeline = cpp.Pipeline()
            pipeline.add_listener(self.error_callback)
            inj_module = inj.InjectImage(image_name, input_image)
            inj_module.module_num = 1
            pipeline.add_module(inj_module)
            module = calc.CorrectIllumination_Calculate()
            module.module_num = 2
            pipeline.add_module(module)
            module.image_name.value = image_name
            module.illumination_image_name.value = "OutputImage"
            module.intensity_choice.value = calc.IC_REGULAR
            module.each_or_all.value == calc.EA_EACH
            module.smoothing_method.value = calc.SM_FIT_POLYNOMIAL
            module.rescale_option.value = cps.NO
            module.dilate_objects.value = False
            image_set_list = pipeline.prepare_run(None)
            image_set = image_set_list.get_image_set(0)
            object_set = cpo.ObjectSet()
            measurements = cpm.Measurements()
            workspace = cpw.Workspace(pipeline,
                                      inj_module,
                                      image_set,
                                      object_set,
                                      measurements,
                                      image_set_list)
            inj_module.run(workspace)
            module.run(workspace)
            image = image_set.get_image("OutputImage")
            self.assertTrue(np.all(np.abs(image.pixel_data-input_image) < .001),
                            "Failed to fit polynomial to %s"%(image_name))
    
    def test_03_02_gaussian_filter(self):
        """Test gaussian filtering a gaussian of a point"""
        input_image = np.zeros((101,101))
        input_image[50,50] = 1
        image_name = "InputImage"
        i,j = np.mgrid[-50:51,-50:51]
        expected_image = np.e ** (- (i**2+j**2)/(2*(10.0/2.35)**2))
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = calc.IC_REGULAR
        module.each_or_all.value == calc.EA_EACH
        module.smoothing_method.value = calc.SM_GAUSSIAN_FILTER
        module.automatic_object_width.value = calc.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cps.NO
        module.dilate_objects.value = False
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  inj_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        ipd   = image.pixel_data[40:61,40:61]
        expected_image = expected_image[40:61,40:61]
        self.assertTrue(np.all(np.abs(ipd / ipd.mean()-
                                      expected_image/ expected_image.mean()) <
                                      .001))
        
    def test_03_03_median_filter(self):
        """Test median filtering of a point"""
        input_image = np.zeros((101,101))
        input_image[50,50] = 1
        image_name = "InputImage"
        expected_image = np.zeros((101,101))
        filter_distance = int(.5 + 10/2.35)
        expected_image[-filter_distance:filter_distance+1,
                       -filter_distance:filter_distance+1] = 1
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = calc.IC_REGULAR
        module.each_or_all.value == calc.EA_EACH
        module.smoothing_method.value = calc.SM_MEDIAN_FILTER
        module.automatic_object_width.value = calc.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cps.NO
        module.dilate_objects.value = False
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  inj_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(np.all(image.pixel_data == expected_image))
    
    def test_03_04_smooth_to_average(self):
        """Test smoothing to an average value"""
        np.random.seed(0)
        input_image = np.random.uniform(size=(10,10))
        image_name = "InputImage"
        expected_image = np.ones((10,10))*input_image.mean()
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = calc.IC_REGULAR
        module.each_or_all.value == calc.EA_EACH
        module.smoothing_method.value = calc.SM_TO_AVERAGE
        module.automatic_object_width.value = calc.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cps.NO
        module.dilate_objects.value = False
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  inj_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(np.all(image.pixel_data == expected_image))
    
    def test_04_01_intermediate_images(self):
        """Make sure the average and dilated image flags work"""
        for average_flag, dilated_flag in ((False,False),
                                           (False,True),
                                           (True, False),
                                           (True,True)):
            pipeline = cpp.Pipeline()
            pipeline.add_listener(self.error_callback)
            inj_module = inj.InjectImage("InputImage", np.zeros((10,10)))
            inj_module.module_num = 1
            pipeline.add_module(inj_module)
            module = calc.CorrectIllumination_Calculate()
            module.module_num = 2
            pipeline.add_module(module)
            module.image_name.value = "InputImage"
            module.illumination_image_name.value = "OutputImage"
            module.save_average_image.value = average_flag
            module.average_image_name.value = "AverageImage"
            module.save_dilated_image.value = dilated_flag
            module.dilated_image_name.value = "DilatedImage"
            image_set_list = pipeline.prepare_run(None)
            image_set = image_set_list.get_image_set(0)
            object_set = cpo.ObjectSet()
            measurements = cpm.Measurements()
            workspace = cpw.Workspace(pipeline,
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
        input_image = np.ones((10,10))
        input_image[0:5,:] *= .5
        image_name = "InputImage"
        expected_image = input_image * 2
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = calc.IC_REGULAR
        module.each_or_all.value == calc.EA_EACH
        module.smoothing_method.value = calc.SM_NONE
        module.automatic_object_width.value = calc.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cps.YES
        module.dilate_objects.value = False
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  inj_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(np.all(image.pixel_data == expected_image))
    
    def test_05_02_rescale_outlier(self):
        """Test rescaling with one low outlier"""
        input_image = np.ones((10,10))
        input_image[0:5,:] *= .5
        input_image[0,0] = .1
        image_name = "InputImage"
        expected_image = input_image * 2
        expected_image[0,0] = 1
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        inj_module = inj.InjectImage(image_name, input_image)
        inj_module.module_num = 1
        pipeline.add_module(inj_module)
        module = calc.CorrectIllumination_Calculate()
        module.module_num = 2
        pipeline.add_module(module)
        module.image_name.value = image_name
        module.illumination_image_name.value = "OutputImage"
        module.intensity_choice.value = calc.IC_REGULAR
        module.each_or_all.value == calc.EA_EACH
        module.smoothing_method.value = calc.SM_NONE
        module.automatic_object_width.value = calc.FI_MANUALLY
        module.size_of_smoothing_filter.value = 10
        module.rescale_option.value = cps.YES
        module.dilate_objects.value = False
        image_set_list = pipeline.prepare_run(None)
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline,
                                  inj_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        inj_module.run(workspace)
        module.run(workspace)
        image = image_set.get_image("OutputImage")
        self.assertTrue(np.all(image.pixel_data == expected_image))
    
    