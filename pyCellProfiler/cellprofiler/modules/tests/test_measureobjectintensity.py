"""test_measureobjectintensity - test the MeasureObjectIntensity module

"""
__version__="$Revision$"

import base64
import math
import numpy
import StringIO
import unittest

import cellprofiler.modules.injectimage as II
import cellprofiler.modules.measureobjectintensity as MOI
import cellprofiler.pipeline as P

#
# This is a pipeline consisting of Matlab modules for LoadImages,
# IdentifyPrimAutomatic and MeasureObjectIntensity
#
pipeline_data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBXZWQgRmViIDExIDE2OjU4OjUyIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAkgIAAHic7VhPb9MwFHfatPwZmsomJtRTjhzGaAUHjlQqgkqsrdg0iaObuMUoiavEnlo+xaR9EY4cOCDxxYjbpHW8UudvAamWLOs57/3eez87z04OAQCNEwDqwXg/6BWwbLVQ1oTO5QtEKXYnfg3o4Gk4/zPoV9DDcGSjK2gz5INVi+Z77phczqerR+fEYjbqQ0dUDlqfOSPk+YNxZBg+HuIZsi/wVwTiLVL7iK6xj4kb2of48uzKL6GS38OgfztY86BJPFSDfizMc/03YK2vb+CtIeg3wn6JZvT52xk0qeFAan7mOC0FTjWGUwXdfgeUaafH7HTQ7Qx73O61wq4u5cvlPjNthJd85bXPmq/Kb03yy+V26/QV95dknR9K9lzuEsMl1GB+uGGLxEnLw6dgr/+PeVRiOBXQJ/v1KDMPFc6BhMPlAfWZ8c4mI2jvPJ6i1keL4WigHdrtOg55n7RO27E40ubxMqGdXO9bZ612GfmrcB5IOFzuuRS5PqZzIZ5ofK/AeyzhcRm7Fr7GFoO2gR04Wd0CyoivrPq2KY4OoyS4UGAzQxzyvjnLwce2OLLgTTw4901oIwEna73Na6+KP+l7kDSOss+/rHy4L2Ah65CXh8j+pr793i7Wgbx1bVE0Jh5h0+JxojFLXSOjL8ikC0AjqHFomiLPTee6gJcYZ9P3zjrPZVi7jCcpzq552vO95/tv8C2PReWr8vOvj6p6ewTiPHCZMGpjF90puCJuQ/vzuSSfj1nPkQ8EWj3hIpsknycSDpd7FnIpHs+HHnbEO1wSvBMJj8vnCPrMQ4PFdhQuyfJ5X1fwUgmko+N7ub7j0vqrParc+W+kstNDnV/NH83b5tLvd5Bu/Z9t0Y/arvR/A0/mSFM='

class TestMeasureObjects(unittest.TestCase):
    def test_01_01_load(self):
        pipeline = P.Pipeline()
        fd = StringIO.StringIO(base64.b64decode(pipeline_data))
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module,MOI.MeasureObjectIntensity))
        self.assertEqual(len(module.object_names),1)
        self.assertEqual(module.object_names[0].value,'Nuclei')
        self.assertEqual(module.image_names[0].value,'DNA')
        
    def test_02_01_supplied_measurements(self):
        """Test the get_category / get_measurements, get_measurement_images functions"""
        
        moi = MOI.MeasureObjectIntensity()
        moi.image_names[0].value='MyImage'
        moi.add_cb()
        moi.object_names[0].value = 'MyObjects1'
        moi.object_names[1].value = 'MyObjects2'
        
        self.assertEqual(moi.get_categories(None, 'MyObjects1'),[MOI.INTENSITY])
        self.assertEqual(moi.get_categories(None, 'Foo'),[])
        measurements = moi.get_measurements(None,'MyObjects1',MOI.INTENSITY)
        self.assertEqual(len(measurements),len(MOI.ALL_MEASUREMENTS))
        self.assertTrue(all([m in MOI.ALL_MEASUREMENTS for m in measurements]))
        self.assertTrue(moi.get_measurement_images(None,'MyObjects1',
                                                   MOI.INTENSITY,
                                                   MOI.MAX_INTENSITY),
                        ['MyImage'])
    
    def test_03_01_zero(self):
        """Make sure we can process a blank image"""
        ii = II.InjectImage('MyImage',numpy.zeros((10,10)))
        ii.module_num = 1
        io = II.InjectObjects('MyObjects',numpy.zeros((10,10),int))
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.image_names[0].value = 'MyImage'
        moi.object_names[0].value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        for meas_name in MOI.ALL_MEASUREMENTS:
            feature_name = "%s_%s_%s"%(MOI.INTENSITY, meas_name, 'MyImage')
            data = m.get_current_measurement('MyObjects',feature_name)
            self.assertEqual(numpy.product(data.shape),0,"Got data for feature %s"%(feature_name))
    
    def test_03_02_one(self):
        """Check measurements on a 3x3 square of 1's"""
        img = numpy.array([[0,0,0,0,0,0,0],
                           [0,0,1,1,1,0,0],
                           [0,0,1,1,1,0,0],
                           [0,0,1,1,1,0,0],
                           [0,0,0,0,0,0,0]])
        ii = II.InjectImage('MyImage',img.astype(float))
        ii.module_num = 1
        io = II.InjectObjects('MyObjects',img.astype(int))
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.image_names[0].value = 'MyImage'
        moi.object_names[0].value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        for meas_name,value in ((MOI.INTEGRATED_INTENSITY,9),
                                (MOI.MEAN_INTENSITY,1),
                                (MOI.STD_INTENSITY,0),
                                (MOI.MIN_INTENSITY,1),
                                (MOI.MAX_INTENSITY,1),
                                (MOI.INTEGRATED_INTENSITY_EDGE,8),
                                (MOI.MEAN_INTENSITY_EDGE,1),
                                (MOI.STD_INTENSITY_EDGE,0),
                                (MOI.MIN_INTENSITY_EDGE,1),
                                (MOI.MAX_INTENSITY_EDGE,1),
                                (MOI.MASS_DISPLACEMENT,0),
                                (MOI.LOWER_QUARTILE_INTENSITY,1),
                                (MOI.MEDIAN_INTENSITY,1),
                                (MOI.UPPER_QUARTILE_INTENSITY,1)):
            feature_name = "%s_%s_%s"%(MOI.INTENSITY, meas_name, 'MyImage')
            data = m.get_current_measurement('MyObjects',feature_name)
            self.assertEqual(numpy.product(data.shape),1)
            self.assertEqual(data[0],value)
        
    def test_03_03_mass_displacement(self):
        """Check the mass displacement of three squares"""
        
        labels = numpy.array([[0,0,0,0,0,0,0],
                              [0,1,1,1,1,1,0],
                              [0,1,1,1,1,1,0],
                              [0,1,1,1,1,1,0],
                              [0,1,1,1,1,1,0],
                              [0,1,1,1,1,1,0],
                              [0,0,0,0,0,0,0],
                              [0,2,2,2,2,2,0],
                              [0,2,2,2,2,2,0],
                              [0,2,2,2,2,2,0],
                              [0,2,2,2,2,2,0],
                              [0,2,2,2,2,2,0],
                              [0,0,0,0,0,0,0],
                              [0,3,3,3,3,3,0],
                              [0,3,3,3,3,3,0],
                              [0,3,3,3,3,3,0],
                              [0,3,3,3,3,3,0],
                              [0,3,3,3,3,3,0],
                              [0,0,0,0,0,0,0]])
        image = numpy.zeros(labels.shape,dtype=float)
        #
        # image # 1 has a single value in one of the corners
        # whose distance is sqrt(8) from the center
        #
        image[1,1] = 1
        # image # 2 has a single value on the top edge
        # and should have distance 2
        #
        image[7,3] = 1
        # image # 3 has a single value on the left edge
        # and should have distance 2
        image[15,1] = 1
        ii = II.InjectImage('MyImage',image)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects',labels)
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.image_names[0].value = 'MyImage'
        moi.object_names[0].value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        feature_name = '%s_%s_%s'%(MOI.INTENSITY,MOI.MASS_DISPLACEMENT,'MyImage')
        mass_displacement = m.get_current_measurement('MyObjects', feature_name)
        self.assertEqual(numpy.product(mass_displacement.shape),3)
        self.assertAlmostEqual(mass_displacement[0],math.sqrt(8.0))
        self.assertAlmostEqual(mass_displacement[1],2.0)
        self.assertAlmostEqual(mass_displacement[2],2.0)
        
    def test_03_04_quartiles(self):
        """test quartile values on a 250x250 square filled with uniform values"""
        labels = numpy.ones((250,250),int)
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(250,250))
        ii = II.InjectImage('MyImage',image)
        ii.module_num = 1
        io = II.InjectObjects('MyObjects',labels)
        io.module_num = 2
        moi = MOI.MeasureObjectIntensity()
        moi.image_names[0].value = 'MyImage'
        moi.object_names[0].value = 'MyObjects'
        moi.module_num = 3
        pipeline = P.Pipeline()
        pipeline.add_module(ii)
        pipeline.add_module(io)
        pipeline.add_module(moi)
        m = pipeline.run()
        feature_name = '%s_%s_%s'%(MOI.INTENSITY,MOI.LOWER_QUARTILE_INTENSITY,'MyImage')
        data = m.get_current_measurement('MyObjects',feature_name)
        self.assertAlmostEqual(data[0],.25,2)
        feature_name = '%s_%s_%s'%(MOI.INTENSITY,MOI.MEDIAN_INTENSITY,'MyImage')
        data = m.get_current_measurement('MyObjects',feature_name)
        self.assertAlmostEqual(data[0],.50,2)
        feature_name = '%s_%s_%s'%(MOI.INTENSITY,MOI.UPPER_QUARTILE_INTENSITY,'MyImage')
        data = m.get_current_measurement('MyObjects',feature_name)
        self.assertAlmostEqual(data[0],.75,2)
        
        