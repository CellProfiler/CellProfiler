"""test_Image.py - test the CellProfiler.Image module
"""
import cellprofiler.cpimage
import unittest
import numpy
import math

class TestImage(unittest.TestCase):
    def test_00_00_init(self):
        cellprofiler.cpimage.Image()
    
    def test_01_01_init_image(self):
        x=cellprofiler.cpimage.Image(numpy.zeros((10,10)))
    
    def test_01_02_init_image_mask(self):
        x=cellprofiler.cpimage.Image(image=numpy.zeros((10,10)),
                                   mask=numpy.ones((10,10),dtype=numpy.bool))
    
    def test_02_01_set_image(self):
        x=cellprofiler.cpimage.Image()
        x.Image = numpy.ones((10,10))
        
    def test_02_02_set_mask(self):
        x=cellprofiler.cpimage.Image()
        x.Mask = numpy.ones((10,10))
    
    def test_03_01_image_casts(self):
        one_target  = numpy.ones((10,10),dtype=numpy.float64)
        zero_target = numpy.zeros((10,10),dtype=numpy.float64)
        tests = [(numpy.float64,0,1.0),
                 (numpy.float32,0,1.0),
                 (numpy.uint32,0,math.pow(2.0,32.0)-1),
                 (numpy.uint16,0,math.pow(2.0,16.0)-1),
                 (numpy.uint8,0,math.pow(2.0,8.0)-1),
                 (numpy.int32,-math.pow(2.0,31.0),math.pow(2.0,31.0)-1),
                 (numpy.int16,-math.pow(2.0,15.0),math.pow(2.0,15.0)-1),
                 (numpy.int8,-math.pow(2.0,7.0),math.pow(2.0,7.0)-1)]
        for dtype,zval,oval in tests:
            x=cellprofiler.cpimage.Image()
            x.set_image((one_target*zval).astype(dtype))
            self.assertTrue((x.image==zero_target).all(),msg="Failed setting %s to min"%(repr(dtype)))
            x.set_image((one_target*oval).astype(dtype))
            y=(x.image==one_target)
            self.assertTrue((x.image==one_target).all(),msg="Failed setting %s to max"%(repr(dtype)))

    def test_04_01_image_mask_missize(self):
        x = cellprofiler.cpimage.Image()
        x.image = numpy.ones((10,10))
        self.assertRaises(AssertionError,x.set_mask,numpy.ones((5,5)))
    
    def test_05_01_mask_of3D(self):
        """The mask of a 3-d image should be 2-d"""
        x=cellprofiler.cpimage.Image()
        x.image = numpy.ones((10,10,3))
        self.assertTrue(x.mask.ndim==2)
    
    def test_06_01_cropping(self):
        x = cellprofiler.cpimage.Image()
        x.image = numpy.ones((8,8))
        crop_mask = numpy.zeros((10,10),bool)
        crop_mask[1:-1,1:-1] = True
        x.crop_mask = crop_mask
        i,j = numpy.mgrid[0:10,0:10]
        test = i+j*10
        test_out = x.crop_image_similarly(test)
        self.assertTrue(numpy.all(test_out == test[1:-1,1:-1]))

    
class TestImageSetList(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.cpimage.ImageSetList()
        self.assertEqual(x.count(),0,"# of elements of an empty image set list is %d, not zero"%(x.count()))
    
    def test_01_01_add_image_set_by_number(self):
        x = cellprofiler.cpimage.ImageSetList()
        y = x.get_image_set(0)
        self.assertEqual(x.count(),1,"# of elements was %d, should be 1"%(x.count()))
        self.assertEqual(y.get_number(),0,"The image set should be #0, was %d"%(y.get_number()))
        self.assertTrue(y.get_keys().has_key("number"),"The image set was missing a number key")
        self.assertEqual(y.get_keys()["number"],0,"The number key should be zero, was %s"%(repr(y.get_keys()["number"])))
    
    def test_01_02_add_image_set_by_key(self):
        x = cellprofiler.cpimage.ImageSetList()
        key = {"key":"value"}
        y = x.get_image_set(key)
        self.assertEqual(x.count(),1,"# of elements was %d, should be 1"%(x.count()))
        self.assertEqual(y.get_number(),0,"The image set should be #0, was %d"%(y.get_number()))
        self.assertEquals(y,x.get_image_set(0),"The image set should be retrievable by index")
        self.assertEquals(y,x.get_image_set(key),"The image set should be retrievable by key")
        self.assertEquals(repr(key),repr(y.get_keys()))
        
    def test_01_03_add_two_image_sets(self):
        x = cellprofiler.cpimage.ImageSetList()
        y = x.get_image_set(0)
        z = x.get_image_set(1)
        self.assertEqual(x.count(),2,"# of elements was %d, should be 2"%(x.count()))
        self.assertEqual(y.number,0,"The image set should be #0, was %d"%(y.get_number()))
        self.assertEqual(z.number,1,"The image set should be #1, was %d"%(y.get_number()))
        self.assertEquals(y,x.get_image_set(0),"The first image set was not retrieved by index")
        self.assertEquals(z,x.get_image_set(1),"The second image set was not retrieved by index")
    
    def test_02_01_add_image_provider(self):
        x = cellprofiler.cpimage.ImageSetList()
        y = x.get_image_set(0)
        img = cellprofiler.cpimage.Image(numpy.ones((10,10)))
        def fn(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image provider")
            return img
        z = cellprofiler.cpimage.CallbackImageProvider("TestImageProvider",fn)
        y.providers.append(z)
        self.assertEquals(img,y.get_image("TestImageProvider"))
    
    def test_02_02_add_two_image_providers(self):
        x = cellprofiler.cpimage.ImageSetList()
        y = x.get_image_set(0)
        img1 = cellprofiler.cpimage.Image(numpy.ones((10,10)))
        def fn1(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image set")
            return img1
        img2 = cellprofiler.cpimage.Image(numpy.ones((5,5)))
        def fn2(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image set")
            return img2
        y.providers.append(cellprofiler.cpimage.CallbackImageProvider("IP1",fn1))
        y.providers.append(cellprofiler.cpimage.CallbackImageProvider("IP2",fn2))
        self.assertEquals(img1,y.get_image("IP1"),"Failed to get correct first image")
        self.assertEquals(img2,y.get_image("IP2"),"Failed to get correct second image")

if __name__ == "__main__":
    unittest.main()
