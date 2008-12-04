"""test_Image.py - test the CellProfiler.Image module
"""
import CellProfiler.Image
import unittest
import numpy
import math

class TestImage(unittest.TestCase):
    def test_00_00_Init(self):
        CellProfiler.Image.Image()
    
    def test_01_01_InitImage(self):
        x=CellProfiler.Image.Image(numpy.zeros((10,10)))
    
    def test_01_02_InitImageMask(self):
        x=CellProfiler.Image.Image(image=numpy.zeros((10,10)),
                                   mask=numpy.ones((10,10),dtype=numpy.bool))
    
    def test_02_01_SetImage(self):
        x=CellProfiler.Image.Image()
        x.Image = numpy.ones((10,10))
        
    def test_02_02_SetMask(self):
        x=CellProfiler.Image.Image()
        x.Mask = numpy.ones((10,10))
    
    def test_03_01_ImageCasts(self):
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
            x=CellProfiler.Image.Image()
            x.SetImage((one_target*zval).astype(dtype))
            self.assertTrue((x.Image==zero_target).all(),msg="Failed setting %s to min"%(repr(dtype)))
            x.SetImage((one_target*oval).astype(dtype))
            y=(x.Image==one_target)
            self.assertTrue((x.Image==one_target).all(),msg="Failed setting %s to max"%(repr(dtype)))

    def test_04_01_ImageMaskMissize(self):
        x = CellProfiler.Image.Image()
        x.Image = numpy.ones((10,10))
        self.assertRaises(AssertionError,x.SetMask,numpy.ones((5,5)))
    
class TestImageSetList(unittest.TestCase):
    def test_00_00_Init(self):
        x = CellProfiler.Image.ImageSetList()
        self.assertEqual(x.Count(),0,"# of elements of an empty image set list is %d, not zero"%(x.Count()))
    
    def test_01_01_AddImageSetByNumber(self):
        x = CellProfiler.Image.ImageSetList()
        y = x.GetImageSet(0)
        self.assertEqual(x.Count(),1,"# of elements was %d, should be 1"%(x.Count()))
        self.assertEqual(y.GetNumber(),0,"The image set should be #0, was %d"%(y.GetNumber()))
        self.assertTrue(y.GetKeys().has_key("number"),"The image set was missing a number key")
        self.assertEqual(y.GetKeys()["number"],0,"The number key should be zero, was %s"%(repr(y.GetKeys()["number"])))
    
    def test_01_02_AddImageSetByKey(self):
        x = CellProfiler.Image.ImageSetList()
        key = {"key":"value"}
        y = x.GetImageSet(key)
        self.assertEqual(x.Count(),1,"# of elements was %d, should be 1"%(x.Count()))
        self.assertEqual(y.GetNumber(),0,"The image set should be #0, was %d"%(y.GetNumber()))
        self.assertEquals(y,x.GetImageSet(0),"The image set should be retrievable by index")
        self.assertEquals(y,x.GetImageSet(key),"The image set should be retrievable by key")
        self.assertEquals(repr(key),repr(y.GetKeys()))
        
    def test_01_03_AddTwoImageSets(self):
        x = CellProfiler.Image.ImageSetList()
        y = x.GetImageSet(0)
        z = x.GetImageSet(1)
        self.assertEqual(x.Count(),2,"# of elements was %d, should be 2"%(x.Count()))
        self.assertEqual(y.Number,0,"The image set should be #0, was %d"%(y.GetNumber()))
        self.assertEqual(z.Number,1,"The image set should be #1, was %d"%(y.GetNumber()))
        self.assertEquals(y,x.GetImageSet(0),"The first image set was not retrieved by index")
        self.assertEquals(z,x.GetImageSet(1),"The second image set was not retrieved by index")
    
    def test_02_01_AddImageProvider(self):
        x = CellProfiler.Image.ImageSetList()
        y = x.GetImageSet(0)
        img = CellProfiler.Image.Image(numpy.ones((10,10)))
        def fn(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image provider")
            return img
        z = CellProfiler.Image.CallbackImageProvider("TestImageProvider",fn)
        y.Providers.append(z)
        self.assertEquals(img,y.GetImage("TestImageProvider"))
    
    def test_02_02_AddTwoImageProviders(self):
        x = CellProfiler.Image.ImageSetList()
        y = x.GetImageSet(0)
        img1 = CellProfiler.Image.Image(numpy.ones((10,10)))
        def fn1(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image set")
            return img1
        img2 = CellProfiler.Image.Image(numpy.ones((5,5)))
        def fn2(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image set")
            return img2
        y.Providers.append(CellProfiler.Image.CallbackImageProvider("IP1",fn1))
        y.Providers.append(CellProfiler.Image.CallbackImageProvider("IP2",fn2))
        self.assertEquals(img1,y.GetImage("IP1"),"Failed to get correct first image")
        self.assertEquals(img2,y.GetImage("IP2"),"Failed to get correct second image")

if __name__ == "__main__":
    unittest.main()
