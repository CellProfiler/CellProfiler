"""test_Image.py - test the CellProfiler.Image module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import cellprofiler.cpimage as cpi
import unittest
import numpy as np
import math

class TestImage(unittest.TestCase):
    def test_00_00_init(self):
        cpi.Image()
    
    def test_01_01_init_image(self):
        x=cpi.Image(np.zeros((10,10)))
    
    def test_01_02_init_image_mask(self):
        x=cpi.Image(image=np.zeros((10,10)),
                                   mask=np.ones((10,10),dtype=np.bool))
    
    def test_02_01_set_image(self):
        x=cpi.Image()
        x.Image = np.ones((10,10))
        
    def test_02_02_set_mask(self):
        x=cpi.Image()
        x.Mask = np.ones((10,10))
    
    def test_03_01_image_casts(self):
        one_target  = np.ones((10,10),dtype=np.float64)
        zero_target = np.zeros((10,10),dtype=np.float64)
        tests = [(np.float64,0,1.0),
                 (np.float32,0,1.0),
                 (np.uint32,0,math.pow(2.0,32.0)-1),
                 (np.uint16,0,math.pow(2.0,16.0)-1),
                 (np.uint8,0,math.pow(2.0,8.0)-1),
                 (np.int32,-math.pow(2.0,31.0),math.pow(2.0,31.0)-1),
                 (np.int16,-math.pow(2.0,15.0),math.pow(2.0,15.0)-1),
                 (np.int8,-math.pow(2.0,7.0),math.pow(2.0,7.0)-1)]
        for dtype,zval,oval in tests:
            x=cpi.Image()
            x.set_image((one_target*zval).astype(dtype))
            self.assertTrue((x.image==zero_target).all(),msg="Failed setting %s to min"%(repr(dtype)))
            x.set_image((one_target*oval).astype(dtype))
            y=(x.image==one_target)
            self.assertTrue((x.image==one_target).all(),msg="Failed setting %s to max"%(repr(dtype)))

    def test_04_01_image_mask_missize(self):
        x = cpi.Image()
        x.image = np.ones((10,10))
        self.assertRaises(AssertionError,x.set_mask,np.ones((5,5)))
    
    def test_05_01_mask_of3D(self):
        """The mask of a 3-d image should be 2-d"""
        x=cpi.Image()
        x.image = np.ones((10,10,3))
        self.assertTrue(x.mask.ndim==2)
    
    def test_06_01_cropping(self):
        x = cpi.Image()
        x.image = np.ones((7,7))
        crop_mask = np.zeros((10,10),bool)
        crop_mask[2:-1,1:-2] = True
        x.crop_mask = crop_mask
        i,j = np.mgrid[0:10,0:10]
        test = i+j*10
        test_out = x.crop_image_similarly(test)
        self.assertTrue(np.all(test_out == test[2:-1,1:-2]))
        
    def test_07_02_cache(self):
        import h5py
        import os
        import tempfile
        
        r = np.random.RandomState()
        r.seed(72)
        test_cases = (
            { "image": r.uniform(size=(10, 20)) },
            { "image": r.uniform(size=(20, 10, 3)) },
            { "image": r.uniform(size=(10, 20)),
              "mask": r.uniform(size=(10, 20)) > .5},
            { "image": r.uniform(size=(10, 20)),
              "crop_mask": np.all(np.mgrid[0:10, 0:20] > 3, 0) })
        h, path = tempfile.mkstemp(suffix=".h5")
        hdf_file = h5py.File(path, "w")
        os.close(h)
        
        for test_case in test_cases:
            image = cpi.Image(**test_case)
            image.cache("foo", hdf_file)
            expected = test_case["image"].astype(np.float32)
            np.testing.assert_array_equal(image.pixel_data, expected)
            if "mask" in test_case:
                np.testing.assert_array_equal(image.mask, test_case["mask"])
            else:
                np.testing.assert_equal(image.mask, True)
                np.testing.assert_equal(
                    tuple(image.mask.shape), tuple(expected.shape[:2]))
            if "crop_mask" in test_case:
                np.testing.assert_array_equal(
                    image.crop_mask, test_case["crop_mask"])
            else:
                np.testing.assert_equal(
                    tuple(image.crop_mask.shape), tuple(expected.shape[:2]))
            
                
IMAGE_NAME = "image"
class TestImageSet(unittest.TestCase):
    def test_01_01_add(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20))))
        self.assertEqual(len(x.providers), 1)
        self.assertEqual(x.providers[0].name, IMAGE_NAME)
        
    def test_01_02_get_image(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20))))
        image = x.get_image(IMAGE_NAME)
        self.assertEqual(tuple(image.pixel_data.shape), (10,20))
        
    def test_02_01_must_be_binary(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20), bool)))
        image = x.get_image(IMAGE_NAME, must_be_binary=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10,20))
        
    def test_02_02_must_be_binary_throws(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20), float)))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, 
                          must_be_binary=True)
        
    def test_03_01_must_be_gray(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20), float)))
        image = x.get_image(IMAGE_NAME, must_be_grayscale=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10,20))
        
    def test_03_02_must_be_gray_throws(self):
        x = cpi.ImageSet(0, {}, {})
        np.random.seed(22)
        x.add(IMAGE_NAME, cpi.Image(np.random.uniform(size=(10,20,3))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, 
                          must_be_grayscale=True)
        
    def test_03_03_must_be_gray_color(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20,3), float)))
        image = x.get_image(IMAGE_NAME, must_be_grayscale=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10,20))
        
    def test_04_01_must_be_color(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20,3), float)))
        image = x.get_image(IMAGE_NAME, must_be_color=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10,20,3))
        
    def test_04_02_must_be_color_throws(self):
        x = cpi.ImageSet(0, {}, {})
        np.random.seed(22)
        x.add(IMAGE_NAME, cpi.Image(np.random.uniform(size=(10,20))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, 
                          must_be_color=True)
        
    def test_05_01_must_be_rgb(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10, 20, 3), float)))
        image = x.get_image(IMAGE_NAME, must_be_rgb=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10,20,3))
        
    def test_05_02_must_be_rgb_throws_gray(self):
        x = cpi.ImageSet(0, {}, {})
        np.random.seed(22)
        x.add(IMAGE_NAME, cpi.Image(np.random.uniform(size=(10,20))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, 
                          must_be_rgb=True)
        
    def test_05_03_must_be_rgb_throws_5_channel(self):
        x = cpi.ImageSet(0, {}, {})
        np.random.seed(22)
        x.add(IMAGE_NAME, cpi.Image(np.random.uniform(size=(10,20,5))))
        self.assertRaises(ValueError, x.get_image, IMAGE_NAME, 
                          must_be_rgb=True)
        
    def test_05_04_must_be_rgb_alpha(self):
        x = cpi.ImageSet(0, {}, {})
        x.add(IMAGE_NAME, cpi.Image(np.zeros((10,20,4), float)))
        image = x.get_image(IMAGE_NAME, must_be_rgb=True)
        self.assertEqual(tuple(image.pixel_data.shape), (10,20,3))
        
        
class TestImageSetList(unittest.TestCase):
    def test_00_00_init(self):
        x = cpi.ImageSetList()
        self.assertEqual(x.count(),0,"# of elements of an empty image set list is %d, not zero"%(x.count()))
    
    def test_01_01_add_image_set_by_number(self):
        x = cpi.ImageSetList()
        y = x.get_image_set(0)
        self.assertEqual(x.count(),1,"# of elements was %d, should be 1"%(x.count()))
        self.assertEqual(y.get_number(),0,"The image set should be #0, was %d"%(y.get_number()))
        self.assertTrue(y.get_keys().has_key("number"),"The image set was missing a number key")
        self.assertEqual(y.get_keys()["number"],0,"The number key should be zero, was %s"%(repr(y.get_keys()["number"])))
    
    def test_01_02_add_image_set_by_key(self):
        x = cpi.ImageSetList()
        key = {"key":"value"}
        y = x.get_image_set(key)
        self.assertEqual(x.count(),1,"# of elements was %d, should be 1"%(x.count()))
        self.assertEqual(y.get_number(),0,"The image set should be #0, was %d"%(y.get_number()))
        self.assertEquals(y,x.get_image_set(0),"The image set should be retrievable by index")
        self.assertEquals(y,x.get_image_set(key),"The image set should be retrievable by key")
        self.assertEquals(repr(key),repr(y.get_keys()))
        
    def test_01_03_add_two_image_sets(self):
        x = cpi.ImageSetList()
        y = x.get_image_set(0)
        z = x.get_image_set(1)
        self.assertEqual(x.count(),2,"# of elements was %d, should be 2"%(x.count()))
        self.assertEqual(y.number,0,"The image set should be #0, was %d"%(y.get_number()))
        self.assertEqual(z.number,1,"The image set should be #1, was %d"%(y.get_number()))
        self.assertEquals(y,x.get_image_set(0),"The first image set was not retrieved by index")
        self.assertEquals(z,x.get_image_set(1),"The second image set was not retrieved by index")
    
    def test_02_01_add_image_provider(self):
        x = cpi.ImageSetList()
        y = x.get_image_set(0)
        img = cpi.Image(np.ones((10,10)))
        def fn(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image provider")
            return img
        z = cpi.CallbackImageProvider("TestImageProvider",fn)
        y.providers.append(z)
        self.assertEquals(img,y.get_image("TestImageProvider"))
    
    def test_02_02_add_two_image_providers(self):
        x = cpi.ImageSetList()
        y = x.get_image_set(0)
        img1 = cpi.Image(np.ones((10,10)))
        def fn1(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image set")
            return img1
        img2 = cpi.Image(np.ones((5,5)))
        def fn2(image_set,image_provider):
            self.assertEquals(y,image_set,"Callback was not called with the correct image set")
            return img2
        y.providers.append(cpi.CallbackImageProvider("IP1",fn1))
        y.providers.append(cpi.CallbackImageProvider("IP2",fn2))
        self.assertEquals(img1,y.get_image("IP1"),"Failed to get correct first image")
        self.assertEquals(img2,y.get_image("IP2"),"Failed to get correct second image")
    
    def test_03_01_serialize_no_key(self):
        '''Serialize an image list with no keys in the image sets'''
        x = cpi.ImageSetList()
        for i in range(5):
            x.get_image_set(i)
        s = x.save_state()
        
        y = cpi.ImageSetList()
        y.load_state(s)
        self.assertEquals(y.count(), 5)
    
    def test_03_02_serialize_key(self):
        x = cpi.ImageSetList()
        values = (('A','B'),('C','D'),('E','F'))
        for value1, value2 in values:
            d = { 'K1':value1, 'K2':value2 }
            x.get_image_set(d)
        s = x.save_state()
        
        y = cpi.ImageSetList()
        y.load_state(s)
        self.assertEquals(y.count(), len(values))
        for i in range(len(values)):
            image_set = y.get_image_set(i)
            self.assertTrue(isinstance(image_set, cpi.ImageSet))
            value1, value2 = values[i]
            for key, value in (('K1',value1),('K2',value2)):
                self.assertEqual(image_set.keys[key], value)
    
    def test_03_03_serialize_legacy_fields(self):
        x = cpi.ImageSetList()
        for i in range(5):
            x.get_image_set(i)
        d = { 'foo':'bar', 'test':'suite' }
        x.legacy_fields['dictionary'] = d
        s = x.save_state()
        
        y = cpi.ImageSetList()
        y.load_state(s)
        self.assertEquals(y.count(), 5)
        self.assertTrue(y.legacy_fields.has_key('dictionary'))
        for key in d.keys():
            self.assertTrue(y.legacy_fields['dictionary'].has_key(key))
            self.assertEqual(y.legacy_fields['dictionary'][key], d[key])

