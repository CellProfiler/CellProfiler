import imaging.eukaryote.image.image as image
import math
import numpy as np
import unittest

IMAGE_NAME = "image"


class TestImage(unittest.TestCase):
    def test_00_00_init(self):
        image.Image()

    def test_01_01_init_image(self):
        x = image.Image(np.zeros((10, 10)))

    def test_01_02_init_image_mask(self):
        x = image.Image(image=np.zeros((10, 10)),
                      mask=np.ones((10, 10), dtype=np.bool))

    def test_02_01_set_image(self):
        x = image.Image()
        x.Image = np.ones((10, 10))

    def test_02_02_set_mask(self):
        x = image.Image()
        x.Mask = np.ones((10, 10))

    def test_03_01_image_casts(self):
        one_target = np.ones((10, 10), dtype=np.float64)
        zero_target = np.zeros((10, 10), dtype=np.float64)
        tests = [(np.float64, 0, 1.0),
                 (np.float32, 0, 1.0),
                 (np.uint32, 0, math.pow(2.0, 32.0) - 1),
                 (np.uint16, 0, math.pow(2.0, 16.0) - 1),
                 (np.uint8, 0, math.pow(2.0, 8.0) - 1),
                 (np.int32, -math.pow(2.0, 31.0), math.pow(2.0, 31.0) - 1),
                 (np.int16, -math.pow(2.0, 15.0), math.pow(2.0, 15.0) - 1),
                 (np.int8, -math.pow(2.0, 7.0), math.pow(2.0, 7.0) - 1)]
        for dtype, zval, oval in tests:
            x = image.Image()
            x.set_image((one_target * zval).astype(dtype))
            self.assertTrue((x.image == zero_target).all(), msg="Failed setting %s to min" % (repr(dtype)))
            x.set_image((one_target * oval).astype(dtype))
            y = (x.image == one_target)
            self.assertTrue((x.image == one_target).all(), msg="Failed setting %s to max" % (repr(dtype)))

    def test_04_01_image_mask_missize(self):
        x = image.Image()
        x.image = np.ones((10, 10))
        self.assertRaises(AssertionError, x.set_mask, np.ones((5, 5)))

    def test_05_01_mask_of3D(self):
        """The mask of a 3-d image should be 2-d"""
        x = image.Image()
        x.image = np.ones((10, 10, 3))
        self.assertTrue(x.mask.ndim == 2)

    def test_06_01_cropping(self):
        x = image.Image()
        x.image = np.ones((7, 7))
        crop_mask = np.zeros((10, 10), bool)
        crop_mask[2:-1, 1:-2] = True
        x.crop_mask = crop_mask
        i, j = np.mgrid[0:10, 0:10]
        test = i + j * 10
        test_out = x.crop_image_similarly(test)
        self.assertTrue(np.all(test_out == test[2:-1, 1:-2]))

    def test_07_02_cache(self):
        import h5py
        import os
        import tempfile

        r = np.random.RandomState()
        r.seed(72)
        test_cases = (
            {"image": r.uniform(size=(10, 20))},
            {"image": r.uniform(size=(20, 10, 3))},
            {"image": r.uniform(size=(10, 20)),
             "mask": r.uniform(size=(10, 20)) > .5},
            {"image": r.uniform(size=(10, 20)),
             "crop_mask": np.all(np.mgrid[0:10, 0:20] > 3, 0)})
        h, path = tempfile.mkstemp(suffix=".h5")
        hdf_file = h5py.File(path, "w")
        os.close(h)

        for test_case in test_cases:
            sample = image.Image(**test_case)
            sample.cache("foo", hdf_file)
            expected = test_case["image"].astype(np.float32)
            np.testing.assert_array_equal(sample.pixel_data, expected)
            if "mask" in test_case:
                np.testing.assert_array_equal(sample.mask, test_case["mask"])
            else:
                np.testing.assert_equal(sample.mask, True)
                np.testing.assert_equal(
                    tuple(sample.mask.shape), tuple(expected.shape[:2]))
            if "crop_mask" in test_case:
                np.testing.assert_array_equal(
                    sample.crop_mask, test_case["crop_mask"])
            else:
                np.testing.assert_equal(
                    tuple(sample.crop_mask.shape), tuple(expected.shape[:2]))
