from __future__ import absolute_import
from __future__ import print_function
from six.moves import range

__version__ = "$Revision$"

import numpy
import struct
import unittest
import time

import centrosome.propagate
from centrosome import _propagate

# class Test_Propagate(unittest.TestCase):
#     def test_01_01_test_convert_to_ints(self):
#         """Make sure the ordering in convert_to_ints is invariant"""
#         numpy.random.seed(0)
#         for i in range(10000):
#             bytes1 = numpy.random.bytes(8)
#             d1 = struct.unpack("d",bytes1)[0]
#             bytes2 = numpy.random.bytes(8)
#             d2 = struct.unpack("d",bytes2)[0]
#             if d1 != d1 or d2 != d2: # NaN test
#                 continue
#             t1 = _propagate.convert_to_ints(d1)
#             t2 = _propagate.convert_to_ints(d2)
#             self.assertEqual((d1<d2), (t1<t2),"%s:%s %f<%f != (%d,%d)<(%d,%d)"%(struct.unpack("BBBBBBBB",bytes1),struct.unpack("BBBBBBBB",bytes2),d1,d2,t1[0],t1[1],t2[0],t2[1]))
#
#     def test_01_02_test_convert_to_ints(self):
#         """Test particular corner cases for convert_to_ints"""
#         for bytes1,bytes2 in (((0x80,0,0,0   ,0,0,0,0x7f),(0x80,0,0,0   ,0,0,0,0x80)),
#                               ((0   ,0,0,0x80,0,0,0,0x80),(   0,0,0,0x7f,0,0,0,0x80)),
#                               ((0   ,0,0,0   ,0,0,0,0x81),(0x80,0,0,0   ,0,0,0,0x80)),
#                               ((0x02,0,0,0   ,0,0,0,0x81),(0x04,0,0,0   ,0,0,0,0x81)),
#                               ((0x02,0,0,0   ,0,0,0,0x01),(0x04,0,0,0   ,0,0,0,0x01))):
#             d1 = struct.unpack("d",struct.pack("BBBBBBBB",bytes1[0],bytes1[1],bytes1[2],bytes1[3],bytes1[4],bytes1[5],bytes1[6],bytes1[7]))[0]
#             d2 = struct.unpack("d",struct.pack("BBBBBBBB",bytes2[0],bytes2[1],bytes2[2],bytes2[3],bytes2[4],bytes2[5],bytes2[6],bytes2[7]))[0]
#             if d1 != d1 or d2 != d2: # NaN test
#                 continue
#             t1 = _propagate.convert_to_ints(d1)
#             t2 = _propagate.convert_to_ints(d2)
#             self.assertEqual((d1<d2), (t1<t2),"%s:%s %f<%f != (%d,%d)<(%d,%d)"%(repr(bytes1),repr(bytes2),d1,d2,t1[0],t1[1],t2[0],t2[1]))


class TestPropagate(unittest.TestCase):
    def test_01_01_zeros(self):
        image = numpy.zeros((10, 10))
        labels = numpy.zeros((10, 10), int)
        mask = numpy.ones((10, 10), bool)
        result, distances = centrosome.propagate.propagate(image, labels, mask, 1.0)
        self.assertTrue(numpy.all(result == 0))

    def test_01_02_one_label(self):
        image = numpy.zeros((10, 10))
        mask = numpy.ones((10, 10), bool)
        labels = numpy.zeros((10, 10), int)
        labels[5, 5] = 1
        result, distances = centrosome.propagate.propagate(image, labels, mask, 1.0)
        self.assertTrue(numpy.all(result == 1))

    def test_01_03_two_labels(self):
        image = numpy.zeros((10, 10))
        labels = numpy.zeros((10, 10), int)
        mask = numpy.ones((10, 10), bool)
        labels[0, 5] = 1
        labels[9, 5] = 2
        result, distances = centrosome.propagate.propagate(image, labels, mask, 1.0)
        self.assertTrue(numpy.all(result[:5, :] == 1))
        self.assertTrue(numpy.all(result[5:, :] == 2))

    def test_01_04_barrier(self):
        image = numpy.zeros((10, 10))
        image[5, :5] = 1
        image[:6, 5] = 1
        labels = numpy.zeros((10, 10), int)
        labels[0, 0] = 1
        labels[9, 0] = 2
        mask = numpy.ones((10, 10), bool)
        result, distances = centrosome.propagate.propagate(image, labels, mask, 0.1)
        x, y = numpy.mgrid[0:10, 0:10]
        self.assertTrue(numpy.all(result[numpy.logical_and(x < 5, y < 5)] == 1))
        self.assertTrue(numpy.all(result[numpy.logical_or(x > 5, y > 5)] == 2))

    def test_01_05_leaky_barrier(self):
        image = numpy.zeros((10, 10))
        image[4, 1:4] = 1
        image[:4, 4] = 1
        labels = numpy.zeros((10, 10), int)
        labels[0, 0] = 1
        labels[9, 0] = 2
        mask = numpy.ones((10, 10), bool)
        result, distances = centrosome.propagate.propagate(image, labels, mask, 0.1)
        x, y = numpy.mgrid[0:10, 0:10]
        self.assertTrue(numpy.all(result[numpy.logical_and(x < 4, y < 4)] == 1))
        self.assertTrue(result[4, 0] == 1)

    def test_01_06_mask(self):
        image = numpy.zeros((10, 10))
        labels = numpy.zeros((10, 10), int)
        mask = numpy.ones((10, 10), bool)
        mask[2, 2] = False
        mask[7, 2] = False
        labels[0, 5] = 1
        labels[9, 5] = 2
        result, distances = centrosome.propagate.propagate(image, labels, mask, 1.0)
        self.assertEqual(result[2, 2], 0)
        self.assertEqual(result[7, 2], 0)
        x, y = numpy.mgrid[0:10, 0:10]
        mask_one = x < 5
        mask_one[2, 2] = False
        mask_two = x >= 5
        mask_two[7, 2] = False
        self.assertTrue(numpy.all(result[mask_one] == 1))
        self.assertTrue(numpy.all(result[mask_two] == 2))

    def test_02_01_time_propagate(self):
        image = numpy.random.uniform(size=(1000, 1000))
        x_coords = numpy.random.uniform(low=0, high=1000, size=(300,)).astype(int)
        y_coords = numpy.random.uniform(low=0, high=1000, size=(300,)).astype(int)
        labels = numpy.zeros((1000, 1000), dtype=int)
        labels[x_coords, y_coords] = numpy.arange(1, 301)
        mask = numpy.ones((1000, 1000), bool)
        t1 = time.process_time()
        result, distances = centrosome.propagate.propagate(image, labels, mask, 1.0)
        t2 = time.process_time()
        print("Running time: %f sec" % (t2 - t1))
