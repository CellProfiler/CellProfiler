from __future__ import absolute_import
import numpy
import unittest

import centrosome.outline as OL


class TestOutline(unittest.TestCase):
    def test_00_00_zeros(self):
        x = numpy.zeros((10, 10), int)
        result = OL.outline(x)
        self.assertTrue(numpy.all(x == result))

    def test_01_01_single(self):
        x = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        e = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        result = OL.outline(x)
        self.assertTrue(numpy.all(result == e))

    def test_01_02_two_disjoint(self):
        x = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        e = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 2, 0, 2, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        result = OL.outline(x)
        self.assertTrue(numpy.all(result == e))

    def test_01_03_touching(self):
        x = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        e = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 2, 0, 2, 0, 0],
                [0, 0, 2, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        result = OL.outline(x)
        self.assertTrue(numpy.all(result == e))

    def test_02_04_edge(self):
        x = numpy.array(
            [
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        e = numpy.array(
            [
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        result = OL.outline(x)
        self.assertTrue(numpy.all(result == e))

    def test_02_05_diagonal(self):
        x = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
            ]
        )
        e = numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
            ]
        )
        result = OL.outline(x)
        self.assertTrue(numpy.all(result == e))
