"""test_Utils - unit tests of CellProfiler.Matlab.Utils

"""

import os
import unittest
import numpy
import CellProfiler.Matlab.Utils

class TestUtils(unittest.TestCase):
    def test_00_00_Init(self):
        CellProfiler.Matlab.Utils.GetMatlabInstance()
    
    def test_01_01_LoadSimple(self):
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        matlab.test = CellProfiler.Matlab.Utils.LoadIntoMatlab({'foo':'bar'})
        self.assertEqual(matlab.test.foo,'bar')
    
    def test_01_02_LoadArray(self):
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        matlab.test = CellProfiler.Matlab.Utils.LoadIntoMatlab({'foo':numpy.zeros((3,3))})
        self.assertTrue((matlab.test.foo == numpy.zeros((3,3))).all())
    
    def test_01_03_LoadStructure(self):
        matlab = CellProfiler.Matlab.Utils.GetMatlabInstance()
        s = numpy.ndarray((1,1),dtype=([('foo','|O4'),('bar','|O4')]))
        s['foo'][0,0] = 'Hello'
        s['bar'][0,0] = 'World'
        matlab.test = CellProfiler.Matlab.Utils.LoadIntoMatlab({'foo':s})
        self.assertEqual(matlab.test.foo.foo,'Hello')
        self.assertEqual(matlab.test.foo.bar,'World')
        
if __name__ == "__main__":
    unittest.main()
        