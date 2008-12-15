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
    
    def test_02_001_EncapsulateString(self):
        a = CellProfiler.Matlab.Utils.EncapsulateString('Hello')
        self.assertTrue(a.shape == (1,))
        self.assertTrue(a.dtype.kind == 'S')
        self.assertTrue(a[0] == 'Hello')
        
    def test_02_001_EncapsulateUnicode(self):
        a = CellProfiler.Matlab.Utils.EncapsulateString(u'Hello')
        self.assertTrue(a.shape == (1,))
        self.assertTrue(a.dtype.kind == 'U')
        self.assertTrue(a[0] == u'Hello')
        
    def test_02_01_EncapsulateCell(self):
        cell = numpy.ndarray((1,1),dtype=object)
        cell[0,0] = u'Hello, world'
        CellProfiler.Matlab.Utils.EncapsulateStringsInArrays(cell)
        self.assertTrue(isinstance(cell[0,0],numpy.ndarray))
        self.assertTrue(cell[0,0][0] == u'Hello, world')
    
    def test_02_02_EncapsulateStruct(self):
        struct = numpy.ndarray((1,1),dtype=[('foo',object)])
        struct['foo'][0,0] = u'Hello, world'
        CellProfiler.Matlab.Utils.EncapsulateStringsInArrays(struct)
        self.assertTrue(isinstance(struct['foo'][0,0],numpy.ndarray))
        self.assertTrue(struct['foo'][0,0][0] == u'Hello, world')
    
    def test_02_03_EncapsulateCellInStruct(self):
        struct = numpy.ndarray((1,1),dtype=[('foo',object)])
        cell = numpy.ndarray((1,1),dtype=object)
        cell[0,0] = u'Hello, world'
        struct['foo'][0,0] = cell
        CellProfiler.Matlab.Utils.EncapsulateStringsInArrays(struct)
        self.assertTrue(isinstance(cell[0,0],numpy.ndarray))
        self.assertTrue(cell[0,0][0] == u'Hello, world')

    def test_02_04_EncapsulateStructInCell(self):
        struct = numpy.ndarray((1,1),dtype=[('foo',object)])
        cell = numpy.ndarray((1,1),dtype=object)
        cell[0,0] = struct
        struct['foo'][0,0] = u'Hello, world'
        CellProfiler.Matlab.Utils.EncapsulateStringsInArrays(cell)
        self.assertTrue(isinstance(struct['foo'][0,0],numpy.ndarray))
        self.assertTrue(struct['foo'][0,0][0] == u'Hello, world')

if __name__ == "__main__":
    unittest.main()
        