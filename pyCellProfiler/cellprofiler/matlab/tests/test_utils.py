"""test_Utils - unit tests of CellProfiler.Matlab.Utils

"""

import os
import unittest
import numpy
import cellprofiler.matlab.cputils as u

class TestUtils(unittest.TestCase):
    def test_00_00_init(self):
        u.get_matlab_instance()
    
    def test_01_01_load_simple(self):
        matlab = u.get_matlab_instance()
        matlab.test = u.load_into_matlab({'foo':'bar'})
        self.assertEqual(matlab.test.foo,'bar')
    
    def test_01_02_LoadArray(self):
        matlab = u.get_matlab_instance()
        matlab.test = u.load_into_matlab({'foo':numpy.zeros((3,3))})
        self.assertTrue((matlab.test.foo == numpy.zeros((3,3))).all())
    
    def test_01_03_LoadStructure(self):
        matlab = u.get_matlab_instance()
        s = numpy.ndarray((1,1),dtype=([('foo','|O4'),('bar','|O4')]))
        s['foo'][0,0] = 'Hello'
        s['bar'][0,0] = 'World'
        matlab.test = u.load_into_matlab({'foo':s})
        self.assertEqual(matlab.test.foo.foo,'Hello')
        self.assertEqual(matlab.test.foo.bar,'World')
    
    def test_02_001_EncapsulateString(self):
        a = u.encapsulate_string('Hello')
        self.assertTrue(a.shape == (1,))
        self.assertTrue(a.dtype.kind == 'S')
        self.assertTrue(a[0] == 'Hello')
        
    def test_02_001_EncapsulateUnicode(self):
        a = u.encapsulate_string(u'Hello')
        self.assertTrue(a.shape == (1,))
        self.assertTrue(a.dtype.kind == 'U')
        self.assertTrue(a[0] == u'Hello')
        
    def test_02_01_EncapsulateCell(self):
        cell = numpy.ndarray((1,1),dtype=object)
        cell[0,0] = u'Hello, world'
        u.encapsulate_strings_in_arrays(cell)
        self.assertTrue(isinstance(cell[0,0],numpy.ndarray))
        self.assertTrue(cell[0,0][0] == u'Hello, world')
    
    def test_02_02_EncapsulateStruct(self):
        struct = numpy.ndarray((1,1),dtype=[('foo',object)])
        struct['foo'][0,0] = u'Hello, world'
        u.encapsulate_strings_in_arrays(struct)
        self.assertTrue(isinstance(struct['foo'][0,0],numpy.ndarray))
        self.assertTrue(struct['foo'][0,0][0] == u'Hello, world')
    
    def test_02_03_EncapsulateCellInStruct(self):
        struct = numpy.ndarray((1,1),dtype=[('foo',object)])
        cell = numpy.ndarray((1,1),dtype=object)
        cell[0,0] = u'Hello, world'
        struct['foo'][0,0] = cell
        u.encapsulate_strings_in_arrays(struct)
        self.assertTrue(isinstance(cell[0,0],numpy.ndarray))
        self.assertTrue(cell[0,0][0] == u'Hello, world')

    def test_02_04_EncapsulateStructInCell(self):
        struct = numpy.ndarray((1,1),dtype=[('foo',object)])
        cell = numpy.ndarray((1,1),dtype=object)
        cell[0,0] = struct
        struct['foo'][0,0] = u'Hello, world'
        u.encapsulate_strings_in_arrays(cell)
        self.assertTrue(isinstance(struct['foo'][0,0],numpy.ndarray))
        self.assertTrue(struct['foo'][0,0][0] == u'Hello, world')

if __name__ == "__main__":
    unittest.main()
        
