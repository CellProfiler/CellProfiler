import cellprofiler.matlab.utilities
import numpy
import unittest


class TestUtils(unittest.TestCase):
    def test_02_001_EncapsulateString(self):
        a = cellprofiler.matlab.utilities.encapsulate_string('Hello')

        self.assertTrue(a.shape == (1,))

        self.assertTrue(a.dtype.kind == 'S')

        self.assertTrue(a[0] == 'Hello')

    def test_02_001_EncapsulateUnicode(self):
        a = cellprofiler.matlab.utilities.encapsulate_string(u'Hello')

        self.assertTrue(a.shape == (1,))

        self.assertTrue(a.dtype.kind == 'U')

        self.assertTrue(a[0] == u'Hello')

    def test_02_01_EncapsulateCell(self):
        cell = numpy.ndarray((1, 1), dtype=object)

        cell[0, 0] = u'Hello, world'

        cellprofiler.matlab.utilities.encapsulate_strings_in_arrays(cell)

        self.assertTrue(isinstance(cell[0, 0], numpy.ndarray))

        self.assertTrue(cell[0, 0][0] == u'Hello, world')

    def test_02_02_EncapsulateStruct(self):
        struct = numpy.ndarray((1, 1), dtype=[('foo', object)])

        struct['foo'][0, 0] = u'Hello, world'

        cellprofiler.matlab.utilities.encapsulate_strings_in_arrays(struct)

        self.assertTrue(isinstance(struct['foo'][0, 0], numpy.ndarray))

        self.assertTrue(struct['foo'][0, 0][0] == u'Hello, world')

    def test_02_03_EncapsulateCellInStruct(self):
        struct = numpy.ndarray((1, 1), dtype=[('foo', object)])

        cell = numpy.ndarray((1, 1), dtype=object)

        cell[0, 0] = u'Hello, world'

        struct['foo'][0, 0] = cell

        cellprofiler.matlab.utilities.encapsulate_strings_in_arrays(struct)

        self.assertTrue(isinstance(cell[0, 0], numpy.ndarray))

        self.assertTrue(cell[0, 0][0] == u'Hello, world')

    def test_02_04_EncapsulateStructInCell(self):
        struct = numpy.ndarray((1, 1), dtype=[('foo', object)])

        cell = numpy.ndarray((1, 1), dtype=object)

        cell[0, 0] = struct

        struct['foo'][0, 0] = u'Hello, world'

        cellprofiler.matlab.utilities.encapsulate_strings_in_arrays(cell)

        self.assertTrue(isinstance(struct['foo'][0, 0], numpy.ndarray))

        self.assertTrue(struct['foo'][0, 0][0] == u'Hello, world')
