"""test_mode.py - test the "mode" function

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import logging
logger = logging.getLogger(__name__)
import numpy as np
from cellprofiler.cpmath.mode import mode
import unittest

class TestMode(unittest.TestCase):
    def test_00_00_empty(self):
        self.assertEqual(len(mode(np.zeros(0))), 0)
        
    def test_01_01_single_mode(self):
        result = mode([1, 1, 2])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 1)
        
    def test_01_02_two_modes(self):
        result = mode([1, 2, 2, 3, 3])
        self.assertEqual(len(result), 2)
        self.assertIn(2, result)
        self.assertIn(3, result)
        
    def test_02_01_timeit(self):
        try:
            import timeit
            from scipy.stats import mode as scipy_mode
        except:
            pass
        else:
            setup = ("import numpy as np;"
                     "from cellprofiler.cpmath.mode import mode;"
                     "from scipy.stats import mode as scipy_mode;"
                     "r = np.random.RandomState(55);"
                     "a = r.randint(0, 10, size=(100000));")
            scipy_time = timeit.timeit("scipy_mode(a)", setup, number = 10)
            my_time = timeit.timeit("mode(a)", setup, number = 10)
            self.assertLess(my_time, scipy_time)
            logger.info("cellprofiler.cpmath.mode.mode=%f sec" % my_time)
            logger.info("scipy.stats.mode=%f sec" % scipy_time)