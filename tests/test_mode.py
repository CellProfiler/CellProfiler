from __future__ import absolute_import
import logging

logger = logging.getLogger(__name__)
import numpy as np
from centrosome.mode import mode
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
            setup = (
                "import numpy as np;"
                "from centrosome.mode import mode;"
                "from scipy.stats import mode as scipy_mode;"
                "r = np.random.RandomState(55);"
                "a = r.randint(0, 10, size=(100000));"
            )
            scipy_time = timeit.timeit("scipy_mode(a)", setup, number=10)
            my_time = timeit.timeit("mode(a)", setup, number=10)
            self.assertLess(my_time, scipy_time)
            logger.info("centrosome.mode.mode=%f sec" % my_time)
            logger.info("scipy.stats.mode=%f sec" % scipy_time)
