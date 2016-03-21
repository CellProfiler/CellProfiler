import sys
import unittest
import warnings
from exceptions import DeprecationWarning


class TestCPMath(unittest.TestCase):
    def setUp(self):
        modules_to_remove = filter(
                (lambda k:
                 k.startswith('cellprofiler.cpmath') and not
                 k.startswith('cellprofiler.cpmath.tests')),
                sys.modules.keys())
        for module_name in modules_to_remove:
            del sys.modules[module_name]

    def check_warning(self, w, count):
        self.assertTrue(issubclass(w[count[0]].category, DeprecationWarning))
        count[0] = len(w)

    def test_01_01_catch_warnings(self):
        count = [0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
            self.check_warning(w, count)
