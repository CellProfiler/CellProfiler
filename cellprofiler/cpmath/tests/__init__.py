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
            import cellprofiler.cpmath
            self.check_warning(w, count)
            from cellprofiler.cpmath.bg_compensate \
                import bg_compensate, MODE_AUTO
            self.check_warning(w, count)
            from cellprofiler.cpmath.cpmorphology import binary_shrink
            self.check_warning(w, count)
            from cellprofiler.cpmath.filter import canny
            self.check_warning(w, count)
            from cellprofiler.cpmath.haralick import Haralick
            self.check_warning(w, count)
            from cellprofiler.cpmath.index import Indexes
            self.check_warning(w, count)
            from cellprofiler.cpmath.lapjv import lapjv
            self.check_warning(w, count)
            from cellprofiler.cpmath.otsu import otsu, otsu3
            self.check_warning(w, count)
            from cellprofiler.cpmath.outline import outline
            self.check_warning(w, count)
            from cellprofiler.cpmath.princomp import princomp
            self.check_warning(w, count)
            from cellprofiler.cpmath.propagate import propagate
            self.check_warning(w, count)
            from cellprofiler.cpmath.radial_power_spectrum import rps
            self.check_warning(w, count)
            from cellprofiler.cpmath.rankorder import rank_order
            self.check_warning(w, count)
            from cellprofiler.cpmath.smooth import fit_polynomial
            self.check_warning(w, count)
            from cellprofiler.cpmath.threshold import get_threshold
            self.check_warning(w, count)
            from cellprofiler.cpmath.watershed import watershed
            self.check_warning(w, count)
            from cellprofiler.cpmath.zernike import zernike
