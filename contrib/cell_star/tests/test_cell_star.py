import unittest
import numpy as np

class test_CellStar(unittest.TestCase):
    def test_fill_holes(self):
        import contrib.cell_star.utils.image_util as image_util

        a = np.ones((30,30), dtype=int)
        a[10:20,20:25] = 0
        expected = a.copy()
        a[0:3,0:3] = 0

        res = image_util.fill_holes(a, 3, 15)

        self.assertTrue((expected == res).all())
