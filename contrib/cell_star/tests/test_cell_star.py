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

    def test_random_seeds(self):
        from contrib.cell_star.core.seed import Seed
        from contrib.cell_star.core.point import Point
        from contrib.cell_star.core.seeder import Seeder
        seed1 = Seed(10,10,'test')
        seed2 = Seed(100,100,'test2')
        new_seeds = Seeder.rand_seeds(5,2,[seed1,seed2])
        self.assertEqual(4, len(new_seeds))

        seed3 = Seed(0,0,'test3')
        new_zero_seeds = Seeder.rand_seeds(10,10,[seed3])
        self.assertTrue(all([seed.euclidean_distance_to(Point(0,0)) < 10 for seed in new_zero_seeds]))