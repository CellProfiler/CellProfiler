class TestSizeSimilarly(unittest.TestCase):
    def test_01_01_size_same(self):
        secondary, mask = cpo.size_similarly(np.zeros((10, 20)),
                                             np.zeros((10, 20)))
        self.assertEqual(tuple(secondary.shape), (10, 20))
        self.assertTrue(np.all(mask))

    def test_01_02_larger_secondary(self):
        secondary, mask = cpo.size_similarly(np.zeros((10, 20)),
                                             np.zeros((10, 30)))
        self.assertEqual(tuple(secondary.shape), (10, 20))
        self.assertTrue(np.all(mask))
        secondary, mask = cpo.size_similarly(np.zeros((10, 20)),
                                             np.zeros((20, 20)))
        self.assertEqual(tuple(secondary.shape), (10, 20))
        self.assertTrue(np.all(mask))

    def test_01_03_smaller_secondary(self):
        secondary, mask = cpo.size_similarly(np.zeros((10, 20), int),
                                             np.zeros((10, 15), np.float32))
        self.assertEqual(tuple(secondary.shape), (10, 20))
        self.assertTrue(np.all(mask[:10, :15]))
        self.assertTrue(np.all(~mask[:10, 15:]))
        self.assertEqual(secondary.dtype, np.dtype(np.float32))

    def test_01_04_size_color(self):
        secondary, mask = cpo.size_similarly(np.zeros((10, 20), int),
                                             np.zeros((10, 15, 3), np.float32))
        self.assertEqual(tuple(secondary.shape), (10, 20, 3))
        self.assertTrue(np.all(mask[:10, :15]))
        self.assertTrue(np.all(~mask[:10, 15:]))
        self.assertEqual(secondary.dtype, np.dtype(np.float32))
