class TestSegmentation(unittest.TestCase):
    def test_01_01_dense(self):
        r = np.random.RandomState()
        r.seed(101)
        labels = r.randint(0, 10, size=(2, 3, 4, 5, 6, 7))
        s = cpo.Segmentation(dense=labels)
        self.assertTrue(s.has_dense())
        self.assertFalse(s.has_sparse())
        np.testing.assert_array_equal(s.get_dense()[0], labels)

    def test_01_02_sparse(self):
        r = np.random.RandomState()
        r.seed(102)
        ijv = np.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [(HDF5ObjectSet.AXIS_Y, np.uint32, 1),
             (HDF5ObjectSet.AXIS_X, np.uint32, 1),
             (HDF5ObjectSet.AXIS_LABELS, np.uint32, 1)])
        s = cpo.Segmentation(sparse=ijv)
        np.testing.assert_array_equal(s.get_sparse(), ijv)
        self.assertFalse(s.has_dense())
        self.assertTrue(s.has_sparse())

    def test_02_01_sparse_to_dense(self):
        #
        # Make 10 circles that might overlap
        #
        r = np.random.RandomState()
        r.seed(201)
        i, j = np.mgrid[0:50, 0:50]
        ii = []
        jj = []
        vv = []
        for idx in range(10):
            x_loc = r.uniform() * 30 + 10
            y_loc = r.uniform() * 30 + 10
            max_radius = np.min([min(loc - 1, 49 - loc) for loc in x_loc, y_loc])
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            ii.append(i[mask])
            jj.append(j[mask])
            vv.append(np.ones(np.sum(mask), np.uint32) * (idx + 1))
        ijv = np.core.records.fromarrays([
                                             np.hstack(x) for x in ii, jj, vv],
                                         [(HDF5ObjectSet.AXIS_Y, np.uint32, 1),
                                          (HDF5ObjectSet.AXIS_X, np.uint32, 1),
                                          (HDF5ObjectSet.AXIS_LABELS, np.uint32, 1)])
        s = cpo.Segmentation(sparse=ijv, shape=(1, 1, 1, 50, 50))
        dense, indices = s.get_dense()
        self.assertEqual(tuple(dense.shape[1:]), (1, 1, 1, 50, 50))
        self.assertEqual(np.sum(dense > 0), len(ijv))
        retrieval = dense[:, 0, 0, 0,
                    ijv[HDF5ObjectSet.AXIS_Y], ijv[HDF5ObjectSet.AXIS_X]]
        matches = (retrieval == ijv[HDF5ObjectSet.AXIS_LABELS][None, :])
        self.assertTrue(np.all(np.sum(matches, 0) == 1))

    def test_02_02_dense_to_sparse(self):
        #
        # Make 10 circles that might overlap
        #
        r = np.random.RandomState()
        r.seed(201)
        i, j = np.mgrid[0:50, 0:50]
        dense = np.zeros((10, 1, 1, 1, 50, 50), np.uint32)
        for idx in range(10):
            x_loc = r.uniform() * 30 + 10
            y_loc = r.uniform() * 30 + 10
            max_radius = np.min([min(loc - 1, 49 - loc) for loc in x_loc, y_loc])
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            dense[idx, 0, 0, 0, mask] = idx + 1
        s = cpo.Segmentation(dense=dense)
        ijv = s.get_sparse()
        self.assertEqual(np.sum(dense > 0), len(ijv))
        retrieval = dense[:, 0, 0, 0,
                    ijv[HDF5ObjectSet.AXIS_Y], ijv[HDF5ObjectSet.AXIS_X]]
        matches = (retrieval == ijv[HDF5ObjectSet.AXIS_LABELS][None, :])
        self.assertTrue(np.all(np.sum(matches, 0) == 1))

    def test_03_01_shape_dense(self):
        r = np.random.RandomState()
        r.seed(101)
        labels = r.randint(0, 10, size=(2, 3, 4, 5, 6, 7))
        s = cpo.Segmentation(dense=labels)
        self.assertTrue(s.has_shape())
        self.assertEqual(tuple(s.shape), tuple(labels.shape[1:]))

    def test_03_02_shape_sparse_explicit(self):
        r = np.random.RandomState()
        r.seed(102)
        shape = (1, 1, 1, 50, 50)
        ijv = np.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [(HDF5ObjectSet.AXIS_Y, np.uint32, 1),
             (HDF5ObjectSet.AXIS_X, np.uint32, 1),
             (HDF5ObjectSet.AXIS_LABELS, np.uint32, 1)])
        s = cpo.Segmentation(sparse=ijv, shape=shape)
        self.assertTrue(s.has_shape())
        self.assertEqual(tuple(s.shape), shape)

    def test_03_02_shape_sparse_implicit(self):
        r = np.random.RandomState()
        r.seed(102)
        shape = (1, 1, 1, 50, 50)
        ijv = np.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [(HDF5ObjectSet.AXIS_Y, np.uint32, 1),
             (HDF5ObjectSet.AXIS_X, np.uint32, 1),
             (HDF5ObjectSet.AXIS_LABELS, np.uint32, 1)])
        ijv[HDF5ObjectSet.AXIS_X] = 11
        ijv[HDF5ObjectSet.AXIS_Y] = 31
        shape = (1, 1, 1, 33, 13)
        s = cpo.Segmentation(sparse=ijv)
        self.assertFalse(s.has_shape())
        self.assertEqual(tuple(s.shape), shape)

    def test_03_03_set_shape(self):
        r = np.random.RandomState()
        r.seed(102)
        shape = (1, 1, 1, 50, 50)
        ijv = np.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [(HDF5ObjectSet.AXIS_Y, np.uint32, 1),
             (HDF5ObjectSet.AXIS_X, np.uint32, 1),
             (HDF5ObjectSet.AXIS_LABELS, np.uint32, 1)])
        ijv[HDF5ObjectSet.AXIS_X] = 11
        ijv[HDF5ObjectSet.AXIS_Y] = 31
        shape = (1, 1, 1, 50, 50)
        s = cpo.Segmentation(sparse=ijv)
        self.assertFalse(s.has_shape())
        s.set_shape(shape)
        self.assertEqual(tuple(s.shape), shape)
