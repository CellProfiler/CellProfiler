import unittest
import cellprofiler.segmentation
import numpy


class TestSegmentation(unittest.TestCase):
    def test_01_01_dense(self):
        r = numpy.random.RandomState()
        r.seed(101)
        labels = r.randint(0, 10, size=(2, 3, 4, 5, 6, 7))
        s = cellprofiler.segmentation.Segmentation(dense=labels)
        self.assertTrue(s.has_dense())
        self.assertFalse(s.has_sparse())
        numpy.testing.assert_array_equal(s.dense[0], labels)

    def test_01_02_sparse(self):
        r = numpy.random.RandomState()
        r.seed(102)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [("y", numpy.uint32, 1), ("x", numpy.uint32, 1), ("label", numpy.uint32, 1)]
        )
        s = cellprofiler.segmentation.Segmentation(sparse=ijv)
        numpy.testing.assert_array_equal(s.sparse, ijv)
        self.assertFalse(s.has_dense())
        self.assertTrue(s.has_sparse())

    def test_02_01_sparse_to_dense(self):
        #
        # Make 10 circles that might overlap
        #
        r = numpy.random.RandomState()
        r.seed(201)
        i, j = numpy.mgrid[0:50, 0:50]
        ii = []
        jj = []
        vv = []
        for idx in range(10):
            x_loc = r.uniform() * 30 + 10
            y_loc = r.uniform() * 30 + 10
            max_radius = numpy.min([min(loc - 1, 49 - loc) for loc in x_loc, y_loc])
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            ii.append(i[mask])
            jj.append(j[mask])
            vv.append(numpy.ones(numpy.sum(mask), numpy.uint32) * (idx + 1))
        ijv = numpy.core.records.fromarrays(
            [numpy.hstack(x) for x in ii, jj, vv],
            [("y", numpy.uint32, 1), ("x", numpy.uint32, 1), ("label", numpy.uint32, 1)]
        )
        s = cellprofiler.segmentation.Segmentation(sparse=ijv, shape=(1, 1, 1, 50, 50))
        dense, indices = s.dense
        self.assertEqual(tuple(dense.shape[1:]), (1, 1, 1, 50, 50))
        self.assertEqual(numpy.sum(dense > 0), len(ijv))
        retrieval = dense[:, 0, 0, 0, ijv["y"], ijv["x"]]
        matches = (retrieval == ijv["label"][None, :])
        self.assertTrue(numpy.all(numpy.sum(matches, 0) == 1))

    def test_02_02_dense_to_sparse(self):
        #
        # Make 10 circles that might overlap
        #
        r = numpy.random.RandomState()
        r.seed(201)
        i, j = numpy.mgrid[0:50, 0:50]
        dense = numpy.zeros((10, 1, 1, 1, 50, 50), numpy.uint32)
        for idx in range(10):
            x_loc = r.uniform() * 30 + 10
            y_loc = r.uniform() * 30 + 10
            max_radius = numpy.min([min(loc - 1, 49 - loc) for loc in x_loc, y_loc])
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            dense[idx, 0, 0, 0, mask] = idx + 1
        s = cellprofiler.segmentation.Segmentation(dense=dense)
        ijv = s.sparse
        self.assertEqual(numpy.sum(dense > 0), len(ijv))
        retrieval = dense[:, 0, 0, 0, ijv["y"], ijv["x"]]
        matches = (retrieval == ijv["label"][None, :])
        self.assertTrue(numpy.all(numpy.sum(matches, 0) == 1))

    def test_03_01_shape_dense(self):
        r = numpy.random.RandomState()
        r.seed(101)
        labels = r.randint(0, 10, size=(2, 3, 4, 5, 6, 7))
        s = cellprofiler.segmentation.Segmentation(dense=labels)
        self.assertTrue(s.has_shape())
        self.assertEqual(tuple(s.shape), tuple(labels.shape[1:]))

    def test_03_02_shape_sparse_explicit(self):
        r = numpy.random.RandomState()
        r.seed(102)
        shape = (1, 1, 1, 50, 50)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [("y", numpy.uint32, 1), ("x", numpy.uint32, 1), ("label", numpy.uint32, 1)]
        )
        s = cellprofiler.segmentation.Segmentation(sparse=ijv, shape=shape)
        self.assertTrue(s.has_shape())
        self.assertEqual(tuple(s.shape), shape)

    def test_03_02_shape_sparse_implicit(self):
        r = numpy.random.RandomState()
        r.seed(102)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [("y", numpy.uint32, 1), ("x", numpy.uint32, 1), ("label", numpy.uint32, 1)]
        )
        ijv["x"] = 11
        ijv["y"] = 31
        shape = (1, 1, 1, 33, 13)
        s = cellprofiler.segmentation.Segmentation(sparse=ijv)
        self.assertFalse(s.has_shape())
        self.assertEqual(tuple(s.shape), shape)

    def test_03_03_set_shape(self):
        r = numpy.random.RandomState()
        r.seed(102)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [("y", numpy.uint32, 1), ("x", numpy.uint32, 1), ("label", numpy.uint32, 1)]
        )
        ijv["x"] = 11
        ijv["y"] = 31
        shape = (1, 1, 1, 50, 50)
        s = cellprofiler.segmentation.Segmentation(sparse=ijv)
        self.assertFalse(s.has_shape())
        s.shape = shape
        self.assertEqual(tuple(s.shape), shape)
