import numpy
import numpy.testing

import cellprofiler_core.image
import cellprofiler_core.object
import cellprofiler_core.utilities.hdf5_dict


class TestSegmentation:
    def test_01_01_dense(self):
        r = numpy.random.RandomState()
        r.seed(101)
        labels = r.randint(0, 10, size=(2, 3, 4, 5, 6, 7))
        s = cellprofiler_core.object.Segmentation(dense=labels)
        assert s.has_dense()
        assert not s.has_sparse()
        numpy.testing.assert_array_equal(s.get_dense()[0], labels)

    def test_01_02_sparse(self):
        r = numpy.random.RandomState()
        r.seed(102)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_LABELS,
                    numpy.uint32,
                    1,
                ),
            ],
        )
        s = cellprofiler_core.object.Segmentation(sparse=ijv)
        numpy.testing.assert_array_equal(s.sparse, ijv)
        assert not s.has_dense()
        assert s.has_sparse()

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
            max_radius = numpy.min([min(loc - 1, 49 - loc) for loc in (x_loc, y_loc)])
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            ii.append(i[mask])
            jj.append(j[mask])
            vv.append(numpy.ones(numpy.sum(mask), numpy.uint32) * (idx + 1))
        ijv = numpy.core.records.fromarrays(
            [numpy.hstack(x) for x in (ii, jj, vv)],
            [
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_LABELS,
                    numpy.uint32,
                    1,
                ),
            ],
        )
        s = cellprofiler_core.object.Segmentation(sparse=ijv, shape=(1, 1, 1, 50, 50))
        dense, indices = s.get_dense()
        assert tuple(dense.shape[1:]) == (1, 1, 1, 50, 50)
        assert numpy.sum(dense > 0) == len(ijv)
        retrieval = dense[
            :,
            0,
            0,
            0,
            ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y],
            ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X],
        ]
        matches = (
            retrieval
            == ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_LABELS][
                None, :
            ]
        )
        assert numpy.all(numpy.sum(matches, 0) == 1)

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
            max_radius = numpy.min([min(loc - 1, 49 - loc) for loc in (x_loc, y_loc)])
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            dense[idx, 0, 0, 0, mask] = idx + 1
        s = cellprofiler_core.object.Segmentation(dense=dense)
        ijv = s.sparse
        assert numpy.sum(dense > 0) == len(ijv)
        retrieval = dense[
            :,
            0,
            0,
            0,
            ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y],
            ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X],
        ]
        matches = (
            retrieval
            == ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_LABELS][
                None, :
            ]
        )
        assert numpy.all(numpy.sum(matches, 0) == 1)

    def test_03_01_shape_dense(self):
        r = numpy.random.RandomState()
        r.seed(101)
        labels = r.randint(0, 10, size=(2, 3, 4, 5, 6, 7))
        s = cellprofiler_core.object.Segmentation(dense=labels)
        assert s.has_shape()
        assert tuple(s.shape) == tuple(labels.shape[1:])

    def test_03_02_shape_sparse_explicit(self):
        r = numpy.random.RandomState()
        r.seed(102)
        shape = (1, 1, 1, 50, 50)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_LABELS,
                    numpy.uint32,
                    1,
                ),
            ],
        )
        s = cellprofiler_core.object.Segmentation(sparse=ijv, shape=shape)
        assert s.has_shape()
        assert tuple(s.shape) == shape

    def test_03_02_shape_sparse_implicit(self):
        r = numpy.random.RandomState()
        r.seed(102)
        shape = (1, 1, 1, 50, 50)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_LABELS,
                    numpy.uint32,
                    1,
                ),
            ],
        )
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X] = 11
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y] = 31
        shape = (1, 1, 1, 33, 13)
        s = cellprofiler_core.object.Segmentation(sparse=ijv)
        assert not s.has_shape()
        assert tuple(s.shape) == shape

    def test_03_03_set_shape(self):
        r = numpy.random.RandomState()
        r.seed(102)
        shape = (1, 1, 1, 50, 50)
        ijv = numpy.core.records.fromarrays(
            [r.randint(0, 10, size=20) for _ in range(3)],
            [
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X,
                    numpy.uint32,
                    1,
                ),
                (
                    cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_LABELS,
                    numpy.uint32,
                    1,
                ),
            ],
        )
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X] = 11
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y] = 31
        shape = (1, 1, 1, 50, 50)
        s = cellprofiler_core.object.Segmentation(sparse=ijv)
        assert not s.has_shape()
        s.shape = shape
        assert tuple(s.shape) == shape
