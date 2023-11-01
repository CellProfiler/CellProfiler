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

    def test_02_01_shape_dense(self):
        r = numpy.random.RandomState()
        r.seed(101)
        labels = r.randint(0, 10, size=(2, 3, 4, 5, 6, 7))
        s = cellprofiler_core.object.Segmentation(dense=labels)
        assert s.has_shape()
        assert tuple(s.shape) == tuple(labels.shape[1:])

    def test_02_02_shape_sparse_explicit(self):
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

    def test_02_02_shape_sparse_implicit(self):
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
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X] = 11
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y] = 31
        shape = (1, 1, 1, 33, 13)
        s = cellprofiler_core.object.Segmentation(sparse=ijv)
        assert not s.has_shape()
        assert tuple(s.shape) == shape

    def test_02_03_set_shape(self):
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
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_X] = 11
        ijv[cellprofiler_core.utilities.hdf5_dict.HDF5ObjectSet.AXIS_Y] = 31
        shape = (1, 1, 1, 50, 50)
        s = cellprofiler_core.object.Segmentation(sparse=ijv)
        assert not s.has_shape()
        s.shape = shape
        assert tuple(s.shape) == shape
