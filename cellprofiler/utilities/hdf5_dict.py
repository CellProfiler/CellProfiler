''' hdf5_dict -- HDF5-backed dictionary for Measurements.

This module implements the HDF5Dict class, which provides a dict-like
interface for measurements, backed by an HDF5 file.
'''

import os
import threading
import numpy as np
import h5py
import time

# h5py is nice, but not being able to make zero-length selections is a pain.
orig_hdf5_getitem = h5py.Dataset.__getitem__
def new_getitem(self, args):
    if isinstance(args, slice) and \
            args.start is not None and args.start == args.stop:
        return np.array([], self.dtype)
    return orig_hdf5_getitem(self, args)
setattr(h5py.Dataset, orig_hdf5_getitem.__name__, new_getitem)

orig_hdf5_setitem = h5py.Dataset.__setitem__
def new_setitem(self, args, val):
    if isinstance(args, slice) and \
            args.start is not None and args.start == args.stop:
        return np.array([], self.dtype)[0:0]
    return orig_hdf5_setitem(self, args, val)
setattr(h5py.Dataset, orig_hdf5_setitem.__name__, new_setitem)

def infer_hdf5_type(val):
    if isinstance(val, str) or isinstance(val, unicode) or np.sctype2char(np.asanyarray(val).dtype) == 'S':
        return h5py.special_dtype(vlen=str)
    return np.asanyarray(val).dtype


class HDF5Dict(object):
    '''The HDF5Dict can be used to store measurements at three
    hierarchical levels:

    measurements = HDF5Dict(hdf5_filename)

    # Experiment-level features
    measurements['Experiment', 'feature1'] = 'a'
    measurements['Experiment', 'feature2'] = 1

    # Image-level features
    measurements['Image', 'imfeature1', 1] = 'foo'
    measurements['Image', 'imfeature2', 1] = 5

    # Object-level features
    measurements['Object1', 'objfeature1', 1, :] = [1, 2, 3]
    measurements['Object1', 'objfeature2', 1, :] = [4.0, 5.0, 6.0]

    The first two axes are are indexed by strings, the other two (if
    present) by integers.  The last may be the a full slice (i.e.,
    [..., :]) when assigning or fetching.  In addition, the second
    axis may be sliced, in which case a structured array will be
    returned.  All indices are 0-indexed, though data can be stored in
    any order.  Note that fetch operations always return either a
    single value, a 1-D array (if the last index is sliced), or a 2-D
    array (if the feature and last index are sliced).

    Partial slicing is not allowed in assignment.

    Integers, floats, and strings can be stored in the measurments.

    Data can be removed with the del operator.  In this case, if there
    are more than three indices, the last must be the full slice:

    del measurements['Experiment', 'feature1']  # ok
    del measurements['Image', 'imfeature1', 2]  # ok
    del measurements['Object1', 'objfeature1', 3, :]  # ok
    del measurements['Object1', 'objfeature2', 3, 5]  # error

    A special feature, '_index', is created for each object.  It is an
    error to attempt to assign to this feature.

    If the 'must_exist' flag is set, it is an error to add a new
    object or feature that does not exist.

    It is an error to write different amounts of data for different
    features on the same object:

    # Assign 3 object measurements to Object1.objectfeature1:
    measurements['Object1', 'objfeature1', 1, :] = [1, 2, 3]
    # Error!  attempting to assign 4 measurements to the same object.
    measurements['Object1', 'objfeature2', 1, :] = [1, 2, 3, 4]

    It is also an error to grow an object-level feature after the
    first assignment:

    measurements['Object1', 'objfeature1', 1, 5] = 2.0  # ok, 6 objects allocated
    measurements['Object1', 'objfeature1', 1, 3] = 1.0  # ok
    measurements['Object1', 'objfeature1', 1, 6] = 1.0  # error, write beyond end of objects array

    XXX - should we relax this reuquirement?  We coudl store a list of
    slices in the level1_indices.
    '''

    # XXX - document how data is stored in hdf5 (basically, /Measurements/Object/Feature)

    def __init__(self, hdf5_filename, top_level_group_name="Measurements"):
        self.filename = hdf5_filename
        # assert not os.path.exists(self.filename)  # currently, don't allow overwrite
        self.hdf5_file = h5py.File(self.filename, 'w')
        self.top_level_group_name = top_level_group_name
        self.top_group = self.hdf5_file.create_group(top_level_group_name)
        self.object_features = {}  # sets of features for each object
        self.level1_indices = {}  # indices for the first numerical coordinate (e.g., ImageNumber)
        self.object_dims = {}
        self.lock = threading.RLock()  # h5py is thread safe, but does not support simultaneous read/write
        self.must_exist = False

    def __del__(self):
        try:
            self.hdf5_file.close()
            os.unlink(self.filename)
        except Exception, e:
            pass

    def __getitem__(self, idxs):
        assert isinstance(idxs, tuple), "Assignment to HDF5_Dict requires at least object and feature names."
        assert isinstance(idxs[0], str) and isinstance(idxs[1], str), "First two indices must be of type str."
        assert not idxs[1] == '_index', "Can't assign to reserved _index feature."

        object_name = idxs[0]
        feature_name = idxs[1]
        full_name = '%s.%s' % (idxs[0], idxs[1])
        feature_exists = self.has_feature(object_name, feature_name)
        assert feature_exists
        assert self.ndim(object_name) == len(idxs) - 2, \
            "%s is %d dimensional, but only %d given" % (object_name, self.ndim(object_name), len(idxs) - 2)

        # find the destination for the data, and check that its
        # the right size for the values.  This may extend the
        # _index feature.
        with self.lock:
            dest = self.find_index_or_slice(object_name, idxs[2:])
            # it's possible we're fetching data from an iamge without
            # any objects, in which case we probably weren't able to
            # infer a type in __setitem__(), which means there may be
            # no dataset, yet.
            if isinstance(dest, slice) and dest.start is not None and dest.start == dest.stop:
                return np.array([])
            dataset = self.get_dataset(object_name, feature_name, create_if_missing=False)
            return dataset[dest]

    def __setitem__(self, idxs, val):
        assert isinstance(idxs, tuple), "Assignment to HDF5_Dict requires at least object and feature names."
        assert isinstance(idxs[0], str) and isinstance(idxs[1], str), "First two indices must be of type str."
        assert not idxs[1] == '_index', "Can't assign to reserved _index feature."

        object_name = idxs[0]
        feature_name = idxs[1]
        full_name = '%s.%s' % (idxs[0], idxs[1])
        feature_exists = self.has_feature(object_name, feature_name)
        assert (not self.must_exist) or feature_exists, \
            "Attempted storing new feature %s, but must_exist=True" % (full_name)

        if not feature_exists:
            if not self.has_object(object_name):
                self.add_object(object_name, len(idxs) - 2)
            self.add_feature(object_name, feature_name)

        assert self.ndim(object_name) == len(idxs) - 2, \
            "%s is %d dimensional, but only %d given" % (object_name, self.ndim(object_name), len(idxs) - 2)

        # find the destination for the data, and check that its
        # the right size for the values.  This may extend the
        # _index feature.
        dest = self.find_index_or_slice(object_name, idxs[2:], val)

        # it's possible we're not actually storing anything.  In
        # this case, we can't infer a type, so return without
        # actually doing anything further.
        # XXX - unless we're passed a numpy array with a given type
        if isinstance(dest, slice) and dest.start is not None and dest.start == dest.stop:
            return

        with self.lock:
            dataset = self.get_dataset(object_name, feature_name, h5type=infer_hdf5_type(val))
            dest_max = (isinstance(dest, slice) and dest.stop) or \
                (not np.isscalar(dest) and np.asanyarray(dest).max()) or \
                dest
            if dataset.shape[0] <= dest_max:
                dataset.resize((dest_max + 1,))
            if np.isscalar(val):
                dataset[dest] = val
            else:
                dataset[dest] = np.asanyarray(val).ravel()

    def get_dataset(self, object_name, feature_name, h5type=None, create_if_missing=True):
        with self.lock:
            object_group = self.top_group.require_group(object_name)
            if feature_name in object_group:
                return object_group[feature_name]
            assert create_if_missing
            return object_group.create_dataset(feature_name, (0,), dtype=h5type, compression='gzip', chunks=(1000,), maxshape=(None,))

    def has_object(self, object_name):
        return object_name in self.object_features

    def add_object(self, object_name, ndims):
        with self.lock:
            object_group = self.top_group.require_group(object_name)
            object_group.require_dataset('_index', (0,), dtype=int, compression='gzip', chunks=(1000,), maxshape=(None,))
            self.level1_indices[object_name] = {}
            self.object_dims[object_name] = ndims

    def ndim(self, object_name):
        return self.object_dims[object_name]

    def has_feature(self, object_name, feature_name):
        return feature_name in self.object_features.get(object_name, set())

    def add_feature(self, object_name, feature_name):
        self.object_features.setdefault(object_name, set()).add(feature_name)

    def find_index_or_slice(self, object_name, idxs, values=None):
        '''Find the linear indexes or slice for a particular set of
        indexes "idxs", and check that values could be stored in that
        linear index or slice.  If the linear index does not exist for
        the given idxs, then it will be created with sufficient size
        to store values (which must not be None, in this case).
        '''
        with self.lock:
            if len(idxs) == 0:
                # Experiment-level features
                assert values is None or np.isscalar(values), \
                    "attempting to assign a sequence to a single element"
                return 0

            level1_index = self.level1_indices[object_name]

            def grow_index(idx, grow_by):
                _index = self.top_group[object_name]['_index']
                start = _index.shape[0]
                _index.resize((start + grow_by,))
                sl = slice(start, start + grow_by, None)
                _index[sl] = idx
                level1_index[idx] = sl
                return start

            if len(idxs) == 1:
                # Image-level features
                idx = idxs[0]

                def lookup_and_maybe_grow(i):
                    if i not in level1_index:
                        return grow_index(i, 1)
                    return level1_index[i].start

                if isinstance(idx, int):
                    assert idx in level1_index or values is not None
                    assert values is None or np.isscalar(values), "attempting to assign a sequence to a single element"
                    return lookup_and_maybe_grow(idx)
                if isinstance(idx, slice):
                    assert idx == slice(None, None, None), "Can only slice with [:]"
                    assert values is None or (not np.isscalar(values) and \
                                                  np.asanyarray(values).size == len(pos_index))
                    # XXX - should we instead return indices sorted by pos_index.keys()?
                    return (self.top_group[object_name]['_index'][:] != -1).flatten()  # remove erased values
                # must be a list of indices
                assert values is None or len(idx) == len(values)
                return [lookup_and_maybe_grow(i) for i in idx]

            # Object-level values
            idx1, idx2 = idxs
            assert isinstance(idx1, int)
            assert idx1 in level1_index or values is not None
            if idx1 not in level1_index:
                if isinstance(idx2, int):
                    assert np.isscalar(values)
                    grow_by = 1
                else:
                    assert idx2 == slice(None, None, None) and not np.isscalar(values)
                    grow_by = np.asanyarray(values).size
                grow_index(idx1, grow_by)

            sl = level1_index[idx1]
            if isinstance(idx2, int):
                assert values is None or np.isscalar(values)
                pos = sl.start + idx2
                assert pos < sl.stop
                return pos
            if isinstance(idx2, slice):
                start, stop, step = idx2.indices(sl.stop - sl.start)
                return slice(start + sl.start, stop + sl.start, step)
            assert np.all(np.asanyarray(indices) < sl.stop - sl.start)
            return [sl.start + v for v in indices]

    def clear(self):
        with self.lock:
            del self.hdf5_file[self.top_level_group_name]
            self.top_group = self.hdf5_file.create_group(self.top_level_group_name)

    def erase(self, object_name, first_idx, mask):
        with self.lock:
            self.top_group[object_name]['_index'][mask] = -1
            self.level1_indices[object_name].pop(first_idx, None)

if __name__ == '__main__':
    h = HDF5Dict('temp.hdf5')
    h['Object1', 'objfeature1', 1, :] = [1, 2, 3]
    h['Object1', 'objfeature2', 1, :] = [1, 2, 3]
    h['Image', 'f1', 1] = 5
    h['Image', 'f2', 1] = 4
    print h['Image', 'f2', 1]
    h['Image', 'f1', 2] = 6
