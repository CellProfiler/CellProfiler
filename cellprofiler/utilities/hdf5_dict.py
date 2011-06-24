''' hdf5_dict -- HDF5-backed dictionary for Measurements.

This module implements the HDF5Dict class, which provides a dict-like
interface for measurements, backed by an HDF5 file.
'''

import os
import threading
import numpy as np
import h5py

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


class HDF5Dict(object):
    '''The HDF5Dict can be used to store measurements at several
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

    The first two indices are strings, all other are integers.  The
    last may be a slice when assigning.  Measurements can be fetched
    using indices, including slices in the last index.  In addition,
    the second axis may be sliced, in which case a structured array
    will be returned.  All indices are 0-indexed, though data can be
    stored in any order (with limitations, see below).

    Integers, floats, and strings can be stored in the measurments.

    Data can be removed with the del operator.

    A special feature, '_index', is created for each object.  It is an
    error to attempt to assign to this feature.

    If the 'must_exist' flag is set, it is an error to add a new
    object or feature that does not exist.

    It is acceptable to assign to specific values (though
    inefficient), as long new assignments do not extend the last
    dimension of the feature data:

    measurements['Object1', 'objfeature1', 1, 5] = 3  # ok
    measurements['Object1', 'objfeature1', 1, 3] = 1  # ok
    measurements['Object1', 'objfeature1', 1, 6] = 3  # error

    It is also an error to write different amounts of data for
    different features on the same object.

    # Assign 3 object measurements to Object1.objectfeature1:
    measurements['Object1', 'objfeature1', 1, :] = [1, 2, 3]
    # Error!  attempting to assign 4 measurements to the same object.
    measurements['Object1', 'objfeature2', 1, :] = [1, 2, 3, 4]

    '''

    # XXX - document how data is stored in hdf5 (basically, /Measurements/Object/Feature)

    def __init__(self, hdf5_filename, top_level_group_name="Measurements"):
        self.filename = hdf5_filename
        # assert not os.path.exists(self.filename)  # currently, don't allow overwrite
        self.hdf5_file = h5py.File(self.filename, 'w')
        self.top_level_group_name = top_level_group_name
        self.top_group = self.hdf5_file.create_group(top_level_group_name)
        self.object_features = {}  # sets of features for each object
        self.indices = {}  # indices for the first numerical coordinate (e.g., ImageNumber)
        self.lock = threading.RLock()  # h5py is thread safe, but does not support simultaneous read/write

        self.must_exist = False

    def __getitem__(self, idxs):
        print idxs
        return 'foo'

    def __setitem__(self, idxs, val):
        assert isinstance(idxs, tuple), "Assignment to HDF5_Dict requires at least object and feature names."
        assert isinstance(idxs[0], str) and isinstance(idxs[1], str), "First two indices must be of type str."
        assert not idxs[1] == '_index', "Can't assign to reserved _index feature."
        if isinstance(idxs[-1], slice):
            assert not np.isscalar(val), "Attempting to assign scalar to slice."
        else:
            assert np.isscalar(val), "Attempting to assign sequence to single element (HDF5_Dict requires slice for last index)."
        for idx in idxs[2:-1]:
            assert isinstance(idx, int) and idx >= 0, "All dimensional indices (except the last) must be integers >= 0"

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
        dest = self.find_index_or_slice(idxs, vals)

        # it's possible we're not actually storing anything.  In
        # this case, we can't infer a type, so return without
        # actually doing anything further.
        # XXX - unless we're passed a numpy array with a given type
        if isinstance(dest, slice) and dest.start == dest.stop:
            return

        # create the data if necessary
        dataset = self.get_dataset(object_name, feature_name, infer_hdf5_type(val)) or \
            self.create_data(object_name, feature_name, infer_hdf5_type(val))
        dest_max = (isinstance(dest, slice) and slice.stop) or dest
        if dataset.shape[0] < dest_max:
            dataset.resize((dest_max,))
        dataset[dest] = vals

    def has_object(self, object_name):
        return object_name in self.object_features

    def add_object(self, object_name, ndims):
        object_group = self.top_group.require_group(object_name)
        object_group.require_dataset('_index', (0, ndims), dtype=int)
        self.indices[object_name] = {}

    def has_feature(self, object_name, feature_name):
        return object_name in self.object_features and feature_name in self.object_features[object_name]

    def add_feature(self, object_name, feature_name):
        self.object_features.setdefault(object_name, set()).add(feature_name)

    def ndim(self, object_name):
        return self.top_group[object_name]['_index'].shape[1]

    def find_index_or_slice(self, idxs, values=None):
        '''Find the linear index or slice for a particular set of
        indexes "idxs", and check that values could be stored in that
        linear index or slice.  If the linear index does not exist for
        the given idxs, then it will be created with sufficient size
        to store values.  In this case, it is an error for values to
        be None.
        '''

        object_name = idxs[0]
        assert self.has_object(object_name)
        assert len(idxs) - 2 == self.ndim(object_name)

        base_idxs, last_idx = idxs[2:-1], idxs[-1]
        lohi = self.indices[object_name].get(base_idxs, None)
        # have these indices been stored before?
        if lohi is not None:
            if len(idxs) == 2 or isinstance(last_idx, int):
                assert values is None or np.isscalar(values)
                if len(idxs) == 2:
                    return 0  # Top-level features
                lo, hi = lohi
                assert last_idx >= 0 and last_idx < hi - lo, "Attempt to access element index %d in a %d length array" % (last_idx, hi - lo)
                return lo + last_idx
            assert not np.isscalar(values)  # None is not a scalar
            if isinstance(last_idx, slice):
                if last_idx.start is None and last_idx.stop is None:
                    assert (hi - lo) // idxs[0].step == np.asanyarray(values).size
                    return slice(lo, hi, idxs[0].step)
            linear_indices = np.arange(lo, hi)[last_idx]  # this also allows for a list index, and checks for valid values
            assert linear_indices.size == np.asanyarray(values).size
            return linear_indices

        # the indices have not been stored before

if __name__ == '__main__':
    h = HDF5Dict('temp.hdf5')
    h['Object1', 'objfeature1', 1, :] = [1, 2, 3]
