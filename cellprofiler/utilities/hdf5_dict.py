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
    if isinstance(val, str) or np.sctype2char(np.asanyarray(val).dtype) == 'S':
        return h5py.special_dtype(vlen=str)
    return np.asanyarray(val).dtype


class HDF5Dict(object):
    '''The HDF5Dict can be used to store data indexed by a tuple of
    two strings and a non-negative integer.

    measurements = HDF5Dict(hdf5_filename)

    # Experiment-level features
    measurements['Experiment', 'feature1', 0] = 'a'
    measurements['Experiment', 'feature2', 0] = 1

    # Image-level features
    measurements['Image', 'imfeature1', 1] = 'foo'
    measurements['Image', 'imfeature2', 1] = 5

    # Object-level features
    measurements['Object1', 'objfeature1', 1] = [1, 2, 3]
    measurements['Object1', 'objfeature2', 1] = [4.0, 5.0, 6.0]

    Note that fetch operations always return either a single value or
    1D array, depending on what was stored.  The last integer can be
    any non-negative value, and does not need to be assigned in order.

    Slicing is not allowed in assignment or fetching.

    Integers, floats, and strings can be stored in the measurments.
    Strings will be returned as utf-8.

    Data can be removed with the del operator.

    del measurements['Experiment', 'feature1', 0]  # ok
    del measurements['Image', 'imfeature1', 2]  # ok

    If the 'must_exist' flag is set, it is an error to add a new
    object or feature that does not exist.

    '''

    # XXX - document how data is stored in hdf5 (basically, /Measurements/Object/Feature)

    def __init__(self, hdf5_filename, top_level_group_name="Measurements"):
        self.filename = hdf5_filename
        # assert not os.path.exists(self.filename)  # currently, don't allow overwrite
        self.hdf5_file = h5py.File(self.filename, 'w')
        self.top_level_group_name = top_level_group_name
        self.top_group = self.hdf5_file.create_group(top_level_group_name)
        self.indices = {}  # nested indices for data slices, indexed by (object, feature) then by numerical index
        self.lock = threading.RLock()  # h5py is thread safe, but does not support simultaneous read/write
        self.must_exist = False

    def __del__(self):
        try:
            self.hdf5_file.close()
            os.unlink(self.filename)
        except Exception, e:
            pass

    def __getitem__(self, idxs):
        assert isinstance(idxs, tuple), "Accessing HDF5_Dict requires a tuple of (object_name, feature_name, integer)"
        assert isinstance(idxs[0], basestring) and isinstance(idxs[1], basestring), "First two indices must be of type str."
        assert isinstance(idxs[2], int) and idxs[2] >= 0, "Third index must be a non-negative integer"

        object_name, feature_name, num_idx = idxs
        feature_exists = self.has_feature(object_name, feature_name)
        assert feature_exists

        if not self.has_data(*idxs):
            return None

        with self.lock:
            dest = self.find_index_or_slice(idxs)
            # it's possible we're fetching data from an image without
            # any objects, in which case we probably weren't able to
            # infer a type in __setitem__(), which means there may be
            # no dataset, yet.
            if isinstance(dest, slice) and dest.start is not None and dest.start == dest.stop:
                return np.array([])
            dataset = self.get_dataset(object_name, feature_name)
            return dataset[dest]

    def __setitem__(self, idxs, val):
        assert isinstance(idxs, tuple), "Assigning to HDF5_Dict requires a tuple of (object_name, feature_name, integer)"
        assert isinstance(idxs[0], basestring) and isinstance(idxs[1], basestring), "First two indices must be of type str."
        assert isinstance(idxs[2], int) and idxs[2] >= 0, "Third index must be a non-negative integer"

        object_name, feature_name, num_idx = idxs
        full_name = '%s.%s' % (idxs[0], idxs[1])
        feature_exists = self.has_feature(object_name, feature_name)
        assert (not self.must_exist) or feature_exists, \
            "Attempted storing new feature %s, but must_exist=True" % (full_name)

        if not feature_exists:
            if not self.has_object(object_name):
                self.add_object(object_name)
            self.add_feature(object_name, feature_name)

        # find the destination for the data, and check that its
        # the right size for the values.  This may extend the
        # _index and data arrays.
        dest = self.find_index_or_slice(idxs, val)

        # it's possible we're not actually storing anything.  In
        # this case, we can't infer a type, so return without
        # actually doing anything further.
        # XXX - unless we're passed a numpy array with a given type
        if dest is None:
            return

        with self.lock:
            # XXX - promote int to float
            dataset = self.get_dataset(object_name, feature_name)
            if np.isscalar(val):
                dataset[dest] = val
            else:
                dataset[dest] = np.asanyarray(val).ravel()

    def __delitem__(self, idxs):
        assert isinstance(idxs, tuple), "Accessing HDF5_Dict requires a tuple of (object_name, feature_name, integer)"
        assert isinstance(idxs[0], basestring) and isinstance(idxs[1], basestring), "First two indices must be of type str."
        assert isinstance(idxs[2], int) and idxs[2] >= 0, "Third index must be a non-negative integer"

        object_name, feature_name, num_idx = idxs
        feature_exists = self.has_feature(object_name, feature_name)
        assert feature_exists

        if not self.has_data(*idxs):
            return

        with self.lock:
            dest = self.find_index_or_slice(idxs)
            # it's possible we're fetching data from an image without
            # any objects, in which case we probably weren't able to
            # infer a type in __setitem__(), which means there may be
            # no dataset, yet.
            del self.indices[object_name, feature_name][num_idx]
            # reserved value of -1 means deleted
            idx = self.top_group[object_name][feature_name]['index']
            idx[idx[:, 0] == num_idx, 0] = -1

    def has_data(self, object_name, feature_name, num_idx):
        return num_idx in self.indices.get((object_name, feature_name), [])

    def get_dataset(self, object_name, feature_name):
        with self.lock:
            return self.top_group[object_name][feature_name]['data']

    def has_object(self, object_name):
        with self.lock:
            return object_name in self.top_group

    def add_object(self, object_name):
        with self.lock:
            object_group = self.top_group.require_group(object_name)

    def has_feature(self, object_name, feature_name):
        return (object_name, feature_name) in self.indices

    def add_feature(self, object_name, feature_name):
        feature_group = self.top_group[object_name].require_group(feature_name)
        self.indices.setdefault((object_name, feature_name), {})

    def find_index_or_slice(self, idxs, values=None):
        '''Find the linear indexes or slice for a particular set of
        indexes "idxs", and check that values could be stored in that
        linear index or slice.  If the linear index does not exist for
        the given idxs, then it will be created with sufficient size
        to store values (which must not be None, in this case).
        '''
        with self.lock:
            object_name, feature_name, num_idx = idxs
            assert isinstance(num_idx, int)
            index = self.indices[object_name, feature_name]
            if (num_idx not in index) and (values is None):
                return None  # no data
            if num_idx not in index:
                grow_by = np.asanyarray(values).ravel().size
                if grow_by == 0:
                    return None  # can't infer type
                # create the measurements if needed
                feature_group = self.top_group.require_group(object_name).require_group(feature_name)
                if not 'data' in feature_group:
                    feature_group.create_dataset('data', (0,), dtype=infer_hdf5_type(values),
                                                 compression='gzip', chunks=(1000,), maxshape=(None,))
                    feature_group.create_dataset('index', (0, 3), dtype=int, compression='gzip',
                                                 chunks=(1000, 3), maxshape=(None, 3))
                # grow data and index
                ds = feature_group['data']
                hdf5_index = feature_group['index']
                cur_size = ds.shape[0]
                ds.resize((cur_size + grow_by,))
                hdf5_index.resize((hdf5_index.shape[0] + 1, 3))
                # store locations for new data
                hdf5_index[-1, :] = (num_idx, cur_size, cur_size + grow_by)
                index[num_idx] = slice(cur_size, cur_size + grow_by)
            return index[num_idx]

    def clear(self):
        with self.lock:
            del self.hdf5_file[self.top_level_group_name]
            self.top_group = self.hdf5_file.create_group(self.top_level_group_name)
            self.indices = {}

    def erase(self, object_name, first_idx, mask):
        with self.lock:
            self.top_group[object_name]['_index'][mask] = -1
            self.level1_indices[object_name].pop(first_idx, None)

    def get_indices(self, object_name, feature_name):
        # CellProfiler expects these in write order
        with self.lock:
            idxs = self.top_group[object_name][feature_name]['index'][:, 0][:]
            return idxs[idxs != -1]

    def top_level_names(self):
        with self.lock:
            return self.top_group.keys()

    def second_level_names(self, object_name):
        with self.lock:
            return self.top_group[object_name].keys()

if __name__ == '__main__':
    h = HDF5Dict('temp.hdf5')
    h['Object1', 'objfeature1', 1] = [1, 2, 3]
    h['Object1', 'objfeature2', 1] = [1, 2, 3]
    h['Image', 'f1', 1] = 5
    h['Image', 'f2', 1] = 4
    print h['Image', 'f2', 1]
    h['Image', 'f1', 2] = 6
    h['Image', 'f2', 1] = 6
    print h['Image', 'f2', 1]
