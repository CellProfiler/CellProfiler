"""hdf5_dict -- HDF5-backed dictionary for Measurements.

This module implements the HDF5Dict class, which provides a dict-like
interface for measurements, backed by an HDF5 file.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
Copyright (c) 2011 Institut Curie
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
from __future__ import with_statement

__version__ = "$Revision$"

import os
import threading
import numpy as np
import h5py
import time
import logging
logger = logging.getLogger(__name__)

version_number = 1
VERSION = "Version"

# h5py is nice, but not being able to make zero-length selections is a pain.
orig_hdf5_getitem = h5py.Dataset.__getitem__
def new_getitem(self, args):
    if (isinstance(args, slice) and \
            args.start is not None and args.start == args.stop):
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
    val = np.asanyarray(val)
    if val.size == 0:
        return int
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
    
    The measurements data is stored in groups corresponding to object names
    (with special objects, "Image" = image set measurements and "Experiment" = 
    experiment measurements. Each object feature has its own group under
    the object group. The feature group has two data sets. The first data set
    is "index" and holds indexes into the second data set whose name is "data".
    "index" is an N x 3 integer array where N is the number of image sets
    with this feature measurement and the three row values are the image number
    of that row's measurements, the offset to the first data element for
    the feature measurement for that image number in the "data" dataset 
    and the offset to one past the last data element.
    '''

    # XXX - document how data is stored in hdf5 (basically, /Measurements/Object/Feature)

    def __init__(self, hdf5_filename, 
                 top_level_group_name = "Measurements",
                 run_group_name = time.strftime("%Y-%m-%d-%H-%m-%S"),
                 is_temporary = False,
                 copy = None):
        self.is_temporary = is_temporary
        self.filename = hdf5_filename
        logger.debug("HDF5Dict.__init__(): %s, temporary=%s, copy=%s", self.filename, self.is_temporary, copy)
        # assert not os.path.exists(self.filename)  # currently, don't allow overwrite
        self.hdf5_file = h5py.File(self.filename, 'w')
        vdataset = self.hdf5_file.create_dataset(
            VERSION, data = np.array([version_number], int))
        self.top_level_group_name = top_level_group_name
        mgroup = self.hdf5_file.create_group(top_level_group_name)
        self.top_group = mgroup.create_group(run_group_name)
        self.indices = {}  # nested indices for data slices, indexed by (object, feature) then by numerical index
        self.string_references = StringReferencer(
            self.top_group.require_group("StringReferences"))
        class HDF5Lock:
            def __init__(self):
                self.lock = threading.RLock()
            def __enter__(self):
                self.lock.acquire()
                if hasattr(h5py.highlevel, "phil"):
                    h5py.highlevel.phil.acquire()
                
            def __exit__(self, t, v, tb):
                if hasattr(h5py.highlevel, "phil"):
                    h5py.highlevel.phil.release()
                self.lock.release()
                
        self.lock = HDF5Lock()
                
        self.must_exist = False
        self.chunksize = 1024
        if copy is not None:
            for object_name in copy.keys():
                object_group = copy[object_name]
                self.top_group.copy(object_group, self.top_group)
                for feature_name in object_group.keys():
                    # some measurement objects are written at a higher level, and don't
                    # have an index (e.g. Relationship).
                    if 'index' in object_group[feature_name].keys():
                        d = self.indices[object_name, feature_name] = {}
                        hdf5_index = object_group[feature_name]['index'][:]
                        for num_idx, start, stop in hdf5_index:
                            d[num_idx] = slice(start, stop)
            self.hdf5_file.flush()

    def __del__(self):
        logger.debug("HDF5Dict.__del__(): %s, temporary=%s", self.filename, self.is_temporary)
        self.close()
        
    def close(self):
        if not hasattr(self, "hdf5_file"):
            # This happens if the constructor could not open the hdf5 file
            return
        if self.is_temporary:
            try:
                self.hdf5_file.flush()  # just in case unlink fails
                self.hdf5_file.close()
                os.unlink(self.filename)
            except Exception, e:
                logger.warn("So sorry. CellProfiler failed to remove the temporary file, %s and there it sits on your disk now." % self.filename)
        else:
            self.hdf5_file.flush()
            self.hdf5_file.close()
        del self.hdf5_file

    def flush(self):
        logger.debug("HDF5Dict.flush(): %s, temporary=%s", self.filename, self.is_temporary)
        self.hdf5_file.flush()

    def __getitem__(self, idxs):
        assert isinstance(idxs, tuple), "Accessing HDF5_Dict requires a tuple of (object_name, feature_name[, integer])"
        assert isinstance(idxs[0], basestring) and isinstance(idxs[1], basestring), "First two indices must be of type str."
        assert ((not np.isscalar(idxs[2]) and np.all(idxs[2] >= 0))
                or (isinstance(idxs[2], int) and idxs[2] >= 0)),\
               "Third index must be a non-negative integer or integer array"

        object_name, feature_name, num_idx = idxs
        feature_exists = self.has_feature(object_name, feature_name)
        assert feature_exists
        if not np.isscalar(num_idx):
            with self.lock:
                indices = self.indices[(object_name, feature_name)]
                dataset = self.get_dataset(object_name, feature_name)
                return [None if (isinstance(dest, slice) and 
                                 dest.start is not None and 
                                 dest.start == dest.stop) else dataset[dest]
                        for dest in [ indices.get(image_number, slice(0,0))
                                      for image_number in num_idx]]

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
        # _index and data arrays. It may also overwrite the old value.
        dest = self.find_index_or_slice(idxs, val)

        with self.lock:
            dataset = self.get_dataset(object_name, feature_name)
            if dataset.dtype.kind == 'i':
                if np.asanyarray(val).dtype.kind == 'f':
                    # it's possible we have only stored integers and now need to promote to float
                    if dataset.shape[0] > 0:
                        vals = dataset[:].astype(float)
                    else:
                        vals = np.array([])
                    del self.top_group[object_name][feature_name]['data']
                    dataset = self.top_group[object_name][feature_name].create_dataset('data', (vals.size,), dtype=float,
                                                                                       compression='gzip', shuffle=True, chunks=(self.chunksize,), maxshape=(None,))
                    if vals.size > 0:
                        dataset[:] = vals
                elif np.asanyarray(val).dtype.kind in ('S', 'a', 'U'):
                    # we created the dataset without any data, so didn't know the type before
                    sz = dataset.shape[0]
                    del self.top_group[object_name][feature_name]['data']
                    dataset = self.top_group[object_name][feature_name].create_dataset('data', (sz,), dtype=h5py.special_dtype(vlen=str),
                                                                                       compression='gzip', shuffle=True, chunks=(self.chunksize,), maxshape=(None,))

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
            idx[np.flatnonzero(idx[:, 0] == num_idx), 0] = -1
            
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
        with self.lock:
            feature_group = self.top_group[object_name].require_group(feature_name)
            self.indices.setdefault((object_name, feature_name), {})
            
    def find_index_or_slice(self, idxs, values=None):
        '''Find the linear indexes or slice for a particular set of
        indexes "idxs", and check that values could be stored in that
        linear index or slice.  If the linear index does not exist for
        the given idxs, then it will be created with sufficient size
        to store values (which must not be None, in this case).  If
        the dataset does not exist, it will be created by this method.
        '''
        with self.lock:
            object_name, feature_name, num_idx = idxs
            assert isinstance(num_idx, int)
            index = self.indices[object_name, feature_name]
            if (num_idx not in index) and (values is None):
                return None  # no data
            if values is not None:
                data_size = np.asanyarray(values).ravel().size
                feature_group = self.top_group.require_group(object_name).require_group(feature_name)
                if num_idx in index:
                    sl = index[num_idx]
                    if data_size > (sl.stop - sl.start):
                        hdf5_index = feature_group['index']
                        hdf5_index[np.flatnonzero(hdf5_index[:, 0] == num_idx), 0] = -1
                        del index[num_idx]
                    elif data_size < (sl.stop - sl.start):
                        hdf5_index = feature_group['index']
                        loc = np.flatnonzero(hdf5_index[:, 0] == num_idx)
                        hdf5_index[loc, 2] = hdf5_index[loc, 1] + data_size
                        index[num_idx] = slice(sl.start, sl.start + data_size)
                if num_idx not in index:
                    grow_by = data_size
                    # create the measurements if needed

                    if not 'data' in feature_group:
                        feature_group.create_dataset('data', (0,), dtype=infer_hdf5_type(values),
                                                     compression='gzip', shuffle=True, chunks=(self.chunksize,), maxshape=(None,))
                        feature_group.create_dataset('index', (0, 3), dtype=int, compression=None,
                                                     chunks=(self.chunksize, 3), maxshape=(None, 3))
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
        if not (self.has_object(object_name) and 
                self.has_feature(object_name, feature_name)):
            return []
        with self.lock:
            if 'index' in self.top_group[object_name][feature_name]:
                idxs = self.top_group[object_name][feature_name]['index'][:, 0][:]
                return idxs[idxs != -1]
            else:
                return []

    def top_level_names(self):
        with self.lock:
            return self.top_group.keys()

    def second_level_names(self, object_name):
        with self.lock:
            return self.top_group[object_name].keys()
        
    def add_all(self, object_name, feature_name, values, idxs = None):
        '''Add all imageset values for a given feature
        
        object_name - name of object supporting the feature
        feature_name - name of the feature
        values - either a list of scalar values or a list of arrays
                 where each array has the values for each of the
                 objects in the corresponding image set.
        idxs - the image set numbers associated with the values. If idxs is
               omitted or None, image set numbers are assumed to go from 1 to N
        '''
        with self.lock:
            self.add_object(object_name)
            if self.has_feature(object_name, feature_name):
                del self.top_group[object_name][feature_name]
                del self.indices[object_name, feature_name]
            self.add_feature(object_name, feature_name)
            if idxs is None:
                idxs = [i+1 for i, value in enumerate(values)
                        if value is not None]
                values = [value for value in values if value is not None]
            if len(values) > 0:
                if np.isscalar(values[0]):
                    idx = np.column_stack((idxs,
                                           np.arange(len(idxs)),
                                           np.arange(len(idxs))+1))
                    assert not isinstance(values[0], unicode), "Unicode must be string encoded prior to call"
                    if isinstance(values[0], str):
                        dataset = np.array(
                            [value for value in values if value is not None],
                            object)
                        dtype = h5py.special_dtype(vlen=str)
                    else:
                        dataset = np.array(values)
                        dtype = dataset.dtype
                else:
                    counts = np.array([len(x) for x in values])
                    offsets = np.hstack([[0], np.cumsum(counts)])
                    idx = np.column_stack((idxs, offsets[:-1], offsets[1:]))
                    dataset = np.hstack(values)
                    dtype = dataset.dtype
                
                self.indices[object_name, feature_name] = dict([
                    (i, slice(start, end)) 
                    for i, start, end in idx])
                feature_group = self.top_group[object_name][feature_name]
                feature_group.create_dataset(
                    'data', data = dataset, 
                    dtype = dtype, compression = 'gzip', shuffle=True,
                    chunks = (self.chunksize, ), 
                    maxshape = (None, ))
                feature_group.create_dataset(
                    'index', data = idx, dtype=int,
                    compression = None, chunks = (self.chunksize, 3),
                    maxshape = (None,3))

def get_top_level_group(filename, group_name = 'Measurements', open_mode='r'):
    '''Open and return the Measurements HDF5 group
    
    filename - path to HDF5 file
    
    group_name - name of top-level group, defaults to Measurements group
    
    open_mode - open mode for file: 'r' for read, 'w' for write
    
    returns the hdf5 file object (which must be closed) and the top-level group
    '''
    f = h5py.File(filename, open_mode)
    return f, f.get(group_name)

class HDFCSV(object):
    '''An HDF representation of a .CSV file
    
    HDF5 structure:
    <hosting group>
       <group name="name", CLASS="CSV">
           <group name=column-name, CLASS="COLUMN">
               <dataset name="index">
               <dataset name="data">
           <group name=column-name, CLASS="COLUMN">
               <dataset name="index">
               <dataset name="data">
    '''
    CURRENT_VERSION = 1
    CLASS = "CLASS"
    COLUMN = "COLUMN"
    VERSION = "VERSION"
    HDFCSV_CLASS = "CSV"
    
    def __init__(self, group, name, lock = None):
        '''Create or bind to a CSV
        
        group - HDF group hosting the CSV
        
        name - name of the CSV within the group
        
        lock - locking object to synchronize hdf access
        '''
        self.parent_group = group
        assert isinstance(group, h5py.Group)
        if lock is None:
            self.lock = NullLock()
        else:
            self.lock = lock
        if name in self.parent_group:
            self.top_level_group = self.parent_group[name]
            assert isinstance(self.top_level_group, h5py.Group)
            assert self.top_level_group.attrs[self.CLASS] == self.HDFCSV_CLASS
            assert self.top_level_group.attrs[self.VERSION] <= self.CURRENT_VERSION
        else:
            self.top_level_group = self.parent_group.create_group(name)
            self.top_level_group.attrs[self.CLASS] = self.HDFCSV_CLASS
            self.top_level_group.attrs[self.VERSION] = self.CURRENT_VERSION
            
    def clear(self):
        '''Clear all columns in the CSV'''
        with self.lock:
            for key in self.top_level_group.keys():
                column = self.top_level_group[key]
                if column.attrs[self.CLASS] == self.COLUMN:
                    del column
                    del self.top_level_group[key]
                
    def add_column(self, name, data = None):
        '''Add a column
        
        name - name of column
        
        data - optional, add the strings for the column
        
        returns the column
        '''
        with self.lock:
            column = self.top_level_group.create_group(name)
            column.attrs[self.CLASS] = self.COLUMN
            kolumn = VStringArray(column, self.lock)
            if data is not None:
                kolumn.set_all(data)
            return kolumn
                
    def set_all(self, d):
        '''Set all of the columns simultaneously
        
        Clears all existing columns and sets up new ones.
        
        d - a dictionary. The keys are used for column names and the
            values should be sequences of strings
        '''
        with self.lock:
            self.clear()
            for k, v in d.iteritems():
                self.add_column(k, v)
                
    def get_column_names(self):
        '''Get the names of the columns'''
        return [
            key for key in self.top_level_group.keys()
            if self.top_level_group[key].attrs[self.CLASS] == self.COLUMN]
        
    def __getitem__(self, key):
        '''Get a column as a VStringArray
        
        key - the name of the column
        
        returns a VStringArray which may be used like a sequence
        '''
        return VStringArray(self.top_level_group[key])
    
    def __len__(self):
        return len(self.get_column_names())
    
    def __iter__(self):
        for x in self.get_column_names():
            yield x
    
    def keys(self):
        return self.get_column_names()
    
    def iterkeys(self):
        return self.get_column_names()
 
class NullLock(object):
    '''A "lock" that does nothing if no locking is needed'''
    def __enter__(self):
        return
        
    def __exit__(self, t, v, tb):
        return
    
class VStringArray(object):
    '''A 1-d array of variable-length strings backed by HDF5 datasets
    
    The structure is an index / length array giving the position within
    the block coupled with a data block. Strings are UTF-8 encoded character
    arrays.
    
    The HDF5 structure:
    <group name>
        dataset["index"] = N x 2 array of starts and ends of UTF-8 strings
        dataset["data"] = 1D array of single characters forming the datablock
        
    None is stored as index[?, 0] > index[?, 1]
    
    The VStringArray is a sequence ducktype - you can get the strings back
    in order by using it as an iterator. The iterator is a thread-safe
    snapshot of the array at the time of the call.
    '''
    VS_NULL = np.iinfo(np.uint32).max

    def __init__(self, group, lock = None):
        '''Initialize or bind to a VStringArray within the named group
        
        group - an HDF5 Group
        lock - a mutex or similar to synchronize access to the array. Default
               is no locking.
        '''
        assert isinstance(group, h5py.Group)
        self.group = group
        if "index" in group:
            self.index = group["index"]
        else:
            self.index = group.create_dataset(
                "index", 
                shape = (0, 2),
                dtype = np.uint32,
                shuffle = True,
                chunks = (256, 2),
                maxshape = (None, 2))
        if "data" in group:
            self.data = group["data"]
        else:
            self.data = group.require_dataset(
                "data", (0, ),
                dtype = "S1",
                shuffle = True,
                compression = "gzip",
                chunks = (32768, ),
                maxshape = (None, ))
        if lock is None:
            self.lock = NullLock()
        else:
            self.lock = lock
        
    def __setitem__(self, idx, value):
        '''Store a single string at a single index
        
        idx - index of string within the array
        value - a UTF-8 encoded string, unicode string, None or object to be
                converted to a string
        '''
        with self.lock:
            if idx < 0:
                idx = self.index.shape[0] - idx
            if value is None:
                if idx >= self.index.shape[0]:
                    self.index.resize(idx+1, 0)
                self.index[idx, :] = (self.VS_NULL, 0)
                return
                
            elif isinstance(value, unicode):
                value = value.encode("utf8")
            else:
                value = str(value)
            if idx >= self.index.shape[0]:
                self.index.resize(idx+1, 0)
                begin = self.data.shape[0]
                self.index[idx, 0] = begin
            elif len(value) <= self.index[idx, 1] - self.index[idx, 0]:
                begin = self.index[idx, 0]
            else:
                begin = self.data.shape[0]
                self.index[idx, 0] = begin
            end = begin + len(value)
            self.index[idx, 1] = end
            if self.data.shape[0] < end:
                self.data.resize(end, 0)
            if begin != end:
                self.data[begin:end] = np.frombuffer(value, "S1")
        
    def __getitem__(self, idx):
        '''Retrieve a single string through the indexing interface.
        
        idx - the index of the string within the array
        
        returns a unicode string or None for empty string
        '''
        with self.lock:
            if idx < 0:
                idx = self.index.shape[0] - idx
            begin, end = self.index[idx, :]
            if begin > end:
                return None
            elif begin == end:
                return u""
            return self.data[begin:end].tostring().decode("utf-8")
        
    def __delitem__(self, idx):
        with self.lock:
            orig_len = self.index.shape[0]
            if idx < 0:
                idx = orig_len - idx
            if idx < orig_len - 1:
                self.index[idx:(orig_len-1), :] = self.index[(idx+1):, :]
            self.index.resize(self.index.shape[0]-1,  0)
    
    def __len__(self):
        '''The number of strings stored in the array'''
        return self.index.shape[0]
    
    def __iter__(self):
        '''Iterates through the items in the array in a threadsafe manner'''
        with self.lock:
            index = self.index[:,:]
            data = self.data[:]
        for begin, end in index:
            yield (None if begin > end else
                   u"" if begin == end else
                   data[begin:end].tostring().decode("utf-8"))
    
    def set_all(self, strings):
        '''Store the strings passed, overwriting any previously stored data'''
        nulls = np.array([s is None for s in strings])
        strings = ["" if s is None 
                   else s.encode("utf-8") if isinstance(s, unicode)
                   else str(s) for s in strings]
        with self.lock:
            target_len = len(strings)
            self.index.resize(target_len, 0)
            if len(strings) > 0:
                self.index[:, 1] = np.cumsum([len(s) for s in strings])
                self.index[0, 0] = 0
                self.index[1:, 0] = self.index[:(target_len-1), 1]
                if np.any(nulls):
                    self.index[nulls, 0] = self.VS_NULL
                self.data.resize(self.index[(target_len-1), 1], 0)
            for s, (begin, end) in zip(strings, self.index):
                if begin < end:
                    self.data[begin:end] = np.frombuffer(s, "S1")

class StringReferencer(object):
    '''This class implements a B-tree of strings within an HDF5 file's group
    
    Usage: 
    sr = StringReferencer(group)
    # Get integer reference IDs to strings
    references = sr.get_string_refs(["foo", "bar", "foo"])
    assert references[0] == references[2] # duplicate strings are stored once
    assert tuple(sr.get_strings(references[:-1])) == ("foo", "bar")
    '''

    '''Default size of a b-tree block'''
    SR_DEFAULT_BLOCKSIZE = 256
    SR_BLOCKSIZE_ATTR = "blocksize"
    '''Size of a data block'''
    SR_DATA_BLOCKSIZE = 4096
    '''Null value (for sub block of leaf or as a result from a search)'''
    SR_NULL = np.iinfo(np.uint32).max
    
    '''The name of the string reference dataset
    
    This dataset is indexed by the string reference number
    and contains the block number, offset and length of
    the referenced string. It's dimensions are N x 3 where
    N is the # of unique strings in the system
    '''
    SR_REF_DS = 'stringref'
    '''The block index of the string'''
    SR_REF_BLOCK = 0
    '''The index of the string within the block'''
    SR_REF_IDX = 1
    
    def __init__(self, group, blocksize=None):
        assert isinstance(group, h5py.Group)
        self.group = group
        self.blocksize = self.group.attrs.get(self.SR_BLOCKSIZE_ATTR,
                                              self.SR_DEFAULT_BLOCKSIZE)
        self.blockdesc = self.get_blockdesc_dataset()
        self.refs = self.get_ref_dataset()
        self.blocks = self.get_block_dataset()
        self.data = self.get_data_dataset()

    def get_ref_dataset(self):
        '''Get the string reference dataset
        
        group - the group housing the dataset
        
        returns the N x 2 string reference dataset. An index into this dataset
        gives the block and offset within the block of the start of the string
        as well as the length of the string.
        '''
        ds_ref = self.group.require_dataset(
            self.SR_REF_DS,
            (0, 2),
            dtype = np.uint64,
            shuffle = True,
            chunks = (self.blocksize * 4, 2),
            maxshape = (None, 2))
        return ds_ref
    
    '''The name of the dataset holding the offset and length of strings in a block
    
    This dataset maintains the offset of a string within the datablock and the
    length of the string. The value at I,J,0 gives the offset in SR_DATA_DS to
    the Jth string in block I and the value at I,J,1 gives its length.
    '''
    SR_BLOCK_DS = 'blocks'
    '''The reference index of the string'''
    SR_BLOCK_REF = 0
    '''The offset of the string in the data block'''
    SR_BLOCK_OFF = 1
    '''The length of the string'''
    SR_BLOCK_LEN = 2
    '''The subblock between this entry and the next (or SR_NULL if a leaf)'''
    SR_BLOCK_SUBBLOCK = 3
    SR_BLOCK_ENTRIES = 4
    
    def get_block_dataset(self):
        '''Get the offset / length dataset
        
        returns the N x M x 3 dataset that, for each of N blocks
        and M entries per block gives the offset and length of the
        Mth string in the Nth block of the string data.
        '''
        ds_ol = self.group.require_dataset(
            self.SR_BLOCK_DS,
            (0, self.blocksize, self.SR_BLOCK_ENTRIES),
            dtype = np.uint32,
            shuffle = True,
            chunks = (4, self.blocksize, self.SR_BLOCK_ENTRIES),
            maxshape = (None, self.blocksize, self.SR_BLOCK_ENTRIES))
        return ds_ol
    
    '''The name of the dataset holding the strings
    
    This dataset has dimensions N x M where N is the block number
    and M is the length of the concatenated strings in the block.
    Note that this compresses well in the M direction since the
    members generally start with the same sequence of characters.
    '''
    SR_DATA_DS = 'data'
    
    def get_data_dataset(self):
        ds_data = self.group.require_dataset(
            self.SR_DATA_DS,
            (0,0),
            dtype = np.uint8,
            shuffle = True,
            compression = 'gzip',
            chunks = (1, self.SR_DATA_BLOCKSIZE),
            maxshape = (None, None))
        return ds_data
    
    '''The dataset holding the block descriptor
    
    This dataset has N x 5 members where N is the number of blocks. It holds
    the current number of indices in each block and the length of the string
    data for the block.
    '''
    SR_BLOCKDESC_DS = 'blockdesc'
    '''Index of the entry giving # of indexes in a block'''
    SR_BLOCKDESC_IDX_LEN = 0
    '''Index of the entry giving # of bytes in the block's data section'''
    SR_BLOCKDESC_DATA_LEN = 1
    '''Block # of this block's parent'''
    SR_BLOCKDESC_PARENT = 2
    '''The index of the node in the parent that is less than us. -1 if we are before all'''
    SR_BLOCKDESC_PARENT_IDX = 3
    '''The index of the leftmost child of this block'''
    SR_BLOCKDESC_LEFTMOST_CHILD = 4
    '''# of entries per block'''
    SR_BLOCKDESC_ENTRIES = 5
    '''The attribute on the blockdesc dataset specifying the current root block of the b-tree'''
    SR_ROOT_ATTR = "root"
    
    def get_blockdesc_dataset(self):
        '''Get the dataset holding the block descriptors for each block'''
        ds_blockdesc  = self.group.require_dataset(
            self.SR_BLOCKDESC_DS,
            (0, self.SR_BLOCKDESC_ENTRIES),
            dtype = np.uint32,
            shuffle = True,
            chunks = (256, self.SR_BLOCKDESC_ENTRIES),
            maxshape = (None, self.SR_BLOCKDESC_ENTRIES))
        return ds_blockdesc
    
    @staticmethod
    def string_to_uint8(s):
        '''Convert a utf-8 encoded string to a np.uint8 array'''
        if isinstance(s, unicode):
            s = s.encode('utf-8')
        elif not isinstance(s, str):
            s = str(s)
        result = np.zeros(len(s), np.uint8)
        result.data[:] = s
        return result
    
    def get_strings(self, refs):
        refs, reverse_indices = np.unique(refs, return_inverse=True)
        strings = [self.get_unicode_from_block_and_idx(*self.refs[ref, :])
                   for ref in refs]
        return [strings[idx] for idx in reverse_indices]
    
    def get_unicode_from_block_and_idx(self, i, j):
        '''Return a unicode string given a block and index within the block
        
        i: block #
        j: index within block
        '''
        data_off, data_len = \
            self.blocks[i, j, self.SR_BLOCK_OFF:(self.SR_BLOCK_LEN+1)]
        s = str(self.data[i, data_off:(data_off + data_len)].data)
        return s.decode("utf-8")
        
    def get_string_refs(self, strings):
        '''Get references to strings
        
        Return an integer per string. The integer can be used later to fetch
        the string. Strings are stored in B-Trees and each unique string is
        stored only one time (so if you have a measurement that stores the same
        string 100,000 times, no big deal).
        
        The strategy is to have a top-level table that gives the index and length
        of a string. The index is 2-d where the first dimension is the block
        holding the string and the second is the offset to the start of the block.
        
        Data is stored as uint8, so strings will be UTF-8 encoded on input and
        reported as unicode on output.
        
        strings - a collection of string / unicode
        group - an HDF5 group used to manage the references
        
        returns a numpy array of ints which are references to the strings
        '''
        strings, reverse_indices = np.unique(np.array(strings, object), 
                                             return_inverse = True)
        strings = [self.string_to_uint8(s) for s in strings]
        indices = []
        if self.blocks.shape[0] == 0:
            block = self.sr_alloc_block()
            self.blockdesc.attrs[self.SR_ROOT_ATTR] = block
            self.refs.resize(self.refs.shape[0]+1, 0)
            #
            # Build the first block
            #
            s0 = strings[0]
            strings = strings[1:]
            indices.append(0)
            self.blockdesc[block, :] = (
                1, # entries
                len(s0), # current length of data section
                self.SR_NULL, # left block ptr
                self.SR_NULL, # parent block
                self.SR_NULL) # parent index
            self.data.resize(len(s0), 1)
            self.data[block, :len(s0)] = s0
            self.blocks[block, 0, :] = (0, 0, len(s0), self.SR_NULL)
            self.refs[block, :] = (0, 0)
        for s in strings:
            i, j, found = self.sr_search(s)
            if not found:
                if self.blockdesc[i][self.SR_BLOCKDESC_IDX_LEN] == self.blocksize:
                    # Need to split the block before insertion
                    sr_split_block(i)
                    i, j, idx = sr_search(s)
                idx = self.refs.shape[0]
                self.refs.resize(idx+1, 0)
                self.refs[idx, : ] = (i, j)
                self.sr_insert(s, idx, i, j)
            else:
                idx = self.blocks[i, j, self.SR_BLOCK_REF]
            indices.append(idx)
        #
        # distribute the indices back onto the original input strings
        #
        indices = np.array(indices)
        return indices[reverse_indices]
    
    def sr_alloc_block(self):
        '''Allocate a new block
        
        returns the block number
        '''
        idx = self.blockdesc.shape[0]
        self.blockdesc.resize(idx+1, 0)
        self.blocks.resize(idx+1, 0)
        self.data.resize(idx+1, 0)
        self.blockdesc[idx, :] = (0, 0, self.SR_NULL, self.SR_NULL, 0)
        return idx
    
    def sr_split_block(self, i):
        '''Split a block in half
        
        i - the block number
        refs - the block / index for string references
        ol - the per-reference data stored as block / index
        data - the storage for the string data.
        '''
        # the index of the ref that's promoted
        i1 = self.sr_alloc_block()
        idx_len, data_len, i0, j0, leftmost_child = self.blockdesc[i, :]
        j = int((idx_len-1) / 2)
        
        j_ref, j_data_idx, j_data_len, j_subblock = self.block[i, j, :]
        j_data = self.data[i, j_data_idx:(j_data_idx+j_data_len)]
        if i0 == SR_NULL:
            # Splitting the root. We need to promote.
            i0 = self.sr_alloc_block(blockdesc, ol, data)
            j0 = 0
            self.blockdesc[i0, self.SR_BLOCKDESC_LEFTMOST_CHILD] = i
        elif self.blockdesc[i0, self.SR_BLOCKDESC_IDX_LEN] == self.blocksize:
            # Parent is full too
            self.sr_splitblock(i0)
            i0 = self.blockdesc[i, self.SR_BLOCKDESC_PARENT]
            j0 = self.blockdesc[i, self.SR_BLOCKDESC_PARENT_IDX]
        # insert the string in the promotion slot
        # the block to the right is the new block
        self.sr_insert(j_data, i0, j0, i1)
        if j_subblock != self.SR_NULL:
            self.blockdesc[j_subblock, 
                           self.SR_BLOCKDESC_PARENT:
                           (self.SR_BLOCKDESC_PARENT_IDX+1)] = (i1, -1)
            self.blockdesc[i1, self.SR_BLOCKDESC_LEFTMOST_CHILD] = j_subblock
        self.refs[j_ref, self.SR_REF_BLOCK:(self.SR_REF_IDX + 1)] = (i0, j0)
        #
        # Copy the right-hand half to the new block.
        #
        j_right = j+1
        rh_idx_len = idx_len - j_right
        rh_data_idx = j_data_idx+j_data_len
        rh_data_end = self.blockdesc[i, self.SR_BLOCKDESC_DATA_LEN] 
        rh_data_len = rh_data_end - rh_data_idx
        self.data[i1, :rh_data_len] = self.data[i, rh_data_idx: rh_data_end]
        #
        # Copy the block data - adjust data pointers at same time
        #
        adjustment = np.array([0, - rh_data_idx, 0, 0])[np.newaxis, :]
        self.block[i1, : rh_idx_len, :] =\
            self.block[i, j_right:idx_len, :] - adjustment
        if leftmost_child != self.SR_NULL:
            # If not a leaf, have to adjust children's parents.
            for new_j, subblock in enumerate(
                self.block[i1, :rh_idx_len, self.SR_BLOCK_SUBBLOCK]):
                self.blockdesc[subblock, 
                               self.SR_BLOCKDESC_PARENT:
                               (self.SR_BLOCKDESC_PARENT_IDX+1)] = (i1, new_j)
        #
        # Readjust old block's blockdesc
        #
        self.blockdesc[i, self.SR_BLOCKDESC_IDX_LEN] = j
        self.blockdesc[i, self.SR_BLOCKDESC_DATA_LEN] = j_data_len
        #
        # Set new block's blockdesc
        #
        self.blockdesc[i1, self.SR_BLOCKDESC_IDX_LEN] = rh_idx_len
        self.blockdesc[i1, self.SR_BLOCKDESC_DATA_LEN] = rh_data_len
    
    def sr_insert(self, s, idx, i, j, next_child = SR_NULL):
        '''Open up a slot in block i at position j and insert string s
        
        s - string to insert
        idx - ref of the string
        i - insert in this block
        j - insert at this position
        refs - reference index -> block and position
        blockdesc - block descriptor for the target block
        ol - string descriptors
        next_child - block # of child after this one
        '''
        len_s = len(s)
        data_len = self.blockdesc[i, self.SR_BLOCKDESC_DATA_LEN]
        if data_len + len_s > self.data.shape[1]:
            self.data.resize(data_len + len_s, 1)
        self.blockdesc[i, self.SR_BLOCKDESC_DATA_LEN] = data_len + len_s
        idx_len = self.blockdesc[i, self.SR_BLOCKDESC_IDX_LEN]
        self.blockdesc[i, self.SR_BLOCKDESC_IDX_LEN] = idx_len + 1
        #
        # Shift right to make space
        #
        if idx_len > j:
            # Increment the index pointer for the reference for
            # all shifted references
            for ref in self.blocks[i, j:idx_len, self.SR_BLOCK_REF]:
                self.refs[ref, self.SR_REF_IDX] += 1
            #
            # shift the per-string information
            #
            self.blocks[i, (j+1):(idx_len+1), :] = self.blocks[i, j:idx_len, :]
            #
            # shift the strings
            #
            self.blocks[i, (j+1):(idx_len+1), self.SR_BLOCK_OFF] += len_s
            data_idx = self.blocks[i, j, self.SR_BLOCK_OFF]
            self.data[i, (data_idx + len_s):(data_len + len_s)] = \
                self.data[i, data_idx:data_len]
        else:
            data_idx = data_len
        self.data[i, data_idx:(data_idx + len_s)] = s
        self.blocks[i, j, :] = (idx, data_idx, len(s), next_child)
        self.refs[idx, :] = (i, j)
        
    def sr_search(self, s):
        '''Search for s in btree
        
        s: a uint8 numpy array string representation, e.g. as returned by
           string_to_uint8
           
        returns the block #, index of entry or insertion point 
                and a True / False indicator of whether there was an exact match
        '''
        #
        # bisection code taken from Python bisect package
        # author: Raymond Hettinger
        #
        block_idx = self.blockdesc.attrs[self.SR_ROOT_ATTR]
        OFF = self.SR_BLOCK_OFF
        LEN = self.SR_BLOCK_LEN
        s_len = len(s)
        while True:
            block_len = self.blockdesc[block_idx, self.SR_BLOCKDESC_IDX_LEN]
            data_len = self.blockdesc[block_idx, self.SR_BLOCKDESC_DATA_LEN]
            block = self.blocks[block_idx, :block_len, :]
            data = self.data[block_idx, :data_len]
            hi = block_len
            lo = 0
            while lo < hi:
                mid = int((lo+hi)/2)
                s1_off = block[mid, OFF]
                s1_len = block[mid, LEN]
                s1 = data[s1_off:(s1_off + s1_len)]
                minlen = min(s1_len, s_len)
                s_s1_ne = s[:minlen] != s1[:minlen]
                if not np.any(s_s1_ne):
                    if s_len == s1_len:
                        return block_idx, mid, True
                    elif s1_len < s_len:
                        lo = mid+1
                    else:
                        hi = mid
                elif s1[s_s1_ne][0] < s[s_s1_ne][0]:
                    lo = mid+1
                else:
                    hi = mid
            if lo == 0:
                next_block_idx = self.blockdesc[
                    block_idx, self.SR_BLOCKDESC_LEFTMOST_CHILD]
            else:
                next_block_idx = block[lo - 1, self.SR_BLOCK_SUBBLOCK]
            if next_block_idx == self.SR_NULL:
                return block_idx, lo, False
        
        
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
    print h['Object1', 'objfeature1', 1]
    h['Object1', 'objfeature1', 2] = 3.0
    print h['Object1', 'objfeature1', 1]
    h['Object1', 'objfeature1', 1] = [1, 2, 3]
    h['Object1', 'objfeature1', 1] = [1, 2, 3, 5, 6]
    h['Object1', 'objfeature1', 1] = [9, 4.0, 2.5]
    print     h['Object1', 'objfeature1', 1]
