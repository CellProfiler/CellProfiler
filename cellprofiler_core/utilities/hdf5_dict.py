"""hdf5_dict -- HDF5-backed dictionary for Measurements.

This module implements the HDF5Dict class, which provides a dict-like
interface for measurements, backed by an HDF5 file.
"""

import bisect
import functools
import logging
import os
import sys
import threading
import time
import uuid

import h5py
import numpy
from future.standard_library import install_aliases

import cellprofiler_core.utilities.legacy

install_aliases()


version_number = 1
VERSION = "Version"

INDEX = "index"
DATA = "data"

# h5py is nice, but not being able to make zero-length selections is a pain.
orig_hdf5_getitem = h5py.Dataset.__getitem__


def new_getitem(self, args):
    if isinstance(args, slice) and args.start is not None and args.start == args.stop:
        return numpy.array([], self.dtype)
    return orig_hdf5_getitem(self, args)


setattr(h5py.Dataset, orig_hdf5_getitem.__name__, new_getitem)

orig_hdf5_setitem = h5py.Dataset.__setitem__


def new_setitem(self, args, val):
    if isinstance(args, slice) and args.start is not None and args.start == args.stop:
        return numpy.array([], self.dtype)[0:0]
    return orig_hdf5_setitem(self, args, val)


setattr(h5py.Dataset, orig_hdf5_setitem.__name__, new_setitem)


def infer_hdf5_type(val):
    if isinstance(val, str) or numpy.sctype2char(numpy.asanyarray(val).dtype) == "S":
        return h5py.special_dtype(vlen=str)
    if all(isinstance(v, str) for v in val):
        return h5py.string_dtype()
    val = numpy.asanyarray(val)
    if val.size == 0:
        return int
    return numpy.asanyarray(val).dtype


FILE_LIST_GROUP = "FileList"
DEFAULT_GROUP = "Default"
TOP_LEVEL_GROUP_NAME = "Measurements"
IMAGES_GROUP = "Images"
OBJECTS_GROUP = "Objects"
FILE_METADATA_GROUP = "FileMetadata"
A_TIMESTAMP = "Timestamp"
"""The attribute on a group or dataset that indicates how the data is organized"""
A_CLASS = "Class"
"""Indicates that a group has a filelist directory structure"""
CLASS_DIRECTORY = "Directory"
CLASS_VSTRING_ARRAY_INDEX = "VStringArrayIndex"
CLASS_VSTRING_ARRAY_DATA = "VStringArrayData"
CLASS_FILELIST_GROUP = "FileListGroup"
CLASS_SEGMENTATION_GROUP = "SegmentationGroup"


class HDF5Dict(object):
    """The HDF5Dict can be used to store data indexed by a tuple of
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
    """

    # XXX - document how data is stored in hdf5 (basically, /Measurements/Object/Feature)

    def __init__(
        self,
        hdf5_filename,
        top_level_group_name=TOP_LEVEL_GROUP_NAME,
        run_group_name=None,
        is_temporary=False,
        copy=None,
        mode="w",
        image_numbers=None,
    ):
        """Initialize the HDF5Dict

        hdf5_filename - name of the file to open / create.
                        If None, a memory-only file is created.

        top_level_group_name - Name of the group hosting all measurements.
                               The default is "Measurements".

        run_group_name - Name of the particular measurements instance. You can
                         store several versions of measurements or measurements
                         from different CellProfiler runs by using a different
                         run group name. If you open the file as
        """
        assert mode in ("r", "r+", "w", "x", "w-", "a")
        self.mode = mode
        file_exists = (hdf5_filename is not None) and os.path.exists(hdf5_filename)
        default_run_group_name = time.strftime("%Y-%m-%d-%H-%m-%S")
        if mode in ("r", "r+"):
            load_measurements = True
        elif mode == "a" and file_exists:
            load_measurements = True
        else:
            load_measurements = False
            run_group_name = default_run_group_name

        self.is_temporary = is_temporary and (hdf5_filename is not None)
        self.filename = hdf5_filename
        self.top_level_group_name = top_level_group_name
        logging.debug(
            "HDF5Dict.__init__(): %s, temporary=%s, copy=%s, mode=%s",
            self.filename,
            self.is_temporary,
            copy,
            mode,
        )
        if self.filename is None:
            # core driver requires a unique filename even if the file
            # never touches the disk... maybe the internals check for the
            # conflict?
            #
            if sys.platform == "darwin":
                raise NotImplementedError(
                    "Sorry, for Mac, the core driver is fatally flawed at "
                    "this point and will cause you sorrow when you try to "
                    "flush or close the h5py file.\n"
                    "See http://code.google.com/p/h5py/issues/detail?id=215"
                )
            name = "%s.h5" % uuid.uuid4()
            self.hdf5_file = h5py.File(name, "a", driver="core", backing_store=False)
        else:
            self.hdf5_file = h5py.File(self.filename, mode)
        try:
            if load_measurements:
                if (
                    VERSION not in list(self.hdf5_file.keys())
                    or top_level_group_name not in self.hdf5_file
                ):
                    load_measurements = False
                    run_group_name = default_run_group_name
                else:
                    mgroup = self.hdf5_file[top_level_group_name]
                    if run_group_name is None:
                        if len(list(mgroup.keys())) > 0:
                            run_group_name = sorted(mgroup.keys())[-1]
                        else:
                            run_group_name = default_run_group_name
                            mgroup.create_group(run_group_name)
                    self.top_group = mgroup[run_group_name]

                if mode == "r" and not load_measurements:
                    raise IOError(
                        "%s was opened read-only but contains no measurements"
                        % hdf5_filename
                    )
            if not load_measurements:
                if VERSION not in list(self.hdf5_file.keys()):
                    vdataset = self.hdf5_file.create_dataset(
                        VERSION, data=numpy.array([version_number], int)
                    )
                self.version = VERSION
                mgroup = self.hdf5_file.create_group(top_level_group_name)
                self.top_group = mgroup.create_group(run_group_name)
                self.indices = (
                    {}
                )  # nested indices for data slices, indexed by (object, feature) then by numerical index
            else:
                self.version = self.hdf5_file[VERSION][0]
                self.indices = {}

            self.lock = HDF5Lock()

            self.chunksize = 1024
            if copy is not None:
                if image_numbers is None:
                    for object_name in list(copy.keys()):
                        object_group = copy[object_name]
                        self.top_group.copy(object_group, self.top_group)
                        for feature_name in list(object_group.keys()):
                            # some measurement objects are written at a higher level, and don't
                            # have an index (e.g., Relationship).
                            if "index" in list(object_group[feature_name].keys()):
                                hdf5_index = object_group[feature_name]["index"][:]
                                self.__cache_index(
                                    object_name, feature_name, hdf5_index
                                )
                            else:
                                self.indices[object_name, feature_name] = {}
                else:
                    image_numbers = numpy.array(image_numbers)
                    mask = numpy.zeros(numpy.max(image_numbers) + 1, bool)
                    mask[image_numbers] = True
                    for object_name in list(copy.keys()):
                        src_object_group = copy[object_name]
                        if object_name == "Experiment":
                            self.top_group.copy(src_object_group, self.top_group)
                            for feature_name in list(src_object_group.keys()):
                                hdf5_index = src_object_group[feature_name]["index"][:]
                                self.__cache_index(
                                    object_name, feature_name, hdf5_index
                                )
                            continue
                        dest_object_group = self.top_group.require_group(object_name)
                        for feature_name in list(src_object_group.keys()):
                            src_feature_group = src_object_group[feature_name]
                            if "index" not in src_object_group[feature_name]:
                                dest_object_group.copy(
                                    src_feature_group, dest_object_group
                                )
                                continue
                            src_index_dataset = src_feature_group["index"][:]
                            src_image_numbers = src_index_dataset[:, 0]
                            max_image_number = numpy.max(src_image_numbers)
                            if max_image_number >= len(mask):
                                tmp = numpy.zeros(max_image_number + 1, bool)
                                tmp[: len(mask)] = mask
                                mask = tmp
                            src_dataset = src_feature_group["data"]
                            src_index_dataset = src_index_dataset[
                                mask[src_index_dataset[:, 0]], :
                            ]
                            #
                            # Almost always, the fast case should work. We can
                            # copy a data chunk from one to the other without
                            # having to restructure.
                            #
                            found_bad_case = False
                            for (
                                (prev_num_idx, prev_start, prev_stop),
                                (next_num_idx, next_start, next_stop),
                            ) in zip(src_index_dataset[:-1], src_index_dataset[1:]):
                                if prev_stop != next_start:
                                    found_bad_case = True
                                    break
                            if found_bad_case:
                                for num_idx, start, stop in src_index_dataset:
                                    self[
                                        object_name, feature_name, num_idx
                                    ] = src_dataset[start:stop]
                            else:
                                src_off = src_index_dataset[0, 1]
                                src_stop = src_index_dataset[-1, 2]
                                dest_index_dataset = src_index_dataset.copy()
                                dest_index_dataset[:, 1:] -= src_off
                                dest_feature_group = dest_object_group.require_group(
                                    feature_name
                                )
                                dest_feature_group.create_dataset(
                                    "index",
                                    data=dest_index_dataset.astype(int),
                                    compression=None,
                                    shuffle=True,
                                    chunks=(self.chunksize, 3),
                                    maxshape=(None, 3),
                                )
                                src_chunk = src_dataset[src_off:src_stop]
                                #
                                # Special handling for strings: create the
                                # dataset using the variable length string type
                                # and then set the data
                                #
                                if h5py.check_dtype(vlen=src_dataset.dtype) is str:
                                    ds = dest_feature_group.create_dataset(
                                        "data",
                                        dtype=h5py.special_dtype(vlen=str),
                                        compression="gzip",
                                        shuffle=True,
                                        chunks=(self.chunksize,),
                                        shape=src_chunk.shape,
                                        maxshape=(None,),
                                    )
                                    if len(src_chunk) > 0:
                                        ds[:] = src_chunk
                                else:
                                    dest_feature_group.create_dataset(
                                        "data",
                                        data=src_chunk,
                                        compression="gzip",
                                        shuffle=True,
                                        chunks=(self.chunksize,),
                                        maxshape=(None,),
                                    )
            self.hdf5_file.flush()
        except Exception as e:
            logging.exception("Failed during initial processing of %s" % self.filename)
            self.hdf5_file.close()
            raise

    def __del__(self):
        logging.debug(
            "HDF5Dict.__del__(): %s, temporary=%s", self.filename, self.is_temporary
        )
        self.close()

    def close(self):
        if not hasattr(self, "hdf5_file"):
            # This happens if the constructor could not open the hdf5 file, or
            # if close is called twice.
            return
        if self.is_temporary:
            try:
                self.hdf5_file.flush()  # just in case unlink fails
                self.hdf5_file.close()
                os.unlink(self.filename)
            except Exception as e:
                logging.warning(
                    "So sorry. CellProfiler failed to remove the temporary file, %s and there it sits on your disk now."
                    % self.filename
                )
        else:
            self.hdf5_file.flush()
            self.hdf5_file.close()
        del self.hdf5_file
        del self.top_group

    def flush(self):
        logging.debug(
            "HDF5Dict.flush(): %s, temporary=%s", self.filename, self.is_temporary
        )
        # 2012-06-29: Ray is seeing a bug where file_contents() returns an
        # invalid HDF if the file is flushed once then read, but with two calls
        # to flush() it works.  h5py version 2.1.0, hdf version 1.8.9
        self.hdf5_file.flush()
        # FIXME: Allen says "wtf"
        # self.hdf5_file.flush()

    def file_contents(self):
        with self.lock:
            self.flush()
            try:
                with open(self.filename, "rb") as f:
                    return memoryview(f.read())
            except PermissionError:
                self.hdf5_file.close()
                with open(self.filename, "rb") as f:
                    mem = memoryview(f.read())
                if 'w' not in self.mode:
                    self.hdf5_file = h5py.File(self.filename, mode=self.mode)
                else:
                    self.hdf5_file = h5py.File(self.filename, mode="a")

                #We need to reopen the old group, not just the old file
                mgroup = self.hdf5_file[self.top_level_group_name]
                run_group_name = sorted(mgroup.keys())[-1]
                self.top_group = mgroup[run_group_name]
                return mem

    @classmethod
    def has_hdf5_dict(cls, h5file):
        """Return True if the HDF file has a HDF5Dict in the usual location

        :param h5file: An open HDF5 file

        :returns: True if it has a HDF5Dict
        """
        return TOP_LEVEL_GROUP_NAME in h5file

    @staticmethod
    def __is_positive_int(idx):
        """Return True if the index is a positive integer suitable for HDF5 indexing"""
        return (isinstance(idx, int) or isinstance(idx, numpy.integer)) and idx >= 0

    def __getitem__(self, idxs):
        assert isinstance(
            idxs, tuple
        ), "Accessing HDF5_Dict requires a tuple of (object_name, feature_name[, integer])"
        assert isinstance(idxs[0], str) and isinstance(
            idxs[1], str
        ), "First two indices must be of type str."

        object_name, feature_name, num_idx = idxs
        if numpy.isscalar(num_idx):
            result = self[object_name, feature_name, [num_idx]]
            return result if result is None else result[0]

        feature_exists = self.has_feature(object_name, feature_name)

        assert feature_exists, "Feature {} for {} does not exist".format(
            feature_name, object_name
        )

        with self.lock:
            indices = self.get_indices(object_name, feature_name)
            dataset = self.get_dataset(object_name, feature_name)
            if dataset is None or dataset.shape[0] == 0:
                return [numpy.array([]) for image_number in num_idx]
            if len(indices) / 2 < len(num_idx):
                #
                # Optimize by fetching complete dataset
                # if fetching more than 1/2 of indices
                #
                dataset = dataset[:]
            if dataset.dtype == object:
                # Strings come back out as bytes, we need to decode them.
                try:
                    return [
                        None
                        if (
                            isinstance(dest, slice)
                            and dest.start is not None
                            and dest.start == dest.stop
                        )
                        else dataset[dest].astype(str)
                        for dest in [
                            indices.get(image_number, (slice(0, 0), 0))[0]
                            for image_number in num_idx
                        ]
                    ]
                except Exception as e:
                    logging.error(
                        "Unable to decode object measurement. You may find bytes in your output sheet."
                    )
            return [
                None
                if (
                    (
                        isinstance(dest, slice)
                        and dest.start is not None
                        and dest.start == dest.stop
                    )
                )
                else dataset[dest]
                for dest in [
                    indices.get(image_number, (slice(0, 0), 0))[0]
                    for image_number in num_idx
                ]
            ]

    @staticmethod
    def __all_null(vals):
        if numpy.isscalar(vals):
            return vals is None
        return all(
            [
                x is None or (not numpy.isscalar(x) and HDF5Dict.__all_null(x))
                for x in vals
            ]
        )

    def __make_empty_feature(
        self, object_name, feature_name, image_numbers=None, dtype=int
    ):
        """Create a feature that has only nulls

        lock must be taken prior to call

        object_name - name of feature's object

        feature_name - name of feature within object

        image_numbers - image numbers of the image sets with no values for
                        the feature.

        dtype - the desired data type for the array
        """
        feature_group = self.top_group.require_group(object_name).require_group(
            feature_name
        )
        if image_numbers is None:
            index_slices = numpy.zeros((0, 3), int)
        else:
            index_slices = numpy.column_stack(
                [image_numbers, numpy.zeros((len(image_numbers), 2), int)]
            )
        self.__create_index(feature_group, index_slices)
        feature_group.create_dataset(
            "data",
            (0,),
            dtype=dtype,
            compression="gzip",
            shuffle=True,
            chunks=(self.chunksize,),
            maxshape=(None,),
        )

    def __create_index(self, feature_group, index_slices):
        """Create an index for a feature group

        lock must be taken prior to call

        feature_group - create the dataset in this group

        index_slices - an N x 3 numpy array of the image number, start and stop
                       for the initial indexes
        """
        assert isinstance(feature_group, h5py.Group)
        _, object_name, feature_name = feature_group.name.rsplit("/", 2)
        feature_group.create_dataset(
            "index",
            data=index_slices,
            dtype=int,
            compression=None,
            chunks=(self.chunksize, 3),
            maxshape=(None, 3),
        )
        self.__cache_index(object_name, feature_name, index_slices)

    def __cache_index(self, object_name, feature_name, index_slices):
        """Cache the contents of an "index" dataset in self.indices

        self.indices is a dictionary indexed by object name and feature name
        whose values are themselves dictionaries, indexed by image number.
        The per-image values are the slice of the data in the "data" dataset
        and the index of the entry in the "index" array. This allows efficient
        retrieval of an image set's data; otherwise a complete scan of the
        "index" array would be necessary.

        object_name, feature_name - names of the object and feature to slice

        index_slices - the contents of an "index" dataset or similarly structured
                       Nx3 numpy array. The first column is the image number
                       and the second and third are start and stop values
                       for the slice.
        """
        self.indices[object_name, feature_name] = dict(
            [
                (image_number, (slice(start, stop), i))
                for i, (image_number, start, stop) in enumerate(index_slices)
            ]
        )

    def __setitem__(self, idxs, vals):
        assert isinstance(
            idxs, tuple
        ), "Assigning to HDF5_Dict requires a tuple of (object_name, feature_name, integer)"
        assert isinstance(idxs[0], str) and isinstance(
            idxs[1], str
        ), "First two indices must be of type str."
        assert not numpy.isscalar(idxs[2]) or self.__is_positive_int(
            idxs[2]
        ), "Third index must be a non-negative integer"
        object_name, feature_name, num_idx = idxs[:3]

        if numpy.isscalar(num_idx):
            # An image or experiment feature, typically
            if vals is None:
                vals = []
            elif numpy.isscalar(vals):
                vals = [vals]
            if len(idxs) > 3:
                return self.__setitem__(
                    (object_name, feature_name, [num_idx], idxs[3]), [vals]
                )
            else:
                return self.__setitem__((object_name, feature_name, [num_idx]), [vals])

        num_idx = numpy.atleast_1d(num_idx)
        if len(num_idx) > 0 and (numpy.isscalar(vals[0]) or vals[0] is None):
            # Convert imageset-style to lists per imageset
            vals = [[] if v is None else [v] if numpy.isscalar(v) else v for v in vals]
        all_null = True

        hdf5_type = None
        if len(idxs) > 3 and idxs[3] is not None:
            hdf5_type = idxs[3]
            all_null = False
            hdf5_type_is_int = False
            hdf5_type_is_float = False
            hdf5_type_is_string = False
        else:
            for vector in vals:
                if len(vector) > 0:
                    all_null = False
                    new_dtype = infer_hdf5_type(vector)
                    if hdf5_type is None or hdf5_type == int:
                        hdf5_type = new_dtype
                    elif hdf5_type == float:
                        if new_dtype != int:
                            hdf5_type = new_dtype
                    else:
                        break
            hdf5_type_is_int = numpy.issubdtype(hdf5_type, numpy.signedinteger) or (
                isinstance(hdf5_type, numpy.dtype) and hdf5_type.kind == "u"
            )
            hdf5_type_is_float = numpy.issubdtype(hdf5_type, numpy.floating)
            hdf5_type_is_string = not (hdf5_type_is_int or hdf5_type_is_float)
        with self.lock:
            if not self.has_feature(object_name, feature_name):
                if all_null:
                    self.__make_empty_feature(object_name, feature_name, num_idx)
                else:
                    self.add_all(
                        object_name,
                        feature_name,
                        vals,
                        idxs=num_idx,
                        data_type=hdf5_type,
                    )
                return

            feature_group = self.top_group[object_name][feature_name]
            dataset = feature_group["data"]
            assert isinstance(dataset, h5py.Dataset)
            if all_null:
                index_slices = numpy.zeros((len(num_idx), 3), int)
                index_slices[:, 0] = num_idx
                self.__write_indices(object_name, feature_name, index_slices)
                return
            if dataset.shape[0] == 0:
                recast_dataset = True
            else:
                recast_dataset = False
                ds_type = dataset.dtype.kind
                ds_type_is_string = ds_type in ("S", "U", "O")

                if hdf5_type_is_float and ds_type in ("i", "u"):
                    recast_dataset = True
                elif hdf5_type_is_string and not ds_type_is_string:
                    recast_dataset = True
            if recast_dataset:
                kwds = {
                    "dtype": hdf5_type,
                    "compression": "gzip",
                    "shuffle": True,
                    "chunks": (self.chunksize,),
                    "maxshape": (None,),
                }
                if dataset.shape[0] > 0:
                    if hdf5_type_is_string:
                        kwds["data"] = numpy.array([str(v) for v in dataset[:]], object)
                    else:
                        kwds["data"] = dataset[:]
                else:
                    kwds["shape"] = (0,)
                del feature_group["data"]

                dataset = feature_group.create_dataset("data", **kwds)
            data_lengths = numpy.array([len(v) for v in vals], int)
            if dataset.dtype.kind in ("S", "U", "O"):
                vals = sum([[str(v) for v in vector] for vector in vals], [])
                vals = numpy.array(vals, object).flatten()
            else:
                vals = numpy.hstack(vals).astype(dataset.dtype)
            old_dataset_len = dataset.shape[0]
            data_offsets = numpy.cumsum(data_lengths)
            dataset.resize(old_dataset_len + len(vals), 0)
            dataset[old_dataset_len:] = vals
            index_slices = numpy.column_stack(
                [
                    num_idx,
                    old_dataset_len + data_offsets - data_lengths,
                    old_dataset_len + data_offsets,
                ]
            )
            self.__write_indices(object_name, feature_name, index_slices)

    def __write_indices(self, object_name, feature_name, index_slices):
        """Update the entries in the "index" dataset for the given index slices

        object_name, feature_name - the measurement being written

        index_slices - an N x 3 array of image number, start and stop

        updates both the "index" dataset and the indices cache.
        """

        if len(index_slices) == 0:
            return
        ds_index = self.top_group[object_name][feature_name][INDEX]
        n_current = ds_index.shape[0]
        slots = []
        all_appended = True
        indices = self.get_indices(object_name, feature_name)
        append_index = n_current
        for image_number, start, stop in index_slices:
            if image_number in indices:
                this_slot = indices[image_number][1]
                all_appended = False
            else:
                this_slot = append_index
                append_index += 1
            slots.append(this_slot)
            indices[image_number] = (slice(start, stop), this_slot)
        ds_index.resize(append_index, 0)
        if all_appended:
            ds_index[n_current:, :] = index_slices
        else:
            for slot, row in zip(slots, index_slices):
                ds_index[slot, :] = row

    def __delitem__(self, idxs):
        assert isinstance(
            idxs, tuple
        ), "Accessing HDF5_Dict requires a tuple of (object_name, feature_name, integer)"
        assert isinstance(idxs[0], str) and isinstance(
            idxs[1], str
        ), "First two indices must be of type str."
        if len(idxs) == 3:
            assert (
                isinstance(idxs[2], (int, numpy.integer)) and idxs[2] >= 0
            ), "Third index must be a non-negative integer"

            object_name, feature_name, num_idx = idxs
            feature_exists = self.has_feature(object_name, feature_name)
            assert feature_exists

            if not self.has_data(*idxs):
                return

            with self.lock:
                del self.get_indices(object_name, feature_name)[num_idx]
                # reserved value of -1 means deleted
                idx = self.top_group[object_name][feature_name][INDEX]
                idx[numpy.flatnonzero(idx[:, 0] == num_idx), 0] = -1
        else:
            # Delete the entire measurement
            object_name, feature_name = idxs
            with self.lock:
                if self.has_feature(object_name, feature_name):
                    group = self.top_group[object_name][feature_name]
                    del group[INDEX]
                    del group[DATA]
                    del self.top_group[object_name][feature_name]
                    if (object_name, feature_name) in self.indices:
                        del self.indices[object_name, feature_name]

    def has_data(self, object_name, feature_name, num_idx):
        return num_idx in self.get_indices(object_name, feature_name)

    def get_dataset(self, object_name, feature_name):
        with self.lock:
            return self.top_group[object_name][feature_name][DATA]

    def has_object(self, object_name):
        with self.lock:
            return object_name in list(self.top_group.keys())

    def add_object(self, object_name):
        with self.lock:
            object_group = self.top_group.require_group(object_name)

    def has_feature(self, object_name, feature_name):
        if (object_name, feature_name) in self.indices:
            return True
        return (
            self.has_object(object_name) and feature_name in self.top_group[object_name]
        )

    def add_feature(self, object_name, feature_name):
        with self.lock:
            feature_group = self.top_group[object_name].require_group(feature_name)
            self.indices.setdefault((object_name, feature_name), {})

    def get_feature_dtype(self, object_name, feature_name):
        """Return the dtype of a feature as represented in the HDF dataset

        object_name - name of object
        feature_name - name of feature
        """
        return self.top_group[object_name][feature_name][DATA].dtype

    def clear(self):
        with self.lock:
            for object_name in self.top_level_names():
                del self.top_group[object_name]
            self.indices = {}

    def erase(self, object_name, first_idx, mask):
        with self.lock:
            self.top_group[object_name]["_index"][mask] = -1
            self.level1_indices[object_name].pop(first_idx, None)

    def get_indices(self, object_name, feature_name):
        if (object_name, feature_name) not in self.indices:
            if not self.has_feature(object_name, feature_name):
                return {}
            index_dataset = self.top_group[object_name][feature_name][INDEX][:, :]
            self.__cache_index(object_name, feature_name, index_dataset)
        return self.indices[object_name, feature_name]

    def top_level_names(self):
        with self.lock:
            return list(self.top_group.keys())

    def second_level_names(self, object_name):
        with self.lock:
            return list(self.top_group[object_name].keys())

    def add_all(self, object_name, feature_name, values, idxs=None, data_type=None):
        """Add all imageset values for a given feature

        object_name - name of object supporting the feature
        feature_name - name of the feature
        values - either a list of scalar values or a list of arrays
                 where each array has the values for each of the
                 objects in the corresponding image set.
        idxs - the image set numbers associated with the values. If idxs is
               omitted or None, image set numbers are assumed to go from 1 to N
        data_type - the data type of the array to be created or None to have
                    it inferred.
        """
        with self.lock:
            self.add_object(object_name)
            if self.has_feature(object_name, feature_name):
                del self.top_group[object_name][feature_name]
                if (object_name, feature_name) in self.indices:
                    del self.indices[object_name, feature_name]
            self.add_feature(object_name, feature_name)
            if len(values) > 0 and (numpy.isscalar(values[0]) or values[0] is None):
                # Convert "images"-style value per imageset to a list
                values = [[v] if v is not None else [] for v in values]
            if idxs is None:
                idxs = numpy.arange(1, len(values) + 1)
            dtype = data_type
            if dtype is None:
                for vector in values:
                    if len(vector) > 0:
                        new_dtype = infer_hdf5_type(vector)
                        if dtype is None or dtype == int:
                            dtype = new_dtype
                        elif dtype == float:
                            if new_dtype != int:
                                dtype = new_dtype
                        else:
                            break
            if dtype is None:
                # empty set
                self.__make_empty_feature(object_name, feature_name, idxs)
                return
            elif not numpy.issubdtype(dtype, numpy.number):
                values = [
                    numpy.array([str(v) for v in vector], object) for vector in values
                ]
            else:
                values = [numpy.atleast_1d(vector) for vector in values]
            counts = numpy.array([len(x) for x in values])
            offsets = numpy.hstack([[0], numpy.cumsum(counts)])
            idx = numpy.column_stack((idxs, offsets[:-1], offsets[1:]))
            dataset = numpy.hstack(values)

            self.__cache_index(object_name, feature_name, idx)
            feature_group = self.top_group[object_name][feature_name]
            feature_group.create_dataset(
                "data",
                data=dataset,
                dtype=dtype,
                compression="gzip",
                shuffle=True,
                chunks=(self.chunksize,),
                maxshape=(None,),
            )
            feature_group.create_dataset(
                "index",
                data=idx,
                dtype=int,
                compression=None,
                chunks=(self.chunksize, 3),
                maxshape=(None, 3),
            )

    def reorder(self, object_name, feature_name, image_numbers):
        """Change the image set order for a feature

        object_name, feature_name - picks out the feature to be modified
        image_numbers - an array that maps old image number to new image number.
                        The value in image_numbers[N] is the new image number
                        for the measurement for old image number N. The array
                        is zero-based even though there may not be an image
                        number zero.

        Note: this is intended primarily for reordering during prepare_run.
              The image numbers will most likely be used as references within
              other measurements at later stages of the pipeline and simply
              remapping here is not sufficient.
        """
        with self.lock:
            feature_group = self.top_group.require_group(object_name).require_group(
                feature_name
            )
            if INDEX not in feature_group:
                # All values are None for the feature
                return
            index_array = feature_group[INDEX][:, :]
            index_array[:, 0] = image_numbers[index_array[:, 0]]
            #
            # Reorder sequentially.
            #
            order = numpy.lexsort((index_array[:, 0],))
            index_array = index_array[order, :]
            feature_group[INDEX][:, :] = index_array
            self.__cache_index(object_name, feature_name, index_array)


class HDF5FileList(object):
    """An HDF5FileList is a hierarchical directory structure backed by HDF5

    The HDFFileList holds a list of URLS in a hierarchical directory structure
    that lets the caller list, add and remove the URLs in a directory. It
    is meant to be used for a list of files curated by the user. The structure
    is the following:

    FileList / group (default = default
       schema name
          directory name
              ....
              sub directory name
                   index
                   data
                   metadata
                       index
                       data

    index and data are parts of a VStringArray (see below) and URLs are
    stored in alpabetical order in the array. The metadata group contains a
    second string array whose indices correspond to those for the file name.
    The metadata is the OME-XML as fetched by Bioformats.

    Schema names and directory names are escape-encoded to allow characters that
    can appear in URLs but could cause problems as group names, most notably,
    forward-slash. Characters other than alphanumerics, and percent ("%"),
    equals ("="), period ("."), underbar ("_") plus ("+") and dash ("-") are
    translated into backslash + 2 hex characters (for instance, "(hello)"
    is encoded as "\50hello\51").

    Pragmatically, aside from perhaps a filename with a true
    backslash in it, the group names will be the same as the parts of the
    url path with the one disturbing exception of the first one, because
    there can be from zero to three consecutive forward slashes at the
    start of the path and DOS file paths often start with c:. So there...
    "c:\foo\bar" becomes "file:///C:/foo/bar" as a URL and becomes
    "file", "\2F\2FC\58". SORRY!
    """

    @classmethod
    def has_file_list(cls, hdf5_file):
        """Return True if the hdf5 file has a file list

        hdf5_file - an h5py.File
        """
        assert isinstance(hdf5_file, h5py.File)
        if not FILE_LIST_GROUP in list(hdf5_file.keys()):
            return False
        flg = hdf5_file[FILE_LIST_GROUP]
        for key in list(flg.keys()):
            g = flg[key]
            if g.attrs.get(A_CLASS, None) == CLASS_FILELIST_GROUP:
                return True
        else:
            return False

    @classmethod
    def copy(cls, src, dest):
        """Copy the file list from one HDF5 file to another

        src - a h5py.File with a file list

        dest - destination for file list

        Any file list in dest will be erased.
        """
        assert isinstance(src, h5py.File)
        assert isinstance(dest, h5py.File)
        if not cls.has_file_list(src):
            return

        flg = src[FILE_LIST_GROUP]
        for key in list(flg.keys()):
            src_g = flg[key]
            if src_g.attrs.get(A_CLASS, None) == CLASS_FILELIST_GROUP:
                break
        dest_flg = dest.require_group(FILE_LIST_GROUP)
        for key in list(dest_flg.keys()):
            g = dest_flg[key]
            if g.attrs.get(A_CLASS, None) == CLASS_FILELIST_GROUP:
                del dest_flg[key]
        dest.copy(src_g, dest_flg)

    def __init__(self, hdf5_file, lock=None, filelist_name=DEFAULT_GROUP):
        """Initialize self with an HDF5 file

        hdf5_file - a h5py.File or a h5py.Group if you are perverse

        lock - a mutex object for locking such as threading.RLock. Default
               is no locking.

        filelist_name - the name of this filelist within the file. Defaults
                        to "Default".
        """
        self.hdf5_file = hdf5_file
        if lock is None:
            self.lock = NullLock()
        else:
            self.lock = lock
        g = self.hdf5_file.require_group(FILE_LIST_GROUP)
        if filelist_name in g:
            g = g[filelist_name]
            # When loading CP 3.1.8 cpprojs, A_CLASS val is in bytes, thus the legacy.equals()
            assert cellprofiler_core.utilities.legacy.equals(
                g.attrs.get(A_CLASS, None), CLASS_FILELIST_GROUP
            )
        else:
            g = g.require_group(filelist_name)
            g.attrs[A_CLASS] = CLASS_FILELIST_GROUP
        self.__top_level_group = g
        self.__cache = {}
        self.__notification_list = []
        self.__generation = uuid.uuid4()

    class __CacheEntry(object):
        """A cache entry in the file list cache

        The cache entry for a directory has the URLS for the directory,
        the HDF5 group for the entry and an array of booleans that indicate
        whether metadata was collected per URL.
        """

        def __init__(self, group, urls, has_metadata):
            self.group = group
            self.urls = tuple(urls)
            self.has_metadata = has_metadata

    def get_generation(self):
        """The generation # of this file list

        The generation # is incremented each time the file list changes (including
        the metadata). Users of the file list can use the generation to determine
        if derivative calculations need to be recalculated.
        """
        return self.__generation

    generation = property(get_generation)

    def add_notification_callback(self, callback):
        """Add a callback that will be called if the file list changes in any way

        callback - a function taking no arguments.
        """
        self.__notification_list.append(callback)

    def remove_notification_callback(self, callback):
        """Remove a previously installed callback"""
        self.__notification_list.remove(callback)

    def get_notification_callbacks(self):
        return list(self.__notification_list)

    def notify(self):
        for callback in self.__notification_list:
            callback()

    def get_filelist_group(self):
        """Get the top-level group of this filelist"""
        return self.__top_level_group

    LEGAL_GROUP_CHARACTERS = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+.%="
    )

    @staticmethod
    def encode(name):
        """Encode a name so it can be used as the name of a group

        Sadness: HDF5 interprets names such as "/foo" as a path from root and
                 doesn't handle things like period, question mark, etc
                 (see http://www.hdfgroup.org/HDF5/doc/UG/UG_frame13Attributes.html)

                 So we need to do yet another lame, arbitrary encode/decode.
                 Apparently, backslash is legal as are letters, numbers,
                 underbar and dash. So the encoding is backslash + 2 hex
                 digits for everything else, including backslash.

                 And if that's not enough, the keywords, "index" and "data"
                 are needed for vstringarrays, so we encode "index" as
                 "\69ndex" and "\64ata"
        """
        #
        # I sure hope this isn't slow...
        #
        if name in ("index", "data", "metadata"):
            return r"\%02x%s" % (ord(name[0]), name[1:])
        return "".join(
            [
                c if c in HDF5FileList.LEGAL_GROUP_CHARACTERS else r"\%02x" % ord(c)
                for c in name
            ]
        )

    @staticmethod
    def decode(name):
        """Decode a name back to plaintext

        see encode for details and editorial commentary
        """
        # Split string at every backslash. Every string after the first
        # begins with two hex digits which contain the character to convert
        parts = name.split("\\")
        return parts[0] + "".join([chr(int(s[:2], 16)) + s[2:] for s in parts[1:]])

    @staticmethod
    def split_url(url, is_directory=False):
        """Split a URL into the pieces that are used to traverse groups

        url - a url

        is_directory - if true, then the name should be treated as a root
                       directory which will have a filename concatenated to
                       it... the soul-deadening use case for this is
                       "file://foo.jpg" which might be schema = "file",
                       first directory = "//", filename = "foo.jpg" or it
                       just might be that the caller wants to know what's
                       in the "//foo.jpg" directory.

        returns a two tuple of schema + path part sequence
        """
        if isinstance(url, str):
            url = url
        else:
            url = str(url)
        import urllib.parse, urllib.error

        split = urllib.parse.urlsplit(str(url))
        schema = split[0]
        if schema == "":
            schema = None
        if schema is not None:
            rest = url.split(schema, 1)[1]
            if rest[0] == ":":
                rest = rest[1:]
        else:
            rest = url

        if schema is not None and schema.lower() == "omero":
            return schema, [rest]
        #
        # The first part always has leading slashes which should be preserved.
        # If there are double slashes after the first, just remove them.
        #
        for i in range(len(rest)):
            if rest[i] != "/":
                parts = [s for s in rest[i:].split("/") if len(s) > 0]
                if not is_directory and len(parts) == 1 and i > 0:
                    return schema, [rest[:i], rest[i:]]
                return schema, [rest[:i] + parts[0]] + parts[1:]
        #
        # If no slashes in url (e.g., http:someplace.org ), return schema + rest
        #
        return schema, [rest]

    def add_files_to_filelist(self, urls):
        self.__generation = uuid.uuid4()
        d = {}
        timestamp = time.time()
        for url in urls:
            schema, parts = self.split_url(url)
            if schema not in d:
                d[schema] = {}
            d1 = d[schema]
            for part in parts[:-1]:
                if part not in d1:
                    d1[part] = {}
                d1 = d1[part]
            if None not in d1:
                d1[None] = []
            d1[None].append(parts[-1])

        def fn(g, d, parts=None):
            if parts is None:
                parts = []
            for k in d:
                if k is None:
                    g.attrs[A_TIMESTAMP] = timestamp
                    metadata_group = g.require_group("metadata")
                    metadata = VStringArray(metadata_group, self.lock)
                    dest = VStringArray(g, self.lock)
                    leaves = list(dest)
                    old_len = len(leaves)
                    to_add = set(d[k]).difference(leaves)
                    if len(to_add) > 0:
                        leaves += to_add
                        dest.extend(to_add)
                        sort_order = sorted(
                            list(range(len(leaves))),
                            key=functools.cmp_to_key(
                                lambda x, y: cellprofiler_core.utilities.legacy.cmp(
                                    leaves[x], leaves[y]
                                )
                            ),
                        )
                        dest.reorder(sort_order)
                        metadata.extend([None] * len(to_add))
                        metadata.reorder(sort_order)
                        self.__cache[tuple(parts)] = self.__CacheEntry(
                            g, [leaves[i] for i in sort_order], metadata.is_not_none()
                        )
                else:
                    g1 = g.require_group(self.encode(k))
                    g1.attrs[A_CLASS] = CLASS_DIRECTORY
                    fn(g1, d[k], parts + [k])

        with self.lock:
            fn(self.get_filelist_group(), d)
            self.hdf5_file.flush()
        self.notify()

    def clear_cache(self):
        self.__cache = {}

    def clear_filelist(self):
        """Remove all files from the filelist"""
        self.__generation = uuid.uuid4()
        group = self.get_filelist_group()
        with self.lock:
            schemas = [
                k
                for k in list(group.keys())
                if group[k].attrs[A_CLASS] == CLASS_DIRECTORY
            ]
            for key in schemas:
                del group[key]
            self.hdf5_file.flush()
        self.clear_cache()
        self.notify()

    def remove_files_from_filelist(self, urls):
        self.__generation = uuid.uuid4()
        group = self.get_filelist_group()
        d = {}
        for url in urls:
            schema, parts = self.split_url(url)
            if schema not in d:
                d[schema] = {}
            d1 = d[schema]
            for part in parts[:-1]:
                if part not in d1:
                    d1[part] = {}
                d1 = d1[part]
            if None not in d1:
                d1[None] = []
            d1[None].append(parts[-1])

        def fn(g, d, parts=None):
            if parts is None:
                parts = []
            for k in d:
                next_parts = parts + [k]
                parts_tuple = tuple(parts)
                if k is None:
                    to_remove = set(d[k])
                    dest = VStringArray(g, self.lock)
                    metadata = VStringArray(g.require_group("metadata"), self.lock)
                    leaves = list(dest)
                    order = [
                        i for i, leaf in enumerate(leaves) if leaf not in to_remove
                    ]
                    if len(order) > 0:
                        dest.reorder(order)
                        metadata.reorder(order)
                        self.__cache[parts_tuple] = self.__CacheEntry(
                            g, [leaves[o] for o in order], metadata.is_not_none()
                        )
                    else:
                        dest.delete()
                        del g["metadata"]
                        self.__cache[parts_tuple] = self.__CacheEntry(g, [], [])
                else:
                    encoded_key = self.encode(k)
                    g1 = g.require_group(encoded_key)
                    has_grandchildren = fn(g1, d[k], next_parts)
                    if not has_grandchildren:
                        del g[encoded_key]
            if VStringArray.has_vstring_array(g):
                return True
            for k in g:
                if g[k].attrs.get(A_CLASS, None) == CLASS_DIRECTORY:
                    return True
            return False

        with self.lock:
            fn(self.get_filelist_group(), d)
            self.hdf5_file.flush()
        self.notify()

    def has_files(self):
        """Return True if there are files in the file list"""
        if any([len(ce.urls) > 0 for ce in list(self.__cache.values())]):
            return True
        group_list = [self.get_filelist_group()]
        path_list = [[]]
        while len(group_list) > 0:
            g = group_list.pop()
            path = path_list.pop()
            if VStringArray.has_vstring_array(g):
                self.cache_urls(g, tuple(path))
                return True
            for k in list(g.keys()):
                g0 = g[k]
                path0 = path + [self.decode(k)]
                if self.is_dir(g0):
                    group_list.append(g0)
                    path_list.append(path0)
        return False

    @staticmethod
    def is_dir(g):
        """Return True if a group is a directory

        g - an hdf5 object which may be a group marked as a file list group
        """
        return (
            isinstance(g, h5py.Group)
            and A_CLASS in g.attrs
            and cellprofiler_core.utilities.legacy.equals(
                g.attrs[A_CLASS], CLASS_DIRECTORY
            )
        )

    def get_filelist(self, root_url=None):
        """Retrieve all URLs from a filelist

        root_url - if present, get the file list below this directory.

        returns a sequence of urls
        """
        group = self.get_filelist_group()
        with self.lock:
            if root_url is None:
                schemas = [
                    k for k in list(group.keys()) if HDF5FileList.is_dir(group[k])
                ]
                roots = [(s + ":", group[s], [s]) for s in schemas]
            else:
                schema, path = self.split_url(root_url, is_directory=True)
                g = group[self.encode(schema)]
                for part in path:
                    g = g[self.encode(part)]
                if not root_url.endswith("/"):
                    root_url += "/"
                roots = [(root_url, g, path)]

            def fn(root, g, path):
                urls = []
                path_tuple = tuple(path)
                if path_tuple in self.__cache:
                    a = cellprofiler_core.utilities.legacy.convert_bytes_to_str(
                        self.__cache[path_tuple].urls
                    )
                    urls += [root + x for x in a]
                elif VStringArray.has_vstring_array(g):
                    a = cellprofiler_core.utilities.legacy.convert_bytes_to_str(
                        self.cache_urls(g, path_tuple)
                    )
                    urls += [root + x for x in a]
                for k in sorted(g.keys()):
                    g0 = g[k]
                    if self.is_dir(g0):
                        decoded_key = self.decode(k)
                        if decoded_key.endswith("/"):
                            # Special case - root of "file://foo.jpg" is
                            # "file://"
                            subroot = root + decoded_key
                        else:
                            subroot = root + decoded_key + "/"
                        next_path = path + [self.decode(k)]
                        urls += fn(subroot, g0, next_path)
                return urls

            urls = []
            for root, g, path in roots:
                urls += fn(root, g, path)
            return urls

    def cache_urls(self, g, path_tuple):
        """Look up the array of URLs in a group and cache that list

        g - the HDF5 group
        path_tuple - the tuple of path parts to get to g

        returns the URL list
        """
        if path_tuple in self.__cache:
            return self.__cache[path_tuple].urls
        a = tuple([x for x in VStringArray(g)])
        is_not_none = VStringArray(g.require_group("metadata")).is_not_none()
        self.__cache[path_tuple] = self.__CacheEntry(g, a, is_not_none)
        return a

    def list_files(self, url):
        """List the files in the directory specified by the URL

        returns just the filename parts of the files in the
        directory.
        """
        schema, parts = self.split_url(url, is_directory=True)
        with self.lock:
            path_tuple = tuple([schema] + parts)
            if path_tuple in self.__cache:
                return self.__cache[path_tuple].urls
            group = self.get_filelist_group()
            for part in [schema] + parts:
                encoded_part = self.encode(part)
                if encoded_part not in group:
                    return []
                group = group[encoded_part]

            if VStringArray.has_vstring_array(group):
                result = self.cache_urls(group, path_tuple)
                return result
            return []

    def list_directories(self, url):
        """List the subdirectories of the specified URL

        url - root directory to be searched.

        returns the directory names of the immediate subdirectories
        at the URL. For instance, if the URLs in the file list are
        "file://foo/bar/image.jpg" and "file://foo/baz/image.jpg",
        then self.list_directories("file://foo") would return
        [ "bar", "baz" ]
        """
        schema, parts = self.split_url(url, is_directory=True)
        with self.lock:
            group = self.get_filelist_group()
            for part in [schema] + parts:
                encoded_part = self.encode(part)
                if encoded_part not in group:
                    return []
                group = group[encoded_part]
        return [self.decode(x) for x in list(group.keys()) if self.is_dir(group[x])]

    """URL is a file"""
    TYPE_FILE = "File"
    """URL is a directory"""
    TYPE_DIRECTORY = "Directory"
    """URL is not present in the file list"""
    TYPE_NONE = "None"

    def get_type(self, url):
        schema, parts = self.split_url(url, is_directory=True)
        with self.lock:
            parts_tuple = tuple([schema] + parts[:-1])
            #
            # Look in cache first
            #
            if parts_tuple in self.__cache:
                a = self.__cache[parts_tuple].urls
                idx = bisect.bisect_left(a, parts[-1])
                if idx < len(a) and a[idx] == parts[-1]:
                    return self.TYPE_FILE
            group = self.get_filelist_group()
            for part in [schema] + parts[:-1]:
                encoded_part = self.encode(part)
                if encoded_part not in group:
                    return self.TYPE_NONE
                group = group[encoded_part]
            last_encoded_part = self.encode(parts[-1])
            if last_encoded_part in group and self.is_dir(group[last_encoded_part]):
                return self.TYPE_DIRECTORY
            else:
                if VStringArray.has_vstring_array(group):
                    a = self.cache_urls(group, parts_tuple)
                else:
                    return self.TYPE_NONE
                idx = bisect.bisect_left(a, parts[-1])
                if idx < len(a) and a[idx] == parts[-1]:
                    return self.TYPE_FILE
            return self.TYPE_NONE

    def add_metadata(self, url, metadata):
        """Add metadata associated with the URL

        url - url of the file. The URL must be present in the file list

        metadata - the OME-XML for the file
        """
        self.__generation = uuid.uuid4()
        group, index, has_metadata = self.find_url(url)
        metadata_array = VStringArray(group.require_group("metadata"))
        metadata_array[index] = metadata
        has_metadata[index] = True
        self.notify()

    def get_metadata(self, url):
        """Get the metadata associated with a URL

        url - url of the file.

        metadata - the OME-XML for the file
        """
        result = self.find_url(url)
        if result is None:
            return None
        group, index, has_metadata = result
        if not has_metadata[index]:
            return None
        metadata = VStringArray(group.require_group("metadata"))
        if len(metadata) <= index:
            # Metadata wasn't initialized...
            return None
        return metadata[index]

    def find_url(self, url):
        """Find the group and index of a URL

        url - the URL to find in the file list

        returns the HDF5 group that represents the URL's
        directory, the index of the URL in the file list
        and the metadata indicators for the directory
        or None if the url is not present.
        """
        schema, parts = self.split_url(url)
        with self.lock:
            path_tuple = tuple([schema] + parts[:-1])
            if path_tuple in self.__cache:
                entry = self.__cache[path_tuple]
                group = entry.group
                a = entry.urls
                has_metadata = entry.has_metadata
            else:
                group = self.get_filelist_group()
                for part in [schema] + parts[:-1]:
                    encoded_part = self.encode(part)
                    if encoded_part not in group:
                        return None
                    group = group[encoded_part]
                a = self.cache_urls(group, path_tuple)
                has_metadata = self.__cache[path_tuple].has_metadata
            idx = bisect.bisect_left(a, parts[-1])
            if idx < len(a) and a[idx] == parts[-1]:
                return group, idx, has_metadata
            return None

    def get_refresh_timestamp(self, url):
        """Get the timestamp of the last refresh of the given directory

        url - url of the directory to reference

        returns None if never, else seconds after the epoch
        """
        group = self.get_filelist_group()
        schema, path = self.split_url(url)
        for part in path:
            encoded_part = self.encode(part)
            if encoded_part not in group:
                return None
            group = group[encoded_part]
        return group.attrs.get(A_TIMESTAMP, None)

    def walk(self, callback):
        """Walk the file list in a manner like os.walk

        callback - function to be called when visiting each directory. The
                   signature is: callback(root, directories, files)
                   where root is the root of the URL being visited,
                   directories is a sequence of subdirectories at the root
                   and files is a sequence of "filenames" (root + file
                   gives a URL rooted in the directory).

        Directories are traversed deepest first and the directory
        list can be trimmed during the callback to prevent traversal of children.
        """
        with self.lock:
            group = self.get_filelist_group()
            stack = [[k for k in group if self.is_dir(group[k])]]
            groups = [group]
            roots = [None]
            path = [None]
            while len(stack):
                current = stack.pop()
                g0 = groups.pop()
                root = roots[-1]
                if len(current):
                    k = current[0]
                    groups.append(g0)
                    stack.append(current[1:])
                    g1 = g0[k]
                    kd = self.decode(k)
                    if len(roots) == 1:
                        root = kd + ":"
                    elif len(roots) == 2:
                        root += kd
                    else:
                        root += "/" + kd
                    directories = [self.decode(k) for k in g1 if self.is_dir(g1[k])]
                    path_tuple = tuple(path[1:] + [kd])
                    filenames = self.cache_urls(g1, path_tuple)
                    callback(root, directories, filenames)
                    if len(directories):
                        stack.append([self.encode(d) for d in directories])
                        groups.append(g1)
                        roots.append(root)
                        path.append(kd)
                else:
                    roots.pop()
                    path.pop()


class HDF5ImageSet(object):
    """An HDF5 backing store for an image set

    Images are stored in the CellH5 dataset shape:
    c, t, z, y, x

    By default, each channel's image is stored in the data set,
    "/Images/<channel-name>"
    """

    def __init__(self, hdf5_file=None, root_name=IMAGES_GROUP):
        """Create an HDF5ImageSet instance

        hdf5_file the file or other group-like object that is the root.
        root_name the name of the root group in the hdf5 file. Defaults to
                      "Images"
        """
        self.hdf5_file = hdf5_file
        if root_name not in self.hdf5_file:
            self.root = self.hdf5_file.create_group(root_name)
        else:
            self.root = self.hdf5_file[root_name]

    def set_image(self, image_name, data):
        """Store the image data in the HDF5 file

        The data should be pre-shaped in c, t, z, y, x form.
        For instance, a monochrome image:
        my_shape = (1, 1, 1, img.shape[0], img.shape[1])
        cache.set_image("monochrome", img.reshape(*my_shape)

        a color image:
        img1 = img.transpose(2, 0, 1).reshape(img.shape[2], 1, 1, img.shape[0], img.shape[1])
        cache.set_image("color", img1)

        image_name - a name for storage and retrieval, the name given to
                     the data set within its group
        data - the 5-d image to be stored.
        """
        #
        # The strategy here is to reuse the dataset. The assumption is
        # that generally, it will be the same size and data type from
        # one image set to the next
        #
        if image_name not in self.root:
            self.root.create_dataset(image_name, data=data)
        else:
            data_set = self.root[image_name]
            if (
                tuple(data_set.shape) == tuple(data.shape)
                and data_set.dtype == data.dtype
            ):
                data_set[:] = data
            else:
                del self.root[image_name]
                self.root.create_dataset(image_name, data=data)

    def get_image(self, image_name):
        """Retrieve the image from the HDF5 file

        image_name - the name of the image for storage and retrieval.

        returns a 5-d array of indeterminate type with the dimensions in
        the order, c, t, z, y, x. The array is dereferenced from the dataset,
        so any changes to it do not propagate back into the cached version.

        raises KeyError if your image was not there.
        """
        return self.root[image_name][:]


class HDF5ObjectSet(object):
    """An HDF5 backing-store for segmentations

    Segmentations are stored in one of two formats:

    A 6-d array composed of one or more 5-d integer labelings of
    each pixel. The dimension order is labeling, c, t, z, y, x. Typically,
    a 2-D non-overlapping segmentation has dimensions of 1, 1, 1, 1, y, x.

    The i, j, v labeling of the pixels. The labeling is stored in a record
    data type with each column having a name of "c", "t", "z", "y", "x" or
    "label". The "label" column is the object number, starting with 1.

    Naming is in 2 parts: object_name, segmentation. One group is reserved
    per 2-part name and the datasets within are named, "dense" and "sparse" with
    "dense" being the 6-d array and "sparse" being the i, j, v format. It is
    the caller's responsibility to populate each and to test to see which is
    present.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    ATTR_STALE = "stale"
    AXIS_LABELS = "label"
    AXIS_C = "c"
    AXIS_T = "t"
    AXIS_Z = "z"
    AXIS_Y = "y"
    AXIS_X = "x"
    AXES = (AXIS_C, AXIS_T, AXIS_Z, AXIS_Y, AXIS_X)

    def __init__(self, hdf5_file, root_name=OBJECTS_GROUP):
        """Create an HDF5ObjectSet instance

        hdf5_file the file or other group-like object that is the root.
        root_name the name of the root group in the hdf5 file. Defaults to
                      "Objects"
        """
        self.hdf5_file = hdf5_file
        if root_name not in self.hdf5_file:
            self.root = self.hdf5_file.create_group(root_name)
        else:
            self.root = self.hdf5_file[root_name]

    def set_dense(self, objects_name, segmentation_name, data):
        """Store the dense 6-d representation of the segmentation

        objects_name - name of the labeled objects
        segmentation_name - name of the segmentation, for instance "segmented"
                            or "small_removed"
        data - a 6-dimensional array with axes of "labeling", "c", "t", "z",
               "y", and "x". Values are unsigned integers starting at one with
               zero signifying unlabeled. The "labeling" axis allows the caller
               to specify multiple labels per pixel by placing their label
               numbers for that pixel in array locations that only differ by
               their position on the "labeling axis".
        """
        segmentation_group = self.__ensure_group(objects_name, segmentation_name)
        if self.DENSE in segmentation_group:
            data_set = segmentation_group[self.DENSE]
            if (
                tuple(data_set.shape) == tuple(data.shape)
                and data_set.dtype == data.dtype
            ):
                data_set[:] = data
            else:
                del segmentation_group[self.DENSE]
                data_set = segmentation_group.create_dataset(self.DENSE, data=data)
        else:
            data_set = segmentation_group.create_dataset(self.DENSE, data=data)
        data_set.attrs[self.ATTR_STALE] = False

    def has_dense(self, objects_name, segmentation_name):
        """Return True if a dense segmentation dataset is available

        objects_name - name of the objects
        segmentation_name - name of the segmentation of these objects
        """
        return self.__has(objects_name, segmentation_name, self.DENSE)

    def get_dense(self, objects_name, segmentation_name):
        """Get the dense representation of a data set

        objects_name - name of the objects
        segmentation_name - name of the segmentation of the objects

        Note that this call does not check and raise an exception if the
        data is stale. Call has_dense beforehand to check this.
        """
        return self.root[objects_name][segmentation_name][self.DENSE][:]

    def set_sparse(self, objects_name, segmentation_name, data):
        """Set the sparse representation of a segmentation

        objects_name - name of the objects
        segmentation_name - name of the segmentation
        data - the per-pixel labeling of the objects. Each row represents
               the labeling of a pixel. The array should have a record data type
               with each of the columns labeled with one of the AXIS_ constants.
               For instance:
               dtype = [(HDF5ObjectSet.AXIS_Y, np.uint32, 1),
                        (HDF5ObjectSet.AXIS_X, np.uint32, 1),
                        (HDF5ObjectSet.AXIS_LABELS, np.uint32, 1)]
               data = np.array([(100, 200, 1)], dtype)
        """
        segmentation_group = self.__ensure_group(objects_name, segmentation_name)
        create = False
        if not self.SPARSE in segmentation_group:
            create = True
        else:
            ds = segmentation_group[self.SPARSE]
            create = data.dtype != ds.dtype
            if create:
                del segmentation_group[self.SPARSE]
        if create:
            ds = segmentation_group.create_dataset(
                self.SPARSE, data=data, chunks=(1024,), maxshape=(None,)
            )
        else:
            ds = segmentation_group[self.SPARSE]
            ds.resize((len(data),))
            if len(data) > 0:
                ds[:] = data
        ds.attrs[self.ATTR_STALE] = False

    def has_sparse(self, objects_name, segmentation_name):
        """Return True if sparse representation of segmentation is available

        objects_name - name of the objects
        segmentation_name - name of the segmentation of these objects
        """
        return self.__has(objects_name, segmentation_name, self.SPARSE)

    def get_sparse(self, objects_name, segmentation_name):
        """Return the sparse-style data records for the segmentation
        objects_name - name of the objects
        segmentation_name - name of the segmentation of these objects

        Returns a Numpy record array with one row per pixel per label
        and columns denoting the pixel coordinates and the label.
        """
        ds = self.root[objects_name][segmentation_name][self.SPARSE]
        if len(ds) == 0:
            return numpy.zeros(0, ds.dtype)
        return ds[:]

    def __ensure_group(self, objects_name, segmentation_name):
        if objects_name not in self.root:
            objects_group = self.root.create_group(objects_name)
        else:
            objects_group = self.root[objects_name]
        if segmentation_name not in objects_group:
            segmentation_group = objects_group.create_group(segmentation_name)
            segmentation_group.attrs[A_CLASS] = CLASS_SEGMENTATION_GROUP
        else:
            segmentation_group = objects_group[segmentation_name]
        return segmentation_group

    def __has(self, objects_name, segmentation_name, data_format):
        if objects_name not in self.root:
            return False
        objects_group = self.root[objects_name]
        if segmentation_name not in objects_group:
            return False
        segmentation_group = objects_group[segmentation_name]
        if segmentation_group.attrs[A_CLASS] != CLASS_SEGMENTATION_GROUP:
            return False
        if data_format not in segmentation_group:
            return False
        return not segmentation_group[data_format].attrs[self.ATTR_STALE]

    def clear(self, objects_name, segmentation_name=None):
        """Remove a segmentation from the object set

        Clearing should be done before adding a dense or sparse segmentation
        to mark the sparse representation of a dense segmentation or vice-versa
        as stale. Conceptually, it is as if the segmentation were deleted,
        but practically, we mark, anticipating a reuse of existing storage.

        objects_name - name of the labeled objects
        segmentation_name - name of the segmentation being cleared or None if
                            all.
        """
        if objects_name not in self.root:
            return
        objects_group = self.root[objects_name]
        if segmentation_name is None:
            for name in objects_group:
                segmentation_group = objects_group[name]
                if segmentation_group.attrs[A_CLASS] == CLASS_SEGMENTATION_GROUP:
                    self.clear(objects_name, name)
        elif segmentation_name not in objects_group:
            return
        else:
            segmentation_group = objects_group[segmentation_name]
            for dataset_name in self.DENSE, self.SPARSE:
                if dataset_name in segmentation_group:
                    dataset = segmentation_group[dataset_name]
                    dataset.attrs[self.ATTR_STALE] = True


def get_top_level_group(filename, group_name="Measurements", open_mode="r"):
    """Open and return the Measurements HDF5 group

    filename - path to HDF5 file

    group_name - name of top-level group, defaults to Measurements group

    open_mode - open mode for file: 'r' for read, 'w' for write

    returns the hdf5 file object (which must be closed) and the top-level group
    """
    f = h5py.File(filename, open_mode)
    return f, f.get(group_name)


class HDFCSV(object):
    """An HDF representation of a .CSV file

    HDF5 structure:
    <hosting group>
       <group name="name", CLASS="CSV">
           <group name=column-name, CLASS="COLUMN">
               <dataset name="index">
               <dataset name="data">
           <group name=column-name, CLASS="COLUMN">
               <dataset name="index">
               <dataset name="data">
    """

    CURRENT_VERSION = 1
    CLASS = "CLASS"
    COLUMN = "COLUMN"
    VERSION = "VERSION"
    HDFCSV_CLASS = "CSV"

    def __init__(self, group, name, lock=None):
        """Create or bind to a CSV

        group - HDF group hosting the CSV

        name - name of the CSV within the group

        lock - locking object to synchronize hdf access
        """
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
        self.columns = {}

    def clear(self):
        """Clear all columns in the CSV"""
        with self.lock:
            for key in list(self.top_level_group.keys()):
                column = self.top_level_group[key]
                if column.attrs[self.CLASS] == self.COLUMN:
                    del column
                    del self.top_level_group[key]
            self.columns = {}

    def add_column(self, name, data=None):
        """Add a column

        name - name of column

        data - optional, add the strings for the column

        returns the column
        """
        with self.lock:
            column = self.top_level_group.create_group(name)
            column.attrs[self.CLASS] = self.COLUMN
            kolumn = VStringArray(column, self.lock)
            if data is not None:
                kolumn.set_all(data)
            self.columns[name] = kolumn
            return kolumn

    def set_all(self, d):
        """Set all of the columns simultaneously

        Clears all existing columns and sets up new ones.

        d - a dictionary. The keys are used for column names and the
            values should be sequences of strings
        """
        with self.lock:
            self.clear()
            for k, v in list(d.items()):
                self.add_column(k, v)

    def get_column_names(self):
        """Get the names of the columns"""
        return [
            key
            for key in list(self.top_level_group.keys())
            if self.top_level_group[key].attrs[self.CLASS] == self.COLUMN
        ]

    def __getitem__(self, key):
        """Get a column as a VStringArray

        key - the name of the column

        returns a VStringArray which may be used like a sequence
        """
        if key not in self.columns:
            self.columns[key] = VStringArray(self.top_level_group[key])
        return self.columns[key]

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
    """A "lock" that does nothing if no locking is needed"""

    def __enter__(self):
        return

    def __exit__(self, t, v, tb):
        return


class HDF5Lock:
    def __init__(self):
        self.lock = threading.RLock()

    def __enter__(self):
        self.lock.acquire()
        if hasattr(h5py, "phil"):
            h5py.phil.acquire()

    def __exit__(self, t, v, tb):
        if hasattr(h5py, "phil"):
            h5py.phil.release()
        self.lock.release()


class VStringArray(object):
    """A 1-d array of variable-length strings backed by HDF5 datasets

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
    """

    VS_NULL = numpy.iinfo(numpy.int32).max

    @staticmethod
    def has_vstring_array(group):
        return (
            ("index" in group)
            and cellprofiler_core.utilities.legacy.equals(
                group["index"].attrs[A_CLASS], CLASS_VSTRING_ARRAY_INDEX
            )
            and ("data" in group)
            and cellprofiler_core.utilities.legacy.equals(
                group["data"].attrs[A_CLASS], CLASS_VSTRING_ARRAY_DATA
            )
        )

    def __init__(self, group, lock=None):
        """Initialize or bind to a VStringArray within the named group

        group - an HDF5 Group
        lock - a mutex or similar to synchronize access to the array. Default
               is no locking.
        """
        assert isinstance(group, h5py.Group)
        self.group = group
        if "index" in group:
            self.index = group["index"]
            # assert self.index.attrs[A_CLASS] == CLASS_VSTRING_ARRAY_INDEX
            assert cellprofiler_core.utilities.legacy.equals(
                self.index.attrs[A_CLASS], CLASS_VSTRING_ARRAY_INDEX
            )
        else:
            self.index = group.create_dataset(
                "index",
                shape=(0, 2),
                dtype=numpy.int32,
                shuffle=True,
                chunks=(256, 2),
                maxshape=(None, 2),
            )
            self.index.attrs[A_CLASS] = CLASS_VSTRING_ARRAY_INDEX
        if "data" in group:
            self.data = group["data"]
            # assert self.data.attrs[A_CLASS] == CLASS_VSTRING_ARRAY_DATA
            assert cellprofiler_core.utilities.legacy.equals(
                self.data.attrs[A_CLASS], CLASS_VSTRING_ARRAY_DATA
            )
        else:
            self.data = group.create_dataset(
                "data",
                (0,),
                dtype="S1",
                shuffle=True,
                compression="gzip",
                chunks=(32768,),
                maxshape=(None,),
            )
            self.data.attrs[A_CLASS] = CLASS_VSTRING_ARRAY_DATA
        if lock is None:
            self.lock = NullLock()
        else:
            self.lock = lock

    def __setitem__(self, idx, value):
        """Store a single string at a single index

        idx - index of string within the array
        value - a UTF-8 encoded string, unicode string, None or object to be
                converted to a string
        """
        with self.lock:
            if idx < 0:
                idx = self.index.shape[0] - idx
            if value is None:
                if idx >= self.index.shape[0]:
                    self.index.resize(idx + 1, 0)
                self.index[idx, :] = (self.VS_NULL, 0)
                return

            elif isinstance(value, str):
                value = value.encode("utf-8")

            if idx >= self.index.shape[0]:
                self.index.resize(idx + 1, 0)
                begin = self.data.shape[0]
                self.index[idx, 0] = begin
            else:
                idx0, idx1 = self.index[idx]
                if idx0 == self.VS_NULL:
                    begin = self.data.shape[0]
                    self.index[idx, 0] = begin
                elif len(value) <= idx1 - idx0:
                    begin = idx0
                else:
                    begin = self.data.shape[0]
                    self.index[idx, 0] = begin
            end = begin + len(value)
            self.index[idx, 1] = end
            if self.data.shape[0] < end:
                self.data.resize(end, 0)
            if begin != end:
                self.data[begin:end] = numpy.frombuffer(value, "S1")

    def __getitem__(self, idx):
        """Retrieve a single string through the indexing interface.

        idx - the index of the string within the array

        returns a unicode string or None for empty string
        """
        with self.lock:
            if idx < 0:
                idx = self.index.shape[0] - idx
            begin, end = self.index[idx, :]
            if begin > end:
                return None
            elif begin == end:
                return ""
            retval = self.data[begin:end].tostring()
            if retval and isinstance(retval, bytes):
                retval = retval.decode("utf-8")
            return retval

    def __delitem__(self, idx):
        with self.lock:
            orig_len = self.index.shape[0]
            if idx < 0:
                idx = orig_len - idx
            if idx < orig_len - 1:
                self.index[idx : (orig_len - 1), :] = self.index[(idx + 1) :, :]
            self.index.resize(self.index.shape[0] - 1, 0)

    def __len__(self):
        """The number of strings stored in the array"""
        return self.index.shape[0]

    def __iter__(self):
        """Iterates through the items in the array in a threadsafe manner"""
        with self.lock:
            if self.index.shape[0] == 0:
                return
            index = self.index[:, :]
            data = self.data[:]
        for begin, end in index:
            yield (
                None
                if begin > end
                else ""
                if begin == end
                else data[begin:end].tostring().decode()
            )

    def set_all(self, strings):
        """Store the strings passed, overwriting any previously stored data"""
        nulls = numpy.array([s is None for s in strings])
        strings = [
            "" if s is None else s if isinstance(s, str) else str(s) for s in strings
        ]
        with self.lock:
            target_len = len(strings)
            self.index.resize(target_len, 0)
            index = numpy.zeros(self.index.shape, self.index.dtype)
            if len(strings) > 0:
                index[:, 1] = numpy.cumsum([len(s) for s in strings])
                index[0, 0] = 0
                if len(strings) > 1:
                    index[1:, 0] = index[: (target_len - 1), 1]
                if numpy.any(nulls):
                    index[nulls, 0] = self.VS_NULL
                self.data.resize(index[(target_len - 1), 1], 0)
                self.index[:, :] = index
            for s, (begin, end) in zip(strings, index):
                if begin < end:
                    s = s.encode("utf-8")
                    self.data[begin:end] = numpy.frombuffer(s, "S1")

    def sort(self):
        """Sort strings in-place

        returns a list containing the old indices as they appear in the
        new array. For instance, if the array looked like
        ( "foo", "bar", "baz")
        before calling sort, it would look like
        ( "bar", "baz", "foo" ) and would return
        ( 1, 2, 0 )
        """
        with self.lock:
            if len(self) == 0:
                return
            index = self.index[:, :]

            def compare(i, j):
                i0, i1 = index[i, :]
                j0, j1 = index[j, :]
                if i0 == self.VS_NULL:
                    if j0 == self.VS_NULL:
                        return 0
                    return -1
                elif j0 == self.VS_NULL:
                    return 1
                li, lj = i1 - i0, j1 - j0
                l = min(li, lj)
                # Read 16 byte chunks
                for idx in range(0, l, 16):
                    idx_end = min(idx + 16, l)
                    di = self.data[(i0 + idx) : (i0 + idx_end)]
                    dj = self.data[(j0 + idx) : (j0 + idx_end)]
                    diff = numpy.argwhere(di != dj).flatten()
                    if len(diff) > 0:
                        return cellprofiler_core.utilities.legacy.cmp(
                            di[diff[0]], dj[diff[0]]
                        )
                return cellprofiler_core.utilities.legacy.cmp(li, lj)

            order = list(range(len(self)))
            order.sort(key=functools.cmp_to_key(compare))
            self.index = index[order, :]
            return order

    def reorder(self, order):
        """Reorder the array

        order - a sequence of the old indices in the desired order. The order
                can have missing indices in which case the corresponding
                strings are deleted and the array is downsized.
        """
        if len(order) == 0:
            self.index.resize(0, 0)
            return
        index = self.index[:]
        order = numpy.array(order, int)
        if index.shape[0] > len(order):
            self.index.resize(len(order), 0)
        self.index[:, :] = index[order, :]

    def insert(self, index, s):
        """Insert a string into the array at an index"""
        with self.lock:
            old_len = self.index.shape[0]
            self.index.resize(old_len + 1, 0)
            if index < old_len:
                self.index[index + 1 :, :] = self.index[index:old_len]
            self.index[index, :] = (self.VS_NULL, 0)
            self[index] = s

    def append(self, s):
        """Append a string to the end of the array

        s - string to append
        """
        with self.lock:
            self.insert(len(self), s)

    def extend(self, strings):
        """Append a sequence of strings to the end of the array

        strings - strings to append
        """
        if len(strings) == 0:
            return
        nulls = numpy.array([s is None for s in strings])
        strings = [
            "" if s is None else s if isinstance(s, str) else str(s) for s in strings
        ]
        with self.lock:
            old_len = len(self)
            old_data_len = self.data.shape[0]
            n_strings = len(strings)
            target_len = n_strings + old_len
            self.index.resize(target_len, 0)
            index = numpy.zeros((n_strings, 2), self.index.dtype)
            index[:, 1] = numpy.cumsum([len(s) for s in strings])
            index[0, 0] = 0
            if len(strings) > 1:
                index[1:, 0] = index[:-1, 1]
            index += old_data_len
            if numpy.any(nulls):
                index[nulls, 0] = self.VS_NULL
            self.data.resize(index[-1, 1], 0)
            self.index[old_len:, :] = index
            idx_not_nulls = numpy.where(~nulls)[0]
            for i in range(0, len(idx_not_nulls), 1000):
                iend = min(i + 1000, len(idx_not_nulls))
                ilast = iend - 1
                begin = index[idx_not_nulls[i], 0]
                end = index[idx_not_nulls[ilast], 1]
                scat = numpy.zeros(end - begin, "S1")
                for idx in idx_not_nulls[i:iend]:
                    sbegin = index[idx, 0] - begin
                    send = index[idx, 1] - begin
                    scat[sbegin:send] = numpy.fromstring(strings[idx], "S1")
                self.data[begin:end] = scat

    def bisect_left(self, s):
        """Return the insertion point for s, assuming the array is sorted"""
        if s is None:
            return 0
        elif isinstance(s, str):
            s = s
        else:
            s = str(s)
        #
        # bisection code taken from Python bisect package
        # author: Raymond Hettinger
        #
        lo = 0
        hi = len(self)
        slen = len(s)
        while lo < hi:
            mid = int((lo + hi) / 2)
            i0, i1 = self.index[mid]
            l = min(slen, i1 - i0)
            for s0, s1 in zip(s, self.data[i0:i1]):
                s0 = s0.encode("utf-8")
                if s0 != s1:
                    break
            if s0 == s1:
                if slen == i1 - i0:
                    return mid
                elif slen < i1 - i0:
                    hi = mid
                else:
                    lo = mid + 1
            elif s0 < s1:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def is_not_none(self, index=slice(0, sys.maxsize)):
        """Return True for indices that are not None

        index - either a single index (in which case, we return a single
                True / False value) or some suitable slicing construct
                that works with Numpy arrays. Default is return an indicator
                per element.
        """
        if isinstance(index, int) or hasattr(index, "__int__"):
            return self[index] is not None

        if len(self) == 0:
            return []
        iii = self.index[:, :]
        return iii[index, 0] <= iii[index, 1]

    def delete(self):
        """Remove the vstringarray from the group"""
        del self.group[self.index.name]
        del self.group[self.data.name]
        del self.index
        del self.data


class StringReferencer(object):
    """This class implements a B-tree of strings within an HDF5 file's group

    Usage:
    sr = StringReferencer(group)
    # Get integer reference IDs to strings
    references = sr.get_string_refs(["foo", "bar", "foo"])
    assert references[0] == references[2] # duplicate strings are stored once
    assert tuple(sr.get_strings(references[:-1])) == ("foo", "bar")
    """

    """Default size of a b-tree block"""
    SR_DEFAULT_BLOCKSIZE = 256
    SR_BLOCKSIZE_ATTR = "blocksize"
    """Size of a data block"""
    SR_DATA_BLOCKSIZE = 4096
    """Null value (for sub block of leaf or as a result from a search)"""
    SR_NULL = numpy.iinfo(numpy.uint32).max

    """The name of the string reference dataset

    This dataset is indexed by the string reference number
    and contains the block number, offset and length of
    the referenced string. It's dimensions are N x 3 where
    N is the # of unique strings in the system
    """
    SR_REF_DS = "stringref"
    """The block index of the string"""
    SR_REF_BLOCK = 0
    """The index of the string within the block"""
    SR_REF_IDX = 1

    def __init__(self, group, blocksize=None):
        assert isinstance(group, h5py.Group)
        self.group = group
        self.blocksize = self.group.attrs.get(
            self.SR_BLOCKSIZE_ATTR, self.SR_DEFAULT_BLOCKSIZE
        )
        self.blockdesc = self.get_blockdesc_dataset()
        self.refs = self.get_ref_dataset()
        self.blocks = self.get_block_dataset()
        self.data = self.get_data_dataset()

    def get_ref_dataset(self):
        """Get the string reference dataset

        group - the group housing the dataset

        returns the N x 2 string reference dataset. An index into this dataset
        gives the block and offset within the block of the start of the string
        as well as the length of the string.
        """
        ds_ref = self.group.require_dataset(
            self.SR_REF_DS,
            (0, 2),
            dtype=numpy.uint64,
            shuffle=True,
            chunks=(self.blocksize * 4, 2),
            maxshape=(None, 2),
        )
        return ds_ref

    """The name of the dataset holding the offset and length of strings in a block

    This dataset maintains the offset of a string within the datablock and the
    length of the string. The value at I,J,0 gives the offset in SR_DATA_DS to
    the Jth string in block I and the value at I,J,1 gives its length.
    """
    SR_BLOCK_DS = "blocks"
    """The reference index of the string"""
    SR_BLOCK_REF = 0
    """The offset of the string in the data block"""
    SR_BLOCK_OFF = 1
    """The length of the string"""
    SR_BLOCK_LEN = 2
    """The subblock between this entry and the next (or SR_NULL if a leaf)"""
    SR_BLOCK_SUBBLOCK = 3
    SR_BLOCK_ENTRIES = 4

    def get_block_dataset(self):
        """Get the offset / length dataset

        returns the N x M x 3 dataset that, for each of N blocks
        and M entries per block gives the offset and length of the
        Mth string in the Nth block of the string data.
        """
        ds_ol = self.group.require_dataset(
            self.SR_BLOCK_DS,
            (0, self.blocksize, self.SR_BLOCK_ENTRIES),
            dtype=numpy.uint32,
            shuffle=True,
            chunks=(4, self.blocksize, self.SR_BLOCK_ENTRIES),
            maxshape=(None, self.blocksize, self.SR_BLOCK_ENTRIES),
        )
        return ds_ol

    """The name of the dataset holding the strings

    This dataset has dimensions N x M where N is the block number
    and M is the length of the concatenated strings in the block.
    Note that this compresses well in the M direction since the
    members generally start with the same sequence of characters.
    """
    SR_DATA_DS = "data"

    def get_data_dataset(self):
        ds_data = self.group.require_dataset(
            self.SR_DATA_DS,
            (0, 0),
            dtype=numpy.uint8,
            shuffle=True,
            compression="gzip",
            chunks=(1, self.SR_DATA_BLOCKSIZE),
            maxshape=(None, None),
        )
        return ds_data

    """The dataset holding the block descriptor

    This dataset has N x 5 members where N is the number of blocks. It holds
    the current number of indices in each block and the length of the string
    data for the block.
    """
    SR_BLOCKDESC_DS = "blockdesc"
    """Index of the entry giving # of indexes in a block"""
    SR_BLOCKDESC_IDX_LEN = 0
    """Index of the entry giving # of bytes in the block's data section"""
    SR_BLOCKDESC_DATA_LEN = 1
    """Block # of this block's parent"""
    SR_BLOCKDESC_PARENT = 2
    """The index of the node in the parent that is less than us. -1 if we are before all"""
    SR_BLOCKDESC_PARENT_IDX = 3
    """The index of the leftmost child of this block"""
    SR_BLOCKDESC_LEFTMOST_CHILD = 4
    """# of entries per block"""
    SR_BLOCKDESC_ENTRIES = 5
    """The attribute on the blockdesc dataset specifying the current root block of the b-tree"""
    SR_ROOT_ATTR = "root"

    def get_blockdesc_dataset(self):
        """Get the dataset holding the block descriptors for each block"""
        ds_blockdesc = self.group.require_dataset(
            self.SR_BLOCKDESC_DS,
            (0, self.SR_BLOCKDESC_ENTRIES),
            dtype=numpy.uint32,
            shuffle=True,
            chunks=(256, self.SR_BLOCKDESC_ENTRIES),
            maxshape=(None, self.SR_BLOCKDESC_ENTRIES),
        )
        return ds_blockdesc

    @staticmethod
    def string_to_uint8(s):
        """Convert a utf-8 encoded string to a np.uint8 array"""
        if isinstance(s, str):
            s = s
        elif not isinstance(s, str):
            s = str(s)
        s = s.encode("utf-8")
        result = numpy.zeros(len(s), numpy.uint8)
        result.data[:] = s
        return result

    def get_strings(self, refs):
        refs, reverse_indices = numpy.unique(refs, return_inverse=True)
        strings = [
            self.get_unicode_from_block_and_idx(*self.refs[ref, :]) for ref in refs
        ]
        return [strings[idx] for idx in reverse_indices]

    def get_unicode_from_block_and_idx(self, i, j):
        """Return a unicode string given a block and index within the block

        i: block #
        j: index within block
        """
        data_off, data_len = self.blocks[
            i, j, self.SR_BLOCK_OFF : (self.SR_BLOCK_LEN + 1)
        ]
        s = (
            self.data[i, data_off : (data_off + data_len)]
            .data.obj.tostring()
            .decode("utf-8")
        )
        return s

    def get_string_refs(self, strings):
        """Get references to strings

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
        """
        strings, reverse_indices = numpy.unique(
            numpy.array(strings, object), return_inverse=True
        )
        strings = [self.string_to_uint8(s) for s in strings]
        indices = []
        if self.blocks.shape[0] == 0:
            block = self.sr_alloc_block()
            self.blockdesc.attrs[self.SR_ROOT_ATTR] = block
            self.refs.resize(self.refs.shape[0] + 1, 0)
            #
            # Build the first block
            #
            s0 = strings[0]
            strings = strings[1:]
            indices.append(0)
            self.blockdesc[block, :] = (
                1,  # entries
                len(s0),  # current length of data section
                self.SR_NULL,  # left block ptr
                self.SR_NULL,  # parent block
                self.SR_NULL,
            )  # parent index
            self.data.resize(len(s0), 1)
            self.data[block, : len(s0)] = s0
            self.blocks[block, 0, :] = (0, 0, len(s0), self.SR_NULL)
            self.refs[block, :] = (0, 0)
        for s in strings:
            i, j, found = self.sr_search(s)
            if not found:
                if self.blockdesc[i][self.SR_BLOCKDESC_IDX_LEN] == self.blocksize:
                    # Need to split the block before insertion
                    self.sr_split_block(i)
                    i, j, idx = self.sr_search(s)
                idx = self.refs.shape[0]
                self.refs.resize(idx + 1, 0)
                self.refs[idx, :] = (i, j)
                self.sr_insert(s, idx, i, j)
            else:
                idx = self.blocks[i, j, self.SR_BLOCK_REF]
            indices.append(idx)
        #
        # distribute the indices back onto the original input strings
        #
        indices = numpy.array(indices)
        return indices[reverse_indices]

    def sr_alloc_block(self):
        """Allocate a new block

        returns the block number
        """
        idx = self.blockdesc.shape[0]
        self.blockdesc.resize(idx + 1, 0)
        self.blocks.resize(idx + 1, 0)
        self.data.resize(idx + 1, 0)
        self.blockdesc[idx, :] = (0, 0, self.SR_NULL, self.SR_NULL, 0)
        return idx

    def sr_split_block(self, i):
        """Split a block in half

        i - the block number
        refs - the block / index for string references
        ol - the per-reference data stored as block / index
        data - the storage for the string data.
        """
        # the index of the ref that's promoted
        i1 = self.sr_alloc_block()
        idx_len, data_len, i0, j0, leftmost_child = self.blockdesc[i, :]
        j = int((idx_len - 1) / 2)

        j_ref, j_data_idx, j_data_len, j_subblock = self.block[i, j, :]
        j_data = self.data[i, j_data_idx : (j_data_idx + j_data_len)]
        if i0 == self.SR_NULL:
            # Splitting the root. We need to promote.
            i0 = self.sr_alloc_block()
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
            self.blockdesc[
                j_subblock,
                self.SR_BLOCKDESC_PARENT : (self.SR_BLOCKDESC_PARENT_IDX + 1),
            ] = (i1, -1)
            self.blockdesc[i1, self.SR_BLOCKDESC_LEFTMOST_CHILD] = j_subblock
        self.refs[j_ref, self.SR_REF_BLOCK : (self.SR_REF_IDX + 1)] = (i0, j0)
        #
        # Copy the right-hand half to the new block.
        #
        j_right = j + 1
        rh_idx_len = idx_len - j_right
        rh_data_idx = j_data_idx + j_data_len
        rh_data_end = self.blockdesc[i, self.SR_BLOCKDESC_DATA_LEN]
        rh_data_len = rh_data_end - rh_data_idx
        self.data[i1, :rh_data_len] = self.data[i, rh_data_idx:rh_data_end]
        #
        # Copy the block data - adjust data pointers at same time
        #
        adjustment = numpy.array([0, -rh_data_idx, 0, 0])[numpy.newaxis, :]
        self.block[i1, :rh_idx_len, :] = self.block[i, j_right:idx_len, :] - adjustment
        if leftmost_child != self.SR_NULL:
            # If not a leaf, have to adjust children's parents.
            for new_j, subblock in enumerate(
                self.block[i1, :rh_idx_len, self.SR_BLOCK_SUBBLOCK]
            ):
                self.blockdesc[
                    subblock,
                    self.SR_BLOCKDESC_PARENT : (self.SR_BLOCKDESC_PARENT_IDX + 1),
                ] = (i1, new_j)
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

    def sr_insert(self, s, idx, i, j, next_child=SR_NULL):
        """Open up a slot in block i at position j and insert string s

        s - string to insert
        idx - ref of the string
        i - insert in this block
        j - insert at this position
        refs - reference index -> block and position
        blockdesc - block descriptor for the target block
        ol - string descriptors
        next_child - block # of child after this one
        """
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
            self.blocks[i, (j + 1) : (idx_len + 1), :] = self.blocks[i, j:idx_len, :]
            #
            # shift the strings
            #
            self.blocks[i, (j + 1) : (idx_len + 1), self.SR_BLOCK_OFF] += len_s
            data_idx = self.blocks[i, j, self.SR_BLOCK_OFF]
            self.data[i, (data_idx + len_s) : (data_len + len_s)] = self.data[
                i, data_idx:data_len
            ]
        else:
            data_idx = data_len
        self.data[i, data_idx : (data_idx + len_s)] = s
        self.blocks[i, j, :] = (idx, data_idx, len(s), next_child)
        self.refs[idx, :] = (i, j)

    def sr_search(self, s):
        """Search for s in btree

        s: a uint8 numpy array string representation, e.g., as returned by
           string_to_uint8

        returns the block #, index of entry or insertion point
                and a True / False indicator of whether there was an exact match
        """
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
                mid = int((lo + hi) / 2)
                s1_off = block[mid, OFF]
                s1_len = block[mid, LEN]
                s1 = data[s1_off : (s1_off + s1_len)]
                minlen = min(s1_len, s_len)
                s_s1_ne = s[:minlen] != s1[:minlen]
                if not numpy.any(s_s1_ne):
                    if s_len == s1_len:
                        return block_idx, mid, True
                    elif s1_len < s_len:
                        lo = mid + 1
                    else:
                        hi = mid
                elif s1[s_s1_ne][0] < s[s_s1_ne][0]:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == 0:
                next_block_idx = self.blockdesc[
                    block_idx, self.SR_BLOCKDESC_LEFTMOST_CHILD
                ]
            else:
                next_block_idx = block[lo - 1, self.SR_BLOCK_SUBBLOCK]
            if next_block_idx == self.SR_NULL:
                return block_idx, lo, False


if __name__ == "__main__":
    h = HDF5Dict("temp.hdf5")
    h["Object1", "objfeature1", 1] = [1, 2, 3]
    h["Object1", "objfeature2", 1] = [1, 2, 3]
    h["Image", "f1", 1] = 5
    h["Image", "f2", 1] = 4
    print(h["Image", "f2", 1])
    h["Image", "f1", 2] = 6
    h["Image", "f2", 1] = 6
    print(h["Image", "f2", 1])
    print(h["Object1", "objfeature1", 1])
    h["Object1", "objfeature1", 2] = 3.0
    print(h["Object1", "objfeature1", 1])
    h["Object1", "objfeature1", 1] = [1, 2, 3]
    h["Object1", "objfeature1", 1] = [1, 2, 3, 5, 6]
    h["Object1", "objfeature1", 1] = [9, 4.0, 2.5]
    print(h["Object1", "objfeature1", 1])

    def randtext():
        return "".join(
            [
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[
                    numpy.random.randint(0, 52)
                ]
                for _ in range(numpy.random.randint(5, 8))
            ]
        )

    for i in range(1, 20):
        h["Image", "t1", i] = randtext()

    for i in numpy.random.permutation(numpy.arange(1, 20)):
        h["Image", "t2", i] = randtext()

    for i in range(3, 20):
        h["Image", "f1", i] = numpy.random.randint(0, 100)
        h["Object1", "objfeature1", i] = numpy.random.randint(0, 100, size=5)

    for i in numpy.random.permutation(numpy.arange(3, 20)):
        h["Image", "f2", i] = numpy.random.randint(0, 100)
        h["Object1", "objfeature2", i] = numpy.random.randint(0, 100, size=5)

    hdest = HDF5Dict("temp1.hdf5", copy=h.top_group, image_numbers=numpy.arange(4, 15))
    for i in range(4, 15):
        for object_name, feature_name in (
            ("Image", "f1"),
            ("Image", "f2"),
            ("Image", "t1"),
            ("Image", "t2"),
            ("Object1", "objfeature1"),
            ("Object1", "objfeature2"),
        ):
            src = numpy.atleast_1d(h[object_name, feature_name, i])
            dest = numpy.atleast_1d(hdest[object_name, feature_name, i])
            numpy.testing.assert_array_equal(src, dest)
