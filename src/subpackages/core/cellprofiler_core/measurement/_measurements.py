import json
import logging
import os
import re
import sys
import urllib.request
from urllib.request import url2pathname as os_url2pathname
import warnings

import h5py
import numpy

from ._metadata_group import MetadataGroup
from ._relationship_key import RelationshipKey
from ..constants.image import CT_OBJECTS
from ..constants.measurement import AGG_MEAN, C_Z, C_T, C_OBJECTS_Z, C_OBJECTS_T
from ..constants.measurement import AGG_MEDIAN
from ..constants.measurement import AGG_NAMES
from ..constants.measurement import AGG_STD_DEV
from ..constants.measurement import COLTYPE_FLOAT
from ..constants.measurement import COLTYPE_INTEGER
from ..constants.measurement import COLTYPE_VARCHAR
from ..constants.measurement import C_C
from ..constants.measurement import C_CHANNEL_TYPE
from ..constants.measurement import C_FILE_NAME
from ..constants.measurement import C_FRAME
from ..constants.measurement import C_METADATA
from ..constants.measurement import C_OBJECTS_CHANNEL
from ..constants.measurement import C_OBJECTS_FILE_NAME
from ..constants.measurement import C_OBJECTS_FRAME
from ..constants.measurement import C_OBJECTS_PATH_NAME
from ..constants.measurement import C_OBJECTS_SERIES
from ..constants.measurement import C_OBJECTS_URL
from ..constants.measurement import C_PATH_NAME
from ..constants.measurement import C_SERIES
from ..constants.measurement import C_URL
from ..constants.measurement import DB_TEMP
from ..constants.measurement import EXPERIMENT
from ..constants.measurement import GROUP_INDEX
from ..constants.measurement import GROUP_NUMBER
from ..constants.measurement import GROUP_LENGTH
from ..constants.measurement import IMAGE
from ..constants.measurement import IMAGE_NUMBER
from ..constants.measurement import K_CASE_SENSITIVE
from ..constants.measurement import K_LOCAL_SEPARATOR
from ..constants.measurement import K_PATH_MAPPINGS
from ..constants.measurement import K_URL2PATHNAME_PACKAGE_NAME
from ..constants.measurement import MEASUREMENTS_GROUP_NAME
from ..constants.measurement import M_GROUPING_TAGS
from ..constants.measurement import M_METADATA_TAGS
from ..constants.measurement import M_PATH_MAPPINGS
from ..constants.measurement import NEIGHBORS
from ..constants.measurement import OBJECT_NUMBER
from ..constants.measurement import RELATIONSHIP
from ..constants.measurement import R_FIRST_IMAGE_NUMBER
from ..constants.measurement import R_FIRST_OBJECT_NUMBER
from ..constants.measurement import R_SECOND_IMAGE_NUMBER
from ..constants.measurement import R_SECOND_OBJECT_NUMBER
from ..utilities.hdf5_dict import HDF5Dict, NullLock
from ..utilities.measurement import agg_ignore_feature
from ..utilities.measurement import get_agg_measurement_name
from ..utilities.measurement import make_temporary_file


LOGGER = logging.getLogger(__name__)

class Measurements:
    """Represents measurements made on images and objects
    """

    def __init__(
        self,
        image_set_start=None,
        filename=None,
        copy=None,
        mode="w",
        image_numbers=None,
        multithread=True,
    ):
        """Create a new measurements collection

        image_set_start - the index of the first image set in the image set list
                          or None to start at the beginning
        filename - store the measurement in an HDF5 file with this name
        copy - initialize by copying measurements from here, either an HDF5Dict
               or an H5py group or file.
        mode - open mode for the HDF5 file.
               "r" for read-only access to an existing measurements file,
               "w" to open a new file or truncate an old file,
               "w-" to open a new file and fail if the file exists,
               "w+" to create a new measurements instance in an existing file,
               "a" to create a new file or open an existing file as read/write
               "r+" to open an existing file as read/write
               "memory" to create an HDF5 memory-backed File
        multithread - True if this measurements structure is used in a
               multithreading context, False to disable locking.
        """
        # XXX - allow saving of partial results
        if mode == "memory" and sys.platform == "darwin":
            # Core driver doesn't work on Mac
            # http://code.google.com/p/h5py/issues/detail?id=215
            filename = None
            mode = "w"
        if mode == "memory":
            filename = None
            mode = "w"
            is_temporary = False
        elif filename is None:
            fd, filename = make_temporary_file()
            is_temporary = True
            LOGGER.debug("Created temporary file %s" % filename)

        else:
            is_temporary = False
        if isinstance(copy, Measurements):
            with copy.hdf5_dict.lock:
                self.hdf5_dict = HDF5Dict(
                    filename,
                    is_temporary=is_temporary,
                    copy=copy.hdf5_dict.top_group,
                    mode=mode,
                    image_numbers=image_numbers,
                )
        elif hasattr(copy, "__getitem__") and hasattr(copy, "keys"):
            self.hdf5_dict = HDF5Dict(
                filename,
                is_temporary=is_temporary,
                copy=copy,
                mode=mode,
                image_numbers=image_numbers,
            )
            if not multithread:
                self.hdf5_dict.lock = NullLock
        elif copy is not None:
            raise ValueError(
                "Copy source for measurments is neither a Measurements or HDF5 group."
            )
        else:
            self.hdf5_dict = HDF5Dict(filename, is_temporary=is_temporary, mode=mode)
        if is_temporary:
            os.close(fd)

        self.image_set_number = image_set_start or 1
        self.image_set_start = image_set_start

        self.__is_first_image = True
        self.__initialized_explicitly = False
        self.__relationships = set()
        self.__images = {}
        self.__image_providers = []
        self.__image_number_relationships = {}
        if RELATIONSHIP in self.hdf5_dict.top_group:
            rgroup = self.hdf5_dict.top_group[RELATIONSHIP]
            for module_number in rgroup:
                try:
                    mnum = int(module_number)
                except:
                    continue
                mrgroup = rgroup[module_number]
                if not isinstance(mrgroup, h5py.Group):
                    continue
                for rname in mrgroup:
                    rmrgroup = mrgroup[rname]
                    if not isinstance(rmrgroup, h5py.Group):
                        continue
                    for o1_name in rmrgroup:
                        rmro1group = rmrgroup[o1_name]
                        if not isinstance(rmro1group, h5py.Group):
                            continue
                        for o2_name in rmro1group:
                            if not isinstance(rmro1group[o2_name], h5py.Group):
                                continue
                            self.__relationships.add((mnum, rname, o1_name, o2_name))

    def __del__(self):
        if hasattr(self, "hdf5_dict"):
            self.close()

    def close(self):
        if hasattr(self, "hdf5_dict"):
            self.hdf5_dict.close()
            del self.hdf5_dict

    def __getitem__(self, key):
        # we support slicing the last dimension for the limited case of [..., :]
        if (
            len(key) == 3
            and isinstance(key[2], slice)
            and key[2] == slice(None, None, None)
        ):
            return self.get_all_measurements(*key[:2])
        return self.get_measurement(*key)

    def __setitem__(self, key, value):
        assert 2 <= len(key) <= 4
        if len(key) == 2:
            self.add_measurement(key[0], key[1], value)
        elif len(key) == 3:
            self.add_measurement(key[0], key[1], value, image_set_number=key[2])
        else:
            self.add_measurement(
                key[0], key[1], value, image_set_number=key[2], data_type=key[3]
            )

    def flush(self):
        if self.hdf5_dict is not None:
            self.hdf5_dict.flush()

    def file_contents(self):
        return self.hdf5_dict.file_contents()

    def initialize(self, measurement_columns):
        """Initialize the measurements with a list of objects and features

        This explicitly initializes the measurements with a list of
        object/feature pairs as would be returned by
        get_measurement_columns()

        measurement_columns - list of 3-tuples: object name, feature, type
        """
        # clear the old data, if any
        self.hdf5_dict.clear()

        def fix_type(t):
            if t == "integer":
                return int
            if t.startswith("varchar"):
                len = t.split("(")[1][:-1]
                return numpy.dtype("a" + len)
            return t

        for object_name, feature, coltype in measurement_columns:
            coltype = fix_type(coltype)
            if object_name == EXPERIMENT:
                dims = 0
            elif object_name == IMAGE:
                dims = 1
            else:
                dims = 2
            self.hdf5_dict.add_object(object_name)
            self.hdf5_dict.add_feature(object_name, feature)
        self.__initialized_explicitly = True

    def next_image_set(self, explicit_image_set_number=None):
        assert explicit_image_set_number is None or explicit_image_set_number > 0
        if explicit_image_set_number is None:
            self.image_set_number += 1
        else:
            self.image_set_number = explicit_image_set_number
        self.__is_first_image = False
        self.__images = {}
        self.__image_providers = []

    @property
    def image_set_count(self):
        """The number of complete image sets measured"""
        # XXX - question for Lee: should this return the minimum number
        # of non-null values across columns in the Image table?
        try:
            return len(self.hdf5_dict.get_indices("Image", "ImageNumber"))
        except KeyError:
            return 0

    def get_is_first_image(self):
        """True if this is the first image in the set"""
        return self.__is_first_image

    def set_is_first_image(self, value):
        if not value:
            raise ValueError("Can only reset to be first image")
        self.__is_first_image = True
        self.image_set_number = self.image_set_start_number

    is_first_image = property(get_is_first_image, set_is_first_image)

    @property
    def image_set_start_number(self):
        """The first image set (one-based) processed by the pipeline"""
        if self.image_set_start is None:
            return 1
        return self.image_set_start

    @property
    def has_image_set_start(self):
        """True if the image set has an explicit start"""
        return self.image_set_start is not None

    def create_from_handles(self, handles):
        """Load measurements from a handles structure"""
        m = handles["handles"][0, 0][MEASUREMENTS_GROUP_NAME][0, 0]
        for object_name in list(m.dtype.fields.keys()):
            omeas = m[object_name][0, 0]
            object_counts = numpy.zeros(0, int)
            for feature_name in list(omeas.dtype.fields.keys()):
                if object_name == IMAGE:
                    values = [
                        None if len(x) == 0 else x.flatten()[0]
                        for x in omeas[feature_name][0]
                    ]
                elif object_name == EXPERIMENT:
                    value = omeas[feature_name][0, 0].flatten()[0]
                    self.add_experiment_measurement(feature_name, value)
                    continue
                else:
                    values = [x.flatten() for x in omeas[feature_name][0].tolist()]
                    #
                    # Keep track of # of objects
                    #
                    if len(object_counts) < len(values):
                        temp, object_counts = (
                            object_counts,
                            numpy.zeros(len(values), int),
                        )
                        if len(temp) > 0:
                            object_counts[: len(temp)] = temp
                    object_counts[: len(values)] = numpy.maximum(
                        object_counts[: len(values)],
                        numpy.array([len(x) for x in values]),
                    )
                self.add_all_measurements(object_name, feature_name, values)
            if object_name not in (EXPERIMENT, IMAGE,) and not self.has_feature(
                object_name, OBJECT_NUMBER
            ):
                self.add_all_measurements(
                    object_name,
                    OBJECT_NUMBER,
                    [numpy.arange(1, x + 1) for x in object_counts],
                )
        #
        # Set the image set number to beyond the last in the handles
        #
        self.image_set_number = self.image_set_count + 1

    def add_image_measurement(self, feature_name, data):
        """Add a measurement to the "Image" category

        """
        self.add_measurement(IMAGE, feature_name, data)

    def add_experiment_measurement(self, feature_name, data):
        """Add an experiment measurement to the measurement

        Experiment measurements have one value per experiment
        """
        if isinstance(data, str):
            data = str(data)
        self.hdf5_dict.add_all(EXPERIMENT, feature_name, [data], [0])

    def get_group_number(self):
        """The number of the group currently being processed"""
        return self.get_current_image_measurement(GROUP_NUMBER)

    def set_group_number(self, group_number):
        self.add_image_measurement(GROUP_NUMBER, group_number)

    group_number = property(get_group_number, set_group_number)

    def get_group_index(self):
        """The within-group index of the current image set"""
        return self.get_current_image_measurement(GROUP_INDEX)

    def set_group_index(self, group_index):
        self.add_image_measurement(GROUP_INDEX, group_index)

    group_index = property(get_group_index, set_group_index)

    def get_group_length(self):
        """The group length of the current image group"""
        return self.get_current_image_measurement(GROUP_LENGTH)

    def set_group_length(self, group_length):
        self.add_image_measurement(GROUP_LENGTH, group_length)

    group_length = property(get_group_length, set_group_length)

    def get_groupings(self, features):
        """Return groupings of image sets based on feature values

        features - a sequence of feature names

        returns groupings suitable for return from CPModule.get_groupings.

        group_list - a sequence composed of two-tuples.
                     the first element of the tuple is a dictionary giving
                     the metadata values for the metadata keys
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ ({'Metadata_Row':'A','Metadata_Column':'01'}, [1,97,193]),
          ({'Metadata_Row':'A','Metadata_Column':'02'), [2,98,194]),... ]
        """
        d = {}
        image_numbers = self.get_image_numbers()
        values = [
            [str(x) for x in self.get_measurement(IMAGE, feature, image_numbers)]
            for feature in features
        ]
        for i, image_number in enumerate(image_numbers):
            key = tuple([(k, v[i]) for k, v in zip(features, values)])
            if key not in d:
                d[key] = []
            d[key].append(image_number)
        return [(dict(k), d[k]) for k in sorted(d.keys())]

    def get_relationship_hdf5_group(
        self, module_number, relationship, object_name1, object_name2
    ):
        """Return the HDF5 group for a relationship"""
        return (
            self.hdf5_dict.top_group.require_group(RELATIONSHIP)
            .require_group(str(module_number))
            .require_group(relationship)
            .require_group(object_name1)
            .require_group(object_name2)
        )

    def add_relate_measurement(
        self,
        module_number,
        relationship,
        object_name1,
        object_name2,
        image_numbers1,
        object_numbers1,
        image_numbers2,
        object_numbers2,
    ):
        """Add object relationships to the measurements

        module_number - the module that generated the relationship

        relationship - the relationship of the two objects, for instance,
                       "Parent" means object # 1 is the parent of object # 2

        object_name1, object_name2 - the name of the segmentation for the first and second objects

        image_numbers1, image_numbers2 - the image number of each object

        object_numbers1, object_numbers2 - for each object, the object number
                                           in the object's object set

        This method lets the caller store any sort of arbitrary relationship
        between objects as long as they are in the same group. To record
        all neighbors within a particular segmentation, call with the same
        object name for object_name1 and object_name2 and the same group
        index - that of the current image. Relating would have different object
        names and TrackObjects would have different group indices.

        The structure in the HDF file:
        Measurements / <date> / Relationship / <module #> /
           <relationship-name> / <object-name-1> / <object-name-2> /
           [ImageNumber_First, ObjectNumber_First,
            ImageNumber_Second, ObjectNumber_Second]

        The leaves are vector datasets.
        """
        if len(image_numbers1) == 0:
            return
        with self.hdf5_dict.lock:
            rgroup = self.get_relationship_hdf5_group(
                module_number, relationship, object_name1, object_name2
            )

            for name, values in (
                (R_FIRST_IMAGE_NUMBER, image_numbers1),
                (R_FIRST_OBJECT_NUMBER, object_numbers1),
                (R_SECOND_IMAGE_NUMBER, image_numbers2),
                (R_SECOND_OBJECT_NUMBER, object_numbers2),
            ):
                if name not in rgroup:
                    current_size = 0
                    rgroup.create_dataset(
                        name,
                        data=values,
                        dtype="int32",
                        chunks=(1024,),
                        maxshape=(None,),
                    )
                else:
                    dset = rgroup[name]
                    current_size = dset.shape[0]
                    dset.resize((current_size + len(values),))
                    dset[current_size:] = values
            key = (module_number, relationship, object_name1, object_name2)
            self.__relationships.add(key)
            if key not in self.__image_number_relationships:
                self.__image_number_relationships[
                    key
                ] = self.init_image_number_relationships(rgroup)
            else:
                d = self.__image_number_relationships[key]
                for image_numbers in (image_numbers1, image_numbers2):
                    self.update_image_number_relationships(
                        image_numbers, current_size, d
                    )

    def get_relationship_groups(self):
        """Return the keys of each of the relationship groupings.

        The value returned is a list composed of objects with the following
        attributes:
        module_number - the module number of the module used to generate the relationship
        relationship - the relationship of the two objects
        object_name1 - the object name of the first object in the relationship
        object_name2 - the object name of the second object in the relationship
        """

        return [
            RelationshipKey(module_number, relationship, obj1, obj2)
            for (module_number, relationship, obj1, obj2) in self.__relationships
        ]

    def get_relationships(
        self,
        module_number,
        relationship,
        object_name1,
        object_name2,
        image_numbers=None,
    ):
        """Get the relationships recorded by a particular module

        module_number - # of module recording the relationship

        relationship - the name of the relationship, e.g., "Parent" for
                       object # 1 is parent of object # 2

        object_name1, object_name2 - the names of the two objects

        image_numbers - if defined, only return relationships with first or
                        second objects in these image numbers.

        returns a recarray with the following fields:
        R_FIRST_IMAGE_NUMBER, R_SECOND_IMAGE_NUMBER, R_FIRST_OBJECT_NUMBER,
        R_SECOND_OBJECT_NUMBER
        """
        features = (
            R_FIRST_IMAGE_NUMBER,
            R_FIRST_OBJECT_NUMBER,
            R_SECOND_IMAGE_NUMBER,
            R_SECOND_OBJECT_NUMBER,
        )
        dt = numpy.dtype([(feature, numpy.int32, ()) for feature in features])
        if (
            not (module_number, relationship, object_name1, object_name2)
            in self.__relationships
        ):
            return numpy.zeros(0, dt).view(numpy.recarray)
        with self.hdf5_dict.lock:
            grp = self.get_relationship_hdf5_group(
                module_number, relationship, object_name1, object_name2
            )
            n_records = grp[R_FIRST_IMAGE_NUMBER].shape[0]
            if n_records == 0:
                return numpy.zeros(0, dt).view(numpy.recarray)
            if image_numbers is None:
                temp = numpy.zeros(n_records, dt)
                for feature in features:
                    temp[feature] = grp[feature]
            else:
                image_numbers = numpy.atleast_1d(image_numbers)
                k = (module_number, relationship, object_name1, object_name2)
                d = self.__image_number_relationships.get(k, None)
                if d is None:
                    d = self.__image_number_relationships[
                        k
                    ] = self.init_image_number_relationships(grp)
                #
                # Find the slice of the hdf5 array that contains all records
                # for the desired image numbers
                #
                t_min = sys.maxsize
                t_max = 0
                for image_number in image_numbers:
                    i_min, i_max = d.get(image_number, (t_min, t_max - 1))
                    t_min = min(i_min, t_min)
                    t_max = max(i_max + 1, t_max)
                if t_min >= t_max:
                    return numpy.zeros(0, dt).view(numpy.recarray)
                #
                # Construct a mask, offset by the minimum index to be addressed
                # of the image numbers to keep in the slice
                #
                in_min = numpy.min(image_numbers)
                in_max = numpy.max(image_numbers)
                to_keep = numpy.zeros(in_max - in_min + 1, bool)
                to_keep[image_numbers - in_min] = True
                mask = numpy.zeros(t_max - t_min, bool)
                for a in (
                    grp[R_FIRST_IMAGE_NUMBER][t_min:t_max],
                    grp[R_SECOND_IMAGE_NUMBER][t_min:t_max],
                ):
                    m1 = (a >= in_min) & (a <= in_max)
                    mask[m1] = mask[m1] | to_keep[a[m1] - in_min]
                #
                # Apply the mask to slices for all of the features
                #
                n_records = numpy.sum(mask)
                temp = numpy.zeros(n_records, dt)
                for feature in features:
                    temp[feature] = grp[feature][t_min:t_max][mask]
            return temp.view(numpy.recarray)

    @staticmethod
    def init_image_number_relationships(grp):
        """Create a dictionary of where to find image numbers in a relationship

        grp - the HDF5 group of the relationship

        returns a dictionary whose key is image number and whose value
        is a pair of the minimum and maximum position in the array of that
        image number.
        """
        d = {}
        chunk_size = 1000000
        for imgnums in (
            grp[R_FIRST_IMAGE_NUMBER],
            grp[R_SECOND_IMAGE_NUMBER],
        ):
            for i in range(0, imgnums.shape[0], chunk_size):
                limit = min(imgnums.shape[0], i + chunk_size)
                Measurements.update_image_number_relationships(imgnums[i:limit], i, d)
        return d

    @staticmethod
    def update_image_number_relationships(imgnums, offset, d):
        """Update an image number indexing dictionary with new image numbers

        imgnums - a vector of image numbers

        offset - the offset of this chunk within the relationships records

        d - the dictionary to update
        """

        offsets = offset + numpy.arange(len(imgnums))
        order = numpy.lexsort((offsets, imgnums))
        imgnums = imgnums[order]
        offsets = offsets[order]
        firsts = numpy.hstack(([True], imgnums[:-1] != imgnums[1:]))
        lasts = numpy.hstack((firsts[1:], [True]))
        for i, f, l in zip(imgnums[firsts], offsets[firsts], offsets[lasts]):
            old_f, old_l = d.get(i, (sys.maxsize, 0))
            d[i] = (min(old_f, f), max(old_l, l))

    def copy_relationships(self, src):
        """Copy the relationships from another measurements file

        src - a Measurements possibly having relationships.
        """
        for rk in src.get_relationship_groups():
            r = src.get_relationships(
                rk.module_number, rk.relationship, rk.object_name1, rk.object_name2
            )
            self.add_relate_measurement(
                rk.module_number,
                rk.relationship,
                rk.object_name1,
                rk.object_name2,
                r[R_FIRST_IMAGE_NUMBER],
                r[R_FIRST_OBJECT_NUMBER],
                r[R_SECOND_IMAGE_NUMBER],
                r[R_SECOND_OBJECT_NUMBER],
            )

    def add_measurement(
        self, object_name, feature_name, data, image_set_number=None, data_type=None
    ):
        """Add a measurement or, for objects, an array of measurements to the set

        This is the classic interface - like CPaddmeasurements:
        ObjectName - either the name of the labeled objects or "Image"
        FeatureName - the feature name, encoded with underbars for category/measurement/image/scale
        Data - the data item to be stored
        image_set_number - write the measurement to this image set or if
                           a sequence of image sets, write the sequence of
                           data values to the sequence of image sets
        data_type - an explicit data type to use when storing the measurements.
        """
        if image_set_number is None:
            image_set_number = self.image_set_number

        # some code adds ImageNumber and ObjectNumber measurements explicitly
        if feature_name in (IMAGE_NUMBER, OBJECT_NUMBER,):
            return

        if object_name == EXPERIMENT:
            if not numpy.isscalar(data) and data is not None and data_type is None:
                data = data[0]
            if data is None:
                data = []
            self.hdf5_dict[
                EXPERIMENT, feature_name, 0, data_type
            ] = Measurements.wrap_string(data)
        elif object_name == IMAGE:
            if numpy.isscalar(image_set_number):
                image_set_number = [image_set_number]
                data = [data]
            data = [
                d
                if d is None or d is numpy.NaN
                else Measurements.wrap_string(d)
                if numpy.isscalar(d)
                else Measurements.wrap_string(d[0])
                if data_type is None
                else d
                for d in data
            ]
            self.hdf5_dict[IMAGE, feature_name, image_set_number, data_type,] = data
            for n in image_set_number:
                if not self.hdf5_dict.has_data(object_name, IMAGE_NUMBER, n):
                    self.hdf5_dict[IMAGE, IMAGE_NUMBER, n,] = n
        else:
            self.hdf5_dict[
                object_name, feature_name, image_set_number, data_type
            ] = data
            for n, d in (
                ((image_set_number, data),)
                if numpy.isscalar(image_set_number)
                else list(zip(image_set_number, data))
            ):
                if not self.hdf5_dict.has_data(IMAGE, IMAGE_NUMBER, n,):
                    self.hdf5_dict[IMAGE, IMAGE_NUMBER, n,] = n
                if (not self.hdf5_dict.has_data(object_name, OBJECT_NUMBER, n)) and (
                    d is not None
                ):
                    self.hdf5_dict[object_name, IMAGE_NUMBER, n,] = [n] * len(d)
                self.hdf5_dict[object_name, OBJECT_NUMBER, n] = numpy.arange(
                    1, len(d) + 1
                )

    def remove_measurement(self, object_name, feature_name, image_number=None):
        """Remove the measurement for the given image number

        object_name - the measurement's object. If other than Image or Experiment,
                      will remove measurements for all objects
        feature_name - name of the measurement feature
        image_number - the image set's image number
        """
        if image_number is None:
            del self.hdf5_dict[object_name, feature_name]
        else:
            del self.hdf5_dict[object_name, feature_name, image_number]

    def clear(self):
        """Remove all measurements"""
        self.hdf5_dict.clear()

    def get_object_names(self):
        """The list of object names (including Image) that have measurements
        """
        return [x for x in self.hdf5_dict.top_level_names() if x not in (DB_TEMP, RELATIONSHIP)]

    object_names = property(get_object_names)

    def get_feature_names(self, object_name):
        """The list of feature names (measurements) for an object
        """
        return [
            name
            for name in self.hdf5_dict.second_level_names(object_name)
            if name not in ("ImageNumber", "ObjectNumber")
        ]

    def get_image_numbers(self):
        """Return the image numbers from the Image table"""
        image_numbers = numpy.array(
            list(self.hdf5_dict.get_indices(IMAGE, IMAGE_NUMBER,).keys()), int,
        )
        image_numbers.sort()
        return image_numbers

    def reorder_image_measurements(self, new_image_numbers):
        """Assign all image measurements to new image numbers

        new_image_numbers - a zero-based array that maps old image number
                            to new image number, e.g., if
                            new_image_numbers = [ 0, 3, 1, 2], then
                            the measurements for old image number 1 will
                            be the measurements for new image number 3, etc.

        Note that this does not handle any image numbers that might be stored
        in the measurements themselves. It is intended for use in
        prepare_run when it is necessary to reorder image numbers because
        of regrouping.
        """
        for feature in self.get_feature_names(IMAGE):
            self.hdf5_dict.reorder(IMAGE, feature, new_image_numbers)

    def has_feature(self, object_name, feature_name):
        return self.hdf5_dict.has_feature(object_name, feature_name)

    def get_current_image_measurement(self, feature_name):
        """Return the value for the named image measurement

        feature_name - the name of the measurement feature to be returned
        """
        return self.get_current_measurement(IMAGE, feature_name)

    def get_current_measurement(self, object_name, feature_name):
        """Return the value for the named measurement for the current image set
        object_name  - the name of the objects being measured or "Image"
        feature_name - the name of the measurement feature to be returned
        """
        return self.get_measurement(object_name, feature_name, self.image_set_number)

    @staticmethod
    def wrap_string(v):
        if isinstance(v, str):
            if getattr(v, "__class__") == str:
                v = v
            return v
        return v

    @staticmethod
    def unwrap_string(v):
        # hdf5 returns string columns as a wrapped type
        # Strings are (sometimes?) returned as numpy.object_ and bizarrely,
        # type(v) == numpy.object_, but v.__class__==str. Additionally,
        # builtin type like number has a __class__ attribute but that can't be
        # referenced with the dot syntax.
        # More Info: http://docs.h5py.org/en/stable/strings.html#how-to-store-text-strings
        #
        if getattr(v, "__class__") == str:
            return v
        elif getattr(v, "__class__") == bytes:
                return v.decode()
        return v

    def get_measurement(self, object_name, feature_name, image_set_number=None):
        """Return the value for the named measurement and indicated image set

        object_name - the name of one of the objects or one of the generic
                      names such as Image or Experiment

        feature_name - the name of the feature to retrieve

        image_set_number - the current image set by default, a single
                           image set number to get measurements for one
                           image set or a sequence of image numbers to
                           return measurements for each of the image sets
                           listed.
        """
        if object_name == EXPERIMENT:
            result = self.hdf5_dict[EXPERIMENT, feature_name, 0]
            if len(result) == 1:
                result = result[0]
            return Measurements.unwrap_string(result)
        if image_set_number is None:
            image_set_number = self.image_set_number
        vals = self.hdf5_dict[object_name, feature_name, image_set_number]
        if object_name == IMAGE:
            if numpy.isscalar(image_set_number):
                if vals is None or len(vals) == 0:
                    return None
                if len(vals) == 1:
                    return Measurements.unwrap_string(vals[0])
                return vals
            else:
                measurement_dtype = self.hdf5_dict.get_feature_dtype(
                    object_name, feature_name
                )
                if h5py.check_dtype(vlen=measurement_dtype) == str:
                    result = [
                        Measurements.unwrap_string(v[0]) if v is not None else None
                        for v in vals
                    ]
                elif measurement_dtype == numpy.uint8:
                    #
                    # Blobs - just pass them through as an array.
                    #
                    result = vals
                else:
                    # numeric expect as numpy array, text as list (or possibly
                    # array of object in order to handle np.NaN
                    #
                    # A missing result is assumed to be "unable to calculate
                    # in this case and we substitute NaN for it.
                    #
                    # Python 2 CellProfiler workspaces will have measurements
                    # stored as bytes rather than strings, so we decode them.
                    #
                    result = numpy.array(
                        [
                            numpy.NaN
                            if v is None or len(v) == 0
                            else v[0].decode("utf-8")
                            if len(v) == 1 and isinstance(v[0], bytes)
                            else v[0]
                            if len(v) == 1
                            else v
                            for v in vals
                        ]
                    )
                return result
        if numpy.isscalar(image_set_number):
            return numpy.array([]) if vals is None else vals.flatten()
        return [numpy.array([]) if v is None else v.flatten() for v in vals]

    def get_measurement_columns(self):
        """Return the measurement columns for the current measurements

        This returns the measurement columns in the style of
        pipeline.get_measurement_columns. It can be used for cases where
        the measurements are loaded from a file and do not reflect
        current module functionality.

        Note that this doesn't correctly differentiate string data and blob
        data.
        """
        result = []
        for object_name in self.get_object_names():
            for feature_name in self.get_feature_names(object_name):
                dtype = self.hdf5_dict.get_feature_dtype(object_name, feature_name)
                if dtype.kind in ["O", "S", "U"]:
                    result.append((object_name, feature_name, COLTYPE_VARCHAR,))
                elif numpy.issubdtype(dtype, float):
                    result.append((object_name, feature_name, COLTYPE_FLOAT,))
                else:
                    result.append((object_name, feature_name, COLTYPE_INTEGER,))
        return result

    def has_measurements(self, object_name, feature_name, image_set_number):
        if object_name == EXPERIMENT:
            return self.hdf5_dict.has_data(EXPERIMENT, feature_name, 0)
        return self.hdf5_dict.has_data(object_name, feature_name, image_set_number)

    def has_current_measurements(self, object_name, feature_name):
        return self.has_measurements(object_name, feature_name, self.image_set_number)

    def get_all_measurements(self, object_name, feature_name):
        warnings.warn(
            "get_all_measurements is deprecated. Please use "
            "get_measurements with an array of image numbers instead",
            DeprecationWarning,
        )
        return self.get_measurement(object_name, feature_name, self.get_image_numbers())

    def add_all_measurements(self, object_name, feature_name, values, data_type=None):
        """Add a list of measurements for all image sets

        object_name - name of object or Images
        feature_name - feature to add
        values - list of either values or arrays of values
        """
        values = [
            []
            if value is None
            else [Measurements.wrap_string(value)]
            if numpy.isscalar(value)
            else value
            for value in values
        ]
        if (not self.hdf5_dict.has_feature(IMAGE, IMAGE_NUMBER,)) or (
            numpy.max(self.get_image_numbers()) < len(values)
        ):
            image_numbers = numpy.arange(1, len(values) + 1)
            self.hdf5_dict.add_all(
                IMAGE, IMAGE_NUMBER, image_numbers,
            )
        else:
            image_numbers = self.get_image_numbers()
        self.hdf5_dict.add_all(
            object_name, feature_name, values, image_numbers, data_type=data_type
        )

    def get_experiment_measurement(self, feature_name):
        """Retrieve an experiment-wide measurement
        """
        result = self.get_measurement(EXPERIMENT, feature_name)
        return "N/A" if result is None else result

    def apply_metadata(self, pattern, image_set_number=None):
        """Apply metadata from the current measurements to a pattern

        pattern - a regexp-like pattern that specifies how to insert
                  metadata into a string. Each token has the form:
                  "\(?<METADATA_TAG>\)" (matlab-style) or
                  "\g<METADATA_TAG>" (Python-style)
        image_name - name of image associated with the metadata (or None
                     if metadata is not associated with an image)
        image_set_number - # of image set to use to retrieve data.
                           None for current.

        C_SERIES and C_FRAME are special cases. We look at the series/frame
        values for all images in the image set and take the one that's the
        highest - assuming that there may be a stack + a single image like
        one used for background subtraction. Admittedly a heuristic, but
        there it is.

        returns a string with the metadata tags replaced by the metadata
        """
        if image_set_number is None:
            image_set_number = self.image_set_number
        result_pieces = []
        double_backquote = "\\\\"
        single_backquote = "\\"
        for piece in pattern.split(double_backquote):
            # Replace tags in piece
            result = ""
            while True:
                # Replace one tag
                m = re.search("\\(\\?[<](.+?)[>]\\)", piece)
                if not m:
                    m = re.search("\\\\g[<](.+?)[>]", piece)
                    if not m:
                        result += piece
                        break
                result += piece[: m.start()]
                feature = m.groups()[0]
                if feature in (C_SERIES, C_FRAME,):
                    max_value = 0
                    for mname in self.get_feature_names(IMAGE):
                        if mname.startswith(feature + "_"):
                            value = self[
                                IMAGE, mname, image_set_number,
                            ]
                            if value and value > max_value:
                                max_value = value
                    result += str(max_value)
                else:
                    measurement = "%s_%s" % (C_METADATA, feature,)
                    result += str(
                        self.get_measurement("Image", measurement, image_set_number)
                    )
                piece = piece[m.end() :]
            result_pieces.append(result)
        return single_backquote.join(result_pieces)

    def has_groups(self):
        """Return True if there is more than one group in the image sets

        Note - this works the dumb way now: it fetches all of the group numbers
               and sees if there is a single unique group number. It involves
               fetching the whole column and it doesn't cache, so it could
               be expensive. Alternatively, this could be an experiment
               measurement, populated after prepare_run.
        """
        if self.has_feature(IMAGE, GROUP_NUMBER,):
            image_numbers = self.get_image_numbers()
            if len(image_numbers) > 0:
                group_numbers = self.get_measurement(
                    IMAGE, GROUP_NUMBER, image_set_number=image_numbers,
                )
                return len(numpy.unique(group_numbers)) > 1
        return False

    def group_by_metadata(self, tags):
        """Return groupings of image sets with matching metadata tags

        tags - a sequence of tags to match.

        Returns a sequence of MetadataGroup objects. Each one represents
        a set of values for the metadata tags along with the image numbers of
        the image sets that match the values
        """
        if len(tags) == 0:
            # if there are no tags, all image sets match each other
            return [MetadataGroup({}, self.get_image_numbers())]

        #
        # The flat_dictionary has a row of tag values as a key
        #
        flat_dictionary = {}
        image_numbers = self.get_image_numbers()
        values = [
            self.get_measurement(IMAGE, "%s_%s" % (C_METADATA, tag), image_numbers,)
            for tag in tags
        ]
        for i, image_number in enumerate(image_numbers):
            key = tuple([(k, v[i]) for k, v in zip(tags, values)])
            if key not in flat_dictionary:
                flat_dictionary[key] = []
            flat_dictionary[key].append(image_number)
        result = []
        for row in list(flat_dictionary.keys()):
            tag_dictionary = dict(row)
            result.append(MetadataGroup(tag_dictionary, flat_dictionary[row]))
        return result

    def match_metadata(self, features, values):
        """Match vectors of metadata values to existing measurements

        This method finds the image sets that match each row in a vector
        of metadata values. Imagine being given an image set with metadata
        values of plate, well and site and annotations for each well
        with metadata values of plate and well and annotation. You'd like
        to match each annotation with all of the sites for it's well. This
        method will return the image numbers that match.

        The method can also be used to match images, for instance when
        different illumination correction functions need to be matched
        against plates or sites.

        features - the measurement names for the incoming metadata

        values - a sequence of vectors, one per feature, giving the
                 metadata values to be matched.

        returns a sequence of vectors of image numbers of equal length
        to the values. An exception is thrown if the metadata for more
        than one row in the values matches the same image set unless the number
        of values in each vector equals the number of image sets - in that case,
        the vectors are assumed to be arranged in the correct order already.
        """
        #
        # Get image features populated by previous modules. If there are any,
        # then we launch the desperate heuristics that attempt to match
        # to them, either by order or by common metadata
        #
        image_set_count = len(self.get_image_numbers())
        by_order = [[i + 1] for i in range(len(values[0]))]
        if image_set_count == 0:
            return by_order

        image_features = self.get_feature_names(IMAGE)
        metadata_features = [
            x for x in image_features if x.startswith(C_METADATA + "_")
        ]
        common_features = [x for x in metadata_features if x in features]
        if len(common_features) == 0:
            if image_set_count > len(values[0]):
                raise ValueError("The measurements and data have no metadata in common")
            return by_order

        #
        # This reduces numberlike things to integers so that they can be
        # more loosely matched.
        #
        def cast(x):
            if isinstance(x, str) and x.isdigit():
                return int(x)
            return x

        common_tags = [f[(len(C_METADATA) + 1) :] for f in common_features]
        groupings = self.group_by_metadata(common_tags)
        groupings = dict(
            [
                (tuple([cast(d[f]) for f in common_tags]), d.image_numbers)
                for d in groupings
            ]
        )
        if image_set_count == len(values[0]):
            #
            # Test whether the common features uniquely identify
            # all image sets. If so, then we can match by metadata
            # and that will be correct, even when the user wants to
            # match by order (assuming the user really did match
            # the metadata)
            #
            if any([len(v) != 1 for v in list(groupings.values())]):
                return by_order
        #
        # Create a list of values that matches the common_features
        #
        result = []
        vv = [values[features.index(c)] for c in common_features]
        for i in range(len(values[0])):
            key = tuple([cast(vvv[i]) for vvv in vv])
            if key not in groupings:
                raise ValueError(
                    (
                        "There was no image set whose metadata matched row %d.\n"
                        % (i + 1)
                    )
                    + "Metadata values: "
                    + ", ".join(
                        ["%s = %s" % (k, v) for k, v in zip(common_features, key)]
                    )
                )
            result.append(groupings[key])
        return result

    @staticmethod
    def agg_ignore_object(object_name):
        """Ignore objects (other than 'Image') if this returns true"""
        if object_name in (EXPERIMENT, NEIGHBORS,):
            return True

    def agg_ignore_feature(self, object_name, feature_name):
        """Return true if we should ignore a feature during aggregation"""

        if self.agg_ignore_object(object_name):
            return True
        if self.hdf5_dict.has_feature(object_name, "SubObjectFlag"):
            return True
        return agg_ignore_feature(feature_name)

    def compute_aggregate_measurements(self, image_set_number, aggs=None):
        """Compute aggregate measurements for a given image set

        returns a dictionary whose key is the aggregate measurement name and
        whose value is the aggregate measurement value
        """
        if aggs is None:
            aggs = AGG_NAMES
        d = {}
        if len(aggs) == 0:
            return d
        for object_name in self.get_object_names():
            if object_name == "Image":
                continue
            for feature in self.get_feature_names(object_name):
                if self.agg_ignore_feature(object_name, feature):
                    continue
                feature_name = "%s_%s" % (object_name, feature)
                values = self.get_measurement(object_name, feature, image_set_number)
                if not numpy.issubdtype(values.dtype, numpy.number):
                    # Can't generate aggregate values for non-numeric measurements
                    values = None
                if values is not None:
                    values = values[numpy.isfinite(values)]
                #
                # Compute the mean and standard deviation
                #
                if AGG_MEAN in aggs:
                    mean_feature_name = get_agg_measurement_name(
                        AGG_MEAN, object_name, feature
                    )
                    mean = numpy.mean(values) if values is not None else numpy.NaN
                    d[mean_feature_name] = mean
                if AGG_MEDIAN in aggs:
                    median_feature_name = get_agg_measurement_name(
                        AGG_MEDIAN, object_name, feature
                    )
                    median = numpy.median(values) if values is not None else numpy.NaN
                    d[median_feature_name] = median
                if AGG_STD_DEV in aggs:
                    stdev_feature_name = get_agg_measurement_name(
                        AGG_STD_DEV, object_name, feature
                    )
                    stdev = numpy.std(values) if values is not None else numpy.NaN
                    d[stdev_feature_name] = stdev
        return d

    def load_image_sets(self, fd_or_file, start=None, stop=None):
        """Load image sets from a .csv file into a measurements file

        fd_or_file - either the path name of the .csv file or a file-like object

        start - the 1-based image set number to start the loading. For instance,
                for start = 2, we skip the first line and write image
                measurements starting at line 2 into image set # 2

        stop - stop loading when this line is reached.
        """
        if isinstance(fd_or_file, str):
            with open(fd_or_file, "r") as fd:
                return self.load_image_sets(fd, start, stop)
        import csv

        reader = csv.reader(fd_or_file)
        header = [x for x in next(reader)]
        columns = [[] for _ in range(len(header))]
        column_is_all_none = numpy.ones(len(header), bool)
        last_image_number = 0
        for i, fields in enumerate(reader):
            fields = [x for x in fields]
            image_number = i + 1
            if start is not None and start < image_number:
                continue
            if stop is not None and image_number == stop:
                break
            for j, (field, column) in enumerate(zip(fields, columns)):
                if field == "None" or len(field) == 0:
                    field = None
                else:
                    column_is_all_none[j] = False
                column.append(field)
            last_image_number = image_number
        if last_image_number == 0:
            LOGGER.warn("No image sets were loaded")
            return
        if start is None:
            image_numbers = list(range(1, last_image_number + 1))
        else:
            image_numbers = list(range(start, last_image_number + 1))
        self.hdf5_dict.add_all(
            IMAGE, IMAGE_NUMBER, image_numbers, image_numbers,
        )
        for feature, column, all_none in zip(header, columns, column_is_all_none):
            if not all_none:
                # try to convert to an integer, then float, then leave as string
                column = numpy.array(column, object)
                try:
                    column = column.astype(int)
                except:
                    try:
                        column = column.astype(float)
                    except:
                        column = numpy.array(
                            [Measurements.wrap_string(x) for x in column], object
                        )
                self.hdf5_dict.add_all(IMAGE, feature, column, image_numbers)

    def write_image_sets(self, fd_or_file, start=None, stop=None):
        if isinstance(fd_or_file, str):
            with open(fd_or_file, "w") as fd:
                return self.write_image_sets(fd, start, stop)

        fd = fd_or_file

        to_save = [
            GROUP_NUMBER,
            GROUP_INDEX,
            GROUP_LENGTH
        ]
        to_save_prefixes = [
            C_URL,
            C_PATH_NAME,
            C_FILE_NAME,
            C_SERIES,
            C_FRAME,
            C_C,
            C_Z,
            C_T,
            C_OBJECTS_URL,
            C_OBJECTS_PATH_NAME,
            C_OBJECTS_FILE_NAME,
            C_OBJECTS_SERIES,
            C_OBJECTS_FRAME,
            C_OBJECTS_CHANNEL,
            C_OBJECTS_Z,
            C_OBJECTS_T,
            C_METADATA,
        ]

        keys = []
        image_features = self.get_feature_names(IMAGE)
        for feature in to_save:
            if feature in image_features:
                keys.append(feature)
        for prefix in to_save_prefixes:
            for feature in image_features:
                if feature.startswith(prefix) and feature not in keys:
                    keys.append(feature)
        header = '"' + '","'.join(keys) + '"\n'
        fd.write(header)
        image_numbers = self.get_image_numbers()
        if start is not None:
            image_numbers = [x for x in image_numbers if x >= start]
        if stop is not None:
            image_numbers = [x for x in image_numbers if x <= stop]

        if len(image_numbers) == 0:
            return

        columns = [
            self.get_measurement(IMAGE, feature_name, image_set_number=image_numbers,)
            for feature_name in keys
        ]
        for i, image_number in enumerate(image_numbers):
            for j, column in enumerate(columns):
                field = column[i]
                if field is numpy.NaN or field is None:
                    field = ""
                if isinstance(field, str):
                    if isinstance(field, str):
                        field = field
                    field = '"' + field.replace('"', '""') + '"'
                else:
                    field = str(field)
                if j > 0:
                    fd.write("," + field)
                else:
                    fd.write(field)
            fd.write("\n")

    def alter_path_for_create_batch(self, name, is_image, fn_alter_path):
        """Alter the path of image location measurements for CreateBatchFiles

        name - name of the image or objects
        is_image - True to load as an image, False to load as objects
        fn_later_path - call this function to alter the path for batch processing
        """
        from cellprofiler_core.utilities.pathname import url2pathname
        from cellprofiler_core.utilities.pathname import pathname2url

        if is_image:
            path_feature = C_PATH_NAME
            file_feature = C_FILE_NAME
            url_feature = C_URL
        else:
            path_feature = C_OBJECTS_PATH_NAME
            file_feature = C_OBJECTS_FILE_NAME
            url_feature = C_OBJECTS_URL
        path_feature, file_feature, url_feature = [
            "_".join((f, name)) for f in (path_feature, file_feature, url_feature)
        ]

        all_image_numbers = self.get_image_numbers()
        urls = self.get_measurement(
            IMAGE, url_feature, image_set_number=all_image_numbers,
        )

        new_urls = []
        for url in urls:
            if url.lower().startswith("file:"):
                full_name = url2pathname(url)
                full_name = fn_alter_path(full_name)
                new_url = pathname2url(full_name)
            else:
                new_url = url
            new_urls.append(new_url)
        if any([url != new_url for url, new_url in zip(urls, new_urls)]):
            self.add_all_measurements(IMAGE, url_feature, new_urls)

        paths = self.get_measurement(
            IMAGE, path_feature, image_set_number=all_image_numbers,
        )
        new_paths = [fn_alter_path(path) for path in paths]
        if any([path != new_path for path, new_path in zip(paths, new_paths)]):
            self.add_all_measurements(IMAGE, path_feature, new_paths)

        filenames = self.get_measurement(
            IMAGE, file_feature, image_set_number=all_image_numbers,
        )
        new_filenames = [fn_alter_path(filename) for filename in filenames]
        if any(
            [
                filename != new_filename
                for filename, new_filename in zip(filenames, new_filenames)
            ]
        ):
            self.add_all_measurements(IMAGE, file_feature, new_filenames)

    def write_path_mappings(self, mappings):
        """Write the mappings of local/remote dirs as an experiment measurement

        This records the mappings of local and remote directories entered
        by the CreateBatchFiles module.

        mappings - a sequence of two-tuples. The first tuple is the local
                   path and the second is the remote path (on the target
                   machine for the run)
        """
        d = {
            K_CASE_SENSITIVE: (os.path.normcase("A") != os.path.normcase("a")),
            K_LOCAL_SEPARATOR: os.path.sep,
            K_PATH_MAPPINGS: tuple([tuple(m) for m in mappings]),
            K_URL2PATHNAME_PACKAGE_NAME: urllib.request.url2pathname.__module__,
        }
        s = json.dumps(d)
        self.add_experiment_measurement(M_PATH_MAPPINGS, s)

    def alter_url_post_create_batch(self, url):
        """Apply CreateBatchFiles path mappings to an unmapped URL

        This method can be run on the measurements output by CreateBatchFiles
        to map the paths of any URL that wasn't mapped by the alter-paths
        mechanism (e.g., URLs encoded in blobs)

        url - the url to map

        returns - a possibly mapped URL
        """
        if not url.lower().startswith("file:"):
            return url
        if not self.has_feature(EXPERIMENT, M_PATH_MAPPINGS,):
            return url
        d = json.loads(self.get_experiment_measurement(M_PATH_MAPPINGS))
        full_name = os_url2pathname(url[5:])
        full_name_c = full_name if d[K_CASE_SENSITIVE] else full_name.lower()
        if d[K_LOCAL_SEPARATOR] != os.path.sep:
            full_name = full_name.replace(d[K_LOCAL_SEPARATOR], os.path.sep)
        for local_directory, remote_directory in d[K_PATH_MAPPINGS]:
            if d[K_CASE_SENSITIVE]:
                if full_name_c.startswith(local_directory):
                    full_name = remote_directory + full_name[len(local_directory) :]
            else:
                if full_name_c.startswith(local_directory.lower()):
                    full_name = remote_directory + full_name[len(local_directory) :]
        url = "file:" + urllib.request.pathname2url(full_name)
        return url

    ###########################################################
    #
    # Ducktyping measurements as image sets
    #
    ###########################################################

    @property
    def image_number(self):
        """The image number of the current image"""
        return self.image_set_number

    @property
    def get_keys(self):
        """The keys that uniquely identify the image set

        Return key/value pairs for the metadata that specifies the site
        for the image set, for instance, plate / well / site. If image set
        was created by matching images by order, the image number will be
        returned.
        """
        #
        # XXX (leek) - save the metadata tags used for matching in the HDF
        #              then use it to look up the values per image set
        #              and cache.
        #
        return {IMAGE_NUMBER: str(self.image_number)}

    def get_grouping_keys(self):
        """Get a key, value dictionary that uniquely defines the group

        returns a dictionary for the current image set's group where the
        key is the image feature name and the value is the value to match
        in the image measurements.

        Note: this is somewhat legacy, from before GROUP_NUMBER was defined
              and the only way to determine which images were in a group
              was to get the metadata colums used to define groups and scan
              them for matches. Now, we just return { GROUP_NUMBER: value }
        """
        return {GROUP_NUMBER: self.get_current_image_measurement(GROUP_NUMBER)}

    def get_image(
        self,
        name,
        must_be_binary=False,
        must_be_color=False,
        must_be_grayscale=False,
        must_be_rgb=False,
        cache=True,
    ):
        """Return the image associated with the given name

        name - name of the image within the image_set
        must_be_color - raise an exception if not a color image
        must_be_grayscale - raise an exception if not a grayscale image
        must_be_rgb - raise an exception if 2-d or if # channels not 3 or 4,
                      discard alpha channel.
        """
        from cellprofiler_core.image.abstract_image.file.url._url_image import URLImage
        from cellprofiler_core.image import GrayscaleImage, RGBImage

        name = str(name)
        if name in self.__images:
            image = self.__images[name]
        else:
            matching_providers = [
                p for p in self.__image_providers if p.get_name() == name
            ]
            if len(matching_providers) == 0:
                #
                # Try looking up the URL in measurements
                #
                url_feature_name = "_".join((C_URL, name))
                series_feature_name = "_".join((C_SERIES, name))
                index_feature_name = "_".join((C_FRAME, name))
                if not self.has_feature(IMAGE, url_feature_name):
                    raise ValueError(
                        "The %s image is missing from the pipeline." % name
                    )
                # URL should be ASCII only
                url = str(self.get_current_image_measurement(url_feature_name))
                if self.has_feature(IMAGE, series_feature_name):
                    series = self.get_current_image_measurement(series_feature_name)
                else:
                    series = None
                if self.has_feature(IMAGE, index_feature_name):
                    index = self.get_current_image_measurement(index_feature_name)
                else:
                    index = None
                #
                # XXX (leek): Rescale needs to be bubbled up into
                #             NamesAndTypes and needs to be harvested
                #             from LoadImages etc.
                #             and stored in the measurements.
                #
                metadata_rescale = True
                provider = URLImage(name, url, series, index, metadata_rescale=metadata_rescale)
                self.__image_providers.append(provider)
                matching_providers.append(provider)
            image = matching_providers[0].provide_image(self)
            if cache:
                self.__images[name] = image

        if image.multichannel:
            if must_be_binary:
                raise ValueError("Image must be binary, but it was color")

            if must_be_grayscale:
                pd = image.pixel_data

                pd = pd.transpose(-1, *list(range(pd.ndim - 1)))

                if (
                    pd.shape[-1] >= 3
                    and numpy.all(pd[0] == pd[1])
                    and numpy.all(pd[0] == pd[2])
                ):
                    return GrayscaleImage(image)

                raise ValueError("Image must be grayscale, but it was color")

            if must_be_rgb:
                if image.pixel_data.shape[-1] not in (3, 4):
                    raise ValueError(
                        "Image must be RGB, but it had %d channels"
                        % image.pixel_data.shape[-1]
                    )

                if image.pixel_data.shape[-1] == 4:
                    LOGGER.warning("Discarding alpha channel.")

                    return RGBImage(image)

            return image

        if must_be_binary and image.pixel_data.dtype != bool:
            raise ValueError("Image was not binary")

        if must_be_grayscale and image.pixel_data.dtype.kind == "b":
            return GrayscaleImage(image)

        if must_be_rgb:
            raise ValueError("Image must be RGB, but it was grayscale")

        if must_be_color:
            raise ValueError("Image must be color, but it was grayscale")

        return image

    def get_providers(self):
        """The list of providers (populated during the image discovery phase)"""
        # return tuple to prevent mutating underlying list
        return list(self.__image_providers)

    providers = property(get_providers)
    
    def add_provider(self, provider):
        self.__image_providers.append(provider)

    def get_image_provider(self, name):
        """Get a named image provider

        name - return the image provider with this name
        """
        providers = [x for x in self.__image_providers if x.name == name]
        assert len(providers) > 0, "No provider of the %s image" % name
        assert len(providers) == 1, "More than one provider of the %s image" % name
        return providers[0]

    def remove_image_provider(self, name):
        """Remove a named image provider
        name - the name of the provider to remove
        """
        self.__image_providers = [x for x in self.__image_providers if x.name != name]

    def clear_image(self, name):
        """Remove the image memory associated with a provider

        name - the name of the provider
        """
        self.get_image_provider(name).release_memory()
        if name in self.__images:
            del self.__images[name]

    def clear_cache(self):
        """Remove all of the cached images"""
        self.__images = {}

    def get_names(self):
        """Get the image provider names
        """
        return [provider.name for provider in self.providers]

    names = property(get_names)

    def add(self, name, image):
        from cellprofiler_core.image import VanillaImage

        old_providers = [
            provider for provider in self.providers if provider.name == name
        ]
        if len(old_providers) > 0:
            self.clear_image(name)
        for provider in old_providers:
            self.remove_image_provider(provider.name)
        provider = VanillaImage(name, image)
        self.add_provider(provider)
        self.__images[name] = image

    def set_channel_descriptors(self, channel_descriptors):
        """Write the names and data types of the channel descriptors

        channel_descriptors - pipeline channel descriptors describing the
                              channels in the image set.
        """
        for name, channel_type in channel_descriptors.items():
            self.add_experiment_measurement(f"{C_CHANNEL_TYPE}_{name}", channel_type)

    def get_channel_descriptors(self):
        """Read the channel descriptors

        Returns pipeline.ImageSetChannelDescriptor instances for each
        channel descriptor specified in the experiment measurements.
        """
        channel_descriptors = {}
        for feature_name in self.get_feature_names(EXPERIMENT):
            if feature_name.startswith(C_CHANNEL_TYPE):
                channel_name = feature_name[(len(C_CHANNEL_TYPE) + 1):]
                channel_type = self.get_experiment_measurement(feature_name)
                if channel_type == CT_OBJECTS:
                    url_feature = "_".join([C_OBJECTS_URL, channel_name])
                else:
                    url_feature = "_".join([C_URL, channel_name])
                if url_feature not in self.get_feature_names(IMAGE):
                    continue
                channel_descriptors[channel_name] = channel_type
        return channel_descriptors

    def get_channel_descriptor(self, name):
        """Return the channel descriptor with the given name"""
        feature_name = f"{C_CHANNEL_TYPE}_{name}"
        if self.has_measurements(EXPERIMENT, feature_name, None):
            return self.get_experiment_measurement(feature_name)
        return None

    def set_metadata_tags(self, metadata_tags):
        """Write the metadata tags that are used to make an image set

        metadata_tags - image feature names of the metadata tags that uniquely
                        define an image set. If metadata matching wasn't used,
                        write the image number feature name.
        """
        data = json.dumps(metadata_tags)
        self.add_experiment_measurement(M_METADATA_TAGS, data)

    def get_metadata_tags(self):
        """Read the metadata tags that are used to make an image set

        returns a list of metadata tags
        """
        if M_METADATA_TAGS not in self.get_feature_names(EXPERIMENT):
            return [IMAGE_NUMBER]
        return json.loads(self.get_experiment_measurement(M_METADATA_TAGS))

    def set_grouping_tags(self, grouping_tags):
        """Write the metadata tags that are used to group an image set

        grouping_tags - image feature names of the metadata tags that
                        uniquely define a group.
        """
        data = json.dumps(grouping_tags)
        self.add_experiment_measurement(M_GROUPING_TAGS, data)

    def get_grouping_tags_or_metadata(self):
        """Get the metadata tags that were used to group the image set

        """
        if not self.has_feature(EXPERIMENT, M_GROUPING_TAGS,):
            return self.get_metadata_tags()

        return json.loads(self.get_experiment_measurement(M_GROUPING_TAGS))

    def get_grouping_tags_only(self):
        """Get the metadata tags that were used to group the image set,
        and only those, not metadata instead
        """
        if not self.has_feature(EXPERIMENT, M_GROUPING_TAGS,):
            return []

        return json.loads(self.get_experiment_measurement(M_GROUPING_TAGS))
