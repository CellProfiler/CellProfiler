class ImageSetCache(object):
    '''An ImageSetCache holds the computed results from an image set calculation
    
    The cache remembers the following things:
    
    * Metadata key names used to match images/objects in the image set.
    
    * Image / object names in the image sets.
    
    * Metadata values used in matching.
    
    * URLs, series #, frame #, channel # for each image
    
    * Rows that have errors: the row, image or object name in error.
    
    The image set cache holds the same sort of information as the image
    URL and metadata measurements. It's purpose is to be a more informal
    collection of image set results for cases where the configuration is
    incomplete or might contain errors.
    '''
    # The name of the group hosting the image set cache
    IMAGE_SET_CACHE_GROUP = "ImageSetCache"
    # The version # of the structure.
    IMAGE_SET_CACHE_VERSION = 1
    # Store the version # in the Version attribute of the top-level group
    VERSION_ATTRIBUTE = "Version"
    # The table that holds the image set rows: an HDFCSV
    IMAGE_SET_TABLE = "ImageSetTable"
    # An Nx2 dataset giving the row and column of errors in the image set table
    ERROR_ROW_AND_COLUMN_DATASET = "ErrorRowAndColumn"
    # A descriptive error message for each error in ErrorRowAndColumn
    ERROR_MESSAGES = "ErrorMessages"
    # All metadata key names - an attribute on the image set cache group
    METADATA_KEYS = "MetadataKeys"
    # All image names - an attribute on the image set cache group
    IMAGE_NAMES = "ImageNames"
    # For each image name, either IMAGE or OBJECT depending on whether
    # the name is a name of images or objects
    IMAGE_OR_OBJECT = "ImageOrObject"
    # The row index field in the errors dataset
    ROW_INDEX = "RowIndex"
    # The index of the image name in the ImageNames attribute or the index
    # of the object name in the ObjectNames attribute
    IO_INDEX = "ImageOrObjectIndex"
    # The number of image sets in the cache
    IMAGE_SET_COUNT = "ImageSetCount"

    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        assert isinstance(hdf5_file, h5py.Group)
        if self.IMAGE_SET_CACHE_GROUP in hdf5_file:
            self.image_set_cache_group = hdf5_file[self.IMAGE_SET_CACHE_GROUP]
            version = self.image_set_cache_group.attrs[self.VERSION_ATTRIBUTE]
            if version != self.IMAGE_SET_CACHE_VERSION:
                # It's a cache... sorry, if version is not correct, destroy
                del self.hdf5_file[self.IMAGE_SET_CACHE_GROUP]
                self.__has_cache = False
            else:
                self.__has_cache = True
        else:
            self.__has_cache = False
        if self.__has_cache:
            self.image_set_count = \
                self.image_set_cache_group.attrs[self.IMAGE_SET_COUNT]
            self.metadata_keys = self.load_strings_attr(self.METADATA_KEYS)
            if len(self.metadata_keys) == 0:
                self.metadata_keys = None
            self.image_names = self.load_strings_attr(self.IMAGE_NAMES)
            self.image_or_object = self.load_strings_attr(self.IMAGE_OR_OBJECT)
            self.image_set_table = HDFCSV(self.image_set_cache_group,
                                          self.IMAGE_SET_TABLE)
            self.error_row_and_column_dataset = \
                self.image_set_cache_group[self.ERROR_ROW_AND_COLUMN_DATASET]
            self.error_messages = \
                VStringArray(self.image_set_cache_group[self.ERROR_MESSAGES])

    def store_strings_attr(self, name, strings):
        '''Store a list of unicode strings in an attribute of the top group
        
        name - name of the attribute
        
        strings - a sequence of strings or unicode strings
        '''
        if len(strings) > 0:
            strings = [unicode(x).encode("utf-8") for x in strings]
            self.image_set_cache_group.attrs.create(name, strings)
        elif name in self.image_set_cache_group.attrs:
            del self.image_set_cache_group.attrs[name]

    def load_strings_attr(self, name):
        '''Load a list of unicode strings from an attribute in the top group
        
        name - name of attribute
        
        returns a sequence of strings if the attribute exists, otherwise
        an empty sequence
        '''
        return [x.decode("utf-8")
                for x in self.image_set_cache_group.attrs.get(name, [])]

    @property
    def has_cache(self):
        '''True if there are image sets in the cache'''
        return self.__has_cache

    class ImageSetData(object):
        '''Represents the information in one image set:
        
        key - the unique metadata values that distinguish this image set
        
        ipds - the image plane descriptors of the images in the set.
        
        errors - a sequence of errors, each of which is a two-tuple
                 of the index of the column causing the error and
                 the error message.
        '''

        def __init__(self, key, ipds, errors):
            self.key = key
            self.ipds = ipds
            self.errors = errors

    class ImageData(object):
        '''An alternative ducktype to IPDs'''

        def __init__(self, url, series, index, channel):
            self.url = url
            self.series = series
            self.index = index
            self.channel = channel

    def cache_image_set(self,
                        image_names,
                        image_set_data,
                        metadata_keys=None):
        '''Cache the current image set
        
        image_names - a sequence two tuples of the image or object name and
                      either IMAGE or OBJECT to distinguish between the
                      two uses.
        
        image_set_data - a sequence of ImageSetData objects
        
        metadata_keys - the names of the columns that provide the unique keys
                        for each image set. Index by order if None.
        '''
        if not self.__has_cache:
            self.image_set_cache_group = self.hdf5_file.require_group(
                self.IMAGE_SET_CACHE_GROUP)
            self.image_set_cache_group.attrs[self.VERSION_ATTRIBUTE] = \
                self.IMAGE_SET_CACHE_VERSION
        self.image_names = [n for n, io in image_names]
        self.image_or_object = [io for n, io in image_names]
        self.store_strings_attr(self.IMAGE_NAMES, self.image_names)
        self.store_strings_attr(self.IMAGE_OR_OBJECT, self.image_or_object)
        if metadata_keys is None:
            self.store_strings_attr(self.METADATA_KEYS, [])
            self.metadata_keys = None
        else:
            self.store_strings_attr(self.METADATA_KEYS, metadata_keys)
            self.metadata_keys = list(metadata_keys)
        self.image_set_table = HDFCSV(self.image_set_cache_group,
                                      self.IMAGE_SET_TABLE)
        if metadata_keys is not None:
            for i, name in enumerate(metadata_keys):
                self.image_set_table.add_column(
                    name,
                    [isd.key[i] for isd in image_set_data])
        for i, name in enumerate(self.image_names):
            self.image_set_table.add_column(
                C_URL + "_" + name,
                [isd.ipds[i].url for isd in image_set_data])
            for feature, values in (
                    (C_SERIES, [isd.ipds[i].series for isd in image_set_data]),
                    (C_FRAME, [isd.ipds[i].index for isd in image_set_data]),
                    (C_CHANNEL, [isd.ipds[i].channel for isd in image_set_data])):
                self.image_set_table.add_column(
                    feature + "_" + name,
                    [str(value) if value is not None else ""
                     for value in values])

        errors = sum(
            [[(i, idx, msg) for idx, msg in isd.errors]
             for i, isd in enumerate(image_set_data)], [])
        if self.ERROR_ROW_AND_COLUMN_DATASET in self.image_set_cache_group:
            self.error_row_and_column_dataset = \
                self.image_set_cache_group[self.ERROR_ROW_AND_COLUMN_DATASET]
            self.error_row_and_column_dataset, resize((len(errors),))
        else:
            self.error_row_and_column_dataset = \
                self.image_set_cache_group.create_dataset(
                    self.ERROR_ROW_AND_COLUMN_DATASET,
                    dtype=np.dtype([(self.ROW_INDEX, np.uint32, 1),
                                    (self.IO_INDEX, np.uint8, 1)]),
                    shape=(len(errors),),
                    chunks=(256,),
                    maxshape=(None,))
            self.error_messages = VStringArray(
                self.image_set_cache_group.require_group(self.ERROR_MESSAGES))

        for i, (image_set_row_number, index, msg) in enumerate(errors):
            self.error_row_and_column_dataset[i] = (
                image_set_row_number, index)
            self.error_messages[i] = msg
        self.image_set_count = len(image_set_data)
        self.image_set_cache_group.attrs[self.IMAGE_SET_COUNT] = \
            self.image_set_count
        self.__has_cache = True

    def get_error_rows(self):
        '''Return the indices of rows with errors
        
        Precondition: some image set must be cached.
        '''
        if self.error_row_and_column_dataset.shape[0] == 0:
            return np.zeros(0, int)
        return np.unique(
            self.error_row_and_column_dataset[self.ROW_INDEX])

    def get_errors(self, idx):
        '''Get errors in the idx'th row of a data set
        
        idx - the index of the data set that might have an error
        
        returns a list of two tuples of the form:
        image_index - index of the name of the image in self.image_names
        msg - descriptive error message
        '''
        if self.error_row_and_column_dataset.shape[0] == 0:
            return []
        errors = np.where(
            self.error_row_and_column_dataset[self.ROW_INDEX] == idx)
        idxs = self.error_row_and_column_dataset[self.IO_INDEX][:]
        return [(idxs[i],
                 self.error_messages[i])
                for i in errors]

    def get_image_set_data(self, idx):
        '''Get an ImageSetData item for the indexed image set'''
        errors = self.get_errors(idx)
        if self.metadata_keys is not None:
            key = tuple([self.image_set_table[k][idx]
                         for k in self.metadata_keys])
        else:
            key = []
        ipds = []
        for name in self.image_names:
            url = self.image_set_table[C_URL + "_" + name][idx]
            series, index, channel = [
                None if len(value) == 0 else int(value)
                for value in [self.image_set_table[ftr + "_" + name][idx]
                              for ftr in (C_SERIES, C_FRAME, C_CHANNEL)]]
            ipds.append(self.ImageData(url, series, index, channel))
        return self.ImageSetData(key, ipds, errors)
