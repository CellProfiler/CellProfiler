class Segmentation(object):
    '''A segmentation of a space into labeled objects
    
    Supports overlapping objects and cacheing. Retrieval can be as a
    single plane (legacy), as multiple planes and as sparse ijv.
    '''
    SEGMENTED = "segmented"
    UNEDITED_SEGMENTED = "unedited segmented"
    SMALL_REMOVED_SEGMENTED = "small removed segmented"
    
    def __init__(self, dense=None, sparse=None, shape=None):
        '''Initialize the segmentation with either a dense or sparse labeling
        
        dense - a 6-D labeling with the first axis allowing for alternative
                labelings of the same hyper-voxel.
        sparse - the sparse labeling as a record array with axes from
                 cellprofiler.utilities.hdf_dict.HDF5ObjectSet
        shape - the 5-D shape of the imaging site if sparse.
        '''

        self.__dense = dense
        self.__sparse = sparse
        if shape is not None:
            self.__shape = shape
            self.__explicit_shape = True
        else:
            self.__shape = None
            self.__explicit_shape = False
        self.__cache = None
        if dense is not None:
            self.__indices = [np.unique(d) for d in dense]
            self.__indices = [
                idx[1:] if idx[0] == 0 else idx for idx in self.__indices]
    
    def cache(self, hdf5_object_set, objects_name, segmentation_name):
        '''Cache the segmentation in the given object set
        
        hdf5_object_set - an HDF5ObjectSet for moving objects out of memory
        objects_name - name to use to store the objects
        segmentation_name - name of this particular segmentation, for instance,
                            Segmentation.SEGMENTED for the user-visible
                            segmentation.
        '''
        if self.__cache is not None:
            return
        self.__objects_name = objects_name
        self.__segmentation_name = segmentation_name
        hdf5_object_set.clear(objects_name, segmentation_name)
        if self.__dense is not None:
            hdf5_object_set.set_dense(objects_name, segmentation_name,
                                      self.__dense)
        if self.__sparse is not None:
            hdf5_object_set.set_sparse(objects_name, segmentation_name,
                                       self.__sparse)
        self.__dense = None
        self.__sparse = None
        self.__cache = hdf5_object_set
        
    def get_shape(self):
        '''Get or estimate the shape of the segmentation matrix
        
        Order of precedence:
        Shape supplied in the constructor
        Shape of the dense representation
        maximum extent of the sparse representation + 1
        '''
        if self.__shape is not None:
            return self.__shape
        if self.has_dense():
            self.__shape = self.get_dense()[0].shape[1:]
        else:
            sparse = self.get_sparse()
            if len(sparse) == 0:
                self.__shape = (1, 1, 1, 1, 1)
            else:
                from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
                self.__shape = tuple(
                    [np.max(sparse[axis])+2 
                     if axis in sparse.dtype.fields.keys() else 1
                     for axis in HDF5ObjectSet.AXES])
        return self.__shape
    
    def set_shape(self, shape):
        '''Set the shape of the segmentation array
        
        shape - the 5D shape of the array
        
        This fixes the shape of the 5D array for sparse representations
        '''
        self.__shape = shape
        self.__explicit_shape = True
    
    shape = property(get_shape, set_shape)
    
    def has_dense(self):
        return self.__dense is not None or (
            self.__cache is not None and self.__cache.has_dense(
                self.__objects_name, self.__segmentation_name))
        
    def has_sparse(self):
        return self.__sparse is not None or (
            self.__cache is not None and self.__cache.has_sparse(
                self.__objects_name, self.__segmentation_name))
    
    def has_shape(self):
        if self.__explicit_shape:
            return True
        
        return self.has_dense()
    
    def get_sparse(self):
        '''Get the sparse representation of the segmentation
        
        returns a Numpy record array where every row represents
        the labeling of a pixel. The dtype record names are taken from
        HDF5ObjectSet.AXIS_[X,Y,Z,C,T] and AXIS_LABELS for the object
        numbers.
        '''
        if self.__sparse is not None:
            return self.__sparse
        if self.__cache is not None and self.__cache.has_sparse(
            self.__objects_name, self.__segmentation_name):
            return self.__cache.get_sparse(
                self.__objects_name, self.__segmentation_name)
        if not self.has_dense():
            raise ValueError(
                "Can't find object, \"%s\", segmentation, \"%s\"." %
                (self.__objects_name, self.__segmentation_name))
        return self.__convert_dense_to_sparse()
    
    sparse = property(get_sparse)
    
    def get_dense(self):
        '''Get the dense representation of the segmentation
        
        return the segmentation as a 6-D array and a sequence of arrays of the
        object numbers in each 5-D hyperplane of the segmentation. The first
        axis of the segmentation allows us to assign multiple labels to 
        individual pixels. Given a 5-D algorithm, the code typically iterates 
        over the first axis:
        
        for labels in self.get_dense():
            # do something
            
        The remaining axes are in the order, C, T, Z, Y and X
        '''
        if self.__dense is not None:
            return (self.__dense, self.__indices)
        if self.__cache is not None and self.__cache.has_dense(
            self.__objects_name, self.__segmentation_name):
            return (self.__cache.get_dense(
                self.__objects_name, self.__segmentation_name),
                    self.__indices)
        if not self.has_sparse():
            raise ValueError(
                "Can't find object, \"%s\", segmentation, \"%s\"." %
                (self.__objects_name, self.__segmentation_name))
        return self.__convert_sparse_to_dense()
        
    def __convert_dense_to_sparse(self):
        dense, indices = self.get_dense()
        from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
        axes = list(HDF5ObjectSet.AXES)
        axes, shape = [
            [a for a, s in zip(aa, self.shape) if s > 1]
            for aa in axes, self.shape]
        #
        # dense.shape[0] is the overlap-axis - it's usually 1
        # except if there are multiply-labeled pixels and overlapping
        # objects. When collecting the coords, we can discard this axis.
        #
        dense = dense.reshape([dense.shape[0]] + shape)
        coords = np.where(dense != 0)
        plane, coords = coords[0], coords[1:]
        if np.max(shape) < 2**16:
            coords_dtype = np.uint16
        else:
            coords_dtype = np.uint32
        if len(plane) > 0:
            labels = dense[tuple([plane]+list(coords))]
            max_label = np.max(indices)
            if max_label < 2**8:
                labels_dtype = np.uint8
            elif max_label < 2**16:
                labels_dtype = np.uint16
            else:
                labels_dtype = np.uint32
        else:
            labels = np.zeros(0, dense.dtype)
            labels_dtype = np.uint8
        dtype = [(axis, coords_dtype, 1) for axis in axes]
        dtype.append((HDF5ObjectSet.AXIS_LABELS, labels_dtype, 1))
        sparse = np.core.records.fromarrays(list(coords) + [labels], dtype=dtype)
        if self.__cache is not None:
            self.__cache.set_sparse(
                self.__objects_name, self.__segmentation_name, sparse)
        else:
            self.__sparse = sparse
        return sparse
    
    def __set_or_cache_dense(self, dense, indices = None):
        if self.__cache is not None:
            self.__cache.set_dense(
                self.__objects_name, self.__segmentation_name, dense)
        else:
            self.__dense = dense
        if indices is not None:
            self.__indices = indices
        else:
            self.__indices = [np.unique(d) for d in dense]
            self.__indices = [
                idx[1:] if idx[0] == 0 else idx for idx in self.__indices]
        return (dense, self.__indices)
    
    def __convert_sparse_to_dense(self):
        from cellprofiler.utilities.hdf5_dict import HDF5ObjectSet
        sparse = self.get_sparse()
        if len(sparse) == 0:
            return self.__set_or_cache_dense(
                np.zeros([1] + list(self.shape), np.uint16))

        #
        # The code below assigns a "color" to each label so that no
        # two labels have the same color
        #
        positional_columns = []
        available_columns = []
        lexsort_columns = []
        for axis in HDF5ObjectSet.AXES:
            if axis in sparse.dtype.fields.keys():
                positional_columns.append(sparse[axis])
                available_columns.append(sparse[axis])
                lexsort_columns.insert(0, sparse[axis])
            else:
                positional_columns.append(0)
        labels = sparse[HDF5ObjectSet.AXIS_LABELS]
        lexsort_columns.insert(0, labels)
        
        sort_order = np.lexsort(lexsort_columns)
        n_labels = np.max(labels)
        #
        # Find the first of a run that's different from the rest
        #
        mask = available_columns[0][sort_order[:-1]] != \
            available_columns[0][sort_order[1:]]
        for column in available_columns[1:]:
            mask = mask | (column[sort_order[:-1]] !=
                           column[sort_order[1:]])
        breaks = np.hstack(([0], np.where(mask)[0]+1, [len(labels)]))
        firsts = breaks[:-1]
        counts = breaks[1:] - firsts
        indexer = Indexes(counts)
        #
        # Eliminate the locations that are singly labeled
        #
        mask = counts > 1
        firsts = firsts[mask]
        counts = counts[mask]
        if len(counts) == 0:
            dense = np.zeros([1]+list(self.shape), labels.dtype)
            dense[[0] + positional_columns] = labels
            return self.__set_or_cache_dense(dense)
        #
        # There are n * n-1 pairs for each coordinate (n = # labels)
        # n = 1 -> 0 pairs, n = 2 -> 2 pairs, n = 3 -> 6 pairs
        #
        pairs = all_pairs(np.max(counts))
        pair_counts = counts * (counts - 1)
        #
        # Create an indexer for the inputs (indexes) and for the outputs
        # (first and second of the pairs)
        #
        # Remember idx points into sort_order which points into labels
        # to get the nth label, grouped into consecutive positions.
        #
        input_indexer = Indexes(counts)
        output_indexer = Indexes(pair_counts)
        #
        # The start of the run of overlaps and the offsets
        #
        run_starts = firsts[output_indexer.rev_idx]
        offs = pairs[output_indexer.idx[0], :]
        first = labels[sort_order[run_starts + offs[:, 0]]]
        second = labels[sort_order[run_starts + offs[:, 1]]]
        #
        # And sort these so that we get consecutive lists for each
        #
        pair_sort_order = np.lexsort((second, first))
        #
        # Eliminate dupes
        #
        to_keep = np.hstack(([True], 
                             (first[1:] != first[:-1]) |
                             (second[1:] != second[:-1])))
        to_keep = to_keep & (first != second)
        pair_idx = pair_sort_order[to_keep]
        first = first[pair_idx]
        second = second[pair_idx]
        #
        # Bincount each label so we can find the ones that have the
        # most overlap. See cpmorphology.color_labels and
        # Welsh, "An upper bound for the chromatic number of a graph and
        # its application to timetabling problems", The Computer Journal, 10(1)
        # p 85 (1967)
        #
        overlap_counts = np.bincount(first.astype(np.int32))
        #
        # The index to the i'th label's stuff
        #
        indexes = np.cumsum(overlap_counts) - overlap_counts
        #
        # A vector of a current color per label. All non-overlapping
        # objects are assigned to plane 1
        #
        v_color = np.ones(n_labels+1, int)
        v_color[0] = 0
        #
        # Clear all overlapping objects
        #
        v_color[np.unique(first)] = 0
        #
        # The processing order is from most overlapping to least
        #
        ol_labels = np.where(overlap_counts > 0)[0]
        processing_order = np.lexsort((ol_labels, overlap_counts[ol_labels]))
        
        for index in ol_labels[processing_order]:
            neighbors = second[
                indexes[index]:indexes[index] + overlap_counts[index]]
            colors = np.unique(v_color[neighbors])
            if colors[0] == 0:
                if len(colors) == 1:
                    # all unassigned - put self in group 1
                    v_color[index] = 1
                    continue
                else:
                    # otherwise, ignore the unprocessed group and continue
                    colors = colors[1:]
            # Match a range against the colors array - the first place
            # they don't match is the first color we can use
            crange = np.arange(1, len(colors)+1)
            misses = crange[colors != crange]
            if len(misses):
                color = misses[0]
            else:
                max_color = len(colors) + 1
                color = max_color
            v_color[index] = color
        #
        # Create the dense matrix by using the color to address the
        # 5-d hyperplane into which we place each label
        #
        result = []
        dense = np.zeros([np.max(v_color)]+list(self.shape), labels.dtype)
        slices = tuple([v_color[labels]-1] + positional_columns)
        dense[slices] = labels
        indices = [
            np.where(v_color == i)[0] for i in range(1, dense.shape[0]+1)]
        
        return self.__set_or_cache_dense(dense, indices)
