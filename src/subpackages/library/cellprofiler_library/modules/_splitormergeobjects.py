from cellprofiler_library.opts.splitormergeobjects import RelabelOption, MergeOption, MergingMethod, C_PARENT, ObjectIntensityMethod
import numpy
import scipy
import centrosome.cpmorphology

def split_or_merge_objects(
        labels, 
        relabel_option, 
        merge_option, 
        distance_threshold, 
        objects_name, 
        parent_name, 
        relaitonship_measurement, 
        merge_using_image, 
        merging_method, 
        image, 
        where_algorithm,
        minimum_intensity_fraction
    ):
    if relabel_option == RelabelOption.SPLIT:
        output_labels, count = scipy.ndimage.label(
            labels > 0, numpy.ones((3, 3), bool)
        )
    else:
        if merge_option == MergeOption.UNIFY_DISTANCE:
            mask = labels > 0
            if distance_threshold > 0:
                #
                # Take the distance transform of the reverse of the mask
                # and figure out what points are less than 1/2 of the
                # distance from an object.
                #
                d = scipy.ndimage.distance_transform_edt(~mask)
                mask = d < distance_threshold / 2 + 1
            output_labels, count = scipy.ndimage.label(
                mask, numpy.ones((3, 3), bool)
            )
            output_labels[labels == 0] = 0
            if merge_using_image:
                output_labels = filter_using_image(labels, image, mask, where_algorithm, minimum_intensity_fraction)
        elif merge_option == MergeOption.UNIFY_PARENT:
            parents_of = relaitonship_measurement[
                objects_name, "_".join((C_PARENT, parent_name))
            ]
            output_labels = labels.copy().astype(numpy.uint32)
            output_labels[labels > 0] = parents_of[labels[labels > 0] - 1]
            if merging_method == MergingMethod.CONVEX_HULL:
                ch_pts, n_pts = centrosome.cpmorphology.convex_hull(output_labels)
                ijv = centrosome.cpmorphology.fill_convex_hulls(ch_pts, n_pts)
                output_labels[ijv[:, 0], ijv[:, 1]] = ijv[:, 2]

            #Renumber to be consecutive
            ## Create an array that maps label indexes to their new values
            ## All labels to be deleted have a value in this array of zero
            indexes = numpy.unique(output_labels)[1:]
            new_object_count = len(indexes)
            max_label = numpy.max(output_labels)
            label_indexes = numpy.zeros((max_label + 1,), int)
            label_indexes[indexes] = numpy.arange(1, new_object_count + 1)

            # Reindex the labels of the old source image
            output_labels = label_indexes[output_labels]
        else: 
            raise NotImplementedError(f"Unimplemented merging method: {merging_method}")
    return output_labels

def filter_using_image(labels, image, mask, where_algorithm, minimum_intensity_fraction):
    """Filter out connections using local intensity minima between objects

    workspace - the workspace for the image set
    mask - mask of background points within the minimum distance
    """
    #
    # NOTE: This is an efficient implementation and an improvement
    #       in accuracy over the Matlab version. It would be faster and
    #       more accurate to eliminate the line-connecting and instead
    #       do the following:
    #     * Distance transform to get the coordinates of the closest
    #       point in an object for points in the background that are
    #       at most 1/2 of the max distance between objects.
    #     * Take the intensity at this closest point and similarly
    #       label the background point if the background intensity
    #       is at least the minimum intensity fraction
    #     * Assume there is a connection between objects if, after this
    #       labeling, there are adjacent points in each object.
    #
    # As it is, the algorithm duplicates the Matlab version but suffers
    # for cells whose intensity isn't high in the centroid and clearly
    # suffers when two cells touch at some point that's off of the line
    # between the two.
    #
    


    #
    # Do a distance transform into the background to label points
    # in the background with their closest foreground object
    #
    i, j = scipy.ndimage.distance_transform_edt(
        labels == 0, return_indices=True, return_distances=False
    )
    confluent_labels = labels[i, j]
    confluent_labels[~mask] = 0
    if where_algorithm == ObjectIntensityMethod.CLOSEST_POINT.value:
        #
        # For the closest point method, find the intensity at
        # the closest point in the object (which will be the point itself
        # for points in the object).
        #
        object_intensity = image[i, j] * minimum_intensity_fraction
        confluent_labels[object_intensity > image] = 0
    count, index, c_j = centrosome.cpmorphology.find_neighbors(confluent_labels)
    if len(c_j) == 0:
        # Nobody touches - return the labels matrix
        return labels
    #
    # Make a row of i matching the touching j
    #
    c_i = numpy.zeros(len(c_j))
    #
    # Eliminate labels without matches
    #
    label_numbers = numpy.arange(1, len(count) + 1)[count > 0]
    index = index[count > 0]
    count = count[count > 0]
    #
    # Get the differences between labels so we can use a cumsum trick
    # to increment to the next label when they change
    #
    label_numbers[1:] = label_numbers[1:] - label_numbers[:-1]
    c_i[index] = label_numbers
    c_i = numpy.cumsum(c_i).astype(int)
    if where_algorithm == ObjectIntensityMethod.CENTROIDS.value:
        #
        # Only connect points > minimum intensity fraction
        #
        center_i, center_j = centrosome.cpmorphology.centers_of_labels(labels)
        indexes, counts, i, j = centrosome.cpmorphology.get_line_pts(
            center_i[c_i - 1],
            center_j[c_i - 1],
            center_i[c_j - 1],
            center_j[c_j - 1],
        )
        #
        # The indexes of the centroids at pt1
        #
        last_indexes = indexes + counts - 1
        #
        # The minimum of the intensities at pt0 and pt1
        #
        centroid_intensities = numpy.minimum(
            image[i[indexes], j[indexes]], image[i[last_indexes], j[last_indexes]]
        )
        #
        # Assign label numbers to each point so we can use
        # scipy.ndimage.minimum. The label numbers are indexes into
        # "connections" above.
        #
        pt_labels = numpy.zeros(len(i), int)
        pt_labels[indexes[1:]] = 1
        pt_labels = numpy.cumsum(pt_labels)
        minima = scipy.ndimage.minimum(
            image[i, j], pt_labels, numpy.arange(len(indexes))
        )
        minima = centrosome.cpmorphology.fixup_scipy_ndimage_result(minima)
        #
        # Filter the connections using the image
        #
        mif = minimum_intensity_fraction
        i = c_i[centroid_intensities * mif <= minima]
        j = c_j[centroid_intensities * mif <= minima]
    else:
        i = c_i
        j = c_j
    #
    # Add in connections from self to self
    #
    unique_labels = numpy.unique(labels)
    i = numpy.hstack((i, unique_labels))
    j = numpy.hstack((j, unique_labels))
    #
    # Run "all_connected_components" to get a component # for
    # objects identified as same.
    #
    new_indexes = centrosome.cpmorphology.all_connected_components(i, j)
    new_labels = numpy.zeros(labels.shape, int)
    new_labels[labels != 0] = new_indexes[labels[labels != 0]]
    return new_labels