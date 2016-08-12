import os
import urllib2

import cellprofiler.preferences
import centrosome.cpmorphology
import centrosome.propagate
import numpy
import scipy.interpolate
import scipy.io
import scipy.ndimage
import scipy.sparse

OO_WITH_OVERLAP = "With overlap"
OO_WITHOUT_OVERLAP = "Without overlap"
OO_BOTH = "Both"
MODE_TRAIN = "Train"
MODE_UNTANGLE = "Untangle"
SCM_ANGLE_SHAPE_MODEL = 'angle_shape_model'
MAX_CONSIDERED = 50000
MAX_PATHS = 400
TRAINING_DATA = "TrainingData"
ATTR_WORM_MEASUREMENTS = "WormMeasurements"
C_WORM = "Worm"
F_LENGTH = "Length"
F_ANGLE = "Angle"
F_CONTROL_POINT_X = "ControlPointX"
F_CONTROL_POINT_Y = "ControlPointY"
T_NAMESPACE = "http://www.cellprofiler.org/linked_files/schemas/UntangleWorms.xsd"
T_TRAINING_DATA = "training-data"
T_VERSION = "version"
T_MIN_AREA = "min-area"
T_MAX_AREA = "max-area"
T_COST_THRESHOLD = "cost-threshold"
T_NUM_CONTROL_POINTS = "num-control-points"
T_MEAN_ANGLES = "mean-angles"
T_INV_ANGLES_COVARIANCE_MATRIX = "inv-angles-covariance-matrix"
T_MAX_SKEL_LENGTH = "max-skel-length"
T_MAX_RADIUS = "max-radius"
T_MIN_PATH_LENGTH = "min-path-length"
T_MAX_PATH_LENGTH = "max-path-length"
T_MEDIAN_WORM_AREA = "median-worm-area"
T_OVERLAP_WEIGHT = "overlap-weight"
T_LEFTOVER_WEIGHT = "leftover-weight"
T_RADII_FROM_TRAINING = "radii-from-training"
T_TRAINING_SET_SIZE = "training-set-size"
T_VALUES = "values"
T_VALUE = "value"
C_ALL = "Process all clusters"
C_ALL_VALUE = numpy.iinfo(int).max
C_MEDIUM = "Medium"
C_MEDIUM_VALUE = 200
C_HIGH = "High"
C_HIGH_VALUE = 600
C_VERY_HIGH = "Very high"
C_VERY_HIGH_VALUE = 1000
C_CUSTOM = "Custom"
complexity_limits = {
    C_ALL: C_ALL_VALUE,
    C_MEDIUM: C_MEDIUM_VALUE,
    C_HIGH: C_HIGH_VALUE,
    C_VERY_HIGH: C_VERY_HIGH_VALUE
}


def read_params(training_set_directory, training_set_file_name, d):
    """Read a training set parameters  file

    training_set_directory - the training set directory setting

    training_set_file_name - the training set file name setting

    d - a dictionary that stores cached parameters
    """

    #
    # The parameters file is a .xml file with the following structure:
    #
    # initial_filter
    #     min_worm_area: float
    # single_worm_determination
    #     max_area: float
    # single_worm_find_path
    #     method: string (=? "dfs_longest_path")
    # single_worm_filter
    #     method: string (=? "angle_shape_cost")
    #     cost_threshold: float
    #     num_control_points: int
    #     mean_angles: float vector (num_control_points -1 entries)
    #     inv_angles_covariance_matrix: float matrix (num_control_points -1)**2
    # cluster_graph_building
    #     method: "large_branch_area_max_skel_length"
    #     max_radius: float
    #     max_skel_length: float
    # cluster_paths_finding
    #     method: string "dfs"
    # cluster_paths_selection
    #     shape_cost_method: "angle_shape_model"
    #     selection_method: "dfs_prune"
    #     overlap_leftover_method: "skeleton_length"
    #     min_path_length: float
    #     max_path_length: float
    #     median_worm__area: float
    #     worm_radius: float
    #     overlap_weight: int
    #     leftover_weight: int
    #     ---- the following are the same as for the single worm filter ---
    #     num_control_points: int
    #     mean_angles: float vector (num_control_points-1)
    #     inv_angles_covariance_matrix: (num_control_points-1)**2
    #     ----
    #     approx_max_search_n: int
    # worm_descriptor_building
    #     method: string = "default"
    #     radii_from_training: vector ?of length num_control_points?
    #
    class X(object):
        """This "class" is used as a vehicle for arbitrary dot notation

        For instance:
        > x = X()
        > x.foo = 1
        > x.foo
        1
        """
        pass

    path = training_set_directory.get_absolute_path()
    file_name = training_set_file_name.value
    if d.has_key(file_name):
        result, timestamp = d[file_name]
        if (timestamp == "URL" or
                    timestamp == os.stat(os.path.join(path, file_name)).st_mtime):
            return d[file_name][0]

    if training_set_directory.dir_choice == cellprofiler.preferences.URL_FOLDER_NAME:
        url = file_name
        fd_or_file = urllib2.urlopen(url)
        is_url = True
        timestamp = "URL"
    else:
        fd_or_file = os.path.join(path, file_name)
        is_url = False
        timestamp = os.stat(fd_or_file).st_mtime
    try:
        from xml.dom.minidom import parse
        doc = parse(fd_or_file)
        result = X()

        def f(tag, attribute, klass):
            elements = doc.documentElement.getElementsByTagName(tag)
            assert len(elements) == 1
            element = elements[0]
            text = "".join([text.data for text in element.childNodes
                            if text.nodeType == doc.TEXT_NODE])
            setattr(result, attribute, klass(text.strip()))

        for tag, attribute, klass in (
                (T_VERSION, "version", int),
                (T_MIN_AREA, "min_worm_area", float),
                (T_MAX_AREA, "max_area", float),
                (T_COST_THRESHOLD, "cost_threshold", float),
                (T_NUM_CONTROL_POINTS, "num_control_points", int),
                (T_MAX_RADIUS, "max_radius", float),
                (T_MAX_SKEL_LENGTH, "max_skel_length", float),
                (T_MIN_PATH_LENGTH, "min_path_length", float),
                (T_MAX_PATH_LENGTH, "max_path_length", float),
                (T_MEDIAN_WORM_AREA, "median_worm_area", float),
                (T_OVERLAP_WEIGHT, "overlap_weight", float),
                (T_LEFTOVER_WEIGHT, "leftover_weight", float)):
            f(tag, attribute, klass)
        elements = doc.documentElement.getElementsByTagName(T_MEAN_ANGLES)
        assert len(elements) == 1
        element = elements[0]
        result.mean_angles = numpy.zeros(result.num_control_points - 1)
        for index, value_element in enumerate(
                element.getElementsByTagName(T_VALUE)):
            text = "".join([text.data for text in value_element.childNodes
                            if text.nodeType == doc.TEXT_NODE])
            result.mean_angles[index] = float(text.strip())
        elements = doc.documentElement.getElementsByTagName(T_RADII_FROM_TRAINING)
        assert len(elements) == 1
        element = elements[0]
        result.radii_from_training = numpy.zeros(result.num_control_points)
        for index, value_element in enumerate(
                element.getElementsByTagName(T_VALUE)):
            text = "".join([text.data for text in value_element.childNodes
                            if text.nodeType == doc.TEXT_NODE])
            result.radii_from_training[index] = float(text.strip())
        result.inv_angles_covariance_matrix = numpy.zeros(
                [result.num_control_points - 1] * 2)
        elements = doc.documentElement.getElementsByTagName(T_INV_ANGLES_COVARIANCE_MATRIX)
        assert len(elements) == 1
        element = elements[0]
        for i, values_element in enumerate(
                element.getElementsByTagName(T_VALUES)):
            for j, value_element in enumerate(
                    values_element.getElementsByTagName(T_VALUE)):
                text = "".join([text.data for text in value_element.childNodes
                                if text.nodeType == doc.TEXT_NODE])
                result.inv_angles_covariance_matrix[i, j] = float(text.strip())
    except:
        if is_url:
            fd_or_file = urllib2.urlopen(url)

        mat_params = scipy.io.loadmat(fd_or_file)["params"][0, 0]
        field_names = mat_params.dtype.fields.keys()

        result = X()

        CLUSTER_PATHS_SELECTION = 'cluster_paths_selection'
        CLUSTER_GRAPH_BUILDING = 'cluster_graph_building'
        SINGLE_WORM_FILTER = 'single_worm_filter'
        INITIAL_FILTER = 'initial_filter'
        SINGLE_WORM_DETERMINATION = 'single_worm_determination'
        CLUSTER_PATHS_FINDING = 'cluster_paths_finding'
        WORM_DESCRIPTOR_BUILDING = 'worm_descriptor_building'
        SINGLE_WORM_FIND_PATH = 'single_worm_find_path'
        METHOD = "method"

        STRING = "string"
        SCALAR = "scalar"
        VECTOR = "vector"
        MATRIX = "matrix"

        def mp(*args, **kwargs):
            """Look up a field from mat_params"""
            x = mat_params
            for arg in args[:-1]:
                x = x[arg][0, 0]
            x = x[args[-1]]
            kind = kwargs.get("kind", SCALAR)
            if kind == SCALAR:
                return x[0, 0]
            elif kind == STRING:
                return x[0]
            elif kind == VECTOR:
                # Work-around for OS/X Numpy bug
                # Copy a possibly mis-aligned buffer
                b = numpy.array([v for v in numpy.frombuffer(x.data, numpy.uint8)], numpy.uint8)
                return numpy.frombuffer(b, x.dtype)
            return x

        result.min_worm_area = mp(INITIAL_FILTER, "min_worm_area")
        result.max_area = mp(SINGLE_WORM_DETERMINATION, "max_area")
        result.cost_threshold = mp(SINGLE_WORM_FILTER, "cost_threshold")
        result.num_control_points = mp(SINGLE_WORM_FILTER, "num_control_points")
        result.mean_angles = mp(SINGLE_WORM_FILTER, "mean_angles", kind=VECTOR)
        result.inv_angles_covariance_matrix = mp(
                SINGLE_WORM_FILTER, "inv_angles_covariance_matrix", kind=MATRIX)
        result.max_radius = mp(CLUSTER_GRAPH_BUILDING,
                               "max_radius")
        result.max_skel_length = mp(CLUSTER_GRAPH_BUILDING,
                                    "max_skel_length")
        result.min_path_length = mp(
                CLUSTER_PATHS_SELECTION, "min_path_length")
        result.max_path_length = mp(
                CLUSTER_PATHS_SELECTION, "max_path_length")
        result.median_worm_area = mp(
                CLUSTER_PATHS_SELECTION, "median_worm_area")
        result.worm_radius = mp(
                CLUSTER_PATHS_SELECTION, "worm_radius")
        result.overlap_weight = mp(
                CLUSTER_PATHS_SELECTION, "overlap_weight")
        result.leftover_weight = mp(
                CLUSTER_PATHS_SELECTION, "leftover_weight")
        result.radii_from_training = mp(
                WORM_DESCRIPTOR_BUILDING, "radii_from_training", kind=VECTOR)
    d[file_name] = (result, timestamp)
    return result


def recalculate_single_worm_control_points(all_labels, ncontrolpoints):
    """Recalculate the control points for labeled single worms

    Given a labeling of single worms, recalculate the control points
    for those worms.

    all_labels - a sequence of label matrices

    ncontrolpoints - the # of desired control points

    returns a two tuple:

    an N x M x 2 array where the first index is the object number,
    the second index is the control point number and the third index is 0
    for the Y or I coordinate of the control point and 1 for the X or J
    coordinate.

    a vector of N lengths.
    """

    all_object_numbers = [
        filter((lambda n: n > 0), numpy.unique(l)) for l in all_labels]
    if all([len(object_numbers) == 0 for object_numbers in all_object_numbers]):
        return numpy.zeros((0, ncontrolpoints, 2), int), numpy.zeros(0, int)

    nobjects = numpy.max(numpy.hstack(all_object_numbers))
    result = numpy.ones((nobjects, ncontrolpoints, 2)) * numpy.nan
    lengths = numpy.zeros(nobjects)
    for object_numbers, labels in zip(all_object_numbers, all_labels):
        for object_number in object_numbers:
            mask = (labels == object_number)
            skeleton = centrosome.cpmorphology.skeletonize(mask)
            graph = get_graph_from_binary(mask, skeleton)
            path_coords, path = get_longest_path_coords(graph, numpy.iinfo(int).max)
            if len(path_coords) == 0:
                # return NaN for the control points
                continue
            cumul_lengths = calculate_cumulative_lengths(path_coords)
            if cumul_lengths[-1] == 0:
                continue
            control_points = sample_control_points(path_coords, cumul_lengths, ncontrolpoints)
            result[(object_number - 1), :, :] = control_points
            lengths[object_number - 1] = cumul_lengths[-1]
    return result, lengths


def get_graph_from_binary(binary_im, skeleton, max_radius=None, max_skel_length=None):
    """Manufacture a graph of the skeleton of the worm

    Given a binary image containing a cluster of worms, returns a structure
    describing the graph structure of the skeleton of the cluster. This graph
    structure can later be used as input to e.g. get_all_paths().

    Input parameters:

    binary_im: A logical image, containing the cluster to be resolved. Must
    contain exactly one connected component.

    Output_parameters:

    graph_struct: An object with attributes

    image_size: Equal to size(binary_im).

    segments: A list describing the segments of
    the skeleton. Each element is an array of i,j coordinates
    of the pixels making up one segment, traced in the right order.

    branch_areas: A list describing the
    branch areas, i.e. the areas where different segments join. Each
    element is an array of i,j coordinates
    of the pixels making up one branch area, in no particular order.
    The branch areas will include all branchpoints,
    followed by a dilation. If max_radius is supplied, all pixels remaining
    after opening the binary image consisting of all pixels further
    than max_pix from the image background. This allows skeleton pixels
    in thick regions to be replaced by branchppoint regions, which increases
    the chance of connecting skeleton pieces correctly.

    incidence_matrix: A num_branch_areas x num_segments logical array,
    describing the incidence relations among the branch areas and
    segments. incidence_matrix(i, j) is set if and only if branch area
    i connects to segment j.

    incidence_directions: A num_branch_areas x num_segments logical
    array, intended to indicate the directions in which the segments
    are traced. incidence_directions(i,j) is set if and only if the
    "start end" (as in the direction in which the pixels are enumerated
    in graph_struct.segments) of segment j is connected to branch point
    i.

    Notes:

    1. Because of a dilatation step in obtaining them, the branch areas need
       not be (in fact, are never, unless binary_im contains all pixels)
       a subset of the foreground pixels of binary_im. However, they are a
       subset of the ones(3,3)-dilatation of binary_im.

    2. The segments are not considered to actually enter the branch areas;
       that is to say, the pixel set of the branch areas is disjoint from
       that of the segments.

    3. Even if one segment is only one pixel long (but still connects to
       two branch areas), its orientation is well-defined, i.e. one branch
       area will be chosen as starting end. (Even though in this case, the
       "positive direction" of the segment cannot be determined from the
       information in graph_struct.segments.)"""
    branch_areas_binary = centrosome.cpmorphology.branchpoints(skeleton)

    if max_radius is not None:
        #
        # Add any points that are more than the worm diameter to
        # the branchpoints. Exclude segments without supporting branchpoints:
        #
        # OK:
        #
        # * * *       * * *
        #       * * *
        # * * *       * * *
        #
        # Not OK:
        #
        # * * * * * * * * * *
        #
        strel = centrosome.cpmorphology.strel_disk(max_radius)
        far = scipy.ndimage.binary_erosion(binary_im, strel)
        far = scipy.ndimage.binary_opening(far, structure=centrosome.cpmorphology.eight_connect)
        far_labels, count = scipy.ndimage.label(far)
        far_counts = numpy.bincount(far_labels.ravel(), branch_areas_binary.ravel())
        far[far_counts[far_labels] < 2] = False
        branch_areas_binary |= far

        del far
        del far_labels

    branch_areas_binary = scipy.ndimage.binary_dilation(branch_areas_binary, structure=centrosome.cpmorphology.eight_connect)

    segments_binary = skeleton & ~ branch_areas_binary

    if max_skel_length is not None and numpy.sum(segments_binary) > 0:
        max_skel_length = max(int(max_skel_length), 2)  # paranoia
        i, j, labels, order, distance, num_segments = trace_segments(segments_binary)
        #
        # Put breakpoints every max_skel_length, but not at end
        #
        max_order = numpy.array(scipy.ndimage.maximum(order, labels, numpy.arange(num_segments + 1)))
        big_segment = max_order >= max_skel_length
        segment_count = numpy.maximum((max_order + max_skel_length - 1) / max_skel_length, 1).astype(int)
        segment_length = ((max_order + 1) / segment_count).astype(int)
        new_bp_mask = ((order % segment_length[labels] == segment_length[labels] - 1) & (order != max_order[labels]) & (big_segment[labels]))
        new_branch_areas_binary = numpy.zeros(segments_binary.shape, bool)
        new_branch_areas_binary[i[new_bp_mask], j[new_bp_mask]] = True
        new_branch_areas_binary = scipy.ndimage.binary_dilation(new_branch_areas_binary, structure=centrosome.cpmorphology.eight_connect)
        branch_areas_binary |= new_branch_areas_binary
        segments_binary &= ~new_branch_areas_binary

    return get_graph_from_branching_areas_and_segments(branch_areas_binary, segments_binary)


def get_graph_from_branching_areas_and_segments(branch_areas_binary, segments_binary):
    """Turn branches + segments into a graph

    branch_areas_binary - binary mask of branch areas

    segments_binary - binary mask of segments != branch_areas

    Given two binary images, one containing "branch areas" one containing
    "segments", returns a structure describing the incidence relations
    between the branch areas and the segments.

    Output is same format as get_graph_from_binary(), so for details, see
    get_graph_from_binary
    """
    branch_areas_labeled, num_branch_areas = scipy.ndimage.label(
        branch_areas_binary, centrosome.cpmorphology.eight_connect)

    i, j, labels, order, distance, num_segments = trace_segments(segments_binary)

    ooo = numpy.lexsort((order, labels))
    i = i[ooo]
    j = j[ooo]
    labels = labels[ooo]
    order = order[ooo]
    distance = distance[ooo]
    counts = (numpy.zeros(0, int) if len(labels) == 0
              else numpy.bincount(labels.flatten())[1:])

    branch_ij = numpy.argwhere(branch_areas_binary)
    if len(branch_ij) > 0:
        ooo = numpy.lexsort([
            branch_ij[:, 0], branch_ij[:, 1],
            branch_areas_labeled[branch_ij[:, 0], branch_ij[:, 1]]])
        branch_ij = branch_ij[ooo]
        branch_labels = branch_areas_labeled[branch_ij[:, 0], branch_ij[:, 1]]
        branch_counts = numpy.bincount(branch_areas_labeled.flatten())[1:]
    else:
        branch_labels = numpy.zeros(0, int)
        branch_counts = numpy.zeros(0, int)
    #
    # "find" the segment starts
    #
    starts = order == 0
    start_labels = numpy.zeros(segments_binary.shape, int)
    start_labels[i[starts], j[starts]] = labels[starts]
    #
    # incidence_directions = True for starts
    #
    incidence_directions = make_incidence_matrix(branch_areas_labeled, num_branch_areas, start_labels, num_segments)
    #
    # Get the incidence matrix for the ends
    #
    ends = numpy.cumsum(counts) - 1
    end_labels = numpy.zeros(segments_binary.shape, int)
    end_labels[i[ends], j[ends]] = labels[ends]
    incidence_matrix = make_incidence_matrix(branch_areas_labeled, num_branch_areas, end_labels, num_segments)
    incidence_matrix |= incidence_directions

    class Result(object):
        """A result graph:

        image_size: size of input image

        segments: a list for each segment of a forward (index = 0) and
                  reverse N x 2 array of coordinates of pixels in a segment

        segment_indexes: the index of label X into segments

        segment_counts: # of points per segment

        segment_order: for each pixel, its order when tracing

        branch_areas: an N x 2 array of branch point coordinates

        branch_area_indexes: index into the branch areas per branchpoint

        branch_area_counts: # of points in each branch

        incidence_matrix: matrix of areas x segments indicating connections

        incidence_directions: direction of each connection
        """

        def __init__(self, branch_areas_binary, counts, i, j,
                     branch_ij, branch_counts, incidence_matrix,
                     incidence_directions):
            self.image_size = tuple(branch_areas_binary.shape)
            self.segment_coords = numpy.column_stack((i, j))
            self.segment_indexes = numpy.cumsum(counts) - counts
            self.segment_counts = counts
            self.segment_order = order
            self.segments = [
                (self.segment_coords[self.segment_indexes[i]:
                (self.segment_indexes[i] +
                 self.segment_counts[i])],
                 self.segment_coords[self.segment_indexes[i]:
                 (self.segment_indexes[i] +
                  self.segment_counts[i])][::-1])
                for i in range(len(counts))]

            self.branch_areas = branch_ij
            self.branch_area_indexes = numpy.cumsum(branch_counts) - branch_counts
            self.branch_area_counts = branch_counts
            self.incidence_matrix = incidence_matrix
            self.incidence_directions = incidence_directions

    return Result(branch_areas_binary, counts, i, j, branch_ij, branch_counts, incidence_matrix, incidence_directions)


def trace_segments(segments_binary):
    """Find distance of every point in a segment from a segment endpoint

    segments_binary - a binary mask of the segments in an image.

    returns a tuple of the following:
    i - the i coordinate of a point in the mask
    j - the j coordinate of a point in the mask
    label - the segment's label
    order - the ordering (from 0 to N-1 where N is the # of points in
            the segment.)
    distance - the propagation distance of the point from the endpoint
    num_segments - the # of labelled segments
    """
    #
    # Break long skeletons into pieces whose maximum length
    # is max_skel_length.
    #
    segments_labeled, num_segments = scipy.ndimage.label(segments_binary, structure=centrosome.cpmorphology.eight_connect)

    if num_segments == 0:
        return (numpy.array([], int), numpy.array([], int), numpy.array([], int), numpy.array([], int), numpy.array([]), 0)

    #
    # Get one endpoint per segment
    #
    endpoints = centrosome.cpmorphology.endpoints(segments_binary)
    #
    # Use a consistent order: pick with lowest i, then j.
    # If a segment loops upon itself, we pick an arbitrary point.
    #
    order = numpy.arange(numpy.prod(segments_binary.shape))
    order.shape = segments_binary.shape
    order[~ endpoints] += numpy.prod(segments_binary.shape)
    labelrange = numpy.arange(num_segments + 1).astype(int)
    endpoint_loc = scipy.ndimage.minimum_position(order, segments_labeled, labelrange)
    endpoint_loc = numpy.array(endpoint_loc, int)
    endpoint_labels = numpy.zeros(segments_labeled.shape, numpy.int16)
    endpoint_labels[endpoint_loc[:, 0], endpoint_loc[:, 1]] = segments_labeled[endpoint_loc[:, 0], endpoint_loc[:, 1]]
    #
    # A corner case - propagate will trace a loop around both ways. So
    # we have to find that last point and remove it so
    # it won't trace in that direction
    #
    loops = ~ endpoints[endpoint_loc[1:, 0], endpoint_loc[1:, 1]]
    if numpy.any(loops):
        # Consider all points around the endpoint, finding the one
        # which is numbered last
        dilated_ep_labels = centrosome.cpmorphology.grey_dilation(
            endpoint_labels, footprint=numpy.ones((3, 3), bool))
        dilated_ep_labels[dilated_ep_labels != segments_labeled] = 0
        loop_endpoints = scipy.ndimage.maximum_position(
            order, dilated_ep_labels.astype(int), labelrange[1:][loops])
        loop_endpoints = numpy.array(loop_endpoints, int)
        segments_binary_temp = segments_binary.copy()
        segments_binary_temp[loop_endpoints[:, 0], loop_endpoints[:, 1]] = False
    else:
        segments_binary_temp = segments_binary
    #
    # Now propagate from the endpoints to get distances
    #
    _, distances = centrosome.propagate.propagate(numpy.zeros(segments_binary.shape),
                                                  endpoint_labels,
                                                  segments_binary_temp, 1)
    if numpy.any(loops):
        # set the end-of-loop distances to be very large
        distances[loop_endpoints[:, 0], loop_endpoints[:, 1]] = numpy.inf
    #
    # Order points by label # and distance
    #
    i, j = numpy.mgrid[0:segments_binary.shape[0],
           0:segments_binary.shape[1]]
    i = i[segments_binary]
    j = j[segments_binary]
    labels = segments_labeled[segments_binary]
    distances = distances[segments_binary]
    order = numpy.lexsort((distances, labels))
    i = i[order]
    j = j[order]
    labels = labels[order]
    distances = distances[order]
    #
    # Number each point in a segment consecutively. We determine
    # where each label starts. Then we subtract the start index
    # of each point's label from each point to get the order relative
    # to the first index of the label.
    #
    segment_order = numpy.arange(len(i))
    areas = numpy.bincount(labels.flatten())
    indexes = numpy.cumsum(areas) - areas
    segment_order -= indexes[labels]

    return i, j, labels, segment_order, distances, num_segments


def make_incidence_matrix(L1, N1, L2, N2):
    """Return an N1+1 x N2+1 matrix that marks all L1 and L2 that are 8-connected

    L1 - a labels matrix
    N1 - # of labels in L1
    L2 - a labels matrix
    N2 - # of labels in L2

    L1 and L2 should have no overlap

    Returns a matrix where M[n,m] is true if there is some pixel in
    L1 with value n that is 8-connected to a pixel in L2 with value m
    """
    #
    # Overlay the two labels matrix
    #
    L = L1.copy()
    L[L2 != 0] = L2[L2 != 0] + N1
    neighbor_count, neighbor_index, n2 = \
        centrosome.cpmorphology.find_neighbors(L)
    if numpy.all(neighbor_count == 0):
        return numpy.zeros((N1, N2), bool)
    #
    # Keep the neighbors of L1 / discard neighbors of L2
    #
    neighbor_count = neighbor_count[:N1]
    neighbor_index = neighbor_index[:N1]
    n2 = n2[:(neighbor_index[-1] + neighbor_count[-1])]
    #
    # Get rid of blanks
    #
    label = numpy.arange(N1)[neighbor_count > 0]
    neighbor_index = neighbor_index[neighbor_count > 0]
    neighbor_count = neighbor_count[neighbor_count > 0]
    #
    # Correct n2 beause we have formerly added N1 to its labels. Make
    # it zero-based.
    #
    n2 -= N1 + 1
    #
    # Create runs of n1 labels
    #
    n1 = numpy.zeros(len(n2), int)
    n1[0] = label[0]
    n1[neighbor_index[1:]] = label[1:] - label[:-1]
    n1 = numpy.cumsum(n1)
    incidence = scipy.sparse.coo.coo_matrix((numpy.ones(n1.shape), (n1, n2)),
                                            shape=(N1, N2)).toarray()
    return incidence != 0


def get_longest_path_coords(graph_struct, max_length):
    """Given a graph describing the structure of the skeleton of an image,
    returns the longest non-self-intersecting (with some caveats, see
    get_all_paths.m) path through that graph, specified as a polyline.

    Inputs:

    graph_struct: A structure describing the graph. Same format as returned
    by get_graph_from_binary(), see that file for details.

    Outputs:

    path_coords: A n x 2 array, where successive columns contains the
    coordinates of successive points on the paths (which when joined with
    line segments form the path itself.)

    path_struct: A structure, with entries 'segments' and 'branch_areas',
    descring the path found, in relation to graph_struct. See
    get_all_paths.m for details."""

    path_list = get_all_paths(graph_struct, 0, max_length)
    current_longest_path_coords = []
    current_max_length = 0
    current_path = None
    for path in path_list:
        path_coords = path_to_pixel_coords(graph_struct, path)
        path_length = calculate_path_length(path_coords)
        if path_length >= current_max_length:
            current_longest_path_coords = path_coords
            current_max_length = path_length
            current_path = path
    return current_longest_path_coords, current_path


def get_all_paths(graph_struct, min_length, max_length):
    """Given a structure describing a graph, returns a cell array containing
    a list of all paths through the graph.

    The format of graph_struct is exactly that outputted by
    get_graph_from_binary()

    Below, "vertex" refers to the "branch areas" of the
    graph_struct, and "edge" to refer to the "segments".

    For the purposes of this function, a path of length n is a sequence of n
    distinct edges

        e_1, ..., e_n

    together with a sequence of n-1 distinct vertices

        v_1, ..., v_{n-1}

    such that e_1 is incident to v_1, v_1 incident to e_2, and so on.

    Note that, since the ends are not considered parts of the paths, cyclic
    paths are allowed (i.e. ones starting and ending at the same vertex, but
    not self-crossing ones.)

    Furthermore, this function also considers two paths identical if one can
    be obtained by a simple reversation of the other.

    This function works by a simple depth-first search. It seems
    unnecessarily complicated compared to what it perhaps could have been;
    this is due to the fact that the endpoints are segments are not
    considered as vertices in the graph model used, and so each edge can be
    incident to less than 2 vertices.

    To explain how the function works, let me define an "unfinished path" to
    be a sequence of n edges e_1,...,e_n and n distinct vertices v_1, ..., v_n,
    where incidence relations e_1 - v_1 - e_2 - ... - e_n - v_n apply, and
    the intention is for the path to be continued through v_n. In constrast,
    call paths as defined in the previous paragraphs (where the last vertex
    is not included) "finished".

    The function first generates all unfinished paths of length 1 by looping
    through all possible edges, and for each edge at most 2 "continuation"
    vertices. It then calls get_all_paths_recur(), which, given an unfinished
    path, recursively generates a list of all possible finished paths
    beginning that unfinished path.

     To ensure that paths are only returned in one of the two possible
     directions, only 1-length paths and paths where the index of the
     first edge is less than that of the last edge are returned.

     To faciliate the processing in get_all_paths_recur, the function
     build_incidence_lists is used to calculate incidence tables in a list
     form.

     The output is a list of objects, "o" of the form

     o.segments - segment indices of the path
     o.branch_areas - branch area indices of the path"""

    graph_struct.incident_branch_areas, graph_struct.incident_segments = build_incidence_lists(graph_struct)
    n = len(graph_struct.segments)

    graph_struct.segment_lengths = numpy.array([calculate_path_length(x[0]) for x in graph_struct.segments])

    for j in range(n):
        current_length = graph_struct.segment_lengths[j]
        # Add all finished paths of length 1
        if current_length >= min_length:
            yield Path([j], [])
        #
        # Start the segment list for each branch area connected with
        # a segment with the segment.
        #
        segment_list = [j]
        branch_areas_list = [
            [k] for k in graph_struct.incident_branch_areas[j]]

        paths_list = get_all_paths_recur(graph_struct, segment_list, branch_areas_list, current_length, min_length, max_length)

        for path in paths_list:
            yield path


def build_incidence_lists(graph_struct):
    """Return a list of all branch areas incident to j for each segment

    incident_branch_areas{j} is a row array containing a list of all those
    branch areas incident to segment j; similary, incident_segments{i} is a
    row array containing a list of all those segments incident to branch area
    i."""
    m = graph_struct.incidence_matrix.shape[1]
    n = graph_struct.incidence_matrix.shape[0]
    incident_segments = [
        numpy.arange(m)[graph_struct.incidence_matrix[i, :]]
        for i in range(n)]
    incident_branch_areas = [
        numpy.arange(n)[graph_struct.incidence_matrix[:, i]]
        for i in range(m)]
    return incident_branch_areas, incident_segments


def calculate_path_length(path_coords):
    """Return the path length, given path coordinates as Nx2"""
    if len(path_coords) < 2:
        return 0
    return numpy.sum(numpy.sqrt(numpy.sum((path_coords[:-1] - path_coords[1:]) ** 2, 1)))


def get_all_paths_recur(graph, unfinished_segment, unfinished_branch_areas, current_length, min_length, max_length):
    """Recursively find paths

    incident_branch_areas - list of all branch areas incident on a segment
    incident_segments - list of all segments incident on a branch
    """
    if len(unfinished_segment) == 0:
        return
    last_segment = unfinished_segment[-1]
    for unfinished_branch in unfinished_branch_areas:
        end_branch_area = unfinished_branch[-1]
        #
        # Find all segments from the end branch
        #
        direction = graph.incidence_directions[end_branch_area, last_segment]
        last_coord = graph.segments[last_segment][direction][-1]
        for j in graph.incident_segments[end_branch_area]:
            if j in unfinished_segment:
                continue  # segment already in the path
            direction = not graph.incidence_directions[end_branch_area, j]
            first_coord = graph.segments[j][direction][0]
            gap_length = numpy.sqrt(numpy.sum((last_coord - first_coord) ** 2))
            next_length = current_length + gap_length + graph.segment_lengths[j]
            if next_length > max_length:
                continue
            next_segment = unfinished_segment + [j]
            if j > unfinished_segment[0] and next_length >= min_length:
                # Only include if end segment index is greater
                # than start
                yield Path(next_segment, unfinished_branch)
            #
            # Can't loop back to "end_branch_area". Construct all of
            # possible branches otherwise
            #
            next_branch_areas = [unfinished_branch + [k]
                                 for k in graph.incident_branch_areas[j]
                                 if (k != end_branch_area) and
                                 (k not in unfinished_branch)]
            for path in get_all_paths_recur(
                    graph, next_segment, next_branch_areas,
                    next_length, min_length, max_length):
                yield path


def path_to_pixel_coords(graph_struct, path):
    """Given a structure describing paths in a graph, converts those to a
    polyline (i.e. successive coordinates) representation of the same graph.

    (This is possible because the graph_struct descriptor contains
    information on where the vertices and edges of the graph were initially
    located in the image plane.)

    Inputs:

    graph_struct: A structure describing the graph. Same format as returned
    by get_graph_from_binary(), so for details, see that file.

    path_struct: A structure which (in relation to graph_struct) describes
    a path through the graph. Same format as (each entry in the list)
    returned by get_all_paths(), so see further get_all_paths.m

    Outputs:

    pixel_coords: A n x 2 double array, where each column contains the
    coordinates of one point on the path. The path itself can be formed
    by joining these points successively to each other.

    Note:

    Because of the way the graph is built, the points in pixel_coords are
    likely to contain segments consisting of runs of pixels where each is
    close to the next (in its 8-neighbourhood), but interleaved with
    reasonably long "jumps", where there is some distance between the end
    of one segment and the beginning of the next."""

    if len(path.segments) == 1:
        return graph_struct.segments[path.segments[0]][0]

    direction = graph_struct.incidence_directions[path.branch_areas[0], path.segments[0]]
    result = [graph_struct.segments[path.segments[0]][direction]]
    for branch_area, segment in zip(path.branch_areas, path.segments[1:]):
        direction = not graph_struct.incidence_directions[branch_area, segment]
        result.append(graph_struct.segments[segment][direction])
    return numpy.vstack(result)


def calculate_cumulative_lengths(path_coords):
    """return a cumulative length vector given Nx2 path coordinates"""
    if len(path_coords) < 2:
        return numpy.array([0] * len(path_coords))
    return numpy.hstack(([0],
                         numpy.cumsum(numpy.sqrt(numpy.sum((path_coords[:-1] - path_coords[1:]) ** 2, 1)))))


def sample_control_points(path_coords, cumul_lengths, num_control_points):
    """Sample equally-spaced control points from the Nx2 path coordinates

    Inputs:

    path_coords: A Nx2 double array, where each column specifies a point
    on the path (and the path itself is formed by joining successive
    points with line segments). Such as returned by
    path_struct_to_pixel_coords().

    cumul_lengths: A vector, where the ith entry indicates the
    length from the first point of the path to the ith in path_coords).
    In most cases, should be calculate_cumulative_lenghts(path_coords).

    n: A positive integer. The number of control points to sample.

    Outputs:

    control_coords: A N x 2 double array, where the jth column contains the
    jth control point, sampled along the path. The first and last control
    points are equal to the first and last points of the path (i.e. the
    points whose coordinates are the first and last columns of
    path_coords), respectively."""
    assert num_control_points > 2
    #
    # Paranoia - eliminate any coordinates with length = 0, esp the last.
    #
    path_coords = path_coords.astype(float)
    cumul_lengths = cumul_lengths.astype(float)
    mask = numpy.hstack(([True], cumul_lengths[1:] != cumul_lengths[:-1]))
    path_coords = path_coords[mask]
    #
    # Create a function that maps control point index to distance
    #

    ncoords = len(path_coords)
    f = scipy.interpolate.interp1d(cumul_lengths, numpy.linspace(0.0, float(ncoords - 1), ncoords))
    #
    # Sample points from f (for the ones in the middle)
    #
    first = float(cumul_lengths[-1]) / float(num_control_points - 1)
    last = float(cumul_lengths[-1]) - first
    findices = f(numpy.linspace(first, last, num_control_points - 2))
    indices = findices.astype(int)
    assert indices[-1] < ncoords - 1
    fracs = findices - indices
    sampled = (path_coords[indices, :] * (1 - fracs[:, numpy.newaxis]) +
               path_coords[(indices + 1), :] * fracs[:, numpy.newaxis])
    #
    # Tack on first and last
    #
    sampled = numpy.vstack((path_coords[:1, :], sampled, path_coords[-1:, :]))
    return sampled


class Path(object):
    def __init__(self, segments, branch_areas):
        self.segments = segments
        self.branch_areas = branch_areas

    def __repr__(self):
        return "{ segments=" + repr(self.segments) + " branch_areas=" + repr(self.branch_areas) + " }"