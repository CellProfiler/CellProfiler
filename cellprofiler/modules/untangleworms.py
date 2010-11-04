'''<b>UntangleWorms</b> untangles overlapping worms
<hr>

UntangleWorms takes a binary image and the results of worm training and
labels the worms in the image, untangling them and associating all of a
worm's pieces together.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np
import os
import scipy.ndimage as scind
from scipy.sparse import coo
from scipy.interpolate import interp1d
from scipy.io import loadmat
import urllib2

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.cpmath.cpmorphology as morph
from cellprofiler.cpmath.propagate import propagate

from cellprofiler.preferences import standardize_default_folder_names, \
     DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, NO_FOLDER_NAME, \
     ABSOLUTE_FOLDER_NAME, IO_FOLDER_CHOICE_HELP_TEXT

OO_WITH_OVERLAP = "With overlap"
OO_WITHOUT_OVERLAP = "Without overlap"
OO_BOTH = "Both"

'''Shape cost method = angle shape model for cluster paths selection'''
SCM_ANGLE_SHAPE_MODEL = 'angle_shape_model'

'''Maximum # of sets of paths considered at any level'''
MAX_CONSIDERED = 50000
'''Maximum # of different paths considered for input'''
MAX_PATHS = 400

class UntangleWorms(cpm.CPModule):
    
    variable_revision_number = 1
    category = "Object Processing"
    module_name = "UntangleWorms"
    def create_settings(self):
        '''Create the settings that parameterize the module'''
        self.image_name = cps.ImageNameSubscriber(
            "Binary image", "None",
            doc = """A binary image where the foreground indicates the worm
            shapes. The binary image can be produced by the <b>ApplyThreshold</b>
            module.""")
        self.overlap = cps.Choice(
            "Overlap style:", [OO_BOTH, OO_WITH_OVERLAP, OO_WITHOUT_OVERLAP],
            doc = """This setting determines which style objects are output.
            If two worms overlap, you have a choice of including the overlapping
            regions in both worms or excluding the overlapping regions from
            both worms. Choose <i>%(OO_WITH_OVERLAP)s</i> to save objects including
            overlapping regions, <i>%(OO_WITHOUT_OVERLAP)s</i> to save only
            the portions of objects that don't overlap or <i>
            %(OO_BOTH)s</i> to save two versions: with and without overlap.""" %
            globals())
        self.overlap_objects = cps.ObjectNameProvider(
            "Overlapping worms object name:", "OverlappingWorms",
            doc = """This setting names the objects representing the overlapping
            worms. When worms cross, they overlap and pixels are shared by
            both of the overlapping worms. The overlapping worm objects share
            these pixels and measurements of both overlapping worms will include
            these pixels in the measurements of both worms.""")
        self.nonoverlapping_objects = cps.ObjectNameProvider(
            "Non-overlapping worms object name:", "NonOverlappingWorms",
            doc = """This setting names the objects representing the worms,
            excluding those regions where the worms overlap. When worms cross,
            there are pixels that cannot be unambiguously assigned to one
            worm or the other. These pixels are excluded from both worms
            in the non-overlapping objects and will not be a part of the
            measurements of either worm.""")
        self.training_set_directory = cps.DirectoryPath(
            "Training set file location", support_urls = True,
            doc = """Select the folder containing the training set to be loaded.
            %(IO_FOLDER_CHOICE_HELP_TEXT)s
            <p>An additional option is the following:
            <ul>
            <li><i>URL</i>: Use the path part of a URL. For instance, your
            training set might be hosted at 
            <i>http://university.edu/~johndoe/TrainingSet.mat</i>
            To access this file, you would choose <i>URL</i> and enter
            <i>https://svn.broadinstitute.org/CellProfiler/trunk/ExampleImages/ExampleSBSImages</i>
            as the path location.</li>
            </ul></p>"""%globals())
        def get_directory_fn():
            '''Get the directory for the CSV file name'''
            return self.training_set_directory.get_absolute_path()
        def set_directory_fn(path):
            dir_choice, custom_path = self.training_set_directory.get_parts_from_path(path)
            self.training_set_directory.join_parts(dir_choice, custom_path)
            
        self.training_set_file_name = cps.FilenameText(
            "Training set file name", "TrainingSet.mat",
            doc = "This is the name of the training set file.",
            metadata = True,
            get_directory_fn = get_directory_fn,
            set_directory_fn = set_directory_fn,
            browse_msg = "Choose training set",
            exts = [("Worm training set (*.mat)", "*.mat"),
                    ("All files (*.*)", "*.*")])
        
        self.wants_training_set_weights = cps.Binary(
            "Use training set weights?", True,
            doc = """Check this setting to use the overlap and leftover
            weights from the training set. Uncheck the setting to override
            these weights.""")
        
        self.override_overlap_weight = cps.Float(
            "Overlap weight", 5, 0,
            doc = """This setting controls how much weight is given to overlaps
            between worms. <b>UntangleWorms</b> charges a penalty to a
            particular putative grouping of worms that overlap equal to the
            length of the overlapping region times the overlap weight. Increase
            the overlap weight to make <b>UntangleWorms</b> avoid overlapping
            portions of worms. Decrease the overlap weight to make
            <b>UntangleWorms</b> ignore overlapping portions of worms.""")
        
        self.override_leftover_weight = cps.Float(
            "Leftover weight", 10, 0,
            doc = """This setting controls how much weight is given to 
            areas not covered by worms.
            <b>UntangleWorms</b> charges a penalty to a
            particular putative grouping of worms that fail to cover all
            of the foreground of a binary image. The penalty is equal to the
            length of the uncovered region times the leftover weight. Increase
            the leftover weight to make <b>UntangleWorms</b> cover more
            foreground with worms. Decrease the overlap weight to make
            <b>UntangleWorms</b> ignore uncovered foreground.""")
        
    def settings(self):
        return [self.image_name, self.overlap, self.overlap_objects,
                self.nonoverlapping_objects, self.training_set_directory,
                self.training_set_file_name, self.wants_training_set_weights,
                self.override_overlap_weight, 
                self.override_leftover_weight]
    
    def visible_settings(self):
        result = [self.image_name, self.overlap]
        if self.overlap in (OO_WITH_OVERLAP, OO_BOTH):
            result += [self.overlap_objects]
        if self.overlap in (OO_WITHOUT_OVERLAP, OO_BOTH):
            result += [self.nonoverlapping_objects]
        result += [self.training_set_directory, self.training_set_file_name,
                   self.wants_training_set_weights]
        if not self.wants_training_set_weights:
            result += [self.override_overlap_weight, 
                       self.override_leftover_weight]
        return result

    def overlap_weight(self, params):
        '''The overlap weight to use in the cost calculation'''
        if not self.wants_training_set_weights:
            return self.override_overlap_weight.value
        else:
            return params.cluster_paths_selection.overlap_weight
        
    def leftover_weight(self, params):
        '''The leftover weight to use in the cost calculation'''
        if not self.wants_training_set_weights:
            return self.override_leftover_weight.value
        else:
            return params.cluster_paths_selection.leftover_weight
        
    def run(self, workspace):
        params = self.read_params(workspace)
        image_name = self.image_name.value
        image = workspace.image_set.get_image(image_name,
                                              must_be_binary = True)
        labels, count = scind.label(image.pixel_data, morph.eight_connect)
        #
        # Skeletonize once, then remove any points in the skeleton
        # that are adjacent to the edge of the image, then skeletonize again.
        #
        # This gets rid of artifacts that cause combinatoric explosions:
        #
        #    * * * * * * * *
        #      *   *   *
        #    * * * * * * * * 
        #
        skeleton = morph.skeletonize(image.pixel_data)
        eroded = scind.binary_erosion(image.pixel_data, morph.eight_connect)
        skeleton = morph.skeletonize(skeleton & eroded)
        #
        # The path skeletons
        #
        all_path_coords = []
        if count != 0:
            areas = np.bincount(labels.flatten())
            current_index = 1
            for i in range(1,count+1):
                if areas[i] < params.min_worm_area:
                    # Completely exclude the worm
                    continue
                elif areas[i] <= params.max_area:
                    path_coords, path_struct = self.single_worm_find_path(
                        workspace, labels, i, skeleton, params)
                    if self.single_worm_filter(
                        workspace, path_coords, params):
                        all_path_coords.append(path_coords)
                else:
                    graph = self.cluster_graph_building(
                        workspace, labels, i, skeleton, params)
                    paths = self.get_all_paths(graph)
                    paths_selected = self.cluster_paths_selection(
                        graph, paths, labels, i, params)
                    all_path_coords += paths_selected
        ijv = self.worm_descriptor_building(all_path_coords, params,
                                            labels.shape)
        if workspace.frame is not None:
            workspace.display_data.input_image = image.pixel_data
            workspace.display_data.ijv = ijv
        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)
        if self.overlap in (OO_WITH_OVERLAP, OO_BOTH):
            o = cpo.Objects()
            o.ijv = ijv
            o.parent_image = image
            object_set.add_objects(o, self.overlap_objects.value)
        if self.overlap in (OO_WITHOUT_OVERLAP, OO_BOTH):
            #
            # Sum up the number of overlaps using a sparse matrix
            #
            overlap_hits = coo.coo_matrix(
                (np.ones(len(ijv)), (ijv[:,0], ijv[:,1])),
                image.pixel_data.shape)
            overlap_hits = overlap_hits.toarray()
            mask = overlap_hits == 1
            labels = coo.coo_matrix((ijv[:,2],(ijv[:,0], ijv[:,1])), mask.shape)
            labels = labels.toarray()
            labels[~ mask] = 0
            o = cpo.Objects()
            o.segmented = labels
            o.parent_image = image
            object_set.add_objects(o, self.nonoverlapping_objects.value)
    
    def is_interactive(self):
        return False
    
    def display(self, workspace):
        figure = workspace.create_or_find_figure(subplots = (2,1))
        axes = figure.subplot_imshow_bw(0, 0, workspace.display_data.input_image,
                                        title = self.image_name.value)
        if self.overlap in (OO_BOTH, OO_WITH_OVERLAP):
            title = self.overlap_objects.value
        else:
            title = self.nonoverlapping_objects.value
        figure.subplot_imshow_ijv(1, 0, workspace.display_data.ijv,
                                  shape = workspace.display_data.input_image.shape,
                                  title = title)
    
    def single_worm_find_path(self, workspace, labels, i, skeleton, params):
        '''Finds the worm's skeleton  as a path.
        
        labels - the labels matrix, labeling single and clusters of worms
        
        i - the labeling of the worm of interest
        
        params - The parameter structure
        
        returns:
        
        path_coords: A 2 x n array, of coordinates for the path found. (Each
              point along the polyline path is represented by a column,
              i coordinates in the first row and j coordinates in the second.)
              
        path_struct: a structure describing the path
        '''
        binary_im = labels == i
        skeleton = skeleton & binary_im
        graph_struct = self.get_graph_from_binary(binary_im, skeleton)
        return self.get_longest_path_coords(graph_struct)
    
    def get_graph_from_binary(self, binary_im, skeleton, max_radius = None, 
                              max_skel_length = None):
        '''Manufacture a graph of the skeleton of the worm
        
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
           information in graph_struct.segments.)'''
        branch_areas_binary = morph.branchpoints(skeleton)
        if max_radius is not None:
            #
            # Add any points that are more than the worm diameter to
            # the branchpoints
            #
            strel = morph.strel_disk(max_radius)
            far = scind.binary_erosion(binary_im, strel)
            far = scind.binary_opening(far, structure = morph.eight_connect)
            branch_areas_binary |= far
            del far
        branch_areas_binary = scind.binary_dilation(
            branch_areas_binary, structure = morph.eight_connect)
        segments_binary = skeleton & ~ branch_areas_binary
        if max_skel_length is not None:
            max_skel_length = max(int(max_skel_length),2) # paranoia
            i, j, labels, order, distance, num_segments = \
             self.trace_segments(segments_binary)
            #
            # Put breakpoints every max_skel_length, but not at end
            #
            max_order = np.array(scind.maximum(order, labels, 
                                               np.arange(num_segments + 1)))
            big_segment = max_order >= max_skel_length
            segment_count = np.maximum((max_order + max_skel_length - 1) / 
                                       max_skel_length, 1).astype(int)
            segment_length = ((max_order + 1) / segment_count).astype(int)
            new_bp_mask = ((order % segment_length[labels] == 
                            segment_length[labels] - 1) &
                           (order != max_order[labels]) &
                           (big_segment[labels]))
            new_branch_areas_binary = np.zeros(segments_binary.shape, bool)
            new_branch_areas_binary[i[new_bp_mask], j[new_bp_mask]] = True
            new_branch_areas_binary = scind.binary_dilation(
                new_branch_areas_binary, structure = morph.eight_connect)
            branch_areas_binary |= new_branch_areas_binary
            segments_binary &= ~new_branch_areas_binary
        return self.get_graph_from_branching_areas_and_segments(
            branch_areas_binary, segments_binary)
    
    def trace_segments(self, segments_binary):
        '''Find distance of every point in a segment from a segment endpoint
        
        segments_binary - a binary mask of the segments in an image.
        
        returns a tuple of the following:
        i - the i coordinate of a point in the mask
        j - the j coordinate of a point in the mask
        label - the segment's label
        order - the ordering (from 0 to N-1 where N is the # of points in
                the segment.)
        distance - the propagation distance of the point from the endpoint
        num_segments - the # of labelled segments
        '''
        #
        # Break long skeletons into pieces whose maximum length
        # is max_skel_length.
        #
        segments_labeled, num_segments = scind.label(
            segments_binary, structure = morph.eight_connect)
        if num_segments == 0:
            return (np.array([], int), np.array([], int), np.array([], int),
                    np.array([], int), np.array([]), 0)
        #
        # Get one endpoint per segment
        #
        endpoints = morph.endpoints(segments_binary)
        #
        # Use a consistent order: pick with lowest i, then j.
        # If a segment loops upon itself, we pick an arbitrary point.
        #
        order = np.arange(np.prod(segments_binary.shape))
        order.shape = segments_binary.shape
        order[~ endpoints] += np.prod(segments_binary.shape)
        labelrange = np.arange(num_segments+1)
        endpoint_loc = scind.minimum_position(order, segments_labeled,
                                              labelrange)
        endpoint_loc = np.array(endpoint_loc, int)
        endpoint_labels = np.zeros(segments_labeled.shape, np.int16)
        endpoint_labels[endpoint_loc[:,0], endpoint_loc[:,1]] =\
            segments_labeled[endpoint_loc[:,0], endpoint_loc[:,1]]
        #
        # A corner case - propagate will trace a loop around both ways. So
        # we have to find that last point and remove it so
        # it won't trace in that direction
        #
        loops = ~ endpoints[endpoint_loc[1:,0], endpoint_loc[1:,1]]
        if np.any(loops):
            # Consider all points around the endpoint, finding the one
            # which is numbered last
            dilated_ep_labels = morph.grey_dilation(
                endpoint_labels, footprint = np.ones((3,3), bool))
            dilated_ep_labels[dilated_ep_labels != segments_labeled] = 0
            loop_endpoints = scind.maximum_position(
                order, dilated_ep_labels, labelrange[1:][loops])
            loop_endpoints = np.array(loop_endpoints, int)
            segments_binary_temp = segments_binary.copy()
            segments_binary_temp[loop_endpoints[:,0], loop_endpoints[:,1]] = False
        else:
            segments_binary_temp = segments_binary
        #
        # Now propagate from the endpoints to get distances
        #
        _, distances = propagate(np.zeros(segments_binary.shape),
                                 endpoint_labels,
                                 segments_binary_temp, 1)
        if np.any(loops):
            # set the end-of-loop distances to be very large
            distances[loop_endpoints[:,0], loop_endpoints[:,1]] = np.inf
        #
        # Order points by label # and distance
        #
        i, j = np.mgrid[0:segments_binary.shape[0], 
                        0:segments_binary.shape[1]]
        i = i[segments_binary]
        j = j[segments_binary]
        labels = segments_labeled[segments_binary]
        distances = distances[segments_binary]
        order = np.lexsort((distances, labels))
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
        segment_order = np.arange(len(i))
        areas = np.bincount(labels.flatten())
        indexes = np.cumsum(areas) - areas
        segment_order -= indexes[labels]
        return i, j, labels, segment_order, distances, num_segments
        
    def get_graph_from_branching_areas_and_segments(
        self, branch_areas_binary, segments_binary):
        '''Turn branches + segments into a graph
        
        branch_areas_binary - binary mask of branch areas
        
        segments_binary - binary mask of segments != branch_areas
        
        Given two binary images, one containing "branch areas" one containing
        "segments", returns a structure describing the incidence relations
        between the branch areas and the segments.

        Output is same format as get_graph_from_binary(), so for details, see
        get_graph_from_binary
        '''
        branch_areas_labeled, num_branch_areas = scind.label(
            branch_areas_binary, morph.eight_connect)
        
        i, j, labels, order, distance, num_segments = self.trace_segments(
            segments_binary)
        
        ooo = np.lexsort((order, labels))
        i = i[ooo]
        j = j[ooo]
        labels = labels[ooo]
        order = order[ooo]
        distance = distance[ooo]
        counts = np.bincount(labels.flatten())[1:]
        
        branch_ij = np.argwhere(branch_areas_binary)
        if len(branch_ij) > 0:
            ooo = np.lexsort([
                branch_ij[:,0], branch_ij[:,1],
                branch_areas_labeled[branch_ij[:,0], branch_ij[:,1]]])
            branch_ij = branch_ij[ooo]
            branch_labels = branch_areas_labeled[branch_ij[:,0], branch_ij[:,1]]
            branch_counts = np.bincount(branch_areas_labeled.flatten())[1:]
        else:
            branch_labels = np.zeros(0, int)
            branch_counts = np.zeros(0, int)
        #
        # "find" the segment starts
        #
        starts = order == 0
        start_labels = np.zeros(segments_binary.shape, int)
        start_labels[i[starts], j[starts]] = labels[starts]
        #
        # incidence_directions = True for starts
        #
        incidence_directions = self.make_incidence_matrix(
            branch_areas_labeled, num_branch_areas, start_labels, num_segments)
        #
        # Get the incidence matrix for the ends
        #
        ends = np.cumsum(counts)-1
        end_labels  = np.zeros(segments_binary.shape, int)
        end_labels[i[ends], j[ends]] = labels[ends]
        incidence_matrix = self.make_incidence_matrix(
            branch_areas_labeled, num_branch_areas, end_labels, num_segments)
        incidence_matrix |= incidence_directions
        
        class Result(object):
            '''A result graph:
            
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
            '''
            def __init__(self, branch_areas_binary, counts, i,j,
                         branch_ij, branch_counts, incidence_matrix,
                         incidence_directions):
                self.image_size = tuple(branch_areas_binary.shape)
                self.segment_coords = np.column_stack((i,j))
                self.segment_indexes = np.cumsum(counts) - counts
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
                self.branch_area_indexes = np.cumsum(branch_counts) - branch_counts
                self.branch_area_counts = branch_counts
                self.incidence_matrix = incidence_matrix
                self.incidence_directions = incidence_directions
        return Result(branch_areas_binary, counts, i,j, branch_ij, branch_counts,
                      incidence_matrix, incidence_directions)
    
    def make_incidence_matrix(self, L1, N1, L2, N2):
        '''Return an N1+1 x N2+1 matrix that marks all L1 and L2 that are 8-connected
        
        L1 - a labels matrix
        N1 - # of labels in L1
        L2 - a labels matrix
        N2 - # of labels in L2
        
        L1 and L2 should have no overlap
        
        Returns a matrix where M[n,m] is true if there is some pixel in
        L1 with value n that is 8-connected to a pixel in L2 with value m
        '''
        #
        # Overlay the two labels matrix
        #
        L = L1.copy()
        L[L2 != 0] = L2[L2 != 0] + N1
        neighbor_count, neighbor_index, n2 = \
                     morph.find_neighbors(L)
        if np.all(neighbor_count == 0):
            return np.zeros((N1, N2), bool)
        #
        # Keep the neighbors of L1 / discard neighbors of L2
        #
        neighbor_count = neighbor_count[:N1]
        neighbor_index = neighbor_index[:N1]
        n2 = n2[:(neighbor_index[-1] + neighbor_count[-1])]
        #
        # Get rid of blanks
        #
        label = np.arange(N1)[neighbor_count > 0]
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
        n1 = np.zeros(len(n2), int)
        n1[0] = label[0]
        n1[neighbor_index[1:]] = label[1:] - label[:-1]
        n1 = np.cumsum(n1)
        incidence = coo.coo_matrix((np.ones(n1.shape), (n1,n2)),
                                   shape = (N1, N2)).toarray()
        return incidence != 0
        
    def get_longest_path_coords(self, graph_struct):
        '''Given a graph describing the structure of the skeleton of an image,
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
        get_all_paths.m for details.'''

        path_list = self.get_all_paths(graph_struct)
        current_longest_path_coords = []
        current_max_length = 0
        for path in path_list:
            path_coords = self.path_to_pixel_coords(graph_struct, path)
            path_length = self.calculate_path_length(path_coords)
            if path_length > current_max_length:
                current_longest_path_coords = path_coords
                current_max_length = path_length
                current_path = path
        return current_longest_path_coords, current_path
    
    def path_to_pixel_coords(self, graph_struct, path):
        '''Given a structure describing paths in a graph, converts those to a 
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
        of one segment and the beginning of the next.'''
        
        if len(path.segments) == 1:
            return graph_struct.segments[path.segments[0]][0]
        
        direction = graph_struct.incidence_directions[path.branch_areas[0],
                                                      path.segments[0]]
        result = [graph_struct.segments[path.segments[0]][direction]]
        for branch_area, segment in zip(path.branch_areas, path.segments[1:]):
            direction = not graph_struct.incidence_directions[branch_area,
                                                              segment]
            result.append(graph_struct.segments[segment][direction])
        return np.vstack(result)

    def calculate_path_length(self, path_coords):
        '''Return the path length, given path coordinates as Nx2'''
        if len(path_coords) < 2:
            return 0
        return np.sum(np.sqrt(np.sum((path_coords[:-1]-path_coords[1:])**2,1)))
    
    def calculate_cumulative_lengths(self, path_coords):
        '''return a cumulative length vector given Nx2 path coordinates'''
        if len(path_coords) < 2:
            return [0] * len(path_coords)
        return np.hstack(([0], 
            np.cumsum(np.sqrt(np.sum((path_coords[:-1]-path_coords[1:])**2,1)))))
    
    def single_worm_filter(self, workspace, path_coords, params):
        '''Given a path representing a single worm, caculates its shape cost, and
        either accepts it as a worm or rejects it, depending on whether or not
        the shape cost is higher than some threshold.

        Inputs:

        path_coords:  A N x 2 array giving the coordinates of the path.

        params: the parameters structure from which we use

            cost_theshold: Scalar double. The maximum cost possible for a worm; 
            paths of shape cost higher than this are rejected.

            num_control_points. Scalar positive integer. The shape cost
            model uses control points sampled at equal intervals along the
            path.

            mean_angles: A (num_control_points-1) x
            1 double array. See calculate_angle_shape_cost() for how this is
            used.

            inv_angles_covariance_matrix: A
            (num_control_points-1)x(num_control_points-1) double matrix. See
            calculate_angle_shape_cost() for how this is used.

         Returns true if worm passes filter'''
        if len(path_coords) < 2:
            return False
        filter_params = params.filter
        cumul_lengths = self.calculate_cumulative_lengths(path_coords)
        total_length = cumul_lengths[-1]
        control_coords = self.sample_control_points(
            path_coords, cumul_lengths, filter_params.num_control_points)
        cost = self.calculate_angle_shape_cost(
            control_coords, total_length, filter_params.mean_angles,
            filter_params.inv_angles_covariance_matrix)
        return cost < filter_params.cost_threshold

    def sample_control_points(self, path_coords, cumul_lengths, num_control_points):
        '''Sample equally-spaced control points from the Nx2 path coordinates

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
        path_coords), respectively.'''
        assert num_control_points > 2
        #
        # Paranoia - eliminate any coordinates with length = 0, esp the last.
        #
        path_coords = path_coords.astype(float)
        cumul_lengths = cumul_lengths.astype(float)
        mask = np.hstack(([True], cumul_lengths[1:] != cumul_lengths[:-1]))
        path_coords = path_coords[mask]
        #
        # Create a function that maps control point index to distance
        #
        
        ncoords = len(path_coords)
        f = interp1d(cumul_lengths, np.linspace(0.0, float(ncoords-1), ncoords))
        #
        # Sample points from f (for the ones in the middle)
        #
        first = float(cumul_lengths[-1]) / float(num_control_points-1)
        last = float(cumul_lengths[-1]) - first
        findices = f(np.linspace(first, last, num_control_points-2))
        indices = findices.astype(int)
        assert indices[-1] < ncoords-1
        fracs = findices - indices
        sampled = (path_coords[indices,:] * (1-fracs[:,np.newaxis]) +
                   path_coords[(indices+1),:] * fracs[:,np.newaxis])
        #
        # Tack on first and last
        #
        sampled = np.vstack((path_coords[:1,:], sampled, path_coords[-1:,:]))
        return sampled

    def calculate_angle_shape_cost(self, control_coords, total_length,
                                   mean_angles, inv_angles_covariance_matrix):
        '''% Calculates a shape cost based on the angle shape cost model.
        
        Given a set of N control points, calculates the N-2 angles between 
        lines joining consecutive control points, forming them into a vector.
        The function then appends the total length of the path formed, as an
        additional value in the now (N-1)-dimensional feature
        vector.
    
        The returned value is the square of the Mahalanobis distance from
        this feature vector, v, to a training set with mean mu and covariance
        matrix C, calculated as
    
        cost = (v - mu)' * C^-1 * (v - mu)
        
        Input parameters:
    
        control_coords: A 2 x N double array, containing the coordinates of
        the control points; one control point in each column. In the same
        format as returned by sample_control_points().
    
        total_length: Scalar double. The total length of the path from which the control
        points are sampled. (I.e. the distance along the path from the
        first control poin to the last. E.g. as returned by
        calculate_path_length().
    
        mean_angles: A (N-1) x 1 double array. The mu in the above formula,
        i.e. the mean of the feature vectors as calculated from the
        training set. Thus, the first N-2 entries are the means of the
        angles, and the last entry is the mean length of the training
        worms. 
    
        inv_angles_covariance_matrix: A (N-1)x(N-1) double matrix. The
        inverse of the covariance matrix of the feature vectors in the
        training set. Thus, this is the C^-1 (nb: not just C) in the
        above formula.
     
        Output parameters:
        
        current_shape_cost: Scalar double. The squared Mahalanobis distance
        calculated. Higher values indicate that the path represented by
        the control points (and length) are less similar to the training
        set.
        
        Note: All the angles in question here are direction angles,
        constrained to lie between -pi and pi. The angle 0 corresponds to
        the case when two adjacnet line segments are parallel (and thus
        belong to the same line); the angles can be thought of as the
        (signed) angles through which the path "turns", and are thus not the
        angles between the line segments as such.'''
        
        num_control_points = len(mean_angles) + 1
        
        segments_delta = control_coords[1:] - control_coords[:-1]
        segment_bearings = np.arctan2(segments_delta[:,0], segments_delta[:,1])
        angles = segment_bearings[1:] - segment_bearings[:-1]
        #
        # Constrain the angles to -pi <= angle <= pi
        #
        angles[angles > np.pi] -= 2 * np.pi
        angles[angles < -np.pi] += 2 * np.pi
        feat_vec = np.hstack((angles, [total_length])) - mean_angles
        return np.dot(np.dot(feat_vec, inv_angles_covariance_matrix), feat_vec)
    
    def cluster_graph_building(self, workspace, labels, i, skeleton, params):
        binary_im = labels == i
        skeleton = skeleton & binary_im

        return self.get_graph_from_binary(
            binary_im, skeleton, params.cluster_graph_building.max_radius,
            params.cluster_graph_building.max_skel_length)
    
    class Path(object):
        def __init__(self, segments, branch_areas):
            self.segments = segments
            self.branch_areas = branch_areas
            
        def __repr__(self):
            return "{ segments="+repr(self.segments)+" branch_areas="+repr(self.branch_areas)+" }"
            
    def get_all_paths(self, graph_struct):
        '''Given a structure describing a graph, returns a cell array containing
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
         directions, only 1-length paths and paths where the the index of the
         first edge is less than that of the last edge are returned.

         To faciliate the processing in get_all_paths_recur, the function
         build_incidence_lists is used to calculate incidence tables in a list
         form.

         The output is a list of objects, "o" of the form

         o.segments - segment indices of the path
         o.branch_areas - branch area indices of the path'''
        
        paths_list = []
        
        incident_branch_areas, incident_segments = self.build_incidence_lists(
            graph_struct)
        n = len(graph_struct.segments)
        
        for j in range(n):
            # Add all finished paths of length 1
            paths_list.append(self.Path([j], []))
            #
            # Start the segment list for each branch area connected with
            # a segment with the segment.
            #
            segment_list = [j]
            branch_areas_list = [[k] for k in incident_branch_areas[j]]
            
            paths_list += self.get_all_paths_recur(
                incident_branch_areas, incident_segments,
                segment_list, branch_areas_list)
            
        return paths_list
        
    def build_incidence_lists(self, graph_struct):
        '''Return a list of all branch areas incident to j for each segment

        incident_branch_areas{j} is a row array containing a list of all those
        branch areas incident to segment j; similary, incident_segments{i} is a
        row array containing a list of all those segments incident to branch area
        i.'''
        m = graph_struct.incidence_matrix.shape[1]
        n = graph_struct.incidence_matrix.shape[0]
        incident_segments = [ 
            np.arange(m)[graph_struct.incidence_matrix[i,:]]
            for i in range(n)]
        incident_branch_areas = [
            np.arange(n)[graph_struct.incidence_matrix[:,i]]
            for i in range(m)]
        return incident_branch_areas, incident_segments

    def get_all_paths_recur(self, incident_branch_areas, incident_segments,
                            unfinished_segment, unfinished_branch_areas):
        '''Recursively find paths
        
        incident_branch_areas - list of all branch areas incident on a segment
        incident_segments - list of all segments incident on a branch
        '''
        paths_list = []
        for unfinished_branch in unfinished_branch_areas:
            end_branch_area = unfinished_branch[-1]
            #
            # Find all segments from the end branch
            #
            for j in incident_segments[end_branch_area]:
                if j in unfinished_segment:
                    continue # segment already in the path
                next_segment = unfinished_segment + [j]
                if j > unfinished_segment[0]:
                    # Only include if end segment index is greater
                    # than start
                    paths_list.append(self.Path(next_segment, unfinished_branch))
                #
                # Can't loop back to "end_branch_area". Construct all of
                # possible branches otherwise
                #
                next_branch_areas = [ unfinished_branch + [k] 
                                      for k in incident_branch_areas[j]
                                      if (k != end_branch_area) and
                                      (k not in unfinished_branch)]
                paths_list += self.get_all_paths_recur(
                    incident_branch_areas, incident_segments,
                    next_segment, next_branch_areas)
        return paths_list
            
    
    def cluster_paths_selection(self, graph, paths, labels, i, params):
        """Select the best paths for worms from the graph
        
        Given a graph representing a worm cluster, and a list of paths in the
        graph, selects a subcollection of paths likely to represent the worms in
        the cluster.

        More specifically, finds (approximately, depending on parameters) a
        subset K of the set P paths, minimising 
        
        Sum, over p in K, of shape_cost(K)
        +  a * Sum, over p,q distinct in K, of overlap(p, q)
        +  b * leftover(K)
 
        Here, shape_cost is a function which calculates how unlikely it is that 
        the path represents a true worm.
 
        overlap(p, q) indicates how much overlap there is between paths p and q
        (we want to assign a cost to overlaps, to avoid picking out essentially
        the same worm, but with small variations, twice in K)

        leftover(K) is a measure of the amount of the cluster "unaccounted for"
        after all of the paths of P have been chosen. We assign a cost to this to
        make sure we pick out all the worms in the cluster.
        
        Shape model:'angle_shape_model'. More information 
        can be found in calculate_angle_shape_cost(),

        Selection method

        'dfs_prune': searches
        through all the combinations of paths (view this as picking out subsets
        of P one element at a time, to make this a search tree) depth-first,
        but by keeping track of the best solution so far (and noting that the
        shape cost and overlap cost terms can only increase as paths are added
        to K), it can prune away large branches of the search tree guaranteed
        to be suboptimal.

        Furthermore, by setting the approx_max_search_n parameter to a finite
        value, this method adopts a "partially greedy" approach, at each step
        searching through only a set number of branches. Setting this parameter
        approx_max_search_n to 1 should in some sense give just the greedy
        algorithm, with the difference that this takes the leftover cost term
        into account in determining how many worms to find.

        Input parameters:

        graph_struct: A structure describing the graph. As returned from e.g.
        get_graph_from_binary().

        path_structs_list: A cell array of structures, each describing one path
        through the graph. As returned by cluster_paths_finding().

        params: The parameters structure. The parameters below should be
        in params.cluster_paths_selection

        min_path_length: Before performing the search, paths which are too
        short or too long are filtered away. This is the minimum length, in
        pixels.

        max_path_length: Before performing the search, paths which are too
        short or too long are filtered away. This is the maximum length, in
        pixels.

        shape_cost_method: 'angle_shape_cost'
 
        num_control_points: All shape cost models samples equally spaced
        control points along the paths whose shape cost are to be
        calculated. This is the number of such control points to sample.
     
        mean_angles: [Only for 'angle_shape_cost']
        
        inv_angles_covariance_matrix: [Only for 'angle_shape_cost']
        
        For these two parameters,  see calculate_angle_shape_cost().

        overlap_leftover_method:
        'skeleton_length'. The overlap/leftover calculation method to use.
        Note that if selection_method is 'dfs_prune', then this must be
        'skeleton_length'.

        selection_method: 'dfs_prune'. The search method
        to be used. 
     
        median_worm_area: Scalar double. The approximate area of a typical
        worm. 
        This approximates the number of worms in the
        cluster. Is only used to estimate the best branching factors in the
        search tree. If approx_max_search_n is infinite, then this is in
        fact not used at all.

        overlap_weight: Scalar double. The weight factor assigned to
        overlaps, i.e. the a in the formula of the cost to be minimised.
        the unit is (shape cost unit)/(pixels as a unit of
        skeleton length). 
        
        leftover_weight:  The
        weight factor assigned to leftover pieces, i.e. the b in the
        formula of the cost to be minimised. In units of (shape cost
        unit)/(pixels of skeleton length).

        approx_max_search_n: [Only used if selection_method is 'dfs_prune']

        Outputs:

        paths_coords_selected: A cell array of worms selected. Each worm is
        represented as 2xm array of coordinates, specifying the skeleton of
        the worm as a polyline path.
"""
        cps_params = params.cluster_paths_selection
        min_path_length = cps_params.min_path_length
        max_path_length = cps_params.max_path_length
        median_worm_area = cps_params.median_worm_area
        num_control_points = params.filter.num_control_points
        
        mean_angles = params.filter.mean_angles
        inv_angles_covariance_matrix = params.filter.inv_angles_covariance_matrix
        
        component = labels == i
        max_num_worms = int(np.ceil(np.sum(component) / median_worm_area))
        num_worms_to_find = min(len(paths), max(max_num_worms, 1))
 
        # First, filter out based on path length 
        # Simultaneously build a vector of shape costs and a vector of
        # reconstructed binaries for each of the (accepted) paths.
        
        #
        # List of tuples of path structs that pass filter + cost of shape
        #
        paths_and_costs = []
        segment_lengths = np.array([self.calculate_path_length(fwd_segment)
                                    for fwd_segment, rev_segment 
                                    in graph.segments])
        for i, path in enumerate(paths):
            current_path_coords = self.path_to_pixel_coords(graph, path)
            cumul_lengths = self.calculate_cumulative_lengths(current_path_coords)
            total_length = cumul_lengths[-1]
            if total_length > max_path_length or total_length < min_path_length:
                continue
            control_coords = self.sample_control_points(
                current_path_coords, cumul_lengths, num_control_points)
            #
            # Calculate the shape cost
            #
            current_shape_cost = self.calculate_angle_shape_cost(
                control_coords, total_length, mean_angles, 
                inv_angles_covariance_matrix)
            paths_and_costs.append((path, current_shape_cost))
        
        if len(paths_and_costs) == 0:
            return []
        
        path_segment_matrix = np.zeros(
            (len(graph.segments), len(paths_and_costs)), bool)
        for i, (path, cost) in enumerate(paths_and_costs):
            path_segment_matrix[path.segments, i] = True
        overlap_weight = self.overlap_weight(params)
        leftover_weight = self.leftover_weight(params)
        #
        # Sort by increasing cost
        #
        costs = np.array([cost for path, cost in paths_and_costs])
        order = np.lexsort([costs])
        if len(order) > MAX_PATHS:
            order = order[:MAX_PATHS]
        costs = costs[order]
        path_segment_matrix = path_segment_matrix[:, order]
        
        current_best_subset, current_best_cost = self.fast_selection(
            costs, path_segment_matrix, segment_lengths, 
            overlap_weight, leftover_weight)
        selected_paths =  [paths_and_costs[order[i]][0]
                           for i in current_best_subset]
        path_coords_selected = [ self.path_to_pixel_coords(graph, path)
                                 for path in selected_paths]
        return path_coords_selected
        
    def fast_selection(self, costs, path_segment_matrix, segment_lengths,
                       overlap_weight, leftover_weight):
        '''Select the best subset of paths using a breadth-first search
        
        costs - the shape costs of every path
        
        path_segment_matrix - an N x M matrix where N are the segments
        and M are the paths. A cell is true if a path includes the segment
        
        segment_lengths - the length of each segment
        
        overlap_weight - the penalty per pixel of an overlap
        
        leftover_weight - the penalty per pixel of an unincluded segment
        '''
        current_best_subset = []
        current_best_cost = np.sum(segment_lengths) * leftover_weight
        current_costs = costs
        current_path_segment_matrix = path_segment_matrix.astype(int)
        current_path_choices = np.eye(len(costs), dtype = bool)
        for i in range(len(costs)):
            current_best_subset, current_best_cost, \
                current_path_segment_matrix, current_path_choices = \
                self.select_one_level(
                    costs, path_segment_matrix, segment_lengths, 
                    current_best_subset, current_best_cost, 
                    current_path_segment_matrix, current_path_choices,
                    overlap_weight, leftover_weight)
            if np.prod(current_path_choices.shape) == 0:
                break
        return current_best_subset, current_best_cost
    
    def select_one_level(self, costs, path_segment_matrix, segment_lengths,
                         current_best_subset, current_best_cost,
                         current_path_segment_matrix, current_path_choices,
                         overlap_weight, leftover_weight):
        '''Select from among sets of N paths
        
        Select the best subset from among all possible sets of N paths,
        then create the list of all sets of N+1 paths
        
        costs - shape costs of each path
        
        path_segment_matrix - a N x M boolean matrix where N are the segments
        and M are the paths and True means that a path has a given segment
        
        segment_lengths - the lengths of the segments (for scoring)
        
        current_best_subset - a list of the paths in the best collection so far
        
        current_best_cost - the total cost of that subset
        
        current_path_segment_matrix - a matrix giving the number of times
        a segment appears in each of the paths to be considered
        
        current_path_choices - an N x M matrix where N is the number of paths
        and M is the number of sets: the value at a cell is True if a path
        is included in that set.
        
        returns the current best subset, the current best cost and
        the current_path_segment_matrix and current_path_choices for the
        next round.
        '''
        #
        # Compute the cost, not considering uncovered segments
        #
        partial_costs = (
            #
            # The sum of the individual costs of the chosen paths
            #
            np.sum(costs[:, np.newaxis] * current_path_choices, 0) +
            #
            # The sum of the multiply-covered segment lengths * penalty
            #
            np.sum(np.maximum(current_path_segment_matrix - 1, 0) * 
                   segment_lengths[:, np.newaxis], 0) * overlap_weight)
        total_costs = (partial_costs +
            #
            # The sum of the uncovered segments * the penalty
            #
            np.sum((current_path_segment_matrix[:,:] == 0) * 
                   segment_lengths[:, np.newaxis], 0) * leftover_weight)

        order = np.lexsort([total_costs])
        if total_costs[order[0]] < current_best_cost:
            current_best_subset = np.argwhere(current_path_choices[:,order[0]]).flatten().tolist()
            current_best_cost = total_costs[order[0]]
        #
        # Weed out any that can't possibly be better
        #
        mask = partial_costs < current_best_cost
        if not np.any(mask):
            return current_best_subset, current_best_cost, \
                   np.zeros((len(costs),0),int), np.zeros((len(costs),0), bool)
        order = order[mask[order]]
        if len(order) * len(costs) > MAX_CONSIDERED:
            # Limit # to consider at next level
            order = order[:(1+MAX_CONSIDERED / len(costs))]
        current_path_segment_matrix = current_path_segment_matrix[:, order]
        current_path_choices = current_path_choices[:, order]
        #
        # Create a matrix of disallowance - you can only add a path
        # that's higher than any existing path
        #
        i,j = np.mgrid[0:len(costs), 0:len(costs)]
        disallow = i >= j
        allowed = np.dot(disallow, current_path_choices) == 0
        if np.any(allowed):
            i,j = np.argwhere(allowed).transpose()
            current_path_choices = (np.eye(len(costs), dtype = bool)[:, i] | 
                                    current_path_choices[:,j])
            current_path_segment_matrix = \
                    path_segment_matrix[:,i] + current_path_segment_matrix[:,j]
            return current_best_subset, current_best_cost, \
                   current_path_segment_matrix, current_path_choices
        else:
            return current_best_subset, current_best_cost, \
                np.zeros((len(costs), 0), int), np.zeros((len(costs), 0), bool)
                
    def search_recur(self, path_segment_matrix, segment_lengths,
                     path_raw_costs, overlap_weight, leftover_weight,
                     current_subset, last_chosen, current_cost,
                     current_segment_coverings, current_best_subset,
                     current_best_cost, branching_factors, current_level):
        '''Perform a recursive depth-first search on sets of paths
        
        Perform a depth-first search recursively,  keeping the best (so far)
        found subset of paths in current_best_subset, current_cost.
        
        path_segment_matrix, segment_lengths, path_raw_costs, overlap_weight,
        leftover_weight, branching_factor are essentially static.

        current_subset is the currently considered subset, as an array of
        indices, each index corresponding to a path in path_segment_matrix.

        To avoid picking out the same subset twice, we insist that in all
        subsets, indices are listed in increasing order.
        
        Note that the shape cost term and the overlap cost term need not be
        re-calculated each time, but can be calculated incrementally, as more
        paths are added to the subset in consideration. Thus, current_cost holds
        the sum of the shape cost and overlap cost terms for current_subset.
        
        current_segments_coverings, meanwhile, is a logical array of length equal
        to the number of segments in the graph, keeping track of the segments
        covered by paths in current_subset.'''

        # The cost of current_subset, including the leftover cost term
        this_cost = current_cost + leftover_weight * np.sum(
            segment_lengths[~ current_segment_coverings])
        if this_cost < current_best_cost:
            current_best_cost = this_cost
            current_best_subset = current_subset
        if current_level < len(branching_factors):
            this_branch_factor = branching_factors[current_level]
        else:
            this_branch_factor = branching_factors[-1]
        # Calculate, for each path after last_chosen, how much cost would be added
        # to current_cost upon adding that path to the current_subset.
        current_overlapped_costs = (
            path_raw_costs[last_chosen:] + 
            np.sum(current_segment_coverings[:, np.newaxis] * 
                   segment_lengths[:, np.newaxis] * 
                   path_segment_matrix[:, last_chosen:], 0) * overlap_weight)
        order = np.lexsort([current_overlapped_costs])
        #
        # limit to number of branches allowed at this level
        #
        order = order[np.arange(len(order))+1 < this_branch_factor]
        for index in order:
            new_cost = current_cost + current_overlapped_costs[index]
            if new_cost >= current_best_cost:
                break # No chance of subseequent better cost
            path_index = last_chosen + index
            current_best_subset, current_best_cost = self.search_recur(
                path_segment_matrix, segment_lengths, path_raw_costs,
                overlap_weight, leftover_weight,
                current_subset + [path_index],
                path_index,
                new_cost,
                current_segment_coverings | path_segment_matrix[:, path_index],
                current_best_subset,
                current_best_cost,
                branching_factors,
                current_level + 1)
        return current_best_subset, current_best_cost
                
    def worm_descriptor_building(self, all_path_coords, params, shape):
        '''Return the coordinates of reconstructed worms in i,j,v form
        
        Given a list of paths found in an image, reconstructs labeled
        worms.

        Inputs:

        worm_paths: A list of worm paths, each entry an N x 2 array 
        containing the coordinates of the worm path. 

        params:  the params structure loaded using read_params()

        Outputs: an Nx3 array where the first two indices are the i,j
        coordinate and the third is the worm's label.
        '''
        if len(all_path_coords) == 0:
            return np.zeros((0,3), int)
        
        worm_radius = params.cluster_paths_selection.worm_radius
        num_control_points = params.filter.num_control_points
        all_i = []
        all_j = []
        for path in all_path_coords:
            cumul_lengths = self.calculate_cumulative_lengths(path)
            control_coords = self.sample_control_points(
                path, cumul_lengths,  num_control_points)
            ii,jj = self.rebuild_worm_from_control_points_approx(
                control_coords, worm_radius, shape)
            all_i.append(ii)
            all_j.append(jj)
        return np.column_stack((
            np.hstack(all_i),
            np.hstack(all_j),
            np.hstack([np.ones(len(ii), int) * (i+1)
                       for i, ii in enumerate(all_i)])))
            
    
    def rebuild_worm_from_control_points_approx(self, control_coords, 
                                                worm_radius, shape):
        '''Rebuild a worm from its control coordinates
         
        Given a worm specified by some control points along its spline,
        reconstructs an approximate binary image representing the worm.

        Specifically, this function generates an image where successive control
        points have been joined by line segments, and then dilates that by a
        certain (specified) radius.

        Inputs:

        control_coords: A N x 2 double array, where each column contains the x
        and y coordinates for a control point.

        worm_radius: Scalar double. Approximate radius of a typical worm; the
        radius by which the reconstructed worm spline is dilated to form the
        final worm.

        Outputs:
        The coordinates of all pixels in the worm in an N x 2 array'''
        index, count, i, j = morph.get_line_pts(control_coords[:-1,0],
                                                control_coords[:-1,1],
                                                control_coords[1:,0],
                                                control_coords[1:,1])
        #
        # Get dilation coordinates
        #
        iworm_radius = int(worm_radius + 1)
        ii, jj = np.mgrid[-iworm_radius:iworm_radius+1,
                          -iworm_radius:iworm_radius+1]
        mask = ii*ii + jj*jj <= worm_radius * worm_radius
        ii = ii[mask]
        jj = jj[mask]
        #
        # All points (with repeats)
        #
        i = (i[:,np.newaxis] + ii[np.newaxis, :]).flatten()
        j = (j[:,np.newaxis] + jj[np.newaxis, :]).flatten()
        #
        # Find repeats by sorting and comparing against next
        #
        order = np.lexsort((i,j))
        i = i[order]
        j = j[order]
        mask = np.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
        i = i[mask]
        j = j[mask]
        mask = (i >= 0) & (j >= 0) & (i < shape[0]) & (j < shape[1])
        return i[mask], j[mask]
    
    def read_params(self, workspace):
        '''Read the parameters file'''
        #
        # The parameters file is a .mat file with the following structure:
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
        class X(object):
            '''This "class" is used as a vehicle for arbitrary dot notation
            
            For instance:
            > x = X()
            > x.foo = 1
            > x.foo
            1
            '''
            pass
        
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        path = self.training_set_directory.get_absolute_path(m)
        file_name = m.apply_metadata(self.training_set_file_name.value)
        if self.training_set_directory.dir_choice == cps.URL_FOLDER_NAME:
            url = path + "/" + file_name
            fd_or_file = urllib2.urlopen(url)
        else:
            fd_or_file = os.path.join(path, file_name)
        mat_params = loadmat(fd_or_file)["params"][0,0]
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
            '''Look up a field from mat_params'''
            x = mat_params
            for arg in args[:-1]:
                x = x[arg][0,0]
            x = x[args[-1]]
            kind = kwargs.get("kind", SCALAR)
            if kind == SCALAR:
                return x[0,0]
            elif kind == STRING:
                return x[0]
            elif kind == VECTOR:
                if x.shape[0] > 1:
                    return x[:,0]
                else:
                    return x[0,:]
            return x
        
        result.min_worm_area = mp(INITIAL_FILTER, "min_worm_area")
        result.max_area = mp(SINGLE_WORM_DETERMINATION, "max_area")
        result.find_path = X()
        result.find_path.method = mp(SINGLE_WORM_FIND_PATH, METHOD, kind=STRING)
        result.filter = X()
        result.filter.method = mp(SINGLE_WORM_FILTER, METHOD, kind = STRING)
        result.filter.cost_threshold = mp(SINGLE_WORM_FILTER, "cost_threshold")
        result.filter.num_control_points = mp(SINGLE_WORM_FILTER, "num_control_points")
        result.filter.mean_angles = mp(SINGLE_WORM_FILTER, "mean_angles", kind = VECTOR)
        result.filter.inv_angles_covariance_matrix = mp(
            SINGLE_WORM_FILTER, "inv_angles_covariance_matrix", kind = MATRIX)
        result.cluster_graph_building = X()
        result.cluster_graph_building.method = mp(
            CLUSTER_GRAPH_BUILDING, METHOD, kind = STRING)
        result.cluster_graph_building.max_radius = mp(CLUSTER_GRAPH_BUILDING,
                                                      "max_radius")
        result.cluster_graph_building.max_skel_length = mp(CLUSTER_GRAPH_BUILDING,
                                                           "max_skel_length")
        result.cluster_paths_finding = X()
        result.cluster_paths_finding.method = mp(CLUSTER_PATHS_FINDING, METHOD,
                                                 kind = STRING)
        result.cluster_paths_selection = X()
        result.cluster_paths_selection.shape_cost_method = mp(
            CLUSTER_PATHS_SELECTION, "shape_cost_method", kind = STRING)
        result.cluster_paths_selection.selection_method = mp(
            CLUSTER_PATHS_SELECTION, "selection_method", kind = STRING)
        result.cluster_paths_selection.overlap_leftover_method = mp(
            CLUSTER_PATHS_SELECTION, "overlap_leftover_method", kind = STRING)
        result.cluster_paths_selection.min_path_length = mp(
            CLUSTER_PATHS_SELECTION, "min_path_length")
        result.cluster_paths_selection.max_path_length = mp(
            CLUSTER_PATHS_SELECTION, "max_path_length")
        result.cluster_paths_selection.median_worm_area = mp(
            CLUSTER_PATHS_SELECTION, "median_worm_area")
        result.cluster_paths_selection.worm_radius = mp(
            CLUSTER_PATHS_SELECTION, "worm_radius")
        result.cluster_paths_selection.overlap_weight = mp(
            CLUSTER_PATHS_SELECTION, "overlap_weight")
        result.cluster_paths_selection.leftover_weight = mp(
            CLUSTER_PATHS_SELECTION, "leftover_weight")
        result.cluster_paths_selection.approx_max_search_n = mp(
            CLUSTER_PATHS_SELECTION, "approx_max_search_n")
        result.worm_descriptor_building = X()
        result.worm_descriptor_building.method = mp(
            WORM_DESCRIPTOR_BUILDING, METHOD, kind = STRING)
        result.worm_descriptor_building.radii_from_training = mp(
            WORM_DESCRIPTOR_BUILDING, "radii_from_training", kind = VECTOR)
        return result
        
        
        
        