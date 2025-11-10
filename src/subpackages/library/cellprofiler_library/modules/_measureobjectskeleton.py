import numpy
import scipy.ndimage
import centrosome.cpmorphology
import centrosome.propagate as propagate
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from scipy.ndimage import grey_dilation, grey_erosion
from typing import Tuple, Optional, Any, Dict, Union

from numpy.typing import NDArray
from cellprofiler_library.types import Image2DBinary, ObjectSegmentation, Image2DColor, Image2D, Image2DGrayscale
from pydantic import Field, validate_call, ConfigDict
from cellprofiler_library.opts.measureobjectskeleton import VF_I, VF_J, VF_LABELS, VF_KIND, EF_V1, EF_V2, EF_LENGTH, EF_TOTAL_INTENSITY

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def get_ring_with_trunks(
        skeleton: Image2DBinary, 
        labels: ObjectSegmentation
        ) -> Tuple[
            ObjectSegmentation,
            Image2DBinary,
            Image2DBinary
        ]:
    #
    # The following code makes a ring around the seed objects with
    # the skeleton trunks sticking out of it.
    #
    # Create a new skeleton with holes at the seed objects
    # First combine the seed objects with the skeleton so
    # that the skeleton trunks come out of the seed objects.
    #
    # Erode the labels once so that all of the trunk branchpoints
    # will be within the labels
    #
    #
    # Dilate the objects, then subtract them to make a ring
    #
    my_disk = centrosome.cpmorphology.strel_disk(1.5).astype(int)
    dilated_labels = grey_dilation(labels, footprint=my_disk)
    seed_mask = dilated_labels > 0
    combined_skel = skeleton | seed_mask

    closed_labels = grey_erosion(dilated_labels, footprint=my_disk)
    seed_center = closed_labels > 0
    combined_skel = combined_skel & (~seed_center)
    return dilated_labels, combined_skel, seed_center

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def measure_object_skeleton(
        skeleton: Image2DBinary, 
        cropped_labels: ObjectSegmentation, 
        labels_count: numpy.integer[Any],
        label_range: NDArray[numpy.int32], 
        fill_small_holes: bool, 
        max_hole_size: int,
        wants_objskeleton_graph: bool,
        intensity_image_pixel_data: Optional[Image2DGrayscale],
        wants_branchpoint_image: bool
        ) -> Tuple[
            NDArray[numpy.float_], # trunk counts
            NDArray[numpy.float_], # branch counts
            NDArray[numpy.float_], # end counts
            NDArray[numpy.float_], # total distance
            Optional[Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]],
            Optional[Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]],
            Optional[Image2DColor],
        ]:
    #
    # The following code makes a ring around the seed objects with
    # the skeleton trunks sticking out of it.
    #
    dilated_labels, combined_skel, seed_center = get_ring_with_trunks(skeleton, cropped_labels)
    
    #
    # Fill in single holes (but not a one-pixel hole made by
    # a one-pixel image)
    #
    if fill_small_holes:
        def size_fn(area, is_object):
            return (~is_object) and (area <= max_hole_size)

        combined_skel = centrosome.cpmorphology.fill_labeled_holes(
            combined_skel, ~seed_center, size_fn
        )
    #
    # Reskeletonize to make true branchpoints at the ring boundaries
    #
    combined_skel = centrosome.cpmorphology.skeletonize(combined_skel)
    #
    # The skeleton outside of the labels
    #
    outside_skel = combined_skel & (dilated_labels == 0)
    #
    # Associate all skeleton points with seed objects
    #
    dlabels, distance_map = propagate.propagate(
        numpy.zeros(cropped_labels.shape), dilated_labels, combined_skel, 1
    )
    #
    # Get rid of any branchpoints not connected to seeds
    #
    combined_skel[dlabels == 0] = False
    #
    # Find the branchpoints
    #
    branch_points = centrosome.cpmorphology.branchpoints(combined_skel)

    #
    # Odd case: when four branches meet like this, branchpoints are not
    # assigned because they are arbitrary. So assign them.
    #
    # .  .
    #  B.
    #  .B
    # .  .
    #
    odd_case = (
        combined_skel[:-1, :-1]
        & combined_skel[1:, :-1]
        & combined_skel[:-1, 1:]
        & combined_skel[1, 1]
    )
    branch_points[:-1, :-1][odd_case] = True
    branch_points[1:, 1:][odd_case] = True
    #
    # Find the branching counts for the trunks (# of extra branches
    # emanating from a point other than the line it might be on).
    #
    branching_counts = centrosome.cpmorphology.branchings(combined_skel)
    branching_counts = numpy.array([0, 0, 0, 1, 2])[branching_counts]
    #
    # Only take branches within 1 of the outside skeleton
    #
    dilated_skel = scipy.ndimage.binary_dilation(
        outside_skel, centrosome.cpmorphology.eight_connect
    )
    branching_counts[~dilated_skel] = 0
    #
    # Find the endpoints
    #
    end_points = centrosome.cpmorphology.endpoints(combined_skel)
    #
    # We use two ranges for classification here:
    # * anything within one pixel of the dilated image is a trunk
    # * anything outside of that range is a branch
    #
    nearby_labels = dlabels.copy()
    nearby_labels[distance_map > 1.5] = 0

    outside_labels = dlabels.copy()
    outside_labels[nearby_labels > 0] = 0
    #
    # The trunks are the branchpoints that lie within one pixel of
    # the dilated image.
    #
    if labels_count > 0:
        trunk_counts = fix(
            scipy.ndimage.sum(branching_counts, nearby_labels, label_range)
        ).astype(int)
    else:
        trunk_counts = numpy.zeros((0,), int)
    #
    # The branches are the branchpoints that lie outside the seed objects
    #
    if labels_count > 0:
        branch_counts = fix(
            scipy.ndimage.sum(branch_points, outside_labels, label_range)
        )
    else:
        branch_counts = numpy.zeros((0,), int)
    #
    # Save the endpoints
    #
    if labels_count > 0:
        end_counts = fix(scipy.ndimage.sum(end_points, outside_labels, label_range))
    else:
        end_counts = numpy.zeros((0,), int)
    #
    # Calculate the distances
    #
    total_distance = centrosome.cpmorphology.skeleton_length(
        dlabels * outside_skel, label_range
    ).astype(numpy.float64)
    edge_graph = None
    vertex_graph = None
    
    branchpoint_image = None
    if wants_objskeleton_graph or wants_branchpoint_image:
        trunk_mask = (branching_counts > 0) & (nearby_labels != 0)
        
        if wants_objskeleton_graph:
            assert intensity_image_pixel_data is not None
            edge_graph, vertex_graph = make_objskeleton_graph(
                combined_skel,
                dlabels,
                trunk_mask,
                branch_points & ~trunk_mask,
                end_points,
                intensity_image_pixel_data,
            )
        if wants_branchpoint_image:
            branchpoint_image = numpy.zeros((skeleton.shape[0], skeleton.shape[1], 3))
            branch_mask = branch_points & (outside_labels != 0)
            end_mask = end_points & (outside_labels != 0)
            branchpoint_image[outside_skel, :] = 1
            branchpoint_image[trunk_mask | branch_mask | end_mask, :] = 0
            branchpoint_image[trunk_mask, 0] = 1
            branchpoint_image[branch_mask, 1] = 1
            branchpoint_image[end_mask, 2] = 1
            branchpoint_image[dilated_labels != 0, :] *= 0.875
            branchpoint_image[dilated_labels != 0, :] += 0.1

    return (
        trunk_counts, 
        branch_counts,
        end_counts,
        total_distance,
        edge_graph, 
        vertex_graph,
        branchpoint_image
    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def make_objskeleton_graph(
    skeleton: Image2DBinary, 
    skeleton_labels: ObjectSegmentation, 
    trunks: Image2DBinary, 
    branchpoints: Image2DBinary, 
    endpoints: Image2DBinary, 
    image: Image2DGrayscale
    ) -> Tuple[
        Dict[str, NDArray[Union[numpy.float_, numpy.int_]]],
        Dict[str, NDArray[Union[numpy.float_, numpy.int_]]]
    ]:
    """Make a table that captures the graph relationship of the skeleton

    skeleton - binary skeleton image + outline of seed objects
    skeleton_labels - labels matrix of skeleton
    trunks - binary image with trunk points as 1
    branchpoints - binary image with branchpoints as 1
    endpoints - binary image with endpoints as 1
    image - image for intensity measurement

    returns two tables.
    Table 1: edge table
    The edge table is a numpy record array with the following named
    columns in the following order:
    v1: index into vertex table of first vertex of edge
    v2: index into vertex table of second vertex of edge
    length: # of intermediate pixels + 2 (for two vertices)
    total_intensity: sum of intensities along the edge

    Table 2: vertex table
    The vertex table is a numpy record array:
    i: I coordinate of the vertex
    j: J coordinate of the vertex
    label: the vertex's label
    kind: kind of vertex = "T" for trunk, "B" for branchpoint or "E" for endpoint.
    """
    i, j = numpy.mgrid[0 : skeleton.shape[0], 0 : skeleton.shape[1]]
    #
    # Give each point of interest a unique number
    #
    points_of_interest = trunks | branchpoints | endpoints
    number_of_points = numpy.sum(points_of_interest)
    #
    # Make up the vertex table
    #
    tbe = numpy.zeros(points_of_interest.shape, "|S1")
    tbe[trunks] = "T"
    tbe[branchpoints] = "B"
    tbe[endpoints] = "E"
    i_idx = i[points_of_interest]
    j_idx = j[points_of_interest]
    poe_labels = skeleton_labels[points_of_interest]
    tbe = tbe[points_of_interest]
    vertex_table = {
        VF_I: i_idx,
        VF_J: j_idx,
        VF_LABELS: poe_labels,
        VF_KIND: tbe,
    }
    #
    # First, break the skeleton by removing the branchpoints, endpoints
    # and trunks
    #
    broken_skeleton = skeleton & (~points_of_interest)
    #
    # Label the broken skeleton: this labels each edge differently
    #
    edge_labels, nlabels = centrosome.cpmorphology.label_skeleton(skeleton)
    #
    # Reindex after removing the points of interest
    #
    edge_labels[points_of_interest] = 0
    if nlabels > 0:
        indexer = numpy.arange(nlabels + 1)
        unique_labels = numpy.sort(numpy.unique(edge_labels))
        nlabels = len(unique_labels) - 1
        indexer[unique_labels] = numpy.arange(len(unique_labels))
        edge_labels = indexer[edge_labels]
        #
        # find magnitudes and lengths for all edges
        #
        magnitudes = fix(
            scipy.ndimage.sum(
                image, edge_labels, numpy.arange(1, nlabels + 1, dtype=numpy.int32)
            )
        )
        lengths = fix(
            scipy.ndimage.sum(
                numpy.ones(edge_labels.shape),
                edge_labels,
                numpy.arange(1, nlabels + 1, dtype=numpy.int32),
            )
        ).astype(int)
    else:
        magnitudes = numpy.zeros(0)
        lengths = numpy.zeros(0, int)
    #
    # combine the edge labels and indexes of points of interest with padding
    #
    edge_mask = edge_labels != 0
    all_labels = numpy.zeros(numpy.array(edge_labels.shape) + 2, int)
    all_labels[1:-1, 1:-1][edge_mask] = edge_labels[edge_mask] + number_of_points
    all_labels[i_idx + 1, j_idx + 1] = numpy.arange(1, number_of_points + 1)
    #
    # Collect all 8 neighbors for each point of interest
    #
    p1 = numpy.zeros(0, int)
    p2 = numpy.zeros(0, int)
    for i_off, j_off in (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ):
        p1 = numpy.hstack((p1, numpy.arange(1, number_of_points + 1)))
        p2 = numpy.hstack((p2, all_labels[i_idx + i_off, j_idx + j_off]))
    #
    # Get rid of zeros which are background
    #
    p1 = p1[p2 != 0]
    p2 = p2[p2 != 0]
    #
    # Find point_of_interest -> point_of_interest connections.
    #
    p1_poi = p1[(p2 <= number_of_points) & (p1 < p2)]
    p2_poi = p2[(p2 <= number_of_points) & (p1 < p2)]
    #
    # Make sure matches are labeled the same
    #
    same_labels = (
        skeleton_labels[i_idx[p1_poi - 1], j_idx[p1_poi - 1]]
        == skeleton_labels[i_idx[p2_poi - 1], j_idx[p2_poi - 1]]
    )
    p1_poi = p1_poi[same_labels]
    p2_poi = p2_poi[same_labels]
    #
    # Find point_of_interest -> edge
    #
    p1_edge = p1[p2 > number_of_points]
    edge = p2[p2 > number_of_points]
    #
    # Now, each value that p2_edge takes forms a group and all
    # p1_edge whose p2_edge are connected together by the edge.
    # Possibly they touch each other without the edge, but we will
    # take the minimum distance connecting each pair to throw out
    # the edge.
    #
    edge, p1_edge, p2_edge = centrosome.cpmorphology.pairwise_permutations(
        edge, p1_edge
    )
    indexer = edge - number_of_points - 1
    lengths = lengths[indexer]
    magnitudes = magnitudes[indexer]
    #
    # OK, now we make the edge table. First poi<->poi. Length = 2,
    # magnitude = magnitude at each point
    #
    poi_length = numpy.ones(len(p1_poi)) * 2
    poi_magnitude = (
        image[i_idx[p1_poi - 1], j_idx[p1_poi - 1]]
        + image[i_idx[p2_poi - 1], j_idx[p2_poi - 1]]
    )
    #
    # Now the edges...
    #
    poi_edge_length = lengths + 2
    poi_edge_magnitude = (
        image[i_idx[p1_edge - 1], j_idx[p1_edge - 1]]
        + image[i_idx[p2_edge - 1], j_idx[p2_edge - 1]]
        + magnitudes
    )
    #
    # Put together the columns
    #
    v1 = numpy.hstack((p1_poi, p1_edge))
    v2 = numpy.hstack((p2_poi, p2_edge))
    lengths = numpy.hstack((poi_length, poi_edge_length))
    magnitudes = numpy.hstack((poi_magnitude, poi_edge_magnitude))
    #
    # Sort by p1, p2 and length in order to pick the shortest length
    #
    indexer = numpy.lexsort((lengths, v1, v2))
    v1 = v1[indexer]
    v2 = v2[indexer]
    lengths = lengths[indexer]
    magnitudes = magnitudes[indexer]
    if len(v1) > 0:
        to_keep = numpy.hstack(([True], (v1[1:] != v1[:-1]) | (v2[1:] != v2[:-1])))
        v1 = v1[to_keep]
        v2 = v2[to_keep]
        lengths = lengths[to_keep]
        magnitudes = magnitudes[to_keep]
    #
    # Put it all together into a table
    #
    edge_table = {
        EF_V1: v1,
        EF_V2: v2,
        EF_LENGTH: lengths,
        EF_TOTAL_INTENSITY: magnitudes,
    }
    return edge_table, vertex_table
