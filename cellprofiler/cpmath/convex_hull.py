import numpy as np

def convex_hull_ijv(labels_ijv, indexes):
    labels_ijv = labels_ijv.astype(np.int32)[np.lexsort(labels_ijv.T), :]  # reorder by v, then j, then i
    #XXX print labels_ijv, indexes
    out = labels_ijv  # we just made a copy, and this is guaranteed to be large enough
    outidx = 0  # this is the next row to be written in the output
    indexes_reorder = np.argsort(indexes)
    num_indexes = len(indexes)
    # find the maximums
    max_i, max_j, max_label = labels_ijv.max(axis=0)
    # allocate the upper and lower vertex buffers, and initialize them to extreme values
    upper = np.empty(max_j + 1, np.int32)
    lower = np.empty(max_j + 1, np.int32)
    vertex_counts = np.zeros(num_indexes, np.int32)
    hull_offsets = np.zeros(num_indexes, np.int32)
    # initialize them to extreme values
    upper[:] = -1
    lower[:] = max_i + 1
    pixidx = 0
    for cur_req in range(num_indexes):
        cur_label = indexes[indexes_reorder[cur_req]]
        while (cur_label <= max_label) and (labels_ijv[pixidx, 2] < cur_label):
            pixidx += 1
            if pixidx == labels_ijv.shape[0]:
                break
        num_vertices = 0
        hull_offsets[cur_req] = outidx
        if (pixidx == labels_ijv.shape[0]) or (cur_label != labels_ijv[pixidx, 2]):
            # cur_label's hull will have 0 points
            continue
        start_j = labels_ijv[pixidx, 1]
        while cur_label == labels_ijv[pixidx, 2]:
            cur_pix_i, cur_pix_j = labels_ijv[pixidx, :2]
            if upper[cur_pix_j] < cur_pix_i:
                upper[cur_pix_j] = cur_pix_i
            if lower[cur_pix_j] > cur_pix_i:
                lower[cur_pix_j] = cur_pix_i
            pixidx += 1
            if pixidx == labels_ijv.shape[0]:
                break
        end_j = labels_ijv[pixidx - 1, 1]
        #XXX print "STARTJ/END_J", start_j, end_j
        #XXX print "LOWER", lower[start_j:end_j + 1]
        #XXX print "UPPER", upper[start_j:end_j + 1]
        # At this point, the upper and lower buffers have the extreme high/low
        # points, so we just need to convexify them.  We have copied them out
        # of the labels_ijv array, so we write out the hull into that array
        # (via its alias "out").  We are careful about memory when we do so, to
        # make sure we don't invalidate the next entry in labels_ijv.
        #
        # We assume that the j-coordinates are dense.  If this assumption is
        # violated, we should re-walk the pixidx values we just copied to only
        # deal with columns that actually have points.
        #
        # Produce hull in counter-clockwise order, starting with lower
        # envelope.  Reset the envelopes as we do so.
        need_last_upper_point = (lower[start_j] != upper[start_j])
        num_emitted = 0
        out_base_idx = outidx

        def CONVEX(pt_i, pt_j, nemitted):
            # We're walking CCW, so left turns are convex
            d_i_prev, d_j_prev, z = out[out_base_idx + nemitted - 1, :] - out[out_base_idx + nemitted - 2, :]
            d_i_cur = pt_i - out[out_base_idx + nemitted - 1, 0]
            d_j_cur = pt_j - out[out_base_idx + nemitted - 1, 1]
            # note that x is j, y is i
            return (d_j_prev * d_i_cur - d_j_cur * d_i_prev) > 0

        def EMIT(pt_i, pt_j, nemitted):
            while (nemitted >= 2) and not CONVEX(pt_i, pt_j, nemitted):
                # The point we emitted just before this one created a
                # concavity (or is co-linear).  Prune it.
                #XXX print "BACKUP"
                nemitted -= 1
            # The point is convex or we haven't emitted enough points to check.
            #XXX print "writing point", nemitted, pt_i, pt_j
            out[out_base_idx + nemitted, :] = (pt_i, pt_j, cur_label)
            return nemitted + 1

        for envelope_j in range(start_j, end_j + 1):
            if lower[envelope_j] < max_i + 1:
                num_emitted = EMIT(lower[envelope_j], envelope_j, num_emitted)
                lower[envelope_j] = max_i + 1
        for envelope_j in range(end_j, start_j, -1):
            if upper[envelope_j] > -1:
                num_emitted = EMIT(upper[envelope_j], envelope_j, num_emitted)
                upper[envelope_j] = -1
        if need_last_upper_point:
            num_emitted = EMIT(upper[start_j], start_j, num_emitted)
        upper[start_j] = -1
        # advance the output index
        vertex_counts[cur_req] = num_emitted
        outidx += num_emitted
    # reorder
    reordered = np.zeros((np.sum(vertex_counts), 3), np.int32)
    reordered_counts = np.zeros(num_indexes, np.int32)
    reordered_idx = 0
    for reordered_num in range(num_indexes):
        count = vertex_counts[indexes_reorder[reordered_num]]
        src_start = hull_offsets[indexes_reorder[reordered_num]]
        src_end = src_start + count
        dest_start = reordered_idx
        dest_end = reordered_idx + count
        reordered[dest_start:dest_end, :] = out[src_start:src_end, :]
        reordered_idx += count
        reordered_counts[reordered_num] = count
    print "OUT", out
    print "C", vertex_counts
    print "REO", reordered
    print "RC", reordered_counts
    return reordered[:, [2, 0, 1]], reordered_counts

def convex_hull(labels, indexes=None):
    """Given a labeled image, return a list of points per object ordered by
    angle from an interior point, representing the convex hull.s

    labels - the label matrix
    indexes - an array of label #s to be processed, defaults to all non-zero
              labels

    Returns a matrix and a vector. The matrix consists of one row per
    point in the convex hull. Each row has three columns, the label #,
    the i coordinate of the point and the j coordinate of the point. The
    result is organized first by label, then the points are arranged
    counter-clockwise around the perimeter.
    The vector is a vector of #s of points in the convex hull per label
    """
    if indexes == None:
        indexes = np.unique(labels)
        indexes.sort()
        indexes=indexes[indexes!=0]
    else:
        indexes=np.array(indexes)
    if len(indexes) == 0:
        return np.zeros((0,2),int),np.zeros((0,),int)
    #
    # Reduce the # of points to consider
    #
    outlines = labels
    coords = np.argwhere(outlines > 0).astype(np.int32)
    if len(coords)==0:
        # Every outline of every image is blank
        return (np.zeros((0,3),int),
                np.zeros((len(indexes),),int))

    i = coords[:,0]
    j = coords[:,1]
    labels_per_point = labels[i,j]
    pixel_labels = np.column_stack((i,j,labels_per_point))
    return convex_hull_ijv(pixel_labels, indexes)


