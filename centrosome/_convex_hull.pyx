from __future__ import print_function
import cython
import numpy as np
cimport numpy as np
ctypedef np.int32_t DTYPE_t  # 32-bit pixel positions and labels


cdef enum:
    DEBUG = 0

# Does the path a->b->c form a convexity in the plane?
cdef inline int CONVEX(int a_i, int a_j,
                       int b_i, int b_j,
                       int c_i, int c_j) nogil:
    cdef int ab_i, ab_j, bc_i, bc_j, cross
    ab_i = b_i - a_i
    ab_j = b_j - a_j
    bc_i = c_i - b_i
    bc_j = c_j - b_j
    # note that x is j, y is i
    cross = (ab_j * bc_i - bc_j * ab_i)
    if cross > 0:
        return 1
    if cross < 0:
        return 0
    # special case for a U-turn at the end of the points
    return ((b_j > a_j) and (b_j > c_j))

@cython.boundscheck(False)
@cython.wraparound(False)
def convex_hull_ijv(in_labels_ijv,
                    indexes_in):
    # reorder by v, then j, then i.  Note: we will overwrite this array with
    # the output.  The sorting allocates a new copy, and it's guaranteed to be
    # large enough for the convex hulls.
    cdef np.ndarray[DTYPE_t, ndim=2] labels_ijv = in_labels_ijv.astype(np.int32)[np.lexsort(in_labels_ijv.T), :]
    assert np.all(labels_ijv >= 0), "All ijv values must be >=0"
    cdef np.ndarray[DTYPE_t, ndim=1] indexes = np.asarray(indexes_in, np.int32).ravel()
    assert np.all(indexes >= 0), "All indexes must be >= 0"
    # declaration of local variables
    cdef int num_indexes, max_i, max_j, max_label, pixidx, outidx, cur_req, cur_label
    cdef int start_j, cur_pix_i, cur_pix_j, end_j, start_idx, num_vertices
    cdef int need_last_upper_point
    cdef int num_emitted
    # an indirect sorting array for indexes
    cdef np.ndarray[DTYPE_t, ndim=1] indexes_reorder = np.argsort(indexes).astype(np.int32)
    num_indexes = len(indexes)
    # find the maximums
    max_i, max_j, max_label = labels_ijv.max(axis=0)
    # allocate the upper and lower vertex buffers, and initialize them to extreme values
    cdef np.ndarray[DTYPE_t, ndim=1] upper = np.empty(max_j + 1, np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] lower = np.empty(max_j + 1, np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] vertex_counts = np.zeros(num_indexes, np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] hull_offsets = np.zeros(num_indexes, np.int32)
    pixidx = 0  # the next input pixel we'll process
    outidx = 0  # the next row to be written in the output
    for cur_req in range(num_indexes):
        cur_label = indexes[indexes_reorder[cur_req]]
        while (cur_label <= max_label) and (labels_ijv[pixidx, 2] < cur_label):
            pixidx += 1
            if pixidx == labels_ijv.shape[0]:
                break
        hull_offsets[cur_req] = outidx
        if (pixidx == labels_ijv.shape[0]) or (cur_label != labels_ijv[pixidx, 2]):
            # cur_label's hull will have 0 points
            continue
        start_idx = pixidx
        while pixidx < labels_ijv.shape[0] and cur_label == labels_ijv[pixidx, 2]:
            pixidx += 1
        num_vertices = pixidx - start_idx
        start_j = labels_ijv[start_idx, 1]
        end_j = labels_ijv[pixidx - 1, 1]
        upper[start_j:(end_j+1)] = -1
        lower[start_j:(end_j+1)] = max_i + 1
        for start_idx <= pixidx < (start_idx + num_vertices):
            cur_pix_i = labels_ijv[pixidx, 0]
            cur_pix_j = labels_ijv[pixidx, 1]
            if upper[cur_pix_j] < cur_pix_i:
                upper[cur_pix_j] = cur_pix_i
            if lower[cur_pix_j] > cur_pix_i:
                lower[cur_pix_j] = cur_pix_i
        pixidx = start_idx + num_vertices
        if DEBUG:
            print("STARTJ/END_J", start_j, end_j)
            print("LOWER", lower[start_j:end_j + 1])
            print("UPPER", upper[start_j:end_j + 1])

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

        # Macro-type functions from Python version
        # def CONVEX(pt_i, pt_j, nemitted):
        #     # We're walking CCW, so left turns are convex
        #     d_i_prev, d_j_prev, z = out[out_base_idx + nemitted - 1, :] - out[out_base_idx + nemitted - 2, :]
        #     d_i_cur = pt_i - out[out_base_idx + nemitted - 1, 0]
        #     d_j_cur = pt_j - out[out_base_idx + nemitted - 1, 1]
        #     # note that x is j, y is i
        #     return (d_j_prev * d_i_cur - d_j_cur * d_i_prev) > 0
        #
        # def EMIT(pt_i, pt_j, nemitted):
        #     while (nemitted >= 2) and not CONVEX(pt_i, pt_j, nemitted):
        #         # The point we emitted just before this one created a
        #         # concavity (or is co-linear).  Prune it.
        #         #XXX print("BACKUP")
        #         nemitted -= 1
        #     # The point is convex or we haven't emitted enough points to check.
        #     #XXX print("writing point", nemitted, pt_i, pt_j)
        #     out[out_base_idx + nemitted, :] = (pt_i, pt_j, cur_label)
        #     return nemitted + 1

        need_last_upper_point = (lower[start_j] != upper[start_j])
        num_emitted = 0
        for envelope_j in range(start_j, end_j + 1):
            if lower[envelope_j] < max_i + 1:
                # MACRO EMIT(lower[envelope_j], envelope_j, num_emitted)
                while (num_emitted >= 2) and not CONVEX(labels_ijv[outidx + num_emitted - 2, 0], labels_ijv[outidx + num_emitted - 2, 1],
                                                        labels_ijv[outidx + num_emitted - 1, 0], labels_ijv[outidx + num_emitted - 1, 1],
                                                        lower[envelope_j], envelope_j):
                    # The point we emitted just before this one created a concavity (or is co-linear).  Prune it.
                    if DEBUG:
                        print("PRUNE")
                    num_emitted -= 1
                labels_ijv[outidx + num_emitted, 0] = lower[envelope_j]
                labels_ijv[outidx + num_emitted, 1] = envelope_j
                if DEBUG:
                    print("ADD", (lower[envelope_j], envelope_j, cur_label))
                num_emitted += 1
                # END MACRO
                lower[envelope_j] = max_i + 1
        for envelope_j in range(end_j, start_j, -1):
            if upper[envelope_j] > -1:
                # MACRO EMIT(upper[envelope_j], envelope_j, num_emitted)
                while (num_emitted >= 2) and not CONVEX(labels_ijv[outidx + num_emitted - 2, 0], labels_ijv[outidx + num_emitted - 2, 1],
                                                        labels_ijv[outidx + num_emitted - 1, 0], labels_ijv[outidx + num_emitted - 1, 1],
                                                        upper[envelope_j], envelope_j):
                    # The point we emitted just before this one created a concavity (or is co-linear).  Prune it.
                    if DEBUG:
                        print("PRUNE")
                    num_emitted -= 1
                if pixidx > outidx+num_emitted:
                    # The last point can get temporarily pushed on the list
                    # twice only to be taken off below. So this is more than
                    # just keeping a segfault from happening.
                    labels_ijv[outidx + num_emitted, 0] = upper[envelope_j]
                    labels_ijv[outidx + num_emitted, 1] = envelope_j
                    if DEBUG:
                        print("ADD",
                              (upper[envelope_j], envelope_j, cur_label))
                    num_emitted += 1
                # END MACRO
                upper[envelope_j] = -1
        # Even if we don't add the start point, we still might need to prune.
        # MACRO EMIT(upper[start_j], envelope_j, num_emitted)
        while (num_emitted >= 2) and not CONVEX(labels_ijv[outidx + num_emitted - 2, 0], labels_ijv[outidx + num_emitted - 2, 1],
                                                labels_ijv[outidx + num_emitted - 1, 0], labels_ijv[outidx + num_emitted - 1, 1],
                                                upper[start_j], start_j):
            # The point we emitted just before this one created a concavity (or is co-linear).  Prune it.
            if DEBUG:
                print("PRUNE")
            num_emitted -= 1
        if need_last_upper_point:
            labels_ijv[outidx + num_emitted, 0] = upper[start_j]
            labels_ijv[outidx + num_emitted, 1] = start_j
            if DEBUG:
                print("ADD", (upper[start_j], start_j, cur_label))
            num_emitted += 1
            # END MACRO
        upper[start_j] = -1
        # advance the output index
        vertex_counts[cur_req] = num_emitted
        outidx += num_emitted
    # sort by the requested order
    cdef np.ndarray[DTYPE_t, ndim=2] reordered = np.empty((outidx, 3), np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] reordered_counts = np.empty(num_indexes, np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] indexes_unreorder = np.argsort(indexes_reorder).astype(np.int32)
    cdef int reordered_idx, reordered_num, count, src_start, dest_start, tmpidx
    reordered_idx = 0
    if DEBUG:
        print("indexes", indexes)
        print("reorder", indexes_reorder)
        print("unreorder", indexes_unreorder)
        print("Hull offsets", hull_offsets)
    # Output in the order that the indices were passed in
    for reordered_num in range(num_indexes):
        cur_label = indexes[reordered_num]
        count = vertex_counts[indexes_unreorder[reordered_num]]
        src_start = hull_offsets[indexes_unreorder[reordered_num]]
        if DEBUG:
            print("writing index", cur_label, count, src_start)
        dest_start = reordered_idx
        for tmpidx in range(count):
            # Reorder columns to match what CellProfiler expects.
            reordered[dest_start + tmpidx, 0] = cur_label
            reordered[dest_start + tmpidx, 1] = labels_ijv[src_start + tmpidx, 0]
            reordered[dest_start + tmpidx, 2] = labels_ijv[src_start + tmpidx, 1]
        reordered_idx += count
        reordered_counts[reordered_num] = count
    return reordered, reordered_counts
