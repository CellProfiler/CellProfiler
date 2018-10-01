from __future__ import absolute_import
import numpy as np

def rank_order(image, nbins=None):
    """Return an image of the same shape where each pixel has the
    rank-order value of the corresponding pixel in the image.
    The returned image's elements are of type np.uint32 which
    simplifies processing in C code.
    """
    flat_image = image.ravel()
    sort_order = flat_image.argsort().astype(np.uint32)
    flat_image = flat_image[sort_order]
    sort_rank  = np.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    np.cumsum(is_different, out=sort_rank[1:])
    original_values = np.zeros((sort_rank[-1]+1,),image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different] 
    int_image = np.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    if nbins is not None:
        max_ranked_data = np.max(int_image)
        while max_ranked_data >= nbins:
            #
            # Decimate the bins until there are fewer than nbins
            #
            hist = np.bincount(int_image)
            #
            # Rank the bins from lowest count to highest
            order = np.argsort(hist)
            #
            # find enough to maybe decimate to nbins
            #
            candidates = order[:max_ranked_data+2-nbins]
            to_delete = np.zeros(max_ranked_data+2, bool)
            to_delete[candidates] = True
            #
            # Choose candidates that are either not next to others
            # or have an even index so as not to delete adjacent bins
            #
            td_mask = to_delete[:-1] & (
                ((np.arange(max_ranked_data+1) & 2) == 0) |
                (~ to_delete[1:]))
            if td_mask[0]:
                td_mask[0] = False
            #
            # A value to be deleted has the same index as the following
            # value and the two end up being merged
            #
            rd_translation = np.cumsum(~td_mask)-1
            #
            # Translate the rankings to the new space
            #
            int_image = rd_translation[int_image]
            #
            # Eliminate the bins with low counts
            #
            original_values = original_values[~td_mask]
            max_ranked_data = len(original_values)-1
        
    return (int_image.reshape(image.shape), original_values)

    
