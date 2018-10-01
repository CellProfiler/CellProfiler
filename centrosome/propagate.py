from __future__ import absolute_import
import numpy as np

from . import _propagate

def propagate(image, labels, mask, weight):
    """Propagate the labels to the nearest pixels
    
    image - gives the Z height when computing distance
    labels - the labeled image pixels
    mask   - only label pixels within the mask
    weight - the weighting of x/y distance vs z distance
             high numbers favor x/y, low favor z
    
    returns a label matrix and the computed distances
    """
    if image.shape != labels.shape:
        raise ValueError("Image shape %s != label shape %s" % (repr(image.shape), repr(labels.shape)))
    if image.shape != mask.shape:
        raise ValueError("Image shape %s != mask shape %s" % (repr(image.shape), repr(mask.shape)))
    labels_out = np.zeros(labels.shape, np.int32)
    distances = -np.ones(labels.shape, np.float64)
    distances[labels > 0] = 0
    labels_and_mask = np.logical_and(labels != 0, mask)
    coords = np.argwhere(labels_and_mask)
    i1,i2 = _propagate.convert_to_ints(0.0)
    ncoords = coords.shape[0]
    pq = np.column_stack((np.ones((ncoords,), int) * i1, np.ones((ncoords,),int) * i2, labels[labels_and_mask], coords))
    _propagate.propagate(np.ascontiguousarray(image, np.float64), np.ascontiguousarray(pq,np.int32), np.ascontiguousarray(mask, np.int8), labels_out, distances, float(weight))
    labels_out[labels > 0] = labels[labels > 0]
    return labels_out, distances
