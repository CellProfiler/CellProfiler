"""Propagate from a set of seed values

This module takes an image and a set of seed labels and assigns a label to
each pixel based on which label is closest to a pixel. Distances are
cumulative: given a path through pixels 1 to N, the distance to pixel N is
the distance from 1 to 2 plus the distance from 2 to 3 and so on. The distance
metric is the following:

given pixels at i1,j1 and i2,j2

height_diff = sum of height differences between nine neighbors of i1,j1 and
              nine neighbors of i2,j2

manhattan_distance = abs(i2-i1)+abs(j2-j1)

distance = sqrt(height_diff**2 + manhattan_distance * weight**2)

where "weight" is an input that weights the importance of image versus
pixel distance.

The priority queue operates using integer values. Doubles can be ordered
lexicographically by their bits - this means that the following
equation converts a double to two ints  that preserve order when 
lexicographically compared:

double x
int *pix = <int *>&x
if big_endian:
    i1 = pix[0]
    i2 = pix[1]-2147483648 # to make -1 into 0x7FFFFFFF
else:
    i1 = pix[1]
    i2 = pix[0]-2147483648

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import numpy as np

import _propagate

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
        raise ValueError("Image shape %s != label shape %s"%(repr(image.shape),repr(labels.shape)))
    if image.shape != mask.shape:
        raise ValueError("Image shape %s != mask shape %s"%(repr(image.shape),repr(mask.shape)))
    labels_out = np.zeros(labels.shape, np.int32)
    distances  = -np.ones(labels.shape,np.float64)
    distances[labels > 0] = 0
    labels_and_mask = np.logical_and(labels != 0, mask)
    coords = np.argwhere(labels_and_mask)
    i1,i2 = _propagate.convert_to_ints(0.0)
    ncoords = coords.shape[0]
    pq = np.column_stack((np.ones((ncoords,),int) * i1,
                             np.ones((ncoords,),int) * i2,
                             labels[labels_and_mask],
                             coords))
    _propagate.propagate(np.ascontiguousarray(image,np.float64),
                         np.ascontiguousarray(pq,np.int32),
                         np.ascontiguousarray(mask,np.int8),
                         labels_out, distances, float(weight))
    labels_out[labels > 0] = labels[labels > 0]
    return labels_out,distances
    
