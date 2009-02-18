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

"""

import numpy

import _propagate

def propagate(image, labels, weight):
    labels_out = numpy.zeros(labels.shape,numpy.int32)
    coords = numpy.argwhere(labels != 0)
    i1,i2 = _propagate.convert_to_ints(0.0)
    ncoords = coords.shape[0]
    pq = numpy.column_stack((numpy.ones((ncoords,),int) * i1,
                             numpy.ones((ncoords,),int) * i2,
                             labels[labels != 0],
                             coords))
    _propagate.propagate(numpy.ascontiguousarray(image,numpy.float64),
                         numpy.ascontiguousarray(pq,numpy.int32),
                         labels_out, float(weight))
    return labels_out
    
