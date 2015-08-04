"""Utils.py - utility functions for manipulating the Matlab blackboard

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import numpy

def new_string_cell_array(shape):
    """Return a numpy.ndarray that looks like {NxM cell} to Matlab
    
    Return a numpy.ndarray that looks like {NxM cell} to Matlab.
    Each of the cells looks empty.
    shape - the shape of the array that's generated, e.g. (5,19) for a 5x19 cell array.
            Currently, this must be a 2-d shape.
    The object returned is a numpy.ndarray with dtype=dtype('object') and the given shape
    with each cell in the array filled with a numpy.ndarray with shape = (1,0) 
    and dtype=dtype('float64'). This appears to be the form that's created in matlab
    for this sort of object. 
    """
    result = numpy.ndarray(shape, dtype=numpy.dtype('object'))
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            result[i, j] = numpy.empty((0, 0))
    return result


def make_cell_struct_dtype(fields):
    """Makes the dtype of a struct composed of cells
    
    fields - the names of the fields in the struct
    """
    return numpy.dtype([(str(x), '|O4') for x in fields])
