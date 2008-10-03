"""Utils.py - utility functions for manipulating the Matlab blackboard
    $Revision$
"""
import numpy

def NewStringCellArray(shape):
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
    result = numpy.ndarray(shape,dtype=numpy.dtype('object'))
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            result[i,j] = numpy.empty((0,0)) 
    return result

