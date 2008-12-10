"""Utils.py - utility functions for manipulating the Matlab blackboard
    $Revision$
"""
import numpy
import tempfile
import scipy.io.matlab.mio
import os

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

def MakeCellStructDType(fields):
    """Makes the dtype of a struct composed of cells
    
    fields - the names of the fields in the struct
    """
    return numpy.dtype([(x,'|O4') for x in fields])

def GetIntFromMatlab(proxy):
    """Given a proxy to an integer in Matlab, return the integer
    """
    return int(GetMatlabInstance().num2str(proxy))

def LoadIntoMatlab(handles):
    """Return a proxy object which is the data structure passed, loaded into Matlab
    """
    (matfd,matpath) = tempfile.mkstemp('.mat')
    matfh = os.fdopen(matfd,'wb')
    closed = False
    try:
        scipy.io.matlab.mio.savemat(matfh,handles,format='5',long_field_names=True)
        matfh.close()
        closed = True
        matlab = GetMatlabInstance()
        return matlab.load(matpath)
    finally:
        if not closed:
            matfh.close()
        os.unlink(matpath)

   
def GetMatlabInstance():
    global __MATLAB
    if not globals().has_key('__MATLAB'):
        import mlabwrap
        __MATLAB = mlabwrap.mlab
        try:
            __MATLAB.desktop()
        except:
            pass
    return __MATLAB

def GCellFun():
    return GetMatlabInstance().eval('@(cell,x) cell{x+1}')

def SCellFun():
    return GetMatlabInstance().eval('@(cell,x,value) {cell{1:x},value,cell{x+2:end}}')
