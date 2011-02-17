import numpy as np
cimport numpy as np
cimport cython

cdef extern from "Python.h":
    ctypedef int Py_intptr_t

cdef extern from "numpy/arrayobject.h":
    ctypedef class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef Py_intptr_t *dimensions
        cdef Py_intptr_t *strides
    cdef void import_array()
    cdef int  PyArray_ITEMSIZE(np.ndarray)

cdef extern from "stdint.h":
    ctypedef char int8_t
    ctypedef unsigned char uint8_t
    ctypedef short int16_t
    ctypedef unsigned short uint16_t
    ctypedef int int32_t
    ctypedef unsigned int uint32_t
    ctypedef long long int64_t
    ctypedef unsigned long long uint64_t

cdef extern from "memory.h":
    void *memset(void *, int, unsigned int)
    
cdef extern from "mexLap.h":
    double lap(int n, double *cc, int *kk, int *first, int *x, int *y, double *u, double *v)
    
def LAP(np.ndarray[dtype=np.int32_t, ndim=1,
                    negative_indices = False,
                    mode = 'c'] kk,
        np.ndarray[dtype=np.int32_t, ndim=1,
                    negative_indices = False,
                    mode = 'c'] first, 
        np.ndarray[dtype=np.float_t, ndim=1,
                    negative_indices = False,
                    mode = 'c'] cc, n):
    cdef:
        np.ndarray[dtype=np.int32_t, ndim=1,
                    negative_indices = False,
                    mode = 'c'] x
        np.ndarray[dtype=np.int32_t, ndim=1,
                    negative_indices = False,
                    mode = 'c'] y
        np.ndarray[dtype=np.float_t, ndim=1,
                    negative_indices = False,
                    mode = 'c'] u
        np.ndarray[dtype=np.float_t, ndim=1,
                    negative_indices = False,
                    mode = 'c'] v
        double *pcc
        int *pkk
        int *pfirst
        int *px
        int *py
        double *pu
        double *pv
    pkk = <int *>(kk.data)
    pfirst = <int *>(first.data)
    pcc = <double *>(cc.data)
    x = np.zeros(first.size, np.int32)
    px = <int *>(x.data)
    y = np.zeros(first.size, np.int32)
    py = <int *>(y.data)
    u = np.zeros(first.size, np.float)
    pu = <double *>(u.data)
    v = np.zeros(first.size, np.float)
    pv = <double *>(v.data)
    d = np.zeros(first.size, np.int32)
    d =  lap(n, pcc, pkk, pfirst, px, py, pu, pv)
    print d
    return x, y
