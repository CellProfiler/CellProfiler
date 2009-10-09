'''_cpmorphology2.pyx - support routines for cpmorphology in Cython

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

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

import_array()

@cython.boundscheck(False)
def skeletonize_loop(np.ndarray[dtype=np.uint8_t, ndim=2, 
                                negative_indices=False, mode='c'] result,
                     np.ndarray[dtype=np.int32_t, ndim=1,
                                negative_indices=False, mode='c'] i,
                     np.ndarray[dtype=np.int32_t, ndim=1,
                                negative_indices=False, mode='c'] j,
                     np.ndarray[dtype=np.int32_t, ndim=1,
                                negative_indices=False, mode='c'] order,
                     np.ndarray[dtype=np.uint8_t, ndim=1,
                                negative_indices=False, mode='c'] table):
    '''Inner loop of skeletonize function

    result - on input, the image to be skeletonized, on output the skeletonized
             image
    i,j    - the coordinates of each foreground pixel in the image
    order  - the index of each pixel, in the order of processing
    table  - the 512-element lookup table of values after transformation

    The loop determines whether each pixel in the image can be removed without
    changing the Euler number of the image. The pixels are ordered by
    increasing distance from the background which means a point nearer to
    the quench-line of the brushfire will be evaluated later than a
    point closer to the edge.
    '''
    cdef:
        np.int32_t accumulator
        np.int32_t index,order_index
        np.int32_t ii,jj

    for index in range(order.shape[0]):
        accumulator = 16
        order_index = order[index]
        ii = i[order_index]
        jj = j[order_index]
        if ii > 0:
            if jj > 0 and result[ii-1,jj-1]:
                accumulator += 1
            if result[ii-1,jj]:
                accumulator += 2
            if jj < result.shape[1]-1 and result[ii-1,jj+1]:
                    accumulator += 4
            if jj > 0 and result[ii,jj-1]:
                accumulator += 8
            if jj < result.shape[1]-1 and result[ii,jj+1]:
                accumulator += 32
            if ii < result.shape[0]-1:
                if jj > 0 and result[ii+1,jj-1]:
                    accumulator += 64
                if result[ii+1,jj]:
                    accumulator += 128
                if jj < result.shape[1]-1 and result[ii+1,jj+1]:
                    accumulator += 256
            result[ii,jj] = table[accumulator]

@cython.boundscheck(False)
def table_lookup_index(np.ndarray[dtype=np.uint8_t, ndim=2,
                                  negative_indices=False, mode='c'] image):
    '''Return an index into a table per pixel of a binary image

    Take the sum of true neighborhood pixel values where the neighborhood
    looks like this:
     1   2   4
     8  16  32
    64 128 256

    This code could be replaced by a convolution with the kernel:
    256 128 64
     32  16  8
      4   2  1
    but this runs about twice as fast because of inlining and the
    hardwired kernel.
    '''
    cdef:
        np.ndarray[dtype=np.int32_t, ndim=2, 
                   negative_indices=False, mode='c'] indexer
        np.int32_t *p_indexer
        np.uint8_t *p_image
        np.int32_t i_stride
        np.int32_t i_shape
        np.int32_t j_shape
        np.int32_t i
        np.int32_t j
        np.int32_t offset

    i_shape   = image.shape[0]
    j_shape   = image.shape[1]
    indexer = np.zeros((i_shape,j_shape),np.int32)
    p_indexer = <np.int32_t *>indexer.data
    p_image   = <np.uint8_t *>image.data
    i_stride  = image.strides[0]
    assert i_shape >= 3 and j_shape >= 3, "Please use the slow method for arrays < 3x3"

    for i in range(1,i_shape-1):
        offset = i_stride*i+1
        for j in range(1,j_shape-1):
            if p_image[offset]:
                p_indexer[offset+i_stride+1] += 1
                p_indexer[offset+i_stride] += 2
                p_indexer[offset+i_stride-1] += 4
                p_indexer[offset+1] += 8
                p_indexer[offset] += 16
                p_indexer[offset-1] += 32
                p_indexer[offset-i_stride+1] += 64
                p_indexer[offset-i_stride] += 128
                p_indexer[offset-i_stride-1] += 256
            offset += 1
    #
    # Do the corner cases (literally)
    #
    if image[0,0]:
        indexer[0,0] += 16
        indexer[0,1] += 8
        indexer[1,0] += 2
        indexer[1,1] += 1

    if image[0,j_shape-1]:
        indexer[0,j_shape-2] += 32
        indexer[0,j_shape-1] += 16
        indexer[1,j_shape-2] += 4
        indexer[1,j_shape-1] += 2

    if image[i_shape-1,0]:
        indexer[i_shape-2,0] += 128
        indexer[i_shape-2,1] += 64
        indexer[i_shape-1,0] += 16
        indexer[i_shape-1,1] += 8

    if image[i_shape-1,j_shape-1]:
        indexer[i_shape-2,j_shape-2] += 256
        indexer[i_shape-2,j_shape-1] += 128
        indexer[i_shape-1,j_shape-2] += 32
        indexer[i_shape-1,j_shape-1] += 16
    #
    # Do the edges
    #
    for j in range(1,j_shape-1):
        if image[0,j]:
            indexer[0,j-1] += 32
            indexer[0,j]   += 16
            indexer[0,j+1] += 8
            indexer[1,j-1] += 4
            indexer[1,j]   += 2
            indexer[1,j+1] += 1
        if image[i_shape-1,j]:
            indexer[i_shape-2,j-1] += 256
            indexer[i_shape-2,j]   += 128
            indexer[i_shape-2,j+1] += 64
            indexer[i_shape-1,j-1] += 32
            indexer[i_shape-1,j]   += 16
            indexer[i_shape-1,j+1] += 8

    for i in range(1,i_shape-1):
        if image[i,0]:
            indexer[i-1,0] += 128
            indexer[i,0]   += 16
            indexer[i+1,0] += 2
            indexer[i-1,1] += 64
            indexer[i,1]   += 8
            indexer[i+1,1] += 1
        if image[i,j_shape-1]:
            indexer[i-1,j_shape-2] += 256
            indexer[i,j_shape-2]   += 32
            indexer[i+1,j_shape-2] += 4
            indexer[i-1,j_shape-1] += 128
            indexer[i,j_shape-1]   += 16
            indexer[i+1,j_shape-1] += 2
    return indexer

@cython.boundscheck(False)
def grey_reconstruction_loop(np.ndarray[dtype=np.uint32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] avalues,
                             np.ndarray[dtype=np.int32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] aprev,
                             np.ndarray[dtype=np.int32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] anext,
                             np.ndarray[dtype=np.int32_t, ndim=1,
                                        negative_indices = False,
                                        mode = 'c'] astrides,
                             np.int32_t current,
                             int image_stride):
    '''The inner loop for grey_reconstruction'''
    cdef:
        np.int32_t neighbor
        np.uint32_t neighbor_value
        np.uint32_t current_value
        np.uint32_t mask_value
        np.int32_t link
        int i
        np.int32_t nprev
        np.int32_t nnext
        int nstrides = astrides.shape[0]
        np.uint32_t *values = <np.uint32_t *>(avalues.data)
        np.int32_t *prev = <np.int32_t *>(aprev.data)
        np.int32_t *next = <np.int32_t *>(anext.data)
        np.int32_t *strides = <np.int32_t *>(astrides.data)
    
    while current != -1:
        if current < image_stride:
            current_value = values[current]
            if current_value == 0:
                break
            for i in range(nstrides):
                neighbor = current + strides[i]
                neighbor_value = values[neighbor]
                # Only do neighbors less than the current value
                if neighbor_value < current_value:
                    mask_value = values[neighbor + image_stride]
                    # Only do neighbors less than the mask value
                    if neighbor_value < mask_value:
                        # Raise the neighbor to the mask value if
                        # the mask is less than current
                        if mask_value < current_value:
                            link = neighbor + image_stride
                            values[neighbor] = mask_value
                        else:
                            link = current
                            values[neighbor] = current_value
                        # unlink the neighbor
                        nprev = prev[neighbor]
                        nnext = next[neighbor]
                        next[nprev] = nnext
                        if nnext != -1:
                            prev[nnext] = nprev
                        # link the neighbor after the link
                        nnext = next[link]
                        next[neighbor] = nnext
                        prev[neighbor] = link
                        if nnext >= 0:
                            prev[nnext] = neighbor
                            next[link] = neighbor
        current = next[current]
