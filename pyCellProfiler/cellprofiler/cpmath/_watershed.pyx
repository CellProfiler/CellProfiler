"""watershed.pyx - scithon implementation of guts of watershed
"""

import numpy as np
cimport numpy as np
cimport cython

DTYPE_INT32 = np.int32
ctypedef np.int32_t DTYPE_INT32_t
DTYPE_BOOL = np.bool
ctypedef np.int8_t DTYPE_BOOL_t

include "heap.pxi"

@cython.boundscheck(False)
def watershed(np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False, mode='c'] image,
              np.ndarray[DTYPE_INT32_t,ndim=2,negative_indices=False, mode='c'] pq,
              DTYPE_INT32_t age,
              np.ndarray[DTYPE_INT32_t,ndim=2,negative_indices=False, mode='c'] structure,
              DTYPE_INT32_t ndim,
              np.ndarray[DTYPE_BOOL_t,ndim=1,negative_indices=False, mode='c'] mask,
              np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False, mode='c'] image_shape,
              np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False, mode='c'] output):
    """Do heavy lifting of watershed algorithm
    
    image - the flattened image pixels, converted to rank-order
    pq    - the priority queue, starts with the marked pixels
            the first element in each row is the image intensity
            the second element is the age at entry into the queue
            the third element is the index into the flattened image or labels
            the remaining elements are the coordinates of the point
    age   - the next age to assign to a pixel
    structure - a numpy int32 array containing the structuring elements
                that define nearest neighbors. For each row, the first
                element is the stride from the point to its neighbor
                in a flattened array. The remaining elements are the
                offsets from the point to its neighbor in the various
                dimensions
    ndim  - # of dimensions in the image
    mask  - numpy boolean (char) array indicating which pixels to consider
            and which to ignore. Also flattened.
    image_shape - the dimensions of the image, for boundary checking,
                  a numpy array of np.int32
    output - put the image labels in here
    """
    cdef Heap *hp = <Heap *> heap_from_numpy2(pq)

    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False, mode='c'] elem = np.zeros((hp.width,),dtype=np.int32)
    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False, mode='c'] new_elem = np.zeros((hp.width,),dtype=np.int32)
    cdef DTYPE_INT32_t nneighbors = structure.shape[0] 
    cdef DTYPE_INT32_t i = 0
    cdef DTYPE_INT32_t j = 0
    cdef DTYPE_INT32_t ok
    cdef DTYPE_INT32_t coord = 0
    cdef DTYPE_INT32_t index = 0
    cdef DTYPE_INT32_t old_index = 0
    cdef DTYPE_INT32_t max_index = image.shape[0]
    cdef DTYPE_INT32_t old_output

    while hp.items > 0:
        #
        # Pop off an item to work on
        #
        heappop(hp, <np.int32_t *> elem.data)
        ####################################################
        # loop through each of the structuring elements
        #
        old_index = elem[2]
        old_output = output[old_index]
        for i in range(nneighbors):
            # get the flattened address of the neighbor
            index = structure[i,0]+old_index
            if index < 0 or index >= max_index or output[index] or not mask[index]:
                continue
            # Fill in and push the neighbor
            ok = 1
            for j in range(ndim):
                # the coordinate is offset by 3 (value, age and index come 
                # first) in the priority queue and by 1 (stride comes first)
                # in the structure
                coord = elem[j+3]+structure[i,j+1]
                if coord < 0 or coord >= image_shape[j]:
                    ok = 0
                    break
                new_elem[j+3] = coord
            if ok == 0:
                continue
            new_elem[0]   = image[index]
            new_elem[1]   = age
            new_elem[2]   = index
            age          += 1
            output[index] = old_output
            #
            # Push the neighbor onto the heap to work on it later
            #
            heappush(hp, <np.int32_t *>new_elem.data)
    heap_done(hp)
