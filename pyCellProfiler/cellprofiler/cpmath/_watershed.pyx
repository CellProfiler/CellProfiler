"""watershed.pyx - scithon implementation of guts of watershed
"""

import numpy as np
cimport numpy as np
cimport cython

DTYPE_INT32 = np.int32
ctypedef np.int32_t DTYPE_INT32_t
DTYPE_BOOL = np.bool
ctypedef np.int8_t DTYPE_BOOL_t

######################################################
# heappop - pop an element off of the heap
#
# pq_data -  pointer to the priority queue ndarray
#            first element is the pixel "value"
#            second is its age on the queue
# pq_width - # of elements in a data record
# items    - # of records on the queue
#
# Returns the new # of items
#
# Algorithm taken from Introduction to Algorithms, Cormen, p 150
# and (heapify(A,1)) p 143
#
######################################################
cdef inline int heappop(unsigned int *pq_data,
                        unsigned int pq_width,
                        unsigned int items,
                        unsigned int *dest):
    cdef unsigned int k,l,r,i,temp,smallest
    cdef unsigned int src_idx,dest_idx
    cdef unsigned int i_idx,l_idx,r_idx,smallest_idx,items_idx
    cdef unsigned int value1,value2
    #
    # Start by copying the first element to the destination
    #
    for k in range(pq_width):
        dest[k] = pq_data[k]
    items -= 1
    if items==0:
        return 0
    items_idx = items*pq_width
    #
    # Then copy the last element to the first
    #
    for k in range(pq_width):
        pq_data[k] = pq_data[items_idx+k]
    #
    # Bubble the first element up using the heapify function
    #
    i = 0
    i_idx = 0
    smallest_idx = 0
    smallest = 0
    while True:
            l = i*2+1 #__left(i)
            r = i*2+2 #__right(i)
            value1 = pq_data[i_idx]
            if l < items:
                l_idx = l*pq_width
                value2 = pq_data[l_idx]
                if value1 == value2:
                    if pq_data[i_idx+1] > pq_data[l_idx+1]:
                        smallest = l
                        smallest_idx = l_idx
                elif value1 > value2:
                    smallest = l
                    smallest_idx = l_idx
                    value1 = value2
            if r < items:
                r_idx = r*pq_width
                value2 = pq_data[r_idx]
                if value1 == value2:
                    if pq_data[smallest_idx+1] > pq_data[r_idx+1]:
                        smallest = r
                        smallest_idx = r_idx
                elif value1 > value2:
                    smallest = r
                    smallest_idx = r_idx
            if smallest == i:
                break
            for k in range(pq_width):
                temp = pq_data[i_idx+k]
                pq_data[i_idx+k] = pq_data[smallest_idx+k]
                pq_data[smallest_idx+k] = temp
            i = smallest
            i_idx = smallest_idx
    return items
        
##################################################
# heappush - inlined
#
# push the element onto the heap, maintaining the heap invariant
#
# Algorithm taken from Introduction to Algorithms, Cormen, p 150
# We use zero-indexing while the algorithm uses 1-indexing.
# Also, the top element in the heap is the least, not the greatest.
#
##################################################
cdef inline int heappush(unsigned int *pq_data,
                         unsigned int items,
                         unsigned int pq_width,
                         unsigned int *new_elem):
  cdef unsigned int ii         = items
  cdef unsigned int elem_value = new_elem[0]
  cdef unsigned int elem_age   = new_elem[1]
  cdef unsigned int ii_idx     = items * pq_width
  cdef unsigned int parent_idx
  cdef unsigned int parent
  cdef unsigned int parent_value
  cdef unsigned int k
  while ii>0:
      parent     = (ii+1)/2 - 1 # __parent(i)
      parent_idx = parent * pq_width
      parent_value = pq_data[parent_idx]
      if parent_value <= elem_value:
          break
      elif parent_value == elem_value and pq_data[parent_idx+1] <= elem_age:
          break
      for k in range(pq_width):
          pq_data[ii_idx+k] = pq_data[parent_idx+k]
      ii     = parent
      ii_idx = parent_idx
  for k in range(pq_width):
      pq_data[ii_idx+k] = new_elem[k]
  return items + 1

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
    cdef DTYPE_INT32_t items = pq.shape[0] # of items in the queue
    cdef DTYPE_INT32_t pq_width = pq.shape[1]
    cdef DTYPE_INT32_t pq_stride = pq.strides[0]
    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False, mode='c'] elem = np.zeros((pq_width,),dtype=np.int32)
    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False, mode='c'] new_elem = np.zeros((pq_width,),dtype=np.int32)
    cdef DTYPE_INT32_t nneighbors = structure.shape[0] 
    cdef DTYPE_INT32_t i = 0
    cdef DTYPE_INT32_t j = 0
    cdef DTYPE_INT32_t coord = 0
    cdef DTYPE_INT32_t index = 0
    cdef DTYPE_INT32_t old_index = 0
    cdef DTYPE_INT32_t ok
    cdef DTYPE_INT32_t max_index = image.shape[0]
    while items > 0:
        #
        # Pop off an item to work on
        #
        items = heappop(<unsigned int *>pq.data,
                        <unsigned int>pq_width,
                        <unsigned int>items,
                        <unsigned int *>elem.data)
        ####################################################
        # loop through each of the structuring elements
        #
        for i in range(nneighbors):
            ok = 1
            old_index = elem[2]
            index = structure[i,0]+old_index
            if index < 0 or index >= max_index:
                continue
            if output[index] != 0:
                continue
            # Fill in and push the neighbor
            if not mask[index]:
                continue
            for j in range(ndim):
                # the coordinate is offset by 2 (value, age and index come 
                # first) in the priority queue and by 1 (stride comes first)
                # in the structure
                coord = elem[j+3]+structure[i,j+1]
                if coord < 0 or coord >= image_shape[j]:
                    ok = 0
                    break
                new_elem[j+3] = coord
            if ok == 0:
                continue
            # get the flattened address of the neighbor
            new_elem[0]   = image[index]
            new_elem[1]   = age
            new_elem[2]   = index
            age          += 1
            output[index] = output[old_index]
            # First, ensure that our heap has room for the next item
            if pq.shape[0] == items:
                new_array = np.ndarray((max(pq.shape[0]*3/2,1000),
                                        pq.shape[1]),dtype=pq.dtype)
                new_array[:items,:] = pq[:,:]
                pq = new_array
            #
            # Push the pixel onto the heap to work on it later
            #
            items = heappush(<unsigned int *>pq.data,
                             items,
                             pq_width,
                             <unsigned int *>new_elem.data)
