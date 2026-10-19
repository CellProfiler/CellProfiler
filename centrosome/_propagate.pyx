import numpy as np
import struct
cimport numpy as np
np.import_array()
cimport cython
cdef extern from "stdlib.h":
    double sqrt(double) nogil

DTYPE_INT32 = np.int32
ctypedef np.int32_t DTYPE_INT32_t
DTYPE_DOUBLE = np.float64
ctypedef np.float64_t DTYPE_DOUBLE_t
DTYPE_BOOL = np.int8
ctypedef np.int8_t DTYPE_BOOL_t

from centrosome.heap cimport *

cdef int little_endian_flag

cdef inline void set_little_endian_flag():
    global little_endian_flag
    x = struct.pack("BBBB",1,0,0,0)
    little_endian_flag = (struct.unpack("I",x)[0]==1 and 1) or -1

###############################################################
#
# Get the most significant signed integer from a double
#
# value - double to be broken into two integers
#
# returns - most significant part of double (first 32 bits)
###############################################################
cdef inline int get_most_significant(double value) nogil:
    cdef unsigned int *pValue = <unsigned int *>&value
    cdef unsigned int ivalue
    if little_endian_flag==1:
        ivalue = pValue[1]
    else:
        ivalue = pValue[0]
    if ivalue & 0x80000000u:
        return -<int>(ivalue & 0x7FFFFFFF)
    else:
        return <int>(ivalue & 0x7FFFFFFF)

##############################################################
#
# Get the least significant signed integer from a double
#
# value - double to be broken into two integers
#
# returns - least significant part of double (second 31 bits)
#
##############################################################

cdef inline int get_least_significant(double value) nogil:
    cdef unsigned int *pValue = <unsigned int *>&value
    cdef unsigned int ivalue1
    cdef unsigned int ivalue2
    if little_endian_flag==1:
        ivalue1=pValue[1]
        ivalue2=pValue[0]
    else:
        ivalue1=pValue[0]
        ivalue2=pValue[1]
    if ivalue1 & 0x80000000u:
        # the number is negative and the sign is reversed here
        return -<int>(ivalue2 >> 1)
    else:
        return <int>(ivalue2 >> 1)

##############################################################
#
# convert_to_ints - return a tuple of (most significant int, least significant int)
#                   given a value
##############################################################
def convert_to_ints(double value):
    return (get_most_significant(value),get_least_significant(value))

##############################################################
#
# clamped_fetch - fetch an image pixel with special handling of 
#                 pixels outside of the boundaries
#
#                 image - get pixel value from this image
#                 i,j   - coordinates of pixel
#                 m,n   - extent of image
#
# If i or j is negative, fetch the value at zero, similarly at right/bottom
# boundary, otherwise fetch the pixel at i,j
###############################################################
cdef inline double clamped_fetch(double *image,
                                 int i,
                                 int j,
                                 int m,
                                 int n) nogil:
    if i<0:
       i=0
    elif i>=m:
       i=m-1
    if j<0:
       j=0
    elif j>=n:
       j=n-1
    return image[i*n+j]

################################################################
#
# distance - return the algorithm's computation of the distance
#            between pixels i1,j1 and i2,j2
#
################################################################
cdef inline double distance(double *image,
                            int i1,
                            int j1,
                            int i2,
                            int j2,
                            int m,
                            int n,
                            double weight) nogil:
    cdef int delta_i
    cdef int delta_j
    cdef double pixel_diff = 0
    cdef double v1
    cdef double v2
    cdef double manhattan_distance

    for delta_i from -1 <= delta_i <= 1:
        for delta_j from -1 <= delta_j <= 1:
            v1 = clamped_fetch(image,i1+delta_i,j1+delta_j,m,n)
            v2 = clamped_fetch(image,i2+delta_i,j2+delta_j,m,n)
            if v1 > v2:
                pixel_diff += v1-v2
            else:
                pixel_diff += v2-v1
    if i1 > i2:
        manhattan_distance = i1-i2
    else:
        manhattan_distance = i2-i1
    if j1 > j2:
        manhattan_distance += j1-j2
    else:
        manhattan_distance += j2-j1
    return sqrt(pixel_diff*pixel_diff + manhattan_distance*weight*weight)

##############################################################
#
# propagate - perform the propagate algorithm given an image
#             and a priority queue of seed labels
#
##############################################################

@cython.boundscheck(False)
def propagate(np.ndarray[DTYPE_DOUBLE_t,ndim=2,negative_indices=False, mode='c'] image,
              np.ndarray[DTYPE_INT32_t,ndim=2,negative_indices=False,mode='c'] pq,
              np.ndarray[DTYPE_BOOL_t,ndim=2,negative_indices=False,mode='c'] mask,
              np.ndarray[DTYPE_INT32_t,ndim=2,negative_indices=False,mode='c'] labels,
              np.ndarray[DTYPE_DOUBLE_t,ndim=2,negative_indices=False,mode='c'] distances,
              DTYPE_DOUBLE_t weight):
    """Perform the propagate algorithm

    image - the image that provides the "heights" for the distance computation
    pq - the initial priority queue that marks the pixels. The first two elements
         hold the distance score. The remaining two elements hold the pixel coordinates.
    labels - the label array to be created
    weight - the relative weighting of the image vs the manhattan distance.
             Higher means more weight to the image
    """
    cdef Heap *hp = <Heap *> heap_from_numpy2(pq)
    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False,mode='c'] elem = np.zeros((5,),dtype=np.int32)
    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False,mode='c'] new_elem = np.zeros((5,),dtype=np.int32)
    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False,mode='c'] delta_i = np.array((-1,-1,-1, 0, 0, 1,1,1),dtype=np.int32)
    cdef np.ndarray[DTYPE_INT32_t,ndim=1,negative_indices=False,mode='c'] delta_j = np.array((-1, 0, 1,-1, 1,-1,0,1),dtype=np.int32)
    cdef DTYPE_INT32_t i1=0
    cdef DTYPE_INT32_t j1=0
    cdef DTYPE_INT32_t idx=0
    cdef DTYPE_INT32_t label=0
    cdef DTYPE_INT32_t i2=0
    cdef DTYPE_INT32_t j2=0
    cdef DTYPE_INT32_t m = image.shape[0]
    cdef DTYPE_INT32_t n = image.shape[1]
    cdef DTYPE_DOUBLE_t d, d0

    set_little_endian_flag()

    with nogil:
        while hp.items > 0:
            heappop(hp, <np.int32_t *> elem.data)
            i1 = elem[3]
            j1 = elem[4]
            if labels[i1,j1] == 0:
                #
                # i,j has not yet been done. 
                #
                label=elem[2]
                labels[i1,j1]=label
                d0 = distances[i1,j1]
                #
                # For each 8-connected neighbor, push
                #
                
                for idx from 0 <= idx < 8:
                    i2 = i1+delta_i[idx]
                    j2 = j1+delta_j[idx]
                    if i2 < 0 or i2 >= m or j2 < 0 or j2 >= n:
                        continue
                    if labels[i2,j2] > 0:
                        continue
                    if not mask[i2,j2]:
                        continue
                    d = distance(<double *>(image.data), i1, j1, i2, j2, m, n, weight)+d0
                    if distances[i2,j2] == -1 or distances[i2,j2] > d:
                        # push the point if no distance recorded or ours is the best
                        distances[i2,j2] = d
                        new_elem[0] = get_most_significant(d)
                        new_elem[1] = get_least_significant(d)
                        new_elem[2] = label
                        new_elem[3] = i2
                        new_elem[4] = j2
                        heappush(hp, <np.int32_t *>new_elem.data)
    heap_done(hp)
