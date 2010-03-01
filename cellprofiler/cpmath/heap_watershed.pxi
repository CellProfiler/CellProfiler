import numpy as np
cimport numpy as np
cimport cython

cdef struct Heapitem:
    np.int32_t value
    np.int32_t age
    np.int32_t index

cdef inline int smaller(Heapitem *a, Heapitem *b):
    if a.value <> b.value:
      return a.value < b.value
    return a.age < b.age

include "heap_general.pxi"
