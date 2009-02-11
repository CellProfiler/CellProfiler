import numpy as np
cimport numpy as np
cimport cython
from sys import stderr

cdef extern from "stdlib.h":
   ctypedef unsigned long size_t
   void free(void *ptr)
   void *malloc(size_t size)
   void *realloc(void *ptr, size_t size)

cdef struct Heap:
    unsigned int items
    unsigned int width
    unsigned int space
    np.int32_t *data


cdef inline Heap *heap_from_numpy2(object np_heap):
    cdef unsigned int k
    cdef Heap *heap 
    heap = <Heap *> malloc(sizeof (Heap))
    heap.items = np_heap.shape[0]
    heap.width = np_heap.shape[1]
    heap.space = heap.items
    heap.data = <np.int32_t *> malloc(heap.space * heap.width * sizeof(np.int32_t))
    tmp = np_heap.astype(np.int32).flatten('C')
    for k in range(heap.items * heap.width):
        heap.data[k] = <np.int32_t> tmp[k]
    return heap

cdef inline void heap_done(Heap *heap):
   free(heap.data)
   free(heap)

cdef inline int smaller(unsigned int a, unsigned int b, Heap *h):
    cdef unsigned int k
    cdef np.int32_t *ap = h.data + a * h.width
    cdef np.int32_t *bp = h.data + b * h.width
    if ap[0] == bp[0]:
        for k in range(1, h.width):
            if ap[k] == bp[k]:
                continue
            if ap[k] < bp[k]:
                return 1
            break
        return 0
    elif ap[0] < bp[0]:
       return 1
    return 0

cdef inline void swap(unsigned int a, unsigned int b, Heap *h):
    cdef unsigned int k
    cdef np.int32_t *ap = h.data + a * h.width
    cdef np.int32_t *bp = h.data + b * h.width
    for k in range(h.width):
        ap[k], bp[k] = bp[k], ap[k]



######################################################
# heappop - inlined
#
# pop an element off the heap, maintaining heap invariant
# 
# Note: heap ordering is the same as python heapq, i.e., smallest first.
######################################################
cdef inline void heappop(Heap *heap,
                  unsigned int *dest):
    cdef unsigned int i, smallest, l, r # heap indices
    cdef unsigned int k
    
    #
    # Start by copying the first element to the destination
    #
    for k in range(heap.width):
        dest[k] = heap.data[k]
    heap.items -= 1

    # if the heap is now empty, we can return, no need to fix heap.
    if heap.items == 0:
        return

    #
    # Move the last element in the heap to the first.
    #
    swap(0, heap.items, heap)

    #
    # Restore the heap invariant.
    #
    i = 0
    smallest = i
    while True:
        # loop invariant here: smallest == i
        
        # find smallest of (i, l, r), and swap it to i's position if necessary
        l = i*2+1 #__left(i)
        r = i*2+2 #__right(i)
        if l < heap.items:
            if smaller(l, i, heap):
                smallest = l
        if r < heap.items:
            if smaller(r, smallest, heap):
                smallest = r
        # the element at i is smaller than either of its children, heap invariant restored.
        if smallest == i:
                break
        # swap
        swap(i, smallest, heap)
        i = smallest
        
##################################################
# heappush - inlined
#
# push the element onto the heap, maintaining the heap invariant
#
# Note: heap ordering is the same as python heapq, i.e., smallest first.
##################################################
cdef inline void heappush(Heap *heap,
                          unsigned int *new_elem):
  cdef unsigned int child         = heap.items
  cdef unsigned int parent
  cdef unsigned int k

  # grow if necessary
  if heap.items == heap.space:
      heap.space = max(heap.space * 2, 1000)
      heap.data = <np.int32_t *> realloc(<void *> heap.data, <size_t> (heap.space * heap.width * sizeof(np.int32_t)))

  # insert new data at child
  for k in range(heap.width):
      heap.data[child * heap.width + k] = new_elem[k]
  heap.items += 1

  # restore heap invariant, all parents <= children
  while child>0:
      parent = (child + 1) / 2 - 1 # __parent(i)
      
      if smaller(child, parent, heap):
          swap(parent, child, heap)
          child = parent
      else:
          break
