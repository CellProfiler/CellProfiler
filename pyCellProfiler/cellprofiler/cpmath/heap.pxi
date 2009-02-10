import numpy as np
cimport numpy as np
cimport cython

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

cdef inline Heap *heap_from_numpy(np_heap):
    cdef unsigned int k
    cdef Heap *heap = <Heap *> malloc(sizeof (Heap))
    heap.items = np_heap.shape[0]
    heap.width = np_heap.shape[1]
    heap.space = np_heap.shape[0] * np_heap.shape[1]
    heap.data = <np.int32_t *> malloc(np_heap.shape[0] * np_heap.shape[1] * sizeof(np.int32_t))
    tmp = np_heap.astype(np.int32).flatten('C')
    for k in range(heap.space):
        heap.data[k] = tmp.data[k]
    return heap

cdef inline void heap_done(Heap *heap):
   free(heap.data)
   free(heap)

######################################################
# heappop - pop an element off of the heap
#
# Algorithm taken from Introduction to Algorithms, Cormen, p 150
# and (heapify(A,1)) p 143
#
######################################################
cdef inline void heappop(Heap *heap,
                  unsigned int *dest):
    cdef unsigned int heap_items = heap.items
    cdef unsigned int heap_width = heap.width
    cdef np.int32_t *heap_data = <np.int32_t *> heap.data
    cdef unsigned int k,l,r,i,smallest # heap indices
    cdef np.int32_t *i_ptr, *smallest_ptr, *r_ptr, *l_ptr # heap element pointers
    
    #
    # Start by copying the first element to the destination
    #
    for k in range(heap_width):
        dest[k] = heap_data[k]
    heap_items -= 1

    # if the heap is now empty, we can return, no need to fix heap.
    if heap_items == 0:
        heap.items = heap_items
        return

    #
    # Move the last element in the heap to the first.
    #
    for k in range(heap_width):
        heap_data[k] = heap_data[heap_items * heap_width + k]
    #
    # Restore the heap invariant.
    #
    i = 0
    i_ptr = heap_data + i
    smallest = i
    smallest_ptr = i_ptr
    while True:
        # loop invariant here: smallest == i
        
        # find smallest of (i, l, r), and swap it to i's position if necessary
        l = i*2+1 #__left(i)
        r = i*2+2 #__right(i)
        if l < heap_items:
            l_ptr = heap_data + l * heap_width
            for k in range(heap_width):
                if i_ptr[k] == l_ptr[k]:
                    continue
                if i_ptr[k] > l_ptr[k]:
                    smallest = l
                    smallest_ptr = l_ptr
                    break
        if r < heap_items:
            r_ptr = heap_data + r * heap_width
            for k in range(heap_width):
                if smallest_ptr[k] == r_ptr[k]:
                    continue
                if smallest_ptr[k] > r_ptr[k]:
                    smallest = r
                    smallest_ptr = r_ptr
                    break
        # the element at i is smaller than either of its children, invariant restored.
        if smallest_ptr == i_ptr:
                break
        for k in range(heap_width):
            i_ptr[k], smallest_ptr[k] = smallest_ptr[k], i_ptr[k]
        i = smallest
        i_ptr = smallest_ptr
        
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
cdef inline void heappush(Heap *heap,
                          unsigned int *new_elem):
  cdef unsigned int heap_width = heap.width
  cdef np.int32_t *heap_data = <np.int32_t *> heap.data
  cdef unsigned int ii         = heap.items
  cdef unsigned int ii_idx     = heap.items * heap_width
  cdef unsigned int parent
  cdef unsigned int parent_idx
  cdef unsigned int k, parent_less_or_equal

  # grow if necessary
  if heap.items == heap.space:
      heap.space = max((heap.space * 3) / 2, 1000)
      heap.data = <np.int32_t *> realloc(heap.data, heap.space * heap_width * sizeof(np.int32_t))
      heap_data = heap.data

  while ii>0:
      parent     = (ii+1)/2 - 1 # __parent(i)
      parent_idx = parent * heap_width
      parent_less_or_equal = 1
      for k in range(heap_width):
          if heap_data[parent_idx + k] == heap_data[ii_idx + k]:
              continue
          if heap_data[parent_idx + k] > heap_data[ii_idx + k]:
              parent_less_or_equal = 0
          break

      if parent_less_or_equal:
          break

      for k in range(heap_width):
          heap_data[ii_idx+k] = heap_data[parent_idx+k]
      ii     = parent
      ii_idx = parent_idx

  for k in range(heap_width):
      heap_data[ii_idx+k] = new_elem[k]
