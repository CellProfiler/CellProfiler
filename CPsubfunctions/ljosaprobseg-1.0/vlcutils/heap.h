#ifndef VLCUTILS_HEAP_H
#define VLCUTILS_HEAP_H

struct heap;
struct heap *make_heap(int capacity, int (*predicate)());
void free_heap(struct heap *heap);
void heap_insert(void *element, struct heap *heap);
void *heap_pop(struct heap *h);
int heap_size(struct heap *heap);

#endif
