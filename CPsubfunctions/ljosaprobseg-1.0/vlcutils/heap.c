#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "error.h"
#include "heap.h"

struct heap {
     int capacity;
     int size;
     int (*predicate)();
     void **elements;
};

struct heap *
make_heap(int capacity, int (*predicate)())
{
     struct heap *heap;

     assert(capacity > 0);
     assert(predicate);
     heap = malloc(sizeof(struct heap));
     if (!heap)
	  fatal_perror("malloc");
     heap->capacity = capacity;
     heap->size = 0;
     heap->predicate = predicate;
     heap->elements = malloc((capacity + 1) * sizeof(void *));
     if (!heap->elements)
	  fatal_perror("malloc");
     return heap;
}

void free_heap(struct heap *heap)
{
     assert(heap);
     free(heap->elements);
     heap->elements = NULL;
     free(heap);
}

int heap_size(struct heap *heap)
{
     assert(heap);
     return heap->size;
}

void heap_insert(void *element, struct heap *heap)
{
     int i;

     assert(heap);
     if (heap->size == heap->capacity)
	  fatal_error("Heap is full.");

     heap->size++;
     for (i = heap->size; 
	  i > 1 && heap->predicate(element, heap->elements[i / 2]); ) 
     {
	  heap->elements[i] = heap->elements[i / 2];
	  i /= 2;
     }
     heap->elements[i] = element;
}

void *heap_pop(struct heap *h)
{
     int i, child;
     void *first_element, *last_element;

     assert(h);
     if (h->size == 0)
	  fatal_error("Heap is empty.");

     first_element = h->elements[1];
     last_element = h->elements[h->size];
     h->size--;
     for (i = 1; i * 2 <= h->size; i = child) {
	  child = i * 2;
	  if (child != h->size && h->predicate(h->elements[child + 1], 
					       h->elements[child]))
	       child++;
	  if (h->predicate(h->elements[child], last_element))
	       h->elements[i] = h->elements[child];
	  else
	       break;
     }
     h->elements[i] = last_element;
     return first_element;
}

int int_lt(int *a, int *b)
{
     return *a < *b;
}

void print_heap(struct heap *heap)
{
     int i;

     for (i = 1; i < heap->size + 1; i++)
     {
	  printf("%d ", *(int *)heap->elements[i]);
     }
     printf("\n");
}

#if 0
int main(int argc, char *argv[])
{
     struct heap *heap;
     int e1 = 3, e2 = 2, e3 = 1;

     heap = make_heap(10, int_lt);
     print_heap(heap);
     heap_insert(&e1, heap); /* assert(heap->elements[1] == &e1); */
     print_heap(heap);
     heap_insert(&e3, heap); /* assert(heap->elements[2] == &e3); */
     print_heap(heap);
     heap_insert(&e2, heap); /* assert(heap->elements[2] == &e2); */
     print_heap(heap);
     printf("Popped: %d\n", *(int *)heap_pop(heap));
     print_heap(heap);
     printf("Popped: %d\n", *(int *)heap_pop(heap));
     print_heap(heap);
     printf("Popped: %d\n", *(int *)heap_pop(heap));
     print_heap(heap);
     free_heap(heap);
     return 0;
}
#endif
