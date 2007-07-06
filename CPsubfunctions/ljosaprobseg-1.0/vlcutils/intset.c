#include <stdlib.h>
#include <assert.h>
#include "error.h"
#include "intset.h"

void init_intset(struct intset *set, int size)
{
     set->size = size;
     set->fill_pointer = 0;
     set->items = malloc(size * sizeof(int));
     if (!set->items)
	  fatal_perror("malloc");
     set->positions = calloc(size, sizeof(int));
     if (!set->positions)
	  fatal_perror("calloc");
}

struct intset *make_intset(int size)
{
     struct intset *set;

     set = malloc(sizeof(struct intset));
     if (!set)
	  fatal_perror("malloc");
     init_intset(set, size);
     return set;
}

void free_intset(struct intset *set)
{
     free(set->items);
     set->items = NULL;
     free(set->positions);
     set->positions = NULL;
     free(set);
}

void intset_add(struct intset *set, int item)
{
     assert(set);
     assert(0 <= set->fill_pointer && set->fill_pointer < set->size);
     assert(0 <= item && item < set->size);
     assert(set->positions[item] == 0); /* Check that item is not in set. */
     set->items[set->fill_pointer] = item;
     set->positions[item] = set->fill_pointer + 1;
     set->fill_pointer++;
}

void intset_remove(struct intset *set, int item)
{
     int pos;
     assert(set);
     assert(set->fill_pointer > 0);
     assert(0 <= item && item < set->size);
     pos = set->positions[item] - 1;
     assert(pos > -1); /* Check that item is in set. */
     set->items[pos] = set->items[set->fill_pointer - 1];
     set->fill_pointer--;
     set->positions[set->items[pos]] = pos + 1;
     set->positions[item] = 0;
}

void intset_flush(struct intset *set)
{
     int i, item;

     for (i = 0; i < set->fill_pointer; i++) {
          item = set->items[i];
          set->positions[item] = 0;
     }
     set->fill_pointer = 0;
}

void intset_fill(struct intset *set)
{
     int i;

     assert(set->fill_pointer == 0);
     for (i = 0; i < set->size; i++) {
          set->items[i] = i;
          set->positions[i] = i + 1;
     }
     set->fill_pointer = set->size;
}

int intset_pop(struct intset *set)
{
     int item;

     assert(set->fill_pointer);
     item = set->items[set->fill_pointer - 1];
     set->positions[item] = 0;
     set->fill_pointer--;
     return item;
}
