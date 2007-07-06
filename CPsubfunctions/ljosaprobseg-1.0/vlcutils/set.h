#ifndef VLCUTILS_SET_H
#define VLCUTILS_SET_H

#include <assert.h>
#include <string.h>

struct set {
     int size;
     int fill_pointer;
     int *items;
     int *positions;  /* Index into items so items can be removed by value. */
};

__inline__ static struct set *make_set(int size)
{
     struct set *set;

     set = malloc(sizeof(struct set));
     if (!set)
	  fatal_perror("malloc");
     set->size = size;
     set->fill_pointer = 0;
     set->items = malloc(size * sizeof(int));
     if (!set->items)
	  fatal_perror("malloc");
     set->positions = calloc(size, sizeof(int));
     if (!set->positions)
	  fatal_perror("make_set(): calloc");
     return set;
}

__inline__ static void set_add(struct set *set, int item)
{
     assert(set);
     assert(0 <= set->fill_pointer && set->fill_pointer < set->size);
     assert(0 <= item && item < set->size);
     if (set->positions[item] != 0)
          return; /* Item is already in the set. */
     set->items[set->fill_pointer] = item;
     set->positions[item] = set->fill_pointer + 1;
     set->fill_pointer++;
}

__inline__ static void set_remove(struct set *set, int item)
{
     int pos;
     assert(set);
     assert(set->fill_pointer > 0);
     assert(0 <= item && item < set->size);
     pos = set->positions[item];
     if (pos == 0)
          return; /* Item is not in set. */
     pos--;
     set->items[pos] = set->items[set->fill_pointer - 1];
     set->fill_pointer--;
     set->positions[set->items[pos]] = pos + 1;
     set->positions[item] = 0;
}

__inline__ static void set_flush(struct set *set)
{
     memset(set->positions, 0, set->size * sizeof(int));
     set->fill_pointer = 0;
}

#endif
