#ifndef VLCUTILS_INTSET_H
#define VLCUTILS_INTSET_H

#include <assert.h>

struct intset {
     int size;
     int fill_pointer;
     int *items;
     int *positions;  /* Index into items so items can be removed by value. */
};

void init_intset(struct intset *set, int size);
struct intset *make_intset(int size);
void free_intset(struct intset *set);

void intset_add(struct intset *set, int item);
void intset_remove(struct intset *set, int item);
void intset_flush(struct intset *set);
void intset_fill(struct intset *set);
int intset_pop(struct intset *set);

#endif
