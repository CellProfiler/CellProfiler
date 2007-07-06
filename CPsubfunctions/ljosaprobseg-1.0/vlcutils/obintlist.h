#ifndef VLCUTILS_OBINTLIST_H
#define VLCUTILS_OBINTLIST_H

struct obstack;

typedef struct {
     struct obstack *obstack;
     int num;
     int _size;
     int _increase;
     int *elements;
} obintlist;

/* Increase can be zero, in which case the size will be doubled. */
obintlist *make_obintlist(struct obstack *obstack, int initial_size, int increase);
void free_obintlist(obintlist *list);
void obintlist_resize(obintlist *list, int new_size);

__inline__ extern void obintlist_append(obintlist *list, int e)
{
     assert(list);
     if (list->num == list->_size) {
          if (list->_increase == 0)
               obintlist_resize(list, 2 * list->_size);
          else
               obintlist_resize(list, list->_size + list->_increase);
     }
     list->elements[list->num] = e;
     list->num++;
}

__inline__ extern void obintlist_append_new(obintlist *list, int e)
{
     assert(list);
     if (list->num == 0 || list->elements[list->num - 1] != e)
          obintlist_append(list, e);
}

obintlist *obintlist_clone(const obintlist *src);
obintlist *obintlist_catenate(const obintlist *list1, const obintlist *list2);
void obintlist_ncatenate(obintlist *dest, const obintlist *src);

#endif
