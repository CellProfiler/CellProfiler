#ifndef VLCUTILS_INTLIST_H
#define VLCUTILS_INTLIST_H

typedef struct {
     int num;
     int _size;
     int _increase;
     int *elements;
} intlist;

typedef struct {
     int num;
     int _size;
     int _increase;
     unsigned short *elements;
} ushortlist;

/* Increase can be zero, in which case the size will be doubled. */
intlist *make_intlist(int initial_size, int increase);
ushortlist *make_ushortlist(int initial_size, int increase);

void free_intlist(intlist *list);
void free_ushortlist(ushortlist *list);

void intlist_resize(intlist *list, int new_size);
void ushortlist_resize(ushortlist *list, int new_size);

__inline__ extern void intlist_append(intlist *list, int e)
{
     assert(list);
     if (list->num == list->_size) {
          if (list->_increase == 0)
               intlist_resize(list, 2 * list->_size);
          else
               intlist_resize(list, list->_size + list->_increase);
     }
     list->elements[list->num] = e;
     list->num++;
}

__inline__ extern void ushortlist_append(ushortlist *list, unsigned short e)
{
     assert(list);
     if (list->num == list->_size) {
          if (list->_increase == 0)
               ushortlist_resize(list, 2 * list->_size);
          else
               ushortlist_resize(list, list->_size + list->_increase);
     }
     list->elements[list->num] = e;
     list->num++;
}

__inline__ extern void intlist_append_new(intlist *list, int e)
{
     assert(list);
     if (list->num == 0 || list->elements[list->num - 1] != e)
          intlist_append(list, e);
}

__inline__ extern void ushortlist_append_new(ushortlist *list, 
                                             unsigned short e)
{
     assert(list);
     if (list->num == 0 || list->elements[list->num - 1] != e)
          ushortlist_append(list, e);
}

intlist *intlist_clone(const intlist *src);
ushortlist *ushortlist_clone(const ushortlist *src);

intlist *intlist_catenate(const intlist *list1, const intlist *list2);
ushortlist *ushortlist_catenate(const ushortlist *list1, 
                                const ushortlist *list2);

void intlist_ncatenate(intlist *dest, const intlist *src);
void ushortlist_ncatenate(ushortlist *dest, const ushortlist *src);

#endif
