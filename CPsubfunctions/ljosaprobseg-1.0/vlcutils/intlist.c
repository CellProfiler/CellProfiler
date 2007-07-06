#include <assert.h>
#include <string.h>
#include <limits.h>
#include "mem.h"
#include "intlist.h"

intlist *make_intlist(int initial_size, int increase)
{
     intlist *list;

     list = mallock(sizeof(intlist));
     list->num = 0;
     assert(initial_size >= 1);
     list->_size = initial_size;
     assert(increase >= 0);
     list->_increase = increase;
     list->elements = mallock(list->_size * sizeof(int));
     return list;
}

ushortlist *make_ushortlist(int initial_size, int increase)
{
     ushortlist *list;

     list = mallock(sizeof(ushortlist));
     list->num = 0;
     assert(initial_size >= 1);
     list->_size = initial_size;
     assert(increase >= 0);
     list->_increase = increase;
     list->elements = mallock(list->_size * sizeof(unsigned short));
     return list;
}

void free_intlist(intlist *list)
{
     assert(list);
     if (list->elements) {
          free(list->elements);
          list->elements = NULL;
     }
     list->num = 0;
     list->_size = 0;
     free(list);
}

void free_ushortlist(ushortlist *list)
{
     assert(list);
     if (list->elements) {
          free(list->elements);
          list->elements = NULL;
     }
     list->num = 0;
     list->_size = 0;
     free(list);
}

void intlist_resize(intlist *list, int new_size)
{
     assert(list);
     assert(new_size >= list->num);
     list->_size = new_size;
     list->elements = reallock(list->elements, list->_size * sizeof(int));
}

void ushortlist_resize(ushortlist *list, int new_size)
{
     assert(list);
     assert(new_size >= list->num);
     list->_size = new_size;
     list->elements = reallock(list->elements, 
                               list->_size * sizeof(unsigned short));
}

/* Note: Inlined version in header file. */
void intlist_append(intlist *list, int e)
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

void ushortlist_append(ushortlist *list, unsigned short e)
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

/* Note: Inlined version in header file. */
void intlist_append_new(intlist *list, int e)
{
     assert(list);
     if (list->num == 0 || list->elements[list->num - 1] != e)
          intlist_append(list, e);
}

void ushortlist_append_new(ushortlist *list, unsigned short e)
{
     assert(list);
     if (list->num == 0 || list->elements[list->num - 1] != e)
          ushortlist_append(list, e);
}

intlist *intlist_clone(const intlist *src)
{
     intlist *dest;
     dest = make_intlist(src->_size, src->_increase);
     dest->num = src->num;
     memcpy(dest->elements, src->elements, src->num * sizeof(int));
     return dest;
}

ushortlist *ushortlist_clone(const ushortlist *src)
{
     ushortlist *dest;
     dest = make_ushortlist(src->_size, src->_increase);
     dest->num = src->num;
     memcpy(dest->elements, src->elements, src->num * sizeof(unsigned short));
     return dest;
}

intlist *intlist_catenate(const intlist *list1, const intlist *list2)
{
     intlist *list;
     list = intlist_clone(list1);
     intlist_ncatenate(list, list2);
     return list;
}

ushortlist *ushortlist_catenate(const ushortlist *list1, 
                                const ushortlist *list2)
{
     ushortlist *list;
     list = ushortlist_clone(list1);
     ushortlist_ncatenate(list, list2);
     return list;
}

void intlist_ncatenate(intlist *dest, const intlist *src)
{
     if (dest->_size < dest->num + src->num)
          intlist_resize(dest, dest->num + src->num);
     memcpy(&dest->elements[dest->num], src->elements, src->num * sizeof(int));
     dest->num += src->num;
}

void ushortlist_ncatenate(ushortlist *dest, const ushortlist *src)
{
     if (dest->_size < dest->num + src->num)
          ushortlist_resize(dest, dest->num + src->num);
     memcpy(&dest->elements[dest->num], src->elements, 
            src->num * sizeof(unsigned short));
     dest->num += src->num;
}
