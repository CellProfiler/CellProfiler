#include <assert.h>
#include <string.h>
#include <obstack.h>
#include "mem.h"
#include "obintlist.h"

obintlist *make_obintlist(struct obstack *obstack, int initial_size, int increase)
{
     obintlist *list;

     list = mallock(sizeof(obintlist));
     list->obstack = obstack;
     list->num = 0;
     assert(initial_size >= 1);
     list->_size = initial_size;
     assert(increase >= 0);
     list->_increase = increase;
     list->elements = obstack_alloc(list->obstack, list->_size * sizeof(int));
     return list;
}

void free_obintlist(obintlist *list)
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

static void *obstack_realloc(struct obstack *obstack, void *old, 
                             size_t old_size, size_t new_size)
{
     void *new; 

     assert(old);
     assert(new_size > old_size);
     new = obstack_alloc(obstack, new_size);
     memcpy(new, old, old_size);
     return new;
}

void obintlist_resize(obintlist *list, int new_size)
{
     assert(list);
     assert(new_size >= list->num);
     list->elements = obstack_realloc(list->obstack, list->elements, 
                                      list->_size * sizeof(int),
                                      new_size * sizeof(int));
     list->_size = new_size;
}

/* Note: Inlined version in header file. */
void obintlist_append(obintlist *list, int e)
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

/* Note: Inlined version in header file. */
void obintlist_append_new(obintlist *list, int e)
{
     assert(list);
     if (list->num == 0 || list->elements[list->num - 1] != e)
          obintlist_append(list, e);
}

obintlist *obintlist_clone(const obintlist *src)
{
     obintlist *dest;
     dest = make_obintlist(src->obstack, src->_size, src->_increase);
     dest->num = src->num;
     memcpy(dest->elements, src->elements, src->num * sizeof(int));
     return dest;
}

obintlist *obintlist_catenate(const obintlist *list1, const obintlist *list2)
{
     obintlist *list;
     list = obintlist_clone(list1);
     obintlist_ncatenate(list, list2);
     return list;
}

void obintlist_ncatenate(obintlist *dest, const obintlist *src)
{
     if (dest->_size < dest->num + src->num)
          obintlist_resize(dest, dest->num + src->num);
     memcpy(&dest->elements[dest->num], src->elements, src->num * sizeof(int));
     dest->num += src->num;
}
