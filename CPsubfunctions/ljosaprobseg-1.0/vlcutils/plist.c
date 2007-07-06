#include <stdlib.h>
#include "error.h"
#include "plist.h"

struct plist {
     int *key;
     void *value;
     plist *next;
};

plist *plist_remove_1(plist *pl, int *key)
{
     plist *rv;

     if (!pl)
          return NULL;
     if (pl->key == key) {
          rv = pl->next;
          free(pl);
          return rv;
     } else {
          pl->next = plist_remove_1(pl->next, key);
          return pl;
     }
}

plist *plist_set(plist *pl, int *key, void *value)
{
     plist *p;
     p = pl;
     while (p) {
          if (p->key == key) {
               p->value = value;
               return pl;
          }
          p = p->next;
     }
     p = malloc(sizeof(plist));
     if (!p) fatal_perror("malloc");
     p->key = key;
     p->value = value;
     p->next = pl;
     return p;
}

void *plist_get(const plist *pl, const int *sym)
{
     for (; pl; pl = pl->next) {
          if (pl->key == sym)
               return pl->value;
     }
     return NULL;
}
