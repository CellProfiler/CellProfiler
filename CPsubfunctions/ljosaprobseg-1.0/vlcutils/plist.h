#ifndef VLCUTILS_PLIST_H
#define VLCUTILS_PLIST_H

typedef struct plist plist;

plist *plist_set(plist *pl, int *key, void *value);
void *plist_get(const plist *pl, const int *sym);

#endif
