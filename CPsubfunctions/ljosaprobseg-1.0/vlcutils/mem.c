#include <stdlib.h>
#include "error.h"
#include "mem.h"

void *mallock(size_t size)
{
     void *m;
     m = malloc(size);
     if (!m && size) fatal_perror("malloc(%u): ", size);
     return m;
}

void *callock(size_t nmemb, size_t size)
{
     void *m;
     m = calloc(nmemb, size);
     if (!m && size) fatal_perror("calloc(%u, %u): ", nmemb, size);
     return m;
}

void *reallock(void *ptr, size_t size)
{
     void *m;
     m = realloc(ptr, size);
     if (!m && size) fatal_perror("realloc(%u): ", size);
     return m;
}

