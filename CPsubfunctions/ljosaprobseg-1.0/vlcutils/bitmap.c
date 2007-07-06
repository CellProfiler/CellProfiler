#include <vlcutils/error.h>
#include "bitmap.h"

bitmap *make_bitmap(int n)
{
     bitmap *map;
     map = calloc(1, ceil(n / 8.0));
     if (!map) fatal_perror("calloc");
     return map;
}

bitmap *bitmap_dup(const bitmap *map, int n)
{
     bitmap *newmap;
     newmap = malloc(ceil(n / 8.0));
     if (!newmap) fatal_perror("malloc");
     memcpy(newmap, map, ceil(n / 8.0));
     return newmap;
}

void bitmap_write(FILE *f, bitmap *map, int n)
{
     if (fwrite(map, ceil(n / 8.0), 1, f) != 1)
          fatal_perror("fwrite");
}

void bitmap_read(FILE *f, bitmap *map, int n)
{
     
     if (fread(map, ceil(n / 8.0), 1, f) != 1) {
          if (feof(f))
               fatal_error("fread: End of file.");
          else
               fatal_perror("fread");
     }
}

bitmap *bitmap_extend(bitmap *oldmap, int oldsize, int newsize)
{
     bitmap *newmap;
     newmap = calloc(1, ceil(newsize / 8.0));
     if (!newmap) fatal_perror("realloc");
     if (oldmap) {
          bitmap_copy(newmap, oldmap, oldsize);
          free(oldmap);
     }
     return newmap;
}

void bitmap_complement(bitmap *map, int n)
{
     unsigned int *uintmap;
     int i, k;

     uintmap = (unsigned int*)map;
     k = floor(ceil(n / 8.0) / (sizeof(unsigned int) * 1.0));
     for (i = 0; i < k; i++)
          uintmap[i] = ~uintmap[i];
     for (i = k * sizeof(unsigned int); i < ceil(n / 8.0); i++)
          map[i] = ~map[i];
     for (i = n; i < ceil(n / 8.0) * 8; i++)
          bitmap_clear(map, i);
}

void bitmap_and(bitmap *d, const bitmap *s1, const bitmap *s2, int n)
{
     int i, k;

     k = floor(ceil(n / 8.0) / (sizeof(unsigned int) * 1.0));
     for (i = 0; i < k; i++)
          ((unsigned int *)d)[i] =
               ((unsigned int *)s1)[i] & ((unsigned int *)s2)[i];
     for (i = k * sizeof(unsigned int); i < ceil(n / 8.0); i++)
          d[i] = s1[i] & s2[i];
}
