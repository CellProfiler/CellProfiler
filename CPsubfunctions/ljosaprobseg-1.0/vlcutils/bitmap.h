#ifndef VLCUTILS_BITMAP_H
#define VLCUTILS_BITMAP_H

#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

typedef unsigned char bitmap;

/* Can be freed with free(). */
bitmap *make_bitmap(int n);

__inline__ static void bitmap_set(bitmap *map, int i)
{
     map[i / 8] = map[i / 8] | 1 << (8 - i % 8 - 1);
}

__inline__ static int bitmap_get(const bitmap *map, int i)
{
     return map[i / 8] & 1 << (8 - i % 8 - 1);
}

bitmap *bitmap_dup(const bitmap *map, int n);

__inline__ static void bitmap_copy(bitmap *destmap, const bitmap *srcmap, int n)
{
     memcpy(destmap, srcmap, ceil(n / 8.0));
}

bitmap *bitmap_extend(bitmap *oldmap, int oldsize, int newsize);

__inline__ static int bitmap_count(const bitmap *map, int n)
{
     int i, count = 0;
     static int count_table[] =
          { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 
            3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 
            3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
            2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
            3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
            5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 
            2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 
            4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
            4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
            5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
            5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8 };
     for (i = 0; i < ceil(n / 8.0); i++)
          count += count_table[(unsigned char)map[i]];
     return count;
}

__attribute__ ((unused))
__inline__ static void bitmap_clear(bitmap *map, int n)
{
     memset(map, 0, ceil(n / 8.0));
     assert(bitmap_count(map, n) == 0);
}

__attribute__ ((unused))
__inline__ static void bitmap_setall(bitmap *map, int n)
{
     memset(map, 0xff, n / 8);
     if (n % 8)
          map[(int)ceil(n / 8.0) - 1] = 0xff << (8 - n % 8);
     assert(bitmap_count(map, n) == n);
}

__attribute__ ((unused))
__inline__ static void bitmap_to_list(int *list, const bitmap *map, int n)
{
     int imax;
     int i, j, f;

     imax = ceil(n / 8.0);
     f = 0;
     for (i = 0; i < imax; i++)
          if (map[i])
               for (j = 0; j < 8; j++)
                    if (map[i] & 1 << (8 - j % 8 - 1)) {
/*                          assert(bitmap_get(map, i * 8 + j)); */
                         list[f++] = i * 8 + j;
                    }
     list[f++] = -1;
}

void bitmap_write(FILE *f, bitmap *map, int n);
void bitmap_read(FILE *f, bitmap *map, int n);
void bitmap_complement(bitmap *map, int n);
void bitmap_and(bitmap *d, const bitmap *s1, const bitmap *s2, int n);

#endif
