#include <vlcutils/mem.h>
#include "rwalk.h"

struct seed *make_seed(int size)
{
     struct seed *seed = mallock(sizeof(struct seed) + 
                                 (size - 1) * sizeof(struct point));
     seed->size = size;
     return seed;
}

void set_seed(struct seed *seed, int pos, struct point point)
{
     seed->points[pos] = point;
}

struct point point(int x, int y)
{
     struct point p;
     p.x = x;
     p.y = y;
     return p;
}
