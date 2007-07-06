/* One piece of work. */

#include <time.h>
#include <stdlib.h>
#include <vlcutils/pgm.h>
#include "rwalk.h"

static __inline__ struct point random_seed_point(unsigned int *rseed, 
                                      const struct seed *seed)
{
     return seed->points[rand_r(rseed) % seed->size];
}

struct pgm16 *rwalk1(const struct pgm16 *image, const struct seed *seed,
                     int nwalks, int nsteps, double restart_prob)
{
     struct pgm16 *trace;
     int w, i, j;
     struct point pos;
     int xoffsets[] = { 1, 0, -1, -1, -1, 0, 1, 1 };
     int yoffsets[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
     int newx[8], newy[8], intensities[8], sum, tally, guess;
     unsigned int rseed;

     rseed = time(NULL);
     trace = make_pgm16(image->width, image->height);
     clear_pgm(trace);
     for (w = 0; w < nwalks; w++) {
          pos = random_seed_point(&rseed, seed);
          for (j = 0; nsteps == 0 || j < nsteps; j++) {
               if (!nsteps && rand_r(&rseed) < RAND_MAX * restart_prob)
                    break; /* Restart */
               for (i = 0; i < 8; i++) {
                    newx[i] = pos.x + xoffsets[i];
                    newy[i] = pos.y + yoffsets[i];
               }
               sum = 0;
               for (i = 0; i < 8; i++) {
                    if (newy[i] >= 0 && newy[i] < image->height && 
                        newx[i] >= 0 && newx[i] < image->width) {
                         intensities[i] = pgm_pixel(image, newx[i], newy[i]);
                         sum += intensities[i];
                    } else
                         intensities[i] = -1;
               }
               guess = (int)(sum * 1.0 * rand_r(&rseed) / (RAND_MAX + 1.0));
               tally = 0;
               for (i = 0; i < 8; i++) {
                    if (intensities[i] == -1)
                         continue;
                    tally += intensities[i];
                    if (tally >= guess)
                         break;
               }
               pos.x = newx[i];
               pos.y = newy[i];
               if (pgm_pixel(trace, pos.x, pos.y) < trace->maxval)
                    pgm_pixel(trace, pos.x, pos.y)++;
          }
     }
     return trace;
}
