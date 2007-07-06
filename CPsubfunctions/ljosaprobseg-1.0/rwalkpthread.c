#include <vlcutils/mem.h>
#include <vlcutils/pgm.h>
#include <vlcutils/error.h>
#include <pthread.h>
#include "rwalk.h"

struct params {
     const struct pgm16 *image;
     const struct seed *seed;
     int nwalks;
     int nsteps;
     double restart_prob;
};

static void *worker_body(void *parameters)
{
     struct params *par;
     struct pgm16 *trace;

     par = (struct params *)parameters;

     trace = rwalk1(par->image, par->seed, par->nwalks, par->nsteps, 
                    par->restart_prob);
     pthread_exit((void *)trace);
}

struct pgm16 *rwalkpthread(const struct pgm16 *image, const struct seed *seed,
                           int nwalks, int nsteps, double restart_prob,
                           int nworkers)
{
     pthread_t *workers;
     pthread_attr_t attr;
     struct params par;
     struct pgm16 *trace, *trace1;
     int w;

     par.image = image;
     par.seed = seed;
     par.nwalks = nwalks / nworkers;
     par.nsteps = nsteps / nworkers;
     par.restart_prob = restart_prob;

     /* Not all implementations of pthreads make the threads joinable
      * by default. */
     pthread_attr_init(&attr);
     pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

     workers = mallock(sizeof(pthread_t) * nworkers);
     for (w = 0; w < nworkers; w++)
          if (pthread_create(&workers[w], &attr, worker_body, (void *)&par))
               fatal_perror("pthread_create");
     pthread_attr_destroy(&attr);
     trace = make_pgm16(image->width, image->height);
     for (w = 0; w < nworkers; w++) {
          if (pthread_join(workers[w], (void **)&trace1))
               fatal_perror("pthread_join");
          pgm_add16(trace, trace1);
          free(trace1);
     }
     free(workers);
     return trace;
}
