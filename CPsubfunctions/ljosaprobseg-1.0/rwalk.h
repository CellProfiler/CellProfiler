#ifndef RWALK_H
#define RWALK_H

struct point { 
     int x; 
     int y; 
};

struct seed { 
     int size;
     struct point points[1];
};

struct seed *make_seed(int size);
void set_seed(struct seed *seed, int pos, struct point point);
struct point point(int x, int y);

struct pgm16 *rwalk1(const struct pgm16 *image, const struct seed *seed,
                     int nwalks, int nsteps, double restart_prob);

struct pgm16 *rwalkpthread(const struct pgm16 *image, const struct seed *seed,
                           int nwalks, int nsteps, double restart_prob,
                           int nworkers);
     
#endif
