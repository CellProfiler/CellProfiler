#ifndef PGM_H
#define PGM_H

#include <stdio.h>

struct pgm {
     char type[4];
     int height, width, maxval, depth;
     void *pixels;
};

struct pgm8 {
     char type[4];
     int height, width, maxval, depth;
     unsigned char *pixels;
};

struct pgm16 {
     char type[4];
     int height, width, maxval, depth;
     unsigned short *pixels;
};


#define pgm_pixel(PGM, X, Y) (PGM)->pixels[(Y) * (PGM)->width + (X)]

struct pgm *cast_pgm(void *buf);
struct pgm8 *cast_pgm8(void *buf);
struct pgm16 *cast_pgm16(void *buf);
const struct pgm *const_cast_pgm(const void *buf);
const struct pgm8 *const_cast_pgm8(const void *buf);
const struct pgm16 *const_cast_pgm16(const void *buf);

void *make_pgm(int height, int width, int maxval);
struct pgm8 *make_pgm8(int height, int width);
struct pgm16 *make_pgm16(int height, int width);

struct pgm *read_pgm(FILE *f);
struct pgm8 *read_pgm8(FILE *f);
struct pgm16 *read_pgm16(FILE *f);

void write_pgm(FILE *f, const void *pgm);
void clear_pgm(void *pgm);

void pgm_scale_maxval(void *pgm, int new_maxval);
struct pgm *pgm_dup(const struct pgm *input);
struct pgm *pgm_change_depth(const struct pgm *input, int new_depth);

int pgm_max(const void *pgm);
void pgm_clip(const void *pgm, int val);

/* Add an image to the next, clipping pixel intensitites at maxval. */
void pgm_add16(struct pgm16 *dest, const struct pgm16 *src);

#endif
