#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <netinet/in.h>
#include <vlcutils/mem.h>
#include <vlcutils/error.h>
#include <vlcutils/io.h>
#include "pgm.h"

struct pgm *cast_pgm(void *buf)
{
     struct pgm *pgm;
     pgm = buf;
     if (strcmp(pgm->type, "PGM"))
          fatal_error("Not a PGM: %p", buf);
     return pgm;
}

struct pgm8 *cast_pgm8(void *buf)
{
     struct pgm *pgm;
     pgm = cast_pgm(buf);
     if (pgm->depth != 1)
          fatal_error("Cannot cast a PGM of depth %d to a PGM8.", pgm->depth);
     return (struct pgm8 *)pgm;
}

struct pgm16 *cast_pgm16(void *buf)
{
     struct pgm *pgm;
     pgm = cast_pgm(buf);
     if (pgm->depth != 2)
          fatal_error("Cannot cast a pgm of depth %d to a pgm16.", pgm->depth);
     return (struct pgm16 *)pgm;
}

const struct pgm *const_cast_pgm(const void *buf)
{
     const struct pgm *pgm;
     pgm = buf;
     if (strcmp(pgm->type, "PGM"))
          fatal_error("Not a PGM: %p", buf);
     return pgm;
}

const struct pgm8 *const_cast_pgm8(const void *buf)
{
     const struct pgm *pgm;
     pgm = const_cast_pgm(buf);
     if (pgm->depth != 1)
          fatal_error("Cannot cast a PGM of depth %d to a PGM8.", pgm->depth);
     return (const struct pgm8 *)pgm;
}

const struct pgm16 *const_cast_pgm16(const void *buf)
{
     const struct pgm *pgm;
     pgm = const_cast_pgm(buf);
     if (pgm->depth != 2)
          fatal_error("Cannot cast a pgm of depth %d to a pgm16.", pgm->depth);
     return (const struct pgm16 *)pgm;
}

void *make_pgm(int width, int height, int depth)
{
     struct pgm *pgm;
     int size;

     if (width <= 0) fatal_error("Illegal width: %d", width);
     if (height <= 0) fatal_error("Illegal height: %d", height);
     if (depth != 1 && depth != 2) fatal_error("Illegal depth: %d", depth);
     size = (sizeof(struct pgm) / 8) * 8;
     if (sizeof(struct pgm) % 8) size += 8;
     pgm = mallock(size + width * height * depth);
     strcpy(pgm->type, "PGM");
     pgm->pixels = (unsigned char *)pgm + size;
     pgm->width = width;
     pgm->height = height;
     pgm->maxval = (1 << depth * 8) - 1;
     pgm->depth = depth;
     return pgm;
}

struct pgm8 *make_pgm8(int width, int height)
{ return make_pgm(width, height, 1); }

struct pgm16 *make_pgm16(int width, int height)
{ return make_pgm(width, height, 2); }

struct pgm *read_pgm(FILE *f)
{
     int ch, i, width, height, maxval;
     char s[255];
     struct pgm *pgm;

     if (getc(f) != 'P') fatal_perror("Not a valid PGM file.");
     if (getc(f) != '5') fatal_perror("Not a valid PGM file.");
     if (!isspace(getc(f))) fatal_perror("Not a valid PGM file.");
     while (isspace(ch = getc(f)))
          ;
     if (!isdigit(ch)) fatal_perror("Not a valid PGM file.");
     i = 0;
     s[i++] = ch;
     while (isdigit(ch = getc(f))) {
          s[i++] = ch;
          if (i == 255) fatal_perror("Not a valid PGM file.");
     }
     s[i] = '\0';
     width = atoi(s);
     if (!isspace(ch)) fatal_perror("Not a valid PGM file.");
     while (isspace(ch = getc(f)))
          ;
     if (!isdigit(ch)) fatal_perror("Not a valid PGM file.");
     i = 0;
     s[i++] = ch;
     while (isdigit(ch = getc(f))) {
          s[i++] = ch;
          if (i == 255) fatal_perror("Not a valid PGM file.");
     }
     s[i] = '\0';
     height = atoi(s);
     if (!isspace(ch)) fatal_perror("Not a valid PGM file.");
     while (isspace(ch = getc(f)))
          ;
     if (!isdigit(ch)) fatal_perror("Not a valid PGM file.");
     i = 0;
     s[i++] = ch;
     while (isdigit(ch = getc(f))) {
          s[i++] = ch;
          if (i == 255) fatal_perror("Not a valid PGM file.");
     }
     s[i] = '\0';
     maxval = atoi(s);
     if (!isspace(ch)) fatal_perror("Not a valid PGM file.");
     if (maxval < 1 || maxval > 65536)
          fatal_error("Invalid maxval: %d", maxval);
     pgm = make_pgm(width, height, maxval <= 255 ? 1 : 2);
     pgm->maxval = maxval;
     freadck(pgm->pixels, width * height * pgm->depth, f);
     if (pgm->depth == 2)
          for (i = 0; i < width * height; i++)
               ((struct pgm16 *)pgm)->pixels[i] = 
                    ntohs(((struct pgm16 *)pgm)->pixels[i]);
     return pgm;
}

void pgm_scale_maxval(void *pgm, int new_maxval)
{
     int i, j;
     struct pgm *im;
     
     im = cast_pgm(pgm);
     if (new_maxval < 1) fatal_error("Cannot scale maxval to %d.", new_maxval);
     if (new_maxval >= 1 << im->depth * 8)
          fatal_error("Cannot scale maxval to %d for a %d-byte PGM.", 
                      new_maxval, im->depth);
#define F(X) round((X) * (new_maxval * 1.0 / (im->maxval * 1.0)))
     if (im->depth == 1) {
          struct pgm8 *pgm8 = cast_pgm8(pgm);
          for (i = 0; i < im->height; i++)
               for (j = 0; j < im->width; j++)
                    pgm_pixel(pgm8, j, i) = F(pgm_pixel(pgm8, j, i));
     } else {
          struct pgm16 *pgm16 = cast_pgm16(pgm);
          for (i = 0; i < im->height; i++)
               for (j = 0; j < im->width; j++)
                    pgm_pixel(pgm16, j, i) = F(pgm_pixel(pgm16, j, i));
     }
     im->maxval = new_maxval;
}

struct pgm *pgm_dup(const struct pgm *input)
{
     struct pgm *output;
     output = make_pgm(input->width, input->height, input->depth);
     output->maxval = input->maxval;
     memcpy(output->pixels, input->pixels, 
            input->width * input->height * input->depth);
     return output;
}

struct pgm *pgm_change_depth(const struct pgm *input, int new_depth)
{
     const struct pgm8 *in8;
     const struct pgm16 *in16;
     struct pgm8 *out8;
     struct pgm16 *out16;
     void *output;
     int i, j;

     if (new_depth != 1 && new_depth != 2)
          fatal_error("Cannot change PGM depth to %d.", new_depth);
     if (input->depth == new_depth)
          return pgm_dup(input);
     if (input->depth == 1) {
          in8 = const_cast_pgm8(input);
          out16 = make_pgm(in8->width, in8->height, 2);
          out16->maxval = in8->maxval;
          for (i = 0; i < in8->height; i++)
               for (j = 0; j < in8->width; j++)
                    pgm_pixel(out16, j, i) = pgm_pixel(in8, j, i);
          output = out16;
     } else {
          in16 = const_cast_pgm16(input);
          out8 = make_pgm(in8->width, in8->height, 1);
          if (in16->maxval > 255)
               fatal_error("Cannot change PGM depth to 2 because maxval is "
                           "%d.", in16->maxval);
          out8->maxval = in16->maxval;
          for (i = 0; i < in8->height; i++)
               for (j = 0; j < in8->width; j++)
                    pgm_pixel(out8, j, i) = pgm_pixel(in16, j, i);
          output = out8;
     }
     return output;
}

struct pgm8 *read_pgm8(FILE *f)
{
     struct pgm *pgm;
     pgm = read_pgm(f);
     if (pgm->depth != 1)
          fatal_error("The PGM file just read is 16-bit; expected 8-bit.");
     return cast_pgm8(pgm);
}

struct pgm16 *read_pgm16(FILE *f)
{
     struct pgm *pgm;
     struct pgm16 *pgm16;

     pgm = read_pgm(f);
     if (pgm->depth == 2)
          pgm16 = cast_pgm16(pgm);
     else {
          pgm16 = cast_pgm16(pgm_change_depth(pgm, 2));
          free(pgm);
     }
     return pgm16;
}

void write_pgm(FILE *f, const void *pgm)
{
     const struct pgm *im;
     unsigned char *bytes;
     unsigned short *shorts;
     int i;

     im = const_cast_pgm(pgm);
     fprintf(f, "P5\n%d %d\n%d\n", im->width, im->height, im->maxval);
     if (im->maxval <= 255) {
          if (im->depth == 2) {
               bytes = mallock(im->width * im->height);
               for (i = 0; i < im->width * im->height; i++)
                    bytes[i] = ((struct pgm16 *)pgm)->pixels[i];
          } else 
               bytes = im->pixels;
          fwriteck(bytes, im->width * im->height, f);
          if (im->depth == 2)
               free(bytes);
     } else {
          shorts = mallock(im->width * im->height * 2);
          for (i = 0; i < im->width * im->height; i++)
               shorts[i] = htons(((struct pgm16 *)pgm)->pixels[i]);
          fwriteck(shorts, im->width * im->height * 2, f);
          free(shorts);
     }
}

void clear_pgm(void *pgm)
{
     struct pgm *im;
     im = cast_pgm(pgm);
     memset(im->pixels, 0, im->width * im->height * im->depth);
}

int pgm_max(const void *pgm)
{
     const struct pgm *im;
     const struct pgm8 *pgm8;
     const struct pgm16 *pgm16;
     int max, i;

     im = const_cast_pgm(pgm);
     if (im->depth == 1) {
          pgm8 = const_cast_pgm8(pgm);
          max = 0;
          for (i = 0; i < pgm8->width * pgm8->height; i++)
               if (pgm8->pixels[i] > max) max = pgm8->pixels[i];
     } else {
          pgm16 = const_cast_pgm16(pgm);
          max = 0;
          for (i = 0; i < pgm16->width * pgm16->height; i++)
               if (pgm16->pixels[i] > max) max = pgm16->pixels[i];
     }
     return max;
}

void pgm_clip(const void *pgm, int val)
{
     const struct pgm *im;
     int i;

     im = const_cast_pgm(pgm);
     if (im->depth == 1) {
          const struct pgm8 *pgm8 = const_cast_pgm8(pgm);
          for (i = 0; i < pgm8->width * pgm8->height; i++)
               if (pgm8->pixels[i] > val) pgm8->pixels[i] = val;
     } else {
          const struct pgm16 *pgm16 = const_cast_pgm16(pgm);
          for (i = 0; i < pgm16->width * pgm16->height; i++)
               if (pgm16->pixels[i] > val) pgm16->pixels[i] = val;
     }
}

void pgm_add16(struct pgm16 *dest, const struct pgm16 *src)
{
     int i, j;

     for (i = 0; i < dest->height; i++)
          for (j = 0; j < dest->width; j++)
               if (pgm_pixel(dest, j, i) + pgm_pixel(src, j, i) <= dest->maxval)
                    pgm_pixel(dest, j, i) += pgm_pixel(src, j, i);
               else
                    pgm_pixel(dest, j, i) = dest->maxval;
}
