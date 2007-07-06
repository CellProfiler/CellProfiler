#define _GNU_SOURCE
#include <getopt.h>
#include <sysexits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <vlcutils/error.h>
#include <vlcutils/mem.h>
#include <vlcutils/pgm.h>
#include <config.h>
#include "rwalk.h"

#define DEFAULT_NWORKERS 1

static const char *filename, *output_filename = NULL;
static int nwalks, nsteps = 0, clip = 0, eightbit = 0;
static int nworkers = DEFAULT_NWORKERS;
static double restart_prob = 0.0;
static struct seed *seed;

static void usage(FILE *stream)
{
     fprintf(stream, "Usage: %s [OPTION]... PGM-IMAGE-FILE NWALKS START-X,START-Y...\n", program_name);
     fprintf(stream, "Segment cell by random walk.\n");
     if (stream == stderr)
          fprintf(stream, "Try `%s -h` for more information.\n", program_name);
}

static void help(void)
{
     usage(stdout);
     printf("\n"
"      -h            Display this help.\n"
"      -V            Display version number.\n"
"      -8            Always write result as an 8-bit PGM.\n"
"      -c VAL        Clip output at VAL.\n"
"      -e FILENAME   Correct according to estimated image in FILENAME.\n"
"      -n NUMBER     Run NUMBER processes in parallel (default: %d).\n"
            , DEFAULT_NWORKERS);
     printf(
"      -o FILENAME   Write output to FILENAME (default: stdout).\n"
"      -r PROB       Restart probability.\n"
"      -s NSTEPS     Walk the specified number of steps.\n"
);
}

int parse_integer(const char *string)
{
     int val;
     char *parse_end;

     val = strtol(string, &parse_end, 10);
     if (parse_end == string)
          fatal_error("Invalid number: %s", string);
     return val;
}

double parse_double(const char *string)
{
     double val;
     char *parse_end;

     val = strtod(string, &parse_end);
     if (parse_end == string)
          fatal_error("Invalid number: %s", string);
     return val;
}

struct point parse_point(const char *string)
{
     char *comma, *tmp;
     struct point point;

     tmp = strdup(string);
     if (!tmp) fatal_perror("strdup");
     comma = strchr(tmp, ',');
     if (!comma)
          fatal_error("Cannot parse point: %s", tmp);
     *comma = '\0';
     point.x = parse_integer(tmp);
     point.y = parse_integer(comma + 1);
     free(tmp);
     if (point.x < 0)
          fatal_error("Points must be non-negative: %d", point.x);
     if (point.y < 0)
          fatal_error("Points must be non-negative: %d", point.y);
     return point;
}

static void parse_options(int argc, char *argv[])
{
     int opt, i;

     while ((opt = getopt(argc, argv, "hV8c:e:n:o:r:s:")) != -1)
          switch (opt) {
          case 'h':
               help();
               exit(EX_OK);
          case 'V':
               printf("%s %s\n", PACKAGE, VERSION);
               exit(EX_OK);
          case '8':
               eightbit = 1;
               break;
          case 'c':
               clip = parse_integer(optarg);
               if (clip < 0)
                    fatal_error("Clip value must be non-negative: %d", clip);
               break;
          case 'n':
               nworkers = parse_integer(optarg);
               if (nworkers < 1)
                    fatal_error("Number of processes must be positive: %d",
                                nworkers);
               break;
          case 'o':
               output_filename = optarg;
               break;
          case 'r':
               restart_prob = parse_double(optarg);
               if (restart_prob < 0.0 || restart_prob > 1.0)
                    fatal_error("Argument to -r must be between 0 and 1.");
               break;
          case 's':
               nsteps = parse_integer(optarg);
               if (nsteps <= 0) 
                    fatal_error("Argument to -n must be positive.");
               break;
          default:
               usage(stderr);
               exit(EX_USAGE);
          }
     if (argc >= optind + 3) {
          filename = argv[optind];
          nwalks = parse_integer(argv[optind + 1]);
          if (nwalks < 1)
               fatal_error("Must take at least one walk.");
          seed = make_seed(argc - optind - 2);
          for (i = 0; i < seed->size; i++)
               set_seed(seed, i, parse_point(argv[optind + 2 + i]));
     } else {
          usage(stderr);
          exit(EX_USAGE);
     }
     if (restart_prob == 0.0 && nsteps == 0)
          fatal_error("Must give either -r or -s.");
}

int main(int argc, char *argv[])
{
     FILE *input, *output;
     struct pgm16 *image;
     struct pgm16 *pmask;
     int i;

     set_program_name(argv[0]);
     parse_options(argc, argv);
     /* srandomdev(); */
     srandom(time(NULL));
     
     input = fopen(filename, "r");
     if (!input) fatal_perror("%s", filename);
     image = read_pgm16(input);
     fclose(input);

     if (output_filename) {
          output = fopen(output_filename, "w");
          if (!output) fatal_perror("%s", output_filename);
     } else
          output = stdout;

     for (i = 0; i < seed->size; i++)
          if (seed->points[i].x >= image->width || 
              seed->points[i].y >= image->height)
               fatal_error("Starting point (%d, %d) is outside of "
                           "%d x %d image.", seed->points[i].x, 
                           seed->points[i].y, image->height, image->width);

     pmask = rwalkpthread(image, seed, nwalks, nsteps, restart_prob, nworkers);

     pmask->maxval = pgm_max(pmask);
     if (clip) {
          pgm_clip(pmask, clip);
          pmask->maxval = clip;
     }
     if (eightbit && pmask->maxval > 255)
          pgm_scale_maxval(pmask, 255);
     write_pgm(output, pmask);
     free(image);
     free(pmask);
     return EX_OK;
}
