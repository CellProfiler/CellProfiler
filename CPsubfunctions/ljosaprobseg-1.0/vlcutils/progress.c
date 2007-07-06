#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <vlcutils/mem.h>
#include "progress.h"

struct progress *make_progress(int n, const char *message)
{
     struct progress *p;
     p = mallock(sizeof(struct progress));
     p->n = n;
     p->completed = 0;
     p->message_length = strlen(message);
     fprintf(stderr, "%s...%3d %% done", message, 0);
     fflush(stderr);
     return p;
}

void update_progress(struct progress *p, int completed)
{
     if (completed == p->completed)
          return;
     assert(completed <= p->n);
     p->completed = completed;
     fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b%3d %% done", 
             100 * p->completed / p->n);
     fflush(stderr);
}

void free_progress(struct progress *p)
{
     int i;
     for (i = 0; i < p->message_length + 13; i++) putc('\b', stderr);
     for (i = 0; i < p->message_length + 13; i++) putc(' ', stderr);
     for (i = 0; i < p->message_length + 13; i++) putc('\b', stderr);
     fflush(stderr);
}

static struct timeval marked_time;

void mark_start_time(struct timeval *start_time)
{
     gettimeofday(start_time ? start_time : &marked_time, NULL);
}

double elapsed_time(struct timeval *start_time)
{
     struct timeval end_time;

     if (!start_time) start_time = &marked_time;
     gettimeofday(&end_time, NULL);
     return end_time.tv_sec - start_time->tv_sec +
          (end_time.tv_usec - start_time->tv_usec) / 1000000.0;
}
