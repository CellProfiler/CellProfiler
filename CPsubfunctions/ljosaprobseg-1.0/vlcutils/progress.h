#ifndef PROGRESS_H
#define PROGRESS_H

#ifdef __cplusplus
extern "C" {
#endif

struct progress {
     int n;
     int completed;
     int message_length;
};

struct progress *make_progress(int n, const char *message);
void update_progress(struct progress *p, int completed);
void free_progress(struct progress *p);

#include <time.h>
#include <sys/time.h>

void mark_start_time(struct timeval *start_time);
double elapsed_time(struct timeval *start_time);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif

#endif
