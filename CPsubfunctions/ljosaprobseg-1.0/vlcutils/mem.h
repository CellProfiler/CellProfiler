#ifndef VLCUTILS_MEM_H
#define VLCUTILS_MEM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

void *mallock(size_t size);
void *callock(size_t nmemb, size_t size);
void *reallock(void *ptr, size_t size);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif

#endif
