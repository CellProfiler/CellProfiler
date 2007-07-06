#ifndef IO_H
#define IO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

FILE *fopenck(const char *path, const char *mode);
void freadck(void *buf, size_t size, FILE *stream);
void fwriteck(const void *buf, size_t size, FILE *stream);
void fseekck(FILE *stream, long offset, int whence);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif

#endif
