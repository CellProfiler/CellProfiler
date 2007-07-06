#ifndef VLCUTILS_ERROR_H
#define VLCUTILS_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

extern const char *program_name;


void error(const char *format, ...);
void abort_error(const char *format, ...);
void fatal_error(const char *format, ...);
void fatal_perror(const char *format, ...);
void warn(const char *format, ...);

void set_program_name(const char *argv0);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif

#endif
