#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <errno.h>
#include <string.h>
#include "error.h"

const char *program_name = NULL;

void set_program_name(const char *argv0)
{
     program_name = strrchr(argv0, '/');
     if (program_name)
          program_name++;
     else
          program_name = argv0;
}

static void
verror(const char *format, va_list ap)
{
  if (program_name)
    fprintf(stderr, "%s: ", program_name);
  vfprintf(stderr, format, ap);
 }

void
error(const char *format, ...)
{
  va_list ap;
 
  va_start(ap, format);
  verror(format, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}

void
abort_error(const char *format, ...)
{
  va_list ap;

  va_start(ap, format);
  verror(format, ap);
  va_end(ap);
  fprintf(stderr, "\n");
  abort();
}

void
fatal_error(const char *format, ...)
{
  va_list ap;

  va_start(ap, format);
  verror(format, ap);
  va_end(ap);
  fprintf(stderr, "\n");
  exit(1);
}

void
fatal_perror(const char *format, ...)
{
  va_list ap;

  va_start(ap, format);
  verror(format, ap);
  va_end(ap);
  fprintf(stderr, ": %s", strerror(errno));
  fprintf(stderr, "\n");
  exit(1);
}

void
warn(const char *format, ...)
{
  va_list ap;

  va_start(ap, format);
  verror(format, ap);
  va_end(ap);
  fprintf(stderr, "\n");
}
