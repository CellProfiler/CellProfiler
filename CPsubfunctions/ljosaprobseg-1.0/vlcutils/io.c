#include <stdio.h>
#include <vlcutils/error.h>
#include <vlcutils/io.h>

FILE *fopenck(const char *path, const char *mode)
{
     FILE *f;
     f = fopen(path, mode);
     if (!f)
          fatal_perror("%s: fopen", path);
     return f;
}

void freadck(void *buf, size_t size, FILE *stream)
{
     if (fread(buf, size, 1, stream) != 1)
          fatal_perror("fread");
}

void fwriteck(const void *buf, size_t size, FILE *stream)
{
     if (fwrite(buf, size, 1, stream) != 1)
          fatal_perror("fwrite");
}

void fseekck(FILE *stream, long offset, int whence)
{
     if (fseek(stream, offset, whence))
          fatal_perror("fseek");
}
