#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <vlcutils/mem.h>
#include <vlcutils/readline.h>
#include <vlcutils/error.h>

#define INITIAL_BUFFER_SIZE 2

static char *buf = NULL;
static int bufsize;

static void double_buffer(void)
{
     bufsize *= 2;
     buf = reallock(buf, bufsize);
}

const char *readline(FILE *input)
{
     int fill, ch;

     if (!buf) {
          bufsize = INITIAL_BUFFER_SIZE;
          buf = mallock(bufsize);
     }
     fill = 0;
     while (ch = getc(input), ch != EOF && ch != '\n') {
          if (fill == bufsize)
               double_buffer();
          buf[fill++] = ch;
     }
     if (ch == EOF && fill == 0)
          return NULL;
     if (fill == bufsize)
          double_buffer();
     buf[fill++] = '\0';
     return buf;
}

void readline_free_buffer(void)
{
     free(buf);
     buf = NULL;
}

long int readline_long(FILE *input, int *status)
{
     const char *line;
     char *parse_end;
     long int number;

     line = readline(input);
     if (!line) {
          if (status) *status = READLINE_EOF;
          return -1;
     }
     number = strtol(line, &parse_end, 10);
     if (parse_end && *parse_end) {
          fprintf(stderr, "parse_end = \"%s\"\n", parse_end);
          if (status) *status = READLINE_PARSE_ERROR;
          return -1;
     }
     if (number == LONG_MIN && errno == ERANGE) {
          if (status) *status = READLINE_UNDERFLOW;
          return -1;
     }
     if (number == LONG_MAX && errno == ERANGE) {
          if (status) *status = READLINE_OVERFLOW;
          return -1;
     }
     if (status)
          *status = READLINE_OK;
     return number;
}

int readline_long_array(FILE *input, long int *array, int n)
{
     const char *line;
     char *parse_end;
     int i;
     long int number;

     line = readline(input);
     if (!line) return READLINE_EOF;
     for (i = 0; i < n; i++) {
          number = strtol(line, &parse_end, 10);
          if (parse_end && !isspace(parse_end[0]) && i < n - 1) 
               return READLINE_PARSE_ERROR;
          if (parse_end && parse_end[0] != '\0' && i == n - 1)
               return READLINE_PARSE_ERROR;
          if (number == LONG_MIN && errno == ERANGE)
               return READLINE_UNDERFLOW;
          if (number == LONG_MAX && errno == ERANGE)
               return READLINE_OVERFLOW;
          line = parse_end ? parse_end + 1 : NULL;
          array[i] = number;
     }
     return READLINE_OK;
}

int parseline_double_array(const char *line, double *array, int n)
{
     char *parse_end;
     int i;
     double number;

     assert(line);
     for (i = 0; i < n; i++) {
          number = strtod(line, &parse_end);
          if (parse_end && !isspace(parse_end[0]) && i < n - 1) 
               return READLINE_PARSE_ERROR;
          if (parse_end && parse_end[0] != '\0' && i == n - 1)
               return READLINE_PARSE_ERROR;
          if (number == LONG_MIN && errno == ERANGE)
               return READLINE_UNDERFLOW;
          if (number == LONG_MAX && errno == ERANGE)
               return READLINE_OVERFLOW;
          line = parse_end ? parse_end + 1 : NULL;
          array[i] = number;
     }
     return READLINE_OK;
}

int readline_double_array(FILE *input, double *array, int n)
{
     return parseline_double_array(readline(input), array, n);
}

char **slurp_file(const char *filename)
{
     char **lines;
     const char *line;
     int i, n;
     FILE *f;
     
     f = fopen(filename, "r");
     if (!f) fatal_perror("%s", filename);
     for (n = 0; (line = readline(f)); n++);
     rewind(f);
     lines = mallock((n + 1) * sizeof(char *));
     for (i = 0; i < n; i++) {
          line = readline(f);
          if (!line)
               fatal_error("%s:%d: Unexpected end of file.", filename, i + 1);
          lines[i] = mallock(strlen(line) + 1);
          strcpy(lines[i], line);
          if (!lines[i]) fatal_perror("strdup");
     }
     fclose(f);
     lines[n] = NULL;
     return lines;
}

int fcount(FILE *f, int c)
{
     int fd, count;
     off_t length;
     char *contents, *pos;
     off_t offset;

     fd = fileno(f);
     if (fd == -1) fatal_perror("fileno");
     length = lseek(fd, 0, SEEK_END);
     if (length == (off_t)-1) fatal_perror("lseek");
     contents = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
     if (contents == MAP_FAILED) fatal_perror("mmap");
     count = 0;
     offset = 0;
     while (offset < length) {
          pos = memchr(contents + offset, c, length - offset);
          if (pos) {
               count++;
               offset = pos - contents + 1;
          } else
               break;
     }
     if (munmap(contents, length) == -1) fatal_perror("munmap");
     return count;
}
