#ifndef VLCUTILS_READLINE_H
#define VLCUTILS_READLINE_H

#ifdef __cplusplus
extern "C" {
#endif

const char *readline(FILE *input);
void readline_free_buffer(void);

#define READLINE_OK 0
#define READLINE_EOF -1
#define READLINE_PARSE_ERROR -2
#define READLINE_UNDERFLOW -3
#define READLINE_OVERFLOW -4

long int readline_long(FILE *input, int *status);
int readline_long_array(FILE *input, long int *array, int n);
int parseline_double_array(const char *line, double *array, int n);
int readline_double_array(FILE *input, double *array, int n);
char **slurp_file(const char *filename);

int fcount(FILE *f, int delimiter);

#ifdef __cplusplus
} /* closing brace for extern "C" */
#endif

#endif
