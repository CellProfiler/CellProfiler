#include <stdio.h>
#include <string.h>
#include <vlcutils/readline.h>


int main(int argc, char *argv[])
{
     const char *s;

     while ((s = readline(stdin)) != NULL) {
          printf("%d '%s'\n", strlen(s), s);
     }
     readline_free_buffer();
     return 0;
}
