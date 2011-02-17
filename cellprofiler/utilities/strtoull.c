/* strtoull.c - replacement for strtoull.c function missing from msvcrt */
#include <ctype.h>

extern unsigned long long int strtoull(const char *s, char **endptr, int base)
{
  unsigned long long int value = 0;
  
  while (isspace(*s)){
    s++;
  }

  if (((base == 0) || (base == 16)) && (s[0] == '0') &&
      (s[1] == 'x' || s[1] == 'X')) {
    s += 2;
    base = 16;
  } else if (((base == 0) || (base == 8)) && (s[0] == '0')) {
    base = 8;
  }
  while (1) {
    unsigned long long int digit = 0;
    if (isdigit(*s)) {
      digit = *s - '0';
    } else if ((base > 10) && isxdigit(*s)) {
      digit = *s + 10 - 'A';
    } else {
      break;
    }
    if (digit >= base) break;
    value += digit;
    s++;
  }

  if (endptr) {
    *endptr = (char *)s;
  }
  return value;
}
