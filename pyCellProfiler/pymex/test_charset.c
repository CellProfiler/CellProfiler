#include <Python.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
     char c_string[] = { (char)0xf8, 0 };
     PyObject *py_string;

     Py_Initialize();
     
     py_string = PyString_Decode(c_string, 1, "latin_1", "replace");
     if (!py_string) {
          printf("PyString_Decode failed.\n");
          return 1;
     }
     return 0;
}
