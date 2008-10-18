#include <Python.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
     char c_string[] = { (char)0x93, 0 };
     PyObject *py_string;

     Py_Initialize();

     py_string = PyString_Decode(c_string, 1, "windows_1252", "replace");
     if (!py_string) {
          printf("PyString_Decode failed.\n");
          PyErr_Print();
          return 1;
     }
     return 0;
}
#if 0

#include <Python.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
     char c_string[] = { (char)0x93, 0 };
     PyObject *raw, *decoded;

     Py_Initialize();

     raw = PyString_FromString(c_string);
     printf("Undecoded: ");
     PyObject_Print(raw, stdout, 0);
     printf("\n");
     decoded = PyObject_CallMethod(raw, "decode", "s", "windows_1252");
     Py_DECREF(raw);
     printf("Decoded: ");
     PyObject_Print(decoded, stdout, 0);
     printf("\n");
     return 0;
}
#endif
