#if 0
#include <Python.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
     char c_string[] = { (char)0x93, 0 };
     PyObject *py_string;

     Py_Initialize();

     py_string = PyString_Decode(c_string, 1, "windows_1252", "replace");
     if (!py_string) {
          PyErr_Print();
          return 1;
     }
     return 0;
}
#endif

#include <Python.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
     char c_string[] = { (char)0x93, 0 };
     PyObject *raw, *decoded;

     Py_Initialize();

     raw = PyString_FromStringAndSize(c_string, 1);
     printf("Undecoded: ");
     PyObject_Print(raw, stdout, 0);
     printf("\n");
     /* decoded = PyObject_CallMethod(raw, "decode", "s", "windows_1252"); */
     decoded = PyString_AsDecodedObject(raw, "windows_1252", "replace");
     Py_DECREF(raw);
     printf("Decoded: ");
     PyObject_Print(decoded, stdout, 0);
     printf("\n");
     return 0;
}

