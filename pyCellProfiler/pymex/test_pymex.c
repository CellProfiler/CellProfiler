#include <stdio.h>
#include <Python.h>
#include "pymex.h"

void error(const char *message)
{
     fprintf(stderr, "%s", message);
     exit(1);
}

int main(int argc, char *argv[])
{
     Py_Initialize();

     PyObject *a = pymex_make_array(2, 3);
     pymex_array_setitem(a, 0, 0, PyInt_FromLong(1));
     pymex_array_setitem(a, 0, 1, PyInt_FromLong(2));
     pymex_array_setitem(a, 0, 2, PyInt_FromLong(3));
     pymex_array_setitem(a, 1, 0, PyInt_FromLong(4));
     pymex_array_setitem(a, 1, 1, PyInt_FromLong(5));
     pymex_array_setitem(a, 1, 2, PyInt_FromLong(6));

     //Py_Finalize();
}
