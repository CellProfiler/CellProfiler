#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

int
main(int argc, char *argv[])
{
     Py_Initialize();
     import_array();

     int dims[2];
     dims[0] = 2;
     dims[1] = 3;

     PyArrayObject *result;
     result = (PyArrayObject *)PyArray_SimpleNew(2, dims, PyArray_OBJECT);
     int i,j;
     int k = 1;
     PyObject *value;
     void **vp = (void **)result->data;
     for (i = 0; i < dims[0]; i++)
          for (j = 0; j < dims[1]; j++) {
               printf("%d\n", k);
               value = PyInt_FromLong(k);
               PyArray_SETITEM(result, vp++, value);
               k++;
          }

     PyObject *module = PyImport_ImportModule("tests");
     if (!module) {
          fprintf(stderr, "Cannot import module.\n");
          exit(1);
     }
     PyObject *function = PyObject_GetAttrString(module, "test_echo");
     if (!PyCallable_Check(function)) {
          fprintf(stderr, "Cannot get function.\n");
          exit(1);
     }
     PyObject_CallFunctionObjArgs(function, (PyObject *)result, NULL);
     
     Py_Finalize();
     return 0;
}
