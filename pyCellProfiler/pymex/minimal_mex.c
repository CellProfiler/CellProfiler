#include <stdio.h>
#include <Python.h>
#include "mex.h"
#include <numpy/arrayobject.h>

void
mexFunction(int anlhs, mxArray *aplhs[], int anrhs, const mxArray *aprhs[])
{
     int dims[2];
     PyArrayObject *result;
     PyObject *value,*module,*function;
     void **vp;
     int i,j;
     int k = 1;

     printf("Initializing...\n");
     if (Py_IsInitialized())
          printf("Already initialized.\n");
     
     pymex_init();
     printf("Calling import_array()...\n");
     import_array();

     dims[0] = 2;
     dims[1] = 3;

     printf("Allocating array...\n");
     result = (PyArrayObject *)PyArray_SimpleNew(2, dims, PyArray_OBJECT);
     vp = (void **)result->data;
     for (i = 0; i < dims[0]; i++)
          for (j = 0; j < dims[1]; j++) {
               printf("%d\n", k);
               value = PyInt_FromLong(k);
               PyArray_SETITEM(result, vp++, value);
               k++;
          }
     printf("Importing \"tests\" module.\n");
     module = PyImport_ImportModule("tests");
     if (!module) {
          fprintf(stderr, "Cannot import module.\n");
          exit(1);
     }
     printf("Getting test_echo function from module...\n");
     function = PyObject_GetAttrString(module, "test_echo");
     if (!PyCallable_Check(function)) {
          fprintf(stderr, "Cannot get function.\n");
          exit(1);
     }
     printf("Calling function\n");
     PyObject_CallFunctionObjArgs(function, (PyObject *)result, NULL);
     printf("Done\n");
     //Py_Finalize();
}
