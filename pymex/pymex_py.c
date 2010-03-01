#include <Python.h>
#include "pymex.h"

PyObject *pymex_make_array(int m, int n)
{
     PyObject *module = PyImport_ImportModule("pymex");
     if (!module) error("Failed to import pymex module.");
     PyObject *class = PyObject_GetAttrString(module, "Array");
     //Py_DECREF(module);
     if (!class) error("Failed to get pymex.Array class.");
     PyObject *args = Py_BuildValue("(i,i)", m, n);
     PyObject *instance = PyObject_CallObject(class, args);
     //Py_DECREF(args);
     //Py_DECREF(class);
     return instance;
}

void pymex_array_setitem(PyObject *array, int i, int j, PyObject *value)
{
     PyObject_CallMethod(array, "__setitem__", "(i,i,o)", i, j, value);
}
