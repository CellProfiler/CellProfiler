#ifndef PYMEX_H
#define PYMEX_H

#if 0
#ifndef Py_PYTHON_H
typedef void PyObject;
#endif

#ifndef mex_h
typedef void mxArray;
#endif
#endif

const char *get_arg_string(int nrhs, const mxArray *prhs[], int i);

PyObject *pymex_make_array(int m, int n);
void pymex_array_setitem(PyObject *array, int i, int j, PyObject *value);
PyObject* mxarray_to_python(const mxArray *a);

#endif
