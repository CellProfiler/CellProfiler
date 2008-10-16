#include <Python.h>
#include "mex.h"
#include "pymex.h"
#if 0
#include <numpy/arrayobject.h>
#endif

void 
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
     Py_Initialize();
     //import_array();

     PyObject *module = PyImport_ImportModule(get_arg_string(nrhs, prhs, 0));
     if (!module)
          mexErrMsgTxt("Cannot import module.");
     PyObject *function = PyObject_GetAttrString(module, 
                                                 get_arg_string(nrhs, prhs, 1));
     if (!PyCallable_Check(function))
          mexErrMsgTxt("Cannot call something that is not callable.");

     PyObject *args = PyTuple_New(nrhs - 2);
     int i;
     for (i = 0; i < nrhs - 2; i++) {
          PyTuple_SetItem(args, i, mxarray_to_python(prhs[i + 2]));
     }

     PyObject *value = PyObject_CallObject(function, args);
     Py_DECREF(args);

     //Py_DECREF(value);
     Py_DECREF(module);

     /* Don't call Py_Finalize. */
}
