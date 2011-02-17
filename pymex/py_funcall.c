#include <Python.h>
#include "mex.h"
#include "pymex.h"
#if 0
#include <numpy/arrayobject.h>
#endif

void 
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
     PyObject *module=NULL;
     PyObject *function=NULL;
     PyObject *args=NULL;
     PyObject *value=NULL;
     int i;

     pymex_init();
     VERBOSEPRINT("Entering mexFunction for py_funcall\n");

     module = PyImport_ImportModule(get_arg_string(nrhs, prhs, 0));
     if (!module)
          mexErrMsgTxt("Cannot import module.\n");
     VERBOSEPRINT1("Imported module %s\n",get_arg_string(nrhs, prhs, 0));

     function = PyObject_GetAttrString(module, 
                                                 get_arg_string(nrhs, prhs, 1));
     if (!PyCallable_Check(function))
          mexErrMsgTxt("Cannot call something that is not callable.\n");
     VERBOSEPRINT1("Found callable function %s\n",get_arg_string(nrhs, prhs,1));

     if (nrhs > 2) {
          args = PyTuple_New(nrhs - 2);
          for (i = 0; i < nrhs - 2; i++) {
               PyTuple_SetItem(args, i, mxarray_to_python(prhs[i + 2]));
               VERBOSEPRINT1("Parsed arg # %d\n",i);
          }
     } else {
          args = NULL;
     }

     value = PyObject_CallObject(function, args);
     if (! value) {
          PrintStackTrace();
          goto exit;
     }
     VERBOSEPRINT2("Called %s\nValue = %x\n",get_arg_string(nrhs, prhs,1),value);

     if (nlhs > 0) {
          plhs[0] = python_to_mxarray(value);
          VERBOSEPRINT("Encoded output value\n");
     }
  exit:
     if (args)
          Py_DECREF(args);
     if (value)
          Py_DECREF(value);
     if (module)
          Py_DECREF(module);
     if (function)
          Py_DECREF(function);

     /* Don't call Py_Finalize. */
     VERBOSEPRINT("Exiting mexFunction for py_funcall\n");
}
