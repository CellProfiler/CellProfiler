#include <Python.h>
#include "mex.h"
#include "pymex.h"

void 
mexFunction(int anlhs, mxArray *aplhs[], int anrhs, const mxArray *aprhs[])
{
     Py_Initialize();

     PyRun_SimpleString(get_arg_string(anrhs, aprhs, 0));

     /* Don't call Py_Finalize. */
}
