#include <Python.h>
#include "mex.h"

void 
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
     Py_Initialize();
     bar();
}
