#include <stdio.h>
#include <Python.h>
#include "mex.h"
#include "pymex.h"
#include <numpy/arrayobject.h>

static int nlhs, nrhs;
static mxArray **plhs;
static const mxArray **prhs;

static PyObject* mex_argin(PyObject *self, PyObject *args)
{
     if (!PyArg_ParseTuple(args, ":mex_argin"))
          return NULL;
     return Py_BuildValue("i", nrhs);
}

static PyMethodDef mex_methods[] = {
     { "mex_argin", mex_argin, METH_VARARGS,
       "Input arguments." },
     { NULL, NULL, 0, NULL }
};

static PyObject* build_value_from_mxarray(const mxArray *a)
{
     PyObject *result;
     const char *format;
     const void *d;
     int dims[2], type, i, j;

     d = mxGetData(a);
     dims[0] = mxGetM(a);
     dims[1] = mxGetN(a);

     switch (mxGetClassID(a)) {
     case mxUNKNOWN_CLASS:
          mexErrMsgTxt("Cannot pass unknown class to Python.");
     case mxCELL_CLASS:
          mexErrMsgTxt("Cannot pass cell to Python.");
     case mxSTRUCT_CLASS:
          //result = PyArray_SimpleNew(2, dims, PyArray_OBJECT);
          result = PyList_New(dims[0] * dims[1]);
          for (j = 0; j < dims[1]; j++) {
               for (i = 0; i < dims[0]; i++) {
                    //Py_INCREF(Py_None);
                    PyList_SetItem(result, j * dims[0] + i,
                                   Py_None);
               }
          }
          return result;
     case mxLOGICAL_CLASS:
          mexErrMsgTxt("Cannot pass logical to Python.");
     case mxCHAR_CLASS:
          mexErrMsgTxt("Cannot pass char to Python.");

          /* XXX: Use PyArray_NewFromDescr to specify column-major
           * order. */
          /* It should be ok to cast away the const here since the
           * mxArray has dynamic extent. */
#define FOO(type) return PyArray_SimpleNewFromData(2, dims, type, (void *)d)
     case mxDOUBLE_CLASS:
          FOO(PyArray_DOUBLE);
     case mxSINGLE_CLASS:
          FOO(PyArray_FLOAT);
     case mxINT8_CLASS:
          FOO(PyArray_BYTE);
     case mxINT16_CLASS:
          FOO(PyArray_SHORT);
     case mxINT32_CLASS:
          FOO(PyArray_INT);
     case mxINT64_CLASS:
          FOO(PyArray_LONGLONG);
     case mxUINT8_CLASS:
          FOO(PyArray_UBYTE);
     case mxUINT16_CLASS:
          FOO(PyArray_USHORT);
     case mxUINT32_CLASS:
          FOO(PyArray_UINT);
     case mxUINT64_CLASS:
          FOO(PyArray_ULONGLONG);

     case mxFUNCTION_CLASS:
          mexErrMsgTxt("Cannot pass function handle to Python.");
     default:
          mexErrMsgTxt("Cannot pass object of unexpected type to Python.");
     }
     return result;
}

static PyObject* build_argin_list(void)
{
     PyObject *result;
     int i;

     result = PyList_New(nrhs);
     for (i = 0; i < nrhs; i++) {
          PyList_SetItem(result, i, build_value_from_mxarray(prhs[i]));
     }
     return result;
}

void 
mexFunction(int anlhs, mxArray *aplhs[], int anrhs, const mxArray *aprhs[])
{
     const char *python_script = get_arg_string(anrhs, aprhs, 0);
     anrhs--;
     aprhs++;
     
     nlhs = anlhs, plhs = aplhs, nrhs = anrhs, prhs = aprhs;

     Py_Initialize();

     PyObject *module = Py_InitModule("mex", mex_methods);
     import_array();

     PyObject *dict = PyModule_GetDict(module);
     PyObject *nargout = Py_BuildValue("i", nlhs);
     PyDict_SetItemString(dict, "nargout", nargout);
     Py_DECREF(nargout);
     PyObject *nargin = Py_BuildValue("i", nrhs);
     PyDict_SetItemString(dict, "nargin", nargin);
     Py_DECREF(nargin);

     PyObject *argin = build_argin_list();
     PyDict_SetItemString(dict, "argin", argin);
     Py_DECREF(argin);

     FILE *f = fopen(python_script, "r");
     if (!f)
          mexErrMsgTxt("Failed to open python script.");
     PyRun_SimpleFile(f, python_script);
     fclose(f);

     /* Don't call Py_Finalize. */
     anlhs = nlhs, aplhs = plhs, anrhs = nrhs, aprhs = prhs;
}
