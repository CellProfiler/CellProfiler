#include <Python.h>
#include "mex.h"
#include "pymex.h"
#include <numpy/arrayobject.h>

void error(const char *message)
{
     mexErrMsgTxt(message);
}

const char *get_arg_string(int nrhs, const mxArray *prhs[], int i)
{
     if (nrhs < i + 1)
          mexErrMsgTxt("Too few input arguments.");
     if (mxIsChar(prhs[i]) != 1)
          mexErrMsgTxt("Input argument must be a string.");
     if (mxGetM(prhs[i]) != 1)
          mexErrMsgTxt("Input argument must be a row vector.");
     return mxArrayToString(prhs[i]);
}

PyObject* mxarray_to_python(const mxArray *a)
{
     PyObject *result, *value;
     const char *format;
     const void *d, *imaginary;
     int dims[2], type, i, j, k, complex;
     double *pr, pi;
     mxArray *cell;

     import_array();

     d = mxGetData(a);
     imaginary = mxGetImagData(a);
     dims[0] = mxGetM(a);
     dims[1] = mxGetN(a);
     mxClassID class_id = mxGetClassID(a);
     
     if (class_id == mxCHAR_CLASS) {
          if (dims[0] != 1)
               mexErrMsgTxt("A char can only be passed to Python if it is a row vector.");
          result = PyString_Decode(d, dims[1] * sizeof(mxChar), "utf_16", "strict");
     } else {
          result = PyArray_SimpleNew(2, dims, PyArray_OBJECT);
          void **vp = (void **)((PyArrayObject *)result)->data;
          for (i = 0; i < dims[0]; i++) {
               for (j = 0; j < dims[1]; j++) {
#define INDEX (j * dims[0] + i)
                    switch (mxGetClassID(a)) {
                    case mxUNKNOWN_CLASS:
                         mexErrMsgTxt("Cannot pass unknown class to Python.");
                    case mxCELL_CLASS:
                         cell = mxGetCell(a, INDEX);
                         if (cell)
                              value = mxarray_to_python(cell);
                         else {
                              value = Py_None;
                              Py_INCREF(value);
                         }
                         break;
                    case mxSTRUCT_CLASS:
                         value = PyDict_New();
                         for (k = 0; k < mxGetNumberOfFields(a); k++) {
                              const char *fn = mxGetFieldNameByNumber(a, k);
                              mxArray *v = mxGetField(a, j * dims[0] + i, fn);
                              PyDict_SetItem(value, PyString_FromString(fn),
                                             mxarray_to_python(v));
                         }
                         break;
#define DREF(TYPE) (((TYPE *)d)[INDEX])
#define IREF(TYPE) (((TYPE *)imaginary)[INDEX])
                    case mxLOGICAL_CLASS: value = PyBool_FromLong(DREF(mxLogical)); break;
                    case mxDOUBLE_CLASS: 
                         if (imaginary)
                              value = PyComplex_FromDoubles(DREF(double), IREF(double));
                         else
                              value = PyFloat_FromDouble(DREF(double)); 
                         break;
                    case mxSINGLE_CLASS: 
                         if (imaginary)
                              value = PyComplex_FromDoubles(DREF(float), IREF(float));
                         else
                              value = PyFloat_FromDouble(DREF(float)); 
                         break;
                    case mxINT8_CLASS: value = PyInt_FromLong(DREF(char)); break;
                    case mxINT16_CLASS: value = PyInt_FromLong(DREF(short)); break;
                    case mxINT32_CLASS: value = PyInt_FromLong(DREF(int)); break;
                    case mxINT64_CLASS: value = PyLong_FromLongLong(DREF(long long)); break;
                    case mxUINT8_CLASS: value = PyInt_FromLong(DREF(unsigned char)); break;
                    case mxUINT16_CLASS: value = PyInt_FromLong(DREF(unsigned short)); break;
                    case mxUINT32_CLASS: value = PyLong_FromLongLong(DREF(unsigned int)); break;
                    case mxUINT64_CLASS: value = PyLong_FromUnsignedLongLong(DREF(unsigned long long)); break;
#undef DREF
#undef IREF
                    case mxFUNCTION_CLASS:
                         /* XXX: Not the right thing! */
                         value = Py_None;
                         Py_INCREF(value);
                         break;
                    default:
                         mexErrMsgTxt("Cannot pass object of unexpected type to Python.");
                    }
                    PyArray_SETITEM((PyArrayObject *)result, vp++, value);
#undef INDEX               
               }
          }
     }

     return result;
}

mxArray *python_to_mxarray(PyObject *object)
{

}
