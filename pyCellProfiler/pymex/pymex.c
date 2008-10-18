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

/* Workaround because I cannot get PyString_Decode to work. */
PyObject* PyString_Decode_workaround(const char *c_string, Py_ssize_t length,
                                     const char *encoding,
                                     const char *errors)
{
     PyObject *raw = PyString_FromString(c_string);
     PyObject *decoded = PyObject_CallMethod(raw, "decode", "s", encoding, 
                                             errors);
     Py_DECREF(raw);
     return decoded;
}

PyObject* mxarray_to_python(const mxArray *mx)
{
     PyObject *result, *value;
     const char *format;
     const void *real, *imag;
     int dims[2], type, i, j, k, complex, n;
     double *pr, pi;
     mxArray *cell;

     import_array();

     real = mxGetData(mx);
     imag = mxGetImagData(mx);
     n = mxGetNumberOfElements(mx);
     dims[0] = mxGetM(mx);
     dims[1] = mxGetN(mx);
     mxClassID class_id = mxGetClassID(mx);
     
     if (class_id == mxCHAR_CLASS) {
          if (dims[0] != 1)
               mexErrMsgTxt("A char can only be passed to Python if it is a row vector.");
          result = PyString_Decode_workaround(real, dims[1] * sizeof(mxChar), "windows_1252", "ignore");
          if (!result)
               mexErrMsgTxt("Failed to decode string.");
          return result;
     }

     int npytype = 0;
     PyArrayObject *a;
     switch (mxGetClassID(mx)) {
     case mxDOUBLE_CLASS:
          npytype = imag ? NPY_CDOUBLE : NPY_DOUBLE;
          if (imag) {
               a = (PyArrayObject *)PyArray_SimpleNew(2, dims, npytype);
               for (i = 0; i < n; i++) {
                    ((double *)a->data)[2 * i] = ((double *)real)[i];
                    ((double *)a->data)[2 * i + 1] = ((double *)imag)[i];
               }
               return (PyObject *)a;
          }
          break;
     case mxSINGLE_CLASS:
          npytype = imag ? NPY_CFLOAT : NPY_FLOAT;
          if (imag) {
               a = (PyArrayObject *)PyArray_SimpleNew(2, dims, npytype);
               for (i = 0; i < n; i++) {
                    ((float *)a->data)[2 * i] = ((float *)real)[i];
                    ((float *)a->data)[2 * i + 1] = ((float *)imag)[i];
               }
               return (PyObject *)a;
          }
          break;
     case mxINT8_CLASS: npytype = NPY_BYTE; break;
     case mxUINT8_CLASS: npytype = NPY_UBYTE; break;
     case mxINT16_CLASS: npytype = NPY_SHORT; break;
     case mxUINT16_CLASS: npytype = NPY_USHORT; break;
     case mxINT32_CLASS: npytype = NPY_INT; break;
     case mxUINT32_CLASS: npytype = NPY_UINT; break;
     case mxINT64_CLASS: npytype = NPY_LONGLONG; break;
     case mxUINT64_CLASS: npytype = NPY_ULONGLONG; break;
     }
     if (npytype != 0) {
          a = (PyArrayObject *)PyArray_SimpleNew(2, dims, npytype);
          memcpy(a->data, real, n * PyArray_ITEMSIZE(a));
          return (PyObject *)a;
     }

     /* XXX: The following is ok for cell arrays, but should be
      * changed for strut and function. */
     result = PyArray_SimpleNew(2, dims, PyArray_OBJECT);
     void **vp = (void **)((PyArrayObject *)result)->data;
     for (i = 0; i < n; i++) {
          switch (mxGetClassID(mx)) {
          case mxUNKNOWN_CLASS:
               mexErrMsgTxt("Cannot pass unknown class to Python.");
          case mxCELL_CLASS:
               cell = mxGetCell(mx, i);
               if (cell)
                    value = mxarray_to_python(cell);
               else {
                    value = Py_None;
                    Py_INCREF(value);
               }
               break;
          case mxSTRUCT_CLASS:
               value = PyDict_New();
               for (k = 0; k < mxGetNumberOfFields(mx); k++) {
                    const char *fn = mxGetFieldNameByNumber(mx, k);
                    mxArray *v = mxGetField(mx, j * dims[0] + i, fn);
                    PyDict_SetItem(value, PyString_FromString(fn),
                                   mxarray_to_python(v));
               }
               break;
          case mxFUNCTION_CLASS:
               /* XXX: Not the right thing! */
               value = Py_None;
               Py_INCREF(value);
               break;
          default:
               mexErrMsgTxt("Cannot pass object of unexpected type to Python.");
          }
          PyArray_SETITEM((PyArrayObject *)result, vp[i], value);
#undef INDEX               
     }

     return result;
}

mxArray *python_to_mxarray(PyObject *object)
{

}
