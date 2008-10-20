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
     PyObject *raw, *decoded;
     raw = PyString_FromStringAndSize(c_string, length);
     //decoded = PyString_AsDecodedString(raw, encoding, errors);
     decoded = PyObject_CallMethod(raw, "decode", "s,s", encoding, errors);
     Py_DECREF(raw);
     return decoded;
}

static PyObject *mxchar_to_python(const mxArray *mx)
 {
      PyObject *result;

      if (mxGetM(mx) != 1)
           mexErrMsgTxt("A char can only be passed to Python if it is a "
                        "row vector.");
      result = PyString_Decode_workaround(mxGetData(mx), 
                                          mxGetN(mx) * sizeof(mxChar),
                                          "utf_16",
                                          "replace");
      if (!result) {
           PyErr_Print();
           mexErrMsgTxt("Failed to decode string.");
      }
      return result;
 }

static PyObject *mxnumeric_to_python(const mxArray *mx)
{
     const void *real, *imag;
     int n, dims[2], npytype, i;
     PyObject *result;

     real = mxGetData(mx);
     imag = mxGetImagData(mx);
     n = mxGetNumberOfElements(mx);
     dims[0] = mxGetM(mx);
     dims[1] = mxGetN(mx);

#define DATA(TYPE) ((TYPE *)PyArray_DATA(result))
     switch (mxGetClassID(mx)) {
     case mxDOUBLE_CLASS:
          npytype = imag ? NPY_CDOUBLE : NPY_DOUBLE;
          if (imag) {
               result = PyArray_SimpleNew(2, dims, npytype);
               for (i = 0; i < n; i++) {
                    DATA(double)[2 * i] = ((double *)real)[i];
                    DATA(double)[2 * i + 1] = ((double *)imag)[i];
               }
               return result;
          }
          break;
     case mxSINGLE_CLASS:
          npytype = imag ? NPY_CFLOAT : NPY_FLOAT;
          if (imag) {
               result = PyArray_SimpleNew(2, dims, npytype);
               for (i = 0; i < n; i++) {
                    DATA(float)[2 * i] = ((float *)real)[i];
                    DATA(float)[2 * i + 1] = ((float *)imag)[i];
               }
               return result;
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
     case mxLOGICAL_CLASS: npytype = NPY_BOOL; break;
     default:
          mexErrMsgTxt("Unexpected class ID.");
     }
#undef DATA

     result = PyArray_SimpleNew(2, dims, npytype);
     memcpy(PyArray_BYTES(result), real, n * PyArray_ITEMSIZE(result));
     return result;
}

static PyArray_Descr *mxstruct_dtype(const mxArray *mx)
{
     PyObject *list;
     int nfields, i;
     PyArray_Descr *descr;

     nfields = mxGetNumberOfFields(mx);
     list = PyList_New(nfields);
     for (i = 0; i < nfields; i++)
          PyList_SetItem(list, i, Py_BuildValue("(s, s)", 
                                                mxGetFieldNameByNumber(mx, i),
                                                "object"));
     if (PyArray_DescrConverter(list, &descr) != NPY_SUCCEED) {
          PyErr_Print();
          mexErrMsgTxt("Failed to determine dtype of struct array.");
     }
     Py_DECREF(list);
     return descr;
}

static PyObject *mxstruct_to_python(const mxArray *mx)
{
     PyArray_Descr *dtype;
     PyObject *result, **itemptr;
     int n, dims[2], nfields, i, k;
     const mxArray *child;
     
     dtype = mxstruct_dtype(mx);
     n = mxGetNumberOfElements(mx);
     dims[0] = mxGetM(mx);
     dims[1] = mxGetN(mx);
     result = PyArray_SimpleNewFromDescr(2, dims, dtype);
     nfields = mxGetNumberOfFields(mx);
     itemptr = PyArray_DATA(result);
     for (i = 0; i < n; i++)
          for (k = 0; k < nfields; k++)
               *itemptr++ = mxarray_to_python(mxGetFieldByNumber(mx, i, k));
     return result;
}

static PyObject *mxcell_to_python(const mxArray *mx)
{
     PyObject *result, **itemptr, *value;
     int dims[2], n, i;
     const mxArray *cell;
     
     n = mxGetNumberOfElements(mx);
     dims[0] = mxGetM(mx);
     dims[1] = mxGetN(mx);
     result = PyArray_SimpleNew(2, dims, PyArray_OBJECT);
     itemptr = PyArray_DATA(result);
     for (i = 0; i < n; i++) {
          cell = mxGetCell(mx, i);
          if (cell)
               value = mxarray_to_python(cell);
          else {
               value = Py_None;
               Py_INCREF(value);
          }
          *itemptr++ = value;
     }
     return result;
}

static PyObject *mxfunction_to_python(const mxArray *mx)
{
     return PyCObject_FromVoidPtr((void *)mx, NULL);
}

PyObject* mxarray_to_python(const mxArray *mx)
{
     import_array();

     switch (mxGetClassID(mx)) {
     case mxCHAR_CLASS: 
          return mxchar_to_python(mx);
     case mxDOUBLE_CLASS:
     case mxSINGLE_CLASS:
     case mxINT8_CLASS:
     case mxUINT8_CLASS:
     case mxINT16_CLASS:
     case mxUINT16_CLASS:
     case mxINT32_CLASS:
     case mxUINT32_CLASS:
     case mxINT64_CLASS:
     case mxUINT64_CLASS:
     case mxLOGICAL_CLASS:
          return mxnumeric_to_python(mx);
     case mxSTRUCT_CLASS:
          return mxstruct_to_python(mx);
     case mxCELL_CLASS:
          return mxcell_to_python(mx);
     case mxFUNCTION_CLASS:
          return mxfunction_to_python(mx);
     case mxUNKNOWN_CLASS:
          mexErrMsgTxt("Cannot pass unknown class to Python.");
     default:
          mexErrMsgTxt("Cannot pass object of unexpected type to Python.");
     }
}

mxArray *python_to_mxarray(PyObject *object)
{

}
