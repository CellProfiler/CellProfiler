#include <Python.h>
#include "mex.h"
#include "pymex.h"
#include <numpy/arrayobject.h>

static PyObject *
mexMsg(PyObject *self, PyObject *args)
{
     char *s;
     int ok;
     ok = PyArg_ParseTuple(args,"s",&s);
     if (ok) {
          mexPrintf(s);
     }
     return Py_None;
}

static PyObject *
mexErrMsg(PyObject *self, PyObject *args)
{
     char *s;
     int ok;

     ok = PyArg_ParseTuple(args,"s",&s);
     if (ok) {
          mexWarnMsgTxt(s);
     }
     return Py_None;
}

long debug_level = 0;
static PyObject *
mexSetDebugLevel(PyObject *self, PyObject *args)
{
     int ok = PyArg_ParseTuple(args,"l", &debug_level);
     if (! ok) {
          mexWarnMsgTxt("mexSetDebugLevel takes an integer argument");
     }
     mexPrintf("Debug level set to %d\n",debug_level);
     return Py_None;
}

static PyObject *
mexGetDebugLevel(PyObject *self, PyObject *args)
{
     return PyInt_FromLong(debug_level);
}

static PyObject *
mexImplementsMapping(PyObject *self, PyObject *args)
{
     PyObject *object;
     int ok = PyArg_ParseTuple(args,"O",&object);
     if (! ok) {
          mexWarnMsgTxt("ImplementsMapping takes a single argument");
          Py_RETURN_FALSE;
     } else if (PyMapping_Check(object)) {
          Py_RETURN_TRUE;
     }
     Py_RETURN_FALSE;
}

static PyObject *
mexAppendPath(PyObject *self, PyObject *args)
{
     PyObject *module=NULL;
     PyObject *path=NULL;
     PyObject *path_append=NULL;
     PyObject *append_arg=NULL;
     PyObject *value=NULL;
     char *errmsg;
     int success = 0;

     VERBOSEPRINT("Entering mexAppendPath\n");
     if (PySequence_Length(args) != 1) {
          errmsg = "mexAppendPath takes a string argument\n";
          goto exit;
     }

     VERBOSEPRINT("Has one argument\n");
     append_arg = PySequence_GetItem(args,0);
     if (! append_arg) {
          errmsg = "Failed to get the argument\n";
          goto exit;
     }

     VERBOSEPRINT("Got the argument\n");
     if ((! PyString_Check(append_arg)) && (! PyUnicode_Check(append_arg))) {
          errmsg = "Argument is not a string\n";
          goto exit;
     }

     VERBOSEPRINT("Argument was a string\n");

     module = PyImport_ImportModule("sys");
     if (!module) {
          errmsg = "Cannot import sys.\n";
          goto exit;
     }

     VERBOSEPRINT("Imported sys\n");
     path = PyObject_GetAttrString(module,"path");
     if (!path) {
          errmsg = "Could not find sys.path.\n";
          goto exit;
     }

     VERBOSEPRINT("Found sys.path\n");
     path_append = PyObject_GetAttrString(path,"append");
     if (! path_append) {
          errmsg = "Could not find sys.path.append.\n";
          goto exit;
     }
     VERBOSEPRINT("Found sys.path.append\n");
     
     if (!PyCallable_Check(path_append)) {
          errmsg = "Cannot call something that is not callable.\n";
          goto exit;
     }
     VERBOSEPRINT("sys.path.append is callable\n");
     value = PyObject_CallObject(path_append, args);
     if (! value) {
          PrintStackTrace();
          goto exit;
     }
     success = 1;
  exit:
     if (append_arg)
          Py_DECREF(append_arg);
     if (module)
          Py_DECREF(module);
     if (path)
          Py_DECREF(path);
     if (path_append)
          Py_DECREF(path_append);
     if (!success)
          mexErrMsgTxt(errmsg);
     return value;
}

static PyObject *
mexReload(PyObject *self, PyObject *args)
{
     PyObject *builtin=NULL;
     PyObject *module =NULL;
     PyObject *reload=NULL;
     PyObject *value=NULL;
     PyObject *tuple=NULL;
     PyObject *module_name=NULL;
     char *errmsg = NULL;

     VERBOSEPRINT("Entering mexAppendPath\n");
     if (PySequence_Length(args) != 1) {
          errmsg = "reload takes a string argument\n";
          goto exit;
     }
     
     module_name = PySequence_GetItem(args,0);
     if (! module_name) {
          errmsg = "Cannot get module name from arguments\n";
          goto exit;
     }
     module = PyImport_Import(module_name);
     if (! module) {
          errmsg = "Cannot import module\n";
          goto exit;
     }

     VERBOSEPRINT("Has one argument\n");
     builtin = PyImport_ImportModule("__builtin__");
     if (!builtin) {
          errmsg = "Cannot import __builtin__.\n";
          goto exit;
     }

     VERBOSEPRINT("Got __builtin__ module\n");
     reload = PyObject_GetAttrString(builtin,"reload");
     if (!reload) {
          errmsg = "Could not find reload.\n";
          goto exit;
     }

     VERBOSEPRINT("Found reload\n");

     if (!PyCallable_Check(reload)) {
          errmsg = "Cannot call something that is not callable.\n";
          goto exit;
     }
     VERBOSEPRINT("reload is callable\n");

     tuple = PyTuple_Pack(1,module);
     if (! tuple) {
          errmsg = "Could not allocate tuple\n";
          goto exit;
     }
     VERBOSEPRINT("Allocated argument tuple\n");
     
     value = PyObject_CallObject(reload, tuple);
     VERBOSEPRINT("Called reload\n");

  exit:
     if (module_name) {
          Py_DECREF(module_name);
     }
     if (module) {
          Py_DECREF(module);
     }
     if (builtin) {
          Py_DECREF(builtin);
     }
     if (reload) {
          Py_DECREF(reload);
     }
     if (errmsg) {
          mexErrMsgTxt(errmsg);
     }
     return value;
}
static PyMethodDef _pymex_functions[] = {
     { "mexMsg", (PyCFunction) mexMsg, METH_VARARGS, "Displays a message in Matlab" },
     { "mexErrMsg", (PyCFunction) mexErrMsg, METH_VARARGS, "Displays an error message in Matlab" },
     { "mexSetDebugLevel", (PyCFunction) mexSetDebugLevel, METH_VARARGS, "Sets the debug level: 0 = none, higher = more verbose" },
     { "mexGetDebugLevel", (PyCFunction) mexGetDebugLevel, METH_VARARGS, "Returns the current debug level" },
     { "ImplementsMapping", (PyCFunction) mexImplementsMapping, METH_VARARGS, "Returns true if the argument supports the mapping interface" },
     { "mexAppendPath", (PyCFunction) mexAppendPath, METH_VARARGS, "Appends a path to PYTHONPATH" },
     { "reload",(PyCFunction) mexReload, METH_VARARGS, "Reloads a module by name rather than by reference" },
     { NULL, NULL, 0}
};

/*********************************
 * pymex_init - initialize python with hooks into Matlab
 *********************************/
void pymex_init(void)
{
     if (!Py_IsInitialized()) {
          Py_Initialize();
          Py_InitModule("pymex",_pymex_functions);
          /*
           * Create a class that implements the "write"
           * function by calling pymex.mexMsg and another
           * that calls pymex.mexErrMsg. Hook these to sys.stdout
           * and sys.stderr
           */
          PyRun_SimpleString(
               "import pymex\n"
               "import sys\n"
               "class __pymex_stdout:\n"
               "    def write(self,message):\n"
               "        pymex.mexMsg(message)\n"
               "    def flush(self):\n"
               "        pass\n"
               "sys.stdout = __pymex_stdout()\n"
               "class __pymex_stderr:\n"
               "    def write(self,message):\n"
               "        pymex.mexErrMsg(message)\n"
               "sys.stderr = __pymex_stderr()\n");
     }
}


void error(const char *message)
{
     mexErrMsgTxt(message);
}

/********************************
 * Print the stack trace of the last error
 ********************************/
void PrintStackTrace(void)
{
     PyRun_SimpleString("import traceback\ntraceback.print_last()\n");
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
     int n, dims[2], i;
     PyObject *result;
     int npytype = 0;

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
     int dims[2], n, d[2],index;
     const mxArray *cell;
     
     n = mxGetNumberOfElements(mx);
     dims[0] = mxGetM(mx);
     dims[1] = mxGetN(mx);
     result = PyArray_SimpleNew(2, dims, PyArray_OBJECT);
     itemptr = PyArray_DATA(result);
     for (d[0] = 0; d[0] < dims[0]; d[0]++) {
          for (d[1] = 0; d[1] < dims[1]; d[1]++) {
               index = mxCalcSingleSubscript(mx, 2, d);
               cell = mxGetCell(mx, index);
               if (cell)
                    value = mxarray_to_python(cell);
               else {
                    value = Py_None;
                    Py_INCREF(value);
               }
               *itemptr++ = value;
          }
     }
     return result;
}

static PyObject *mxfunction_to_python(const mxArray *mx)
{
     return PyCObject_FromVoidPtr((void *)mx, NULL);
}

PyObject* mxarray_to_python(const mxArray *mx)
{
     VERBOSEPRINT("Entering mxarray_to_python\n");
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
     VERBOSEPRINT("Exiting mxarray_to_python\n");
}

static char *dimstrcpy(char *buffer, long ndims,long *dims)
{
     int idx;
     char *ptr = buffer;
     *ptr++='(';
     for (idx = 0; idx<ndims; idx++)
          ptr += sprintf(ptr,"%d,",dims[idx]);
     ptr[-1] = ')';
     return buffer;
}

static long get_dtype_npy_type(PyObject *dtype)
{
     PyObject *dtype_num = PyObject_GetAttrString(dtype,"num");
     long npy_type = NPY_NOTYPE;
     if (! (dtype_num && PyInt_Check(dtype_num))) {
          ERRPRINT("numpy.ndarray.dtype did not have a proper num\n");
          goto exit;
     }
     npy_type = PyInt_AsLong(dtype_num);
  exit:
     Py_DECREF(dtype_num);
     return npy_type;
}

/************************
 * ndstruct_to_mxarray - convert a ndarray containing fields to a structure mxArray
 ************************/
static mxArray *ndstruct_to_mxarray(PyObject *object, PyObject *dtype, int ndims, long *dims,int length)
{
     PyObject *contiguous = NULL;
     PyObject *fields = NULL;
     int nfields, i, k, npy_type;
     char **keys = NULL;
     PyObject **stringized = NULL;
     PyObject *names = NULL;
     PyObject **data;
     mxArray *result;
     
     VERBOSEPRINT("Entering ndstruct_to_mxarray\n");
     fields = PyObject_GetAttrString(dtype,"fields");
     if (! PyMapping_Check(fields)) {
          ERRPRINT("Failed to get fields from dtype\n");
          goto exit;
     }
     names = PyObject_GetAttrString(dtype,"names");
     if (! PyTuple_Check(names)) {
          ERRPRINT("Failed to get names from dtype\n");
          goto exit;
     }
     nfields = PyMapping_Size(fields);
     VERBOSEPRINT1("%d fields detected \n",nfields);

     keys = (char **)mxCalloc(nfields, sizeof(char*));
     stringized = (PyObject **)mxCalloc(nfields, sizeof(PyObject *));
     
     /*
     ** Get the field names from "dtype.names" - that's an ordered tuple, not a
     ** dictionary with unpredictable order.
     */
     for (i=0; i<nfields; i++) {
          PyObject *key = PyTuple_GetItem(names, i);
          if (! key) {
               ERRPRINT1("Failed to get key from tuple for field # %d\n",i);
               goto exit;
          }
          stringized[i] = PyObject_Str(key);
          if (! PyString_Check(stringized[i])) {
               ERRPRINT1("Key # %d couldn't be stringized\n",i);
               goto exit;
          }
          keys[i] = PyString_AsString(stringized[i]);
          VERBOSEPRINT2("Field #%d is %s\n",i,keys[i]);
     }

     /*
     ** Check all fields to make sure they are all objects.
     */
     for (i=0; i<nfields; i++) {
          PyObject *subtype;
          PyObject *item;
          VERBOSEPRINT1("Checking field %s\n",keys[i]);
          item = PyMapping_GetItemString(fields, keys[i]);
          if (! PyTuple_Check(item)) {
               ERRPRINT1("Did not get a tuple item for field %s\n",keys[i]);
               if (item) Py_DECREF(item);
               goto exit;
          }
          subtype = PyTuple_GetItem(item,0);
          if (! subtype) {
               ERRPRINT1("Failed to get sub-type from tuple for field %s\n",keys[i]);
               goto exit;
          }
          if (! PyArray_DescrCheck(subtype)) {
               ERRPRINT2("Subtype for field %s was not an array descriptor.\nIt was type %s\n",keys[i],subtype->ob_type->tp_name);
               Py_DECREF(item);
               goto exit;
          }
          npy_type = get_dtype_npy_type(subtype);
          Py_DECREF(item);
          if (npy_type != NPY_OBJECT) {
               ERRPRINT1("Subtype for field %s is not object - not supported\n",keys[i]);
               goto exit;
          }
     }
     result = mxCreateStructArray(ndims, dims, nfields, keys);
     if (! result) {
          ERRPRINT("Failed to allocate mx struct array");
          goto exit;
     }
     contiguous = PyArray_GETCONTIGUOUS(object);
     data = (PyObject **) PyArray_DATA(contiguous);

     for (i = 0; i < length; i++) {
          VERBOSEPRINT("Encoding element # %d\n",i);
          for (k = 0; k < nfields; k++) {
               mxArray *sub;
               VERBOSEPRINT1("Encoding field %s\n",keys[k]);
               sub = python_to_mxarray(*data++);
               if (! sub) {
                    ERRPRINT2("Failed to encode object at position %d, field %s",i,keys[k]);
                    goto exit;
               }
               mxSetFieldByNumber(result, i, k, sub);
          }
     }
  exit:
     if (fields)
          Py_DECREF(fields);
     if (names)
          Py_DECREF(names);
     if (keys)
          mxFree(keys);
     if (stringized) {
          for (i = 0; i<nfields; i++) {
               if (stringized[i])
                    Py_DECREF(stringized[i]);
          }
          mxFree(stringized);
     }
     if (contiguous) {
          Py_DECREF(contiguous);
     }
     VERBOSEPRINT("Exiting ndstruct_to_mxarray\n");
     return result;
}
/***********************
 * ndarray_to_mxarray - convert an numpy.ndarray to an MX array
 ***********************/
static mxArray *ndarray_to_mxarray(PyObject *object)
{
     Py_ssize_t ndims;
     long *dims = NULL;
     mxArray *result = NULL;
     void *data = NULL;
     int i;
     int length=1;
     PyObject *shape = NULL;
     PyObject *dtype = NULL;
     PyObject *iter = NULL;
     PyObject *contiguous = NULL;
     PyObject *dtype_num = NULL;
     mxClassID mxClass = 0;
     long npy_type = NPY_NOTYPE;
     char buffer[1024];

     /* Get the array dimensions */
     VERBOSEPRINT("Entering ndarray_to_mxarray\n");
     shape = PyObject_GetAttrString(object, "shape");
     if (! shape) {
          ERRPRINT("numpy.ndarray did not have a shape\n");
          goto error;
     }
     if (! PyTuple_Check(shape)) {
          ERRPRINT("numpy.ndarray did not have a tuple shape\n");
          goto error;
     }
     ndims = PyTuple_Size(shape);
     VERBOSEPRINT1("Array has %d dimensions.\n",ndims);
     dims = (long *)mxMalloc(sizeof(long)*ndims);
     for( i=0; i< ndims; i++) {
          PyObject *obDim = PyTuple_GetItem(shape,i);
          if (! PyInt_Check(obDim)) {
               ERRPRINT("Array shape is not a tuple of integers\n");
               goto error;
          }
          dims[i] = PyInt_AsLong(obDim);
          length *= dims[i];
     }
     VERBOSEPRINT1("array dimensions: %s\n",dimstrcpy(buffer,ndims,dims));
     /*
      * Handle different dtypes
      */
     dtype = PyObject_GetAttrString(object, "dtype");
     if (! dtype) {
          ERRPRINT("numpy.ndarray did not have a dtype\n");
          goto error;
     }
     npy_type = get_dtype_npy_type(dtype);
     switch(npy_type) {
     case NPY_VOID: /* This is where structures show up */
          VERBOSEPRINT("numpy array has type NPY_VOID\n");
          if (PyArray_HASFIELDS(object)) {
               result = ndstruct_to_mxarray(object, dtype, ndims, dims, length);
          } else {
               ERRPRINT("NPY_VOID array does not have fields\n");
          }
          break;
     case NPY_OBJECT: {
          PyObject *iter = PyObject_GetAttrString(object,"flat");
          int idx;

          VERBOSEPRINT("Decoding array of objects\n");
          if (! (iter && PyIter_Check(iter))) {
               ERRPRINT("numpy.ndarray.flat did not return a proper iterator\n");
               goto error;
          }
          result = mxCreateCellArray(ndims, dims);
          if (! result) {
               strcpy(buffer,"mxCreateCellArray returned NULL for an array of dimension ");
               strcpy(dimstrcpy(buffer+strlen(buffer), ndims, dims),"\n");
               ERRPRINT(buffer);
               goto error;
          }
          for (idx = 0; idx < length; idx++) {
               PyObject *py_value;
               mxArray *mx_value;
               py_value = (*iter->ob_type->tp_iternext)(iter);
               if (! py_value) {
                    ERRPRINT("numpy.ndarray's __iter__'s next function failed\n");
                    goto error;
               }
               Py_INCREF(py_value);
               mx_value = python_to_mxarray(py_value);
               Py_DECREF(py_value);
               if (! mx_value) {
                    ERRPRINT("ndarray_to_mxarray failed while trying to translate a subobject to an mxarray\n");
                    goto error;
               }
               mxSetCell(result, idx, mx_value);
          }
          break;
     }
     case NPY_DOUBLE:
          mxClass = mxDOUBLE_CLASS;
          goto NPY_ALL;

     case NPY_FLOAT:
          mxClass = mxSINGLE_CLASS;
          goto NPY_ALL;
     case NPY_BYTE:
          mxClass = mxINT8_CLASS;
          goto NPY_ALL;
     case NPY_UBYTE:
          mxClass = mxUINT8_CLASS;
          goto NPY_ALL;
     case NPY_SHORT:
          mxClass = mxINT16_CLASS;
          goto NPY_ALL;
     case NPY_USHORT:
          mxClass = mxUINT16_CLASS;
          goto NPY_ALL;
     case NPY_INT:
     case NPY_LONG:
          mxClass = mxINT32_CLASS;
          goto NPY_ALL;
     case NPY_UINT:
     case NPY_ULONG:
          mxClass = mxUINT32_CLASS;
          goto NPY_ALL;
     case NPY_LONGLONG:
          mxClass = mxINT64_CLASS;
          goto NPY_ALL;
     case NPY_ULONGLONG:
          mxClass = mxUINT64_CLASS;
     NPY_ALL:
          VERBOSEPRINT2("Type #%d, mxClassID = %d\n",npy_type, mxClass);
          contiguous = PyArray_ContiguousFromAny(object, npy_type, 1,1024);
          VERBOSEPRINT2("Length = %d, item size = %d\n",length, PyArray_ITEMSIZE(contiguous));
          result = mxCreateNumericArray(ndims, dims, mxClass,0);
          VERBOSEPRINT("Created mxArray\n");
          memcpy(mxGetData(result), PyArray_BYTES(contiguous), length * PyArray_ITEMSIZE(contiguous));
          VERBOSEPRINT("Copied array\n");
          Py_DECREF(contiguous);
          break;
     case NPY_CFLOAT: {
          int i;
          float *real,*imaginary;
          float *npy_data;

          contiguous = PyArray_ContiguousFromAny(object, NPY_CFLOAT, 1,1024);
          result = mxCreateNumericArray(ndims, dims, mxSINGLE_CLASS, 1);
          real = (float *)mxGetData(result);
          npy_data = (float *)PyArray_BYTES(contiguous);
          imaginary = (float *)mxGetImagData(result);
          for (i = 0; i < length; i++) {
               real[i] = npy_data[2*i];
               imaginary[i] = npy_data[2*i+1];
          }
          break;
     }

      case NPY_CDOUBLE: {
          int i;
          double *real,*imaginary;
          double *npy_data;
          contiguous = PyArray_ContiguousFromAny(object, NPY_CDOUBLE, 1,1024);
          VERBOSEPRINT("Made complex double array contiguous.\n");
          result = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, 1);
          VERBOSEPRINT("Created complex mxArray of doubles.\n");
          real = (double *)mxGetData(result);
          npy_data = (double *)PyArray_BYTES(contiguous);
          imaginary = (double *)mxGetImagData(result);
          for (i = 0; i < length; i++) {
               real[i] = npy_data[2*i];
               imaginary[i] = npy_data[2*i+1];
          }
          break;
      }
     default: {
          PyObject *dtypename=PyObject_GetAttrString(dtype,"str");
          ERRPRINT1("Unhandled dtype: %s",PyString_AsString(dtypename));
          Py_DECREF(dtypename);
          goto error;
     }
     }
     goto exit;
  error:
     if (dims)
          mxFree(dims);
     result=mxCreateNumericMatrix(1,1,mxLOGICAL_CLASS,0);
     data = mxGetData(result);
     *((bool *)data) = 0;
  exit:
     if (dims)
          mxFree(dims);
     if (shape)
          Py_DECREF(shape);
     if (dtype)
          Py_DECREF(dtype);
     if (dtype_num)
          Py_DECREF(dtype_num);
     if (iter)
          Py_DECREF(iter);
     VERBOSEPRINT("Exiting ndarray_to_mxarray\n");
     return result;
}

/******************************
 * Convert a python dictionary to a 1x1 struct matrix
 ******************************/
static mxArray *mapping_to_mxarray(PyObject *object)
{
     int nItems,i,pass;
     PyObject *items=NULL;
     mxArray *result=NULL;
     char **keys = NULL;
     PyObject **stringized = NULL;
     mxArray *mxValue;

     VERBOSEPRINT("Entering mapping_to_mxarray\n");
     nItems = PyMapping_Length(object);
     VERBOSEPRINT1("%d mapping keys\n",nItems);
     keys = (char **)mxCalloc(nItems, sizeof(char *));
     stringized = (PyObject **)mxCalloc(nItems, sizeof(PyObject *));
     if (! keys) {
          ERRPRINT("Not enough memory for keys array\n");
          goto exit;
     }
     VERBOSEPRINT("Allocated memory for keys\n");
     items = PyMapping_Items(object);
     if (! PyList_Check(items)) {
          ERRPRINT("Failed to get list of items from dictionary.\n");
          goto exit;
     }
     /*
     ** Loop twice - once to get the keys from the tuples and make the array
     **              a second time to fill in the values for the array
     */
     for (pass = 0; pass < 2; pass++) {
          VERBOSEPRINT1("Pass %d\n",pass);
          if (pass == 1) {
               /*
               ** Create the array on the second pass (after we have the keys)
               */
               result = mxCreateStructMatrix(1,1,nItems,keys);
          }
          for (i=0; i<nItems; i++) {
               PyObject *kv;
               PyObject *item = PyList_GetItem(items,i);
               if (! PyTuple_Check(item)) {
                    ERRPRINT1("Item # %d was not a tuple.\n",i);
                    goto exit;
               } 
               kv = PyTuple_GetItem(item, pass);
               if (! kv) {
                    ERRPRINT2("Failed to get %s for item # %d\n",pass?"key":"value",i);
                    goto exit;
               }
               switch(pass) {
               case 0:  /* Key */
                    stringized[i] = PyObject_Str(kv);
                    if (! PyString_Check(stringized[i])) {
                         ERRPRINT("Key # %d did not stringize\n",i);
                         goto exit;
                    }
                    keys[i] = PyString_AsString(stringized[i]);
                    break;
               case 1: /* value */
                    mxValue = python_to_mxarray(kv);
                    if (! mxValue) {
                         ERRPRINT1("Value for %s was untranslatable\n",keys[i]);
                         goto exit;
                    }
                    mxSetField(result, 0, keys[i], mxValue);
                    break;
               }
          }
     }
  exit:
     if (items)
          Py_DECREF(items);
     if (keys)
          mxFree(keys);
     if (stringized) {
          for (i=0; i<nItems; i++) {
               if (stringized[i])
                    Py_DECREF(stringized[i]);
          }
          mxFree(stringized);
     }
     VERBOSEPRINT("Exiting mapping_to_mxarray\n");
     return result;
}

/****************************************
 * sequence_to_mxarray - convert an object that follows the sequence protocol
 *                       to an mxArray of the sequence's objects
 ****************************************/
static mxArray *sequence_to_mxarray(PyObject *object)
{
     mxArray *result=NULL;
     int nItems = 0;
     int i;

     VERBOSEPRINT("Entering sequence_to_mxarray\n");
     nItems = PySequence_Size(object);
     if (nItems == -1) {
          ERRPRINT("Failed to get sequence size.\n");
          goto exit;
     }
     VERBOSEPRINT1("%d items found\n",nItems);
     result = mxCreateCellMatrix(1,nItems);
     for (i=0; i<nItems; i++) {
          PyObject *item = PySequence_GetItem(object, i);
          mxArray *mxItem;
          if (! item) {
               ERRPRINT1("Failed to get sequence item # %d",i);
               goto exit;
          }
          mxItem = python_to_mxarray(item);
          Py_DECREF(item);
          if (! mxItem) {
               ERRPRINT1("Failed to encode item # %d",i);
               goto exit;
          }
          mxSetCell(result,i,mxItem);
     }
     VERBOSEPRINT("Exiting sequence_to_mxarray\n");
  exit:
     return result;
}

mxArray *python_to_mxarray(PyObject *object)
{
     void *data;
     mxArray *result=0;
     VERBOSEPRINT("Entering python_to_mxarray\n");
     if (strcmp(object->ob_type->tp_name,"numpy.ndarray")==0) {
          VERBOSEPRINT("Decoding ndarray\n");
          result = ndarray_to_mxarray(object);
     } else if (PyInt_Check(object)) {
          VERBOSEPRINT("Decoding int\n");
          result=mxCreateNumericMatrix(1,1,mxINT32_CLASS,0);
          data = mxGetData(result);
          *((long *)data) = PyInt_AsLong(object);
     } else if (PyFloat_Check(object)) {
          VERBOSEPRINT("Decoding double\n");
          result=mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,0);
          data = mxGetData(result);
          *((double *)data) = PyFloat_AS_DOUBLE(object);
     } else if (! PyObject_IsTrue(object)) {
          VERBOSEPRINT("Decoding False\n");
          result=mxCreateNumericMatrix(1,1,mxLOGICAL_CLASS,0);
          data = mxGetData(result);
          *((bool *)data) = 0;
     } else if (PyString_Check(object)) {
          Py_ssize_t length;
          PyObject *encoded;
          char *ptr;
          VERBOSEPRINT("Decoding string\n");
          encoded = PyObject_CallMethod(object, "encode", "s,s", "utf_16","replace");
          if (! encoded) {
               ERRPRINT("Failed to encode string in python_to_mxarray\n");
               PrintStackTrace();
               return NULL;
          }
          PyString_AsStringAndSize(encoded,&ptr, &length);
          /* First character is the encoding indicator */
          length-=sizeof(mxChar);
          ptr += sizeof(mxChar);
          VERBOSEPRINT1("String length: %d\n",length);
          result=mxCreateNumericMatrix(1,length/sizeof(mxChar),mxCHAR_CLASS,0);
          data = mxGetData(result);
          memcpy(data,ptr,length);
     } else if (PyUnicode_Check(object)) {
          Py_ssize_t length;
          PyObject *encoded;
          char *ptr;
          VERBOSEPRINT("Decoding unicode string.\n");
          encoded = PyUnicode_AsUTF16String(object);
          PyString_AsStringAndSize(encoded,&ptr, &length);
          /* First character is the encoding indicator */
          length-=sizeof(mxChar);
          ptr += sizeof(mxChar);
          VERBOSEPRINT1("String length: %d\n",length);
          result=mxCreateNumericMatrix(1,length/sizeof(mxChar),mxCHAR_CLASS,0);
          data = mxGetData(result);
          memcpy(data,ptr,length);
     } else if (PyMapping_Check(object)) {
          VERBOSEPRINT("Decoding mapping\n");
          result = mapping_to_mxarray(object);
     } else if (PySequence_Check(object)) {
          VERBOSEPRINT("Decoding sequence\n");
          result = sequence_to_mxarray(object);
     } else {
          char buffer[1024];
          sprintf(buffer,"Unknown object type: %s\n",object->ob_type->tp_name);
          ERRPRINT(buffer);
     }
     VERBOSEPRINT("Exiting python_to_mxarray\n");
     return result;
}
