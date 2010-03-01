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

#define DEBUG_PRINT(level, msg) { if (level <= debug_level) mexPrintf(msg); }
#define ERRPRINT(msg) DEBUG_PRINT(0,msg)
#define WARNPRINT(msg) DEBUG_PRINT(1,msg)
#define INFOPRINT(msg) DEBUG_PRINT(2,msg)
#define VERBOSEPRINT(msg) DEBUG_PRINT(3,msg)

#define DEBUG_PRINT1(level, msg, arg) { if ((level) <= debug_level) mexPrintf((msg),(arg)); }
#define ERRPRINT1(msg,arg) DEBUG_PRINT1(0,(msg),(arg))
#define WARNPRINT1(msg,arg) DEBUG_PRINT1(1,(msg),(arg))
#define INFOPRINT1(msg,arg) DEBUG_PRINT1(2,(msg),(arg))
#define VERBOSEPRINT1(msg,arg) DEBUG_PRINT1(3,(msg),(arg))

#define DEBUG_PRINT2(level, msg, arg1,arg2) { if ((level) <= debug_level) mexPrintf((msg),(arg1),(arg2)); }
#define ERRPRINT2(msg,arg1,arg2) DEBUG_PRINT2(0,(msg),(arg1),(arg2))
#define VERBOSEPRINT2(msg,arg1,arg2) DEBUG_PRINT2(3,(msg),(arg1),(arg2))


void pymex_init(void);
const char *get_arg_string(int nrhs, const mxArray *prhs[], int i);

PyObject *pymex_make_array(int m, int n);
void pymex_array_setitem(PyObject *array, int i, int j, PyObject *value);
PyObject* mxarray_to_python(const mxArray *a);
mxArray *python_to_mxarray(PyObject *object);
void PrintStackTrace(void);

long debug_level;

#endif
