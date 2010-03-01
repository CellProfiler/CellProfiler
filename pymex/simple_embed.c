#include <Python.h>

static int numargs;

static PyObject* emb_numargs(PyObject *self, PyObject *args)
{
     if (!PyArg_ParseTuple(args, ":numargs"))
          return NULL;
     return Py_BuildValue("i", numargs);
}

static PyMethodDef emb_methods[] = {
     { "numargs", emb_numargs, METH_VARARGS,
       "Return the number of arguments."
     },
     { NULL, NULL, 0, NULL }
};

int
main(int argc, char *argv[])
{
  Py_Initialize();
  numargs = argc;
  Py_InitModule("emb", emb_methods);
  PyRun_SimpleString("from time import time,ctime\n"
                     "print 'Today is',ctime(time())\n"
                     "import emb\n"
                     "print '%d arguments' % emb.numargs()\n");
  Py_Finalize();
  return 0;
}
