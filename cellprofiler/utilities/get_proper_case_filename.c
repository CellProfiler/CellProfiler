/* get_proper_case_filename.c - retrieve a filename in its correct case
 *
 * CellProfiler is distributed under the GNU General Public License,
 * but this file is licensed under the more permissive BSD license.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2014 Broad Institute
 *
 * Please see the AUTHORS file for credits.
 *
 * Website: http://www.cellprofiler.org
 *
 * Windows file names are case-insensitive, but Unix file names are not.
 * If a user uses the wrong case in Windows, it may not have any effect
 * locally, but will fail on the cluster.
 *
 * get_proper_case_filename retrieves the file name in its correct case.
 *
 */
#define COBMACROS
#define UNICODE
#define _WIN32_WINNT 0x0500
#include <windows.h>
#include <string.h>
#include <shlwapi.h>
#include <shlobj.h>
#include "Python.h"

static PyObject *
get_proper_case_filename(PyObject *self, PyObject *args)
{
    PyObject *arg_1;
    wchar_t filename_in[4096];
    wchar_t filename_out[4096];
    DWORD nchars;
    LPITEMIDLIST pidl;
    LPSHELLFOLDER pDesktopFolder;
    ULONG chEaten;
    HRESULT hr;
    DWORD dwAttributes;
    
    if (PyTuple_Size(args) < 1) {
        Py_RETURN_NONE;
    }
    arg_1 = PyTuple_GetItem(args, 0);
    if (! PyUnicode_Check(arg_1)) {
        Py_RETURN_NONE;
    }
    nchars = PyUnicode_AsWideChar((PyUnicodeObject *)arg_1, filename_in, 4095);
    if (nchars == 0) {
        Py_RETURN_NONE;
    }
    filename_in[nchars] = 0;
    /*
     * The following is taken from http://support.microsoft.com/kb/132750
     */
    if (FAILED(SHGetDesktopFolder(&pDesktopFolder))) {
        Py_RETURN_NONE;
    }
    dwAttributes = SFGAO_VALIDATE;
    hr = pDesktopFolder->lpVtbl->ParseDisplayName(pDesktopFolder,
                                                  NULL,
                                                  NULL,
                                                  filename_in,
                                                  &chEaten,
                                                  &pidl,
                                                  &dwAttributes);
    pDesktopFolder->lpVtbl->Release(pDesktopFolder);
    if (FAILED(hr)) {
        Py_RETURN_NONE;
    }
    if (! SHGetPathFromIDList(pidl, filename_out)) {
        CoTaskMemFree(pidl);
        Py_RETURN_NONE;
    }
    CoTaskMemFree(pidl);
    nchars = wcslen(filename_out);
    return PyUnicode_FromWideChar(filename_out,nchars);
}
        
static PyObject *
set_file_attributes(PyObject *self, PyObject *args)
{
    PyObject *arg_1;
    PyObject *arg_2;
    wchar_t filename_in[4096];
    DWORD nchars;
    DWORD attributes;
    BOOL result;
    
    if (PyTuple_Size(args) < 2) {
        Py_RETURN_NONE;
    }
    arg_1 = PyTuple_GetItem(args, 0);
    arg_1 = PyUnicode_FromObject(arg_1);
    if (! arg_1) {
        Py_RETURN_NONE;
    }
    nchars = PyUnicode_AsWideChar((PyUnicodeObject *)arg_1, filename_in, 4095);
    Py_DECREF(arg_1);
    if (nchars == 0) {
        Py_RETURN_NONE;
    }
    filename_in[nchars] = 0;
    arg_2 = PyTuple_GetItem(args, 1);
    if (! PyInt_Check(arg_2)) {
        Py_RETURN_NONE;
    }
    attributes = PyInt_AsLong(arg_2);
    if (! SetFileAttributesW(filename_in, attributes)) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject *
get_file_attributes(PyObject *self, PyObject *args)
{
    PyObject *arg_1;
    wchar_t filename_in[4096];
    DWORD nchars;
    DWORD attributes;
    
    if (PyTuple_Size(args) < 1) {
        Py_RETURN_NONE;
    }
    arg_1 = PyTuple_GetItem(args, 0);
    arg_1 = PyUnicode_FromObject(arg_1);
    if (! arg_1) {
        Py_RETURN_NONE;
    }
    nchars = PyUnicode_AsWideChar((PyUnicodeObject *)arg_1, filename_in, 4095);
    Py_DECREF(arg_1);
    if (nchars == 0) {
        Py_RETURN_NONE;
    }
    filename_in[nchars] = 0;
    attributes = GetFileAttributesW(filename_in);
    return PyInt_FromLong(attributes);
}

static PyMethodDef methods[] = {
    {"get_proper_case_filename", (PyCFunction)get_proper_case_filename, 
     METH_VARARGS, 
     "Adjust the file name's case to the case recorded in the file system\n\n"
     "For instance, get_proper_case_filename(""c:\\program files"") returns "
     """c:\\Program Files""."},
     {"set_file_attributes", (PyCFunction)set_file_attributes,
      METH_VARARGS,
      "Set a file's attributes\n\n"
      "path - the path to the file\n\n"
      "attributes - the attributes for the file.\n"
      "             FILE_ATTRIBUTE_ARCHIVE, FILE_ATTRIBUTE_HIDDEN,\n"
      "             FILE_ATTRIBUTE_NOT_CONTENT_INDEXED, FILE_ATTRIBUTE_OFFLINE,\n"
      "             FILE_ATTRIBUTE_READONLY, FILE_ATTRIBUTE_SYSTEM,\n"
      "             or FILE_ATTRIBUTE_TEMPORARY\n\n"
      "Returns None if parameters aren't set correctly, False on system failure\n"
      "such as no such file or bad permission and True on success."},
      {"get_file_attributes", (PyCFunction)get_file_attributes,
      METH_VARARGS,
      "Get a file's attributes\n\n"
      "path - the path to the file\n\n"
      "See set_file_attributes for return values"},
     { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC init_get_proper_case_filename(void)
{
    PyObject *module;
    PyObject *module_dictionary;
    PyObject *tmp;
    
    module = Py_InitModule("_get_proper_case_filename", methods);
    
    module_dictionary = PyModule_GetDict(module);
    tmp = PyInt_FromLong(FILE_ATTRIBUTE_ARCHIVE);
    PyDict_SetItemString(module_dictionary, "FILE_ATTRIBUTE_ARCHIVE", tmp);
    Py_DECREF(tmp);
    
    tmp = PyInt_FromLong(FILE_ATTRIBUTE_HIDDEN);
    PyDict_SetItemString(module_dictionary, "FILE_ATTRIBUTE_HIDDEN", tmp);
    Py_DECREF(tmp);
    
    tmp = PyInt_FromLong(FILE_ATTRIBUTE_NOT_CONTENT_INDEXED);
    PyDict_SetItemString(module_dictionary, "FILE_ATTRIBUTE_NOT_CONTENT_INDEXED", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(FILE_ATTRIBUTE_OFFLINE);
    PyDict_SetItemString(module_dictionary, "FILE_ATTRIBUTE_OFFLINE", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(FILE_ATTRIBUTE_READONLY);
    PyDict_SetItemString(module_dictionary, "FILE_ATTRIBUTE_READONLY", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(FILE_ATTRIBUTE_SYSTEM);
    PyDict_SetItemString(module_dictionary, "FILE_ATTRIBUTE_SYSTEM", tmp);
    Py_DECREF(tmp);

    tmp = PyInt_FromLong(FILE_ATTRIBUTE_TEMPORARY);
    PyDict_SetItemString(module_dictionary, "FILE_ATTRIBUTE_TEMPORARY", tmp);
    Py_DECREF(tmp);
}

