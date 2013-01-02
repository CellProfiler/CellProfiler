/* get_proper_case_filename.c - retrieve a filename in its correct case
 *
 * CellProfiler is distributed under the GNU General Public License,
 * but this file is licensed under the more permissive BSD license.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2013 Broad Institute
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
        
    
static PyMethodDef methods[] = {
    {"get_proper_case_filename", (PyCFunction)get_proper_case_filename, 
     METH_VARARGS, 
     "Adjust the file name's case to the case recorded in the file system\n\n"
     "For instance, get_proper_case_filename(""c:\\program files"") returns "
     """c:\\Program Files""."},
     { NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC init_get_proper_case_filename(void)
{
    Py_InitModule("_get_proper_case_filename", methods);
}

