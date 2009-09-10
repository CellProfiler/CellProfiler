/* get_proper_case_filename.c - retrieve a filename in its correct case
 *
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Developed by the Broad Institute
 * Copyright 2003-2009
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
#define UNICODE
#define _WIN32_WINNT 0x0500
#include <windows.h>
#include "Python.h"

static PyObject *
get_proper_case_filename(PyObject *self, PyObject *args)
{
    Py_UNICODE *filename_in;
    Py_UNICODE filename_short[1024];
    Py_UNICODE filename_out[1024];
    DWORD nchars;
    
    PyArg_ParseTuple(args,"u",&filename_in);
    /*
     * The hack here is to convert the path name to the short (DOS)
     * form and then to the long form. More than one person has settled
     * on this as the way to do it, e.g.
     * http://www.west-wind.com/Weblog/posts/202850.aspx
     */
    nchars = GetShortPathName(filename_in, filename_short, 1024);
    if (nchars == 0) {
        Py_RETURN_NONE;
    }
    nchars = GetLongPathName(filename_short, filename_out, 1024);
    if (nchars == 0)
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PyUnicode_FromUnicode(filename_out,nchars);
    }
}
        
    
static PyMethodDef methods[] = {
    {"get_proper_case_filename", (PyCFunction)get_proper_case_filename, 
     METH_VARARGS, 
     "Adjust the file name's case to the case recorded in the file system\n\n"
     "For instance, get_proper_case_filename(""c:\\program files"") returns "
     """c:\\Program Files""."}
};

PyMODINIT_FUNC init_get_proper_case_filename(void)
{
    Py_InitModule("_get_proper_case_filename", methods);
}

