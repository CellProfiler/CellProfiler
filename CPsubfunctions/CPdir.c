/*
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Developed by the Whitehead Institute for Biomedical Research.
 * Copyright 2003--2008.
 *
 * Please see the AUTHORS file for credits.
 *
 * Website: http://www.cellprofiler.org
 *
 * $Revision: 5583 $
 */

#include <dirent.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "mex.h"
#include "string.h"

struct file {
     char *name;
     int is_dir;
};

int cmp_file(const void *a, const void *b)
{
     const struct file *f = a;
     const struct file *g = b;
     return strcmp(f->name, g->name);
}

struct file *dir(const char *dir_name, int *nfiles)
{
     DIR *dir;
     int n, size;
     struct file *files;
     struct dirent *dirent;
     char *name;

     dir = opendir(dir_name);
     if (!dir) {
          if (errno == ENOENT)
               return NULL;
          else
               mexErrMsgTxt(strerror(errno));
     }

     size = 1;
     files = mxMalloc(size * sizeof(struct file));

     n = 0;
     while (dirent = readdir(dir)) {
          if (n == size) {
               size *= 2;
               files = mxRealloc(files, size * sizeof(struct file));
          }
          name = strdup(dirent->d_name);
          if (!name)
               mexErrMsgTxt("Failed to allocate memory for d_name.");
          files[n].name = name;
          files[n].is_dir = dirent->d_type == DT_DIR;
          n++;
     }
     closedir(dir);

     qsort(files, n, sizeof(struct file), cmp_file);
     if (nfiles)
          *nfiles = n;
     return files;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
     char *dir_name;
     struct file *files;
     int nfiles, i;
     const char *field_names[2];

     if (nlhs > 1)
          mexErrMsgTxt("Too many output arguments.");
     
     if (nrhs == 0)
          dir_name = ".";
     else if (nrhs == 1) {
          if (mxIsChar(prhs[0]) != 1)
               mexErrMsgTxt("Input must be a string.");
          if (mxGetM(prhs[0]) != 1)
               mexErrMsgTxt("Input must be a row vector.");
          dir_name = mxArrayToString(prhs[0]);
     } else
          mexErrMsgTxt("Too many input arguments.");

     files = dir(dir_name, &nfiles);

     field_names[0] = "name";
     field_names[1] = "isdir";
     plhs[0] = mxCreateStructMatrix(nfiles, 1, 2, field_names);

     for (i = 0; i < nfiles; i++) {
          mxSetFieldByNumber(plhs[0], i, 0, mxCreateString(files[i].name));
          mxSetFieldByNumber(plhs[0], i, 1,
                             mxCreateLogicalScalar(files[i].is_dir));
     }

     for (i = 0; i < nfiles; i++)
          free(files[i].name);
     mxFree(files);
     if (nrhs > 0)
          mxFree(dir_name);
     return;
}
