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
 * $Revision$
 */
#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#endif

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

/*
 * add_file - add a file to the "file" array
 *    files  - points to current array of files. Initally null.
 *    name   - file name
 *    is_dir - 0 if not a directory otherwise is a directory
 *    pCount - points to place to store # of elements
 *    pSize  - points to array size
 */
struct file *add_file(struct file *files, const char *name, int is_dir,
                      int *pCount, int *pSize)
{
     char *name_copy;

     if (! files) {
          *pSize = 1;
          files = mxMalloc((*pSize) * sizeof(struct file));
          if (! files)
               mexErrMsgTxt("Failed to allocate initial memory for files array");
          *pCount = 0;
     } else if (*pCount == *pSize) {
          *pSize *= 2;
          files = mxRealloc(files, (*pSize) * sizeof(struct file));
          if (! files)
               mexErrMsgTxt("Failed to increase memory for files array");
     }
     name_copy = mxMalloc(strlen(name)+1);
     if (!name_copy)
          mexErrMsgTxt("Failed to allocate memory for d_name.");
     strcpy(name_copy,name);
     files[*pCount].name = name_copy;
     files[*pCount].is_dir = is_dir;
     *pCount = *pCount + 1;
     return files;
}
  
#if (defined(WIN32) || defined(_WIN32))
struct file *dir(const char *dir_name, int *nfiles)
{
     struct file *files;
     int size;
     WIN32_FIND_DATA find_data;
     HANDLE hFind;
     char buffer[1024];
     char *search_path;
     int last_error;
     const char *search_suffix = "\\*";
     const char *alt_search_suffix = "*";

     files = NULL;
     *nfiles=0;
     search_path = mxMalloc(strlen(dir_name) + strlen(search_suffix) + 1);
     if (!search_path)
          mexErrMsgTxt("Failed to allocate memory for search path");
     strcpy(search_path, dir_name);
     /* Append "*" if an empty string or string ending in one of the
      * Windows path separation characters, otherwise add a path
      * separator and "*". */
     if (strlen(dir_name) == 0 || 
         strchr(":/\\", dir_name[strlen(dir_name) - 1]))
          strcat(search_path, alt_search_suffix);
     else
          strcat(search_path, search_suffix);
     
     hFind = FindFirstFile(search_path, &find_data);
     mxFree(search_path);
     if (hFind == INVALID_HANDLE_VALUE) {
          if (GetLastError() == ERROR_FILE_NOT_FOUND)
               return NULL;
          else {
               sprintf(buffer,
                       "Failed when finding directory. Windows error # %d",
                       GetLastError());
               mexErrMsgTxt(buffer);
          }
     }
     do {
          files = add_file(files, find_data.cFileName,
                           (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0,
                           nfiles, &size);
     } while (FindNextFile(hFind, &find_data) != 0);
     last_error = GetLastError();
     FindClose(hFind);
     if (last_error != ERROR_NO_MORE_FILES) {
          sprintf(buffer,
                  "Failed when iterating through directory list. "
                  "Windows error # %d", last_error);
          mexErrMsgTxt(buffer);
     }
     sprintf(buffer, "Found %d files", *nfiles);
     qsort(files, *nfiles, sizeof(struct file), cmp_file);
     return files;
}
#else
/* Stat the file.  Return 1 if it is a directory or a link to a
 * directory.  Return 0 otherwise.  Return -1 on error */
int stat_isdir(const char *dir_name, const char *lnk_name)
{
  struct stat st;
  char *name;
  int result;

  if (asprintf(&name, "%s/%s", dir_name, lnk_name) == -1)
       mexErrMsgTxt("Failed to allocate memory for file name.");
  if (stat(name, &st) == -1)
       result = -1;
  else
       result = S_ISDIR(st.st_mode);
  free(name);
  return result;
}

const char *file_extensions[] = { "bmp", "cur", "fts", "fits", "gif", "hdf", 
                                  "ico", "jpg", "jpeg", "pbm", "pcx", "pgm", 
                                  "png", "pnm", "ppm", "ras", "tif", "tiff", 
                                  "xwd", "dib", "mat", "fig", "zvi", "raw", 
                                  "flex", NULL };

/* Decide whether a file is a directory, stat-ing when necessary.  In
 * order not to stat unnecessarily, assume that nobody is stupid
 * enough to name their directory something that ends in ".tif" or
 * some other common extension. */
int is_dir(const char *dir_name, struct dirent *dirent)
{
     char *last_part;
     const char **extension;

     switch (dirent->d_type) {
     case DT_DIR: 
          return 1;
     case DT_LNK:
     case DT_UNKNOWN:
          last_part = strrchr(dirent->d_name, '.');
          if (!last_part)
               return stat_isdir(dir_name, dirent->d_name);
          last_part++;
          for (extension = file_extensions; *extension; extension++)
               if (strcasecmp(last_part, *extension) == 0)
                    return 0;
          return stat_isdir(dir_name, dirent->d_name);
     default: 
          return 0;
     }
}

struct file *dir(const char *dir_name, int *nfiles)
{
     struct file *files;
     int size;
     DIR *dir;
     struct dirent *dirent;
     
     files = NULL;
     *nfiles = 0;
     dir = opendir(dir_name);
     if (!dir) {
          if (errno == ENOENT)
               return NULL;
          else
               mexErrMsgTxt(strerror(errno));
     }

     while (dirent = readdir(dir))
          files = add_file(files, dirent->d_name, is_dir(dir_name, dirent),
                           nfiles, &size);
     closedir(dir);
     qsort(files, *nfiles, sizeof(struct file), cmp_file);
     return files;
}
#endif

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
          mxFree(files[i].name);
     mxFree(files);
     if (nrhs > 0)
          mxFree(dir_name);
     return;
}

