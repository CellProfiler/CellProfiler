'''get_proper_case_filename.py - convert filename to proper case for Windows

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import sys
import os

if sys.platform.startswith("win") and False:
    import _get_proper_case_filename
    def get_proper_case_filename(path):
        result = _get_proper_case_filename.get_proper_case_filename(unicode(path))
        if result is None:
            return path
        if (len(result) and len(path) and
            path[-1] == os.path.sep and result[-1] != os.path.sep):
            result += os.path.sep
        return result
else:
    def get_proper_case_filename(path):
        return path
