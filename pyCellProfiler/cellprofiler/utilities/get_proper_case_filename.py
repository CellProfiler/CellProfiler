'''get_proper_case_filename.py - convert filename to proper case for Windows

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import sys

if sys.platform.startswith("win"):
    import _get_proper_case_filename
    def get_proper_case_filename(path):
        return _get_proper_case_filename.get_proper_case_filename(unicode(path))
else:
    def get_proper_case_filename(path):
        return path
