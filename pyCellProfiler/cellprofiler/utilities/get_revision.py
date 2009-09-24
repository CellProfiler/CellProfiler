'''get_revision.py - find the maximum revision number in the cellprofiler tree

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import numpy as np
import re
import sys

def __get_revision_of_module(module):
    version = getattr(module, '__version__', None)
    if version is None:
        return 0
    match = re.match('\\$Revision:\\s+([0-9]+)\\s\\$', version)
    if match:
        return int(match.groups()[0])
    return 0

'''SVN revision - only valid after call to get_revision'''
version = None

def get_revision():
    '''Return the maximum revision number from CellProfiler Python modules
    
    Starting with cellprofiler, find the maximum revision number by looking
    at __version__ in all modules.
    '''
    global version
    if version is not None:
        return version
    version = 0
    for module_name in sys.modules.keys():
        if (module_name.lower().startswith('cellprofiler') and 
            module_name.find('tests') == -1):
            sub_version = __get_revision_of_module(sys.modules[module_name])
            version = max(version, sub_version)
    return version
    
if __name__ == '__main__':
    import cellprofiler
    print get_revision()

    