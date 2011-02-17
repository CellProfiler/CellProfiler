'''get_revision.py - find the maximum revision number in the cellprofiler tree

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import numpy as np
import re
import sys
import cellprofiler # to force import of many other modules for their version number
import cellprofiler.modules as cpmodules

def __get_revision_of_module(module):
    version = getattr(module, '__version__', None)
    if version is None:
        return 0
    match = re.match('\\$Revision:\\s+([0-9]+)\\s\\$', version)
    if match:
        return int(match.groups()[0])
    return 0


def get_revision():
    '''Return the maximum revision number from CellProfiler Python modules.
    Starting with cellprofiler, find the maximum revision number by looking
    at __version__ in all modules.
    '''
    version = 0
    for module_name in sys.modules.keys():
        if module_name.lower().startswith('cellprofiler') and ('plugins' not in module_name):
            sub_version = __get_revision_of_module(sys.modules[module_name])
            version = max(version, sub_version)
    for module in cpmodules.pymodules:
        sub_version = __get_revision_of_module(module)
        version = max(version, sub_version)
    return version

'''SVN revision'''
version = get_revision()
    
if __name__ == '__main__':
    print version

    
