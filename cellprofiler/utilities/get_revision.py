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
    '''The release version is 10997 - it just is'''
    return 10997

'''SVN revision'''
version = get_revision()
    
if __name__ == '__main__':
    print version

    
