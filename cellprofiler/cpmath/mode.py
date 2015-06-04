"""mode.py - compute the mode of an array

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import numpy as np

def mode(a):
    '''Compute the mode of an array
    
    a: an array
    
    returns a vector of values which are the most frequent (more than one
    if there is a tie).
    '''
    a = np.asanyarray(a)
    if a.size == 0:
        return np.zeros(0, a.dtype)
    aa = a.flatten()
    aa.sort()
    indices = np.hstack([[0], np.where(aa[:-1] != aa[1:])[0]+1, [aa.size]])
    counts = indices[1:] - indices[:-1]
    best_indices = indices[:-1][counts == np.max(counts)]
    return aa[best_indices]
    
    