""" tests - tests of CellProfiler-level modules

Also test numpy and scipy here

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version = "$Revision$"

if __name__ == "__main__":
    import scipy.io.matlab
    import unittest
    import nose
    import sys

    nose.main()
    
