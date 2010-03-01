'''test_get_proper_case_filename - test the "get_proper_case_filename" function

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import os
import sys
import tempfile
import unittest

from cellprofiler.utilities.get_proper_case_filename\
     import get_proper_case_filename

if sys.platform.startswith('win'):
    '''Windows has case-sensitive filenames'''
    
    
    class TestGetProperCaseFilename(unittest.TestCase):
        '''Test the "get_proper_case_filename" function'''
        def test_01_01_SystemRoot(self):
            '''Test using the system root directory which should be a short name'''
            path = os.environ['SystemRoot']
            self.assertEqual(path, get_proper_case_filename(path))
        
        def test_01_02_local_same_case(self):
            '''Test using a local file name with the correct case'''
            
            path = os.environ['APPDATA']
            self.assertEqual(path, get_proper_case_filename(path))
        
        def test_01_03_local_uppercase(self):
            '''Test a mixed-case filename converted to upper case'''
            path = os.environ['APPDATA']
            self.assertEqual(path, get_proper_case_filename(path.upper()))
            
        def test_01_04_local_lowercase(self):
            '''Test a mixed-case filename converted to upper case'''
            path = os.environ['APPDATA']
            self.assertEqual(path, get_proper_case_filename(path.lower()))
            
        def test_02_01_remote(self):
            '''Test on Iodine'''
            path = '\\\\iodine\\imaging_analysis\\People'
            self.assertEqual(path, get_proper_case_filename(path))

        def test_02_02_remote_upper(self):
            '''Test on Iodine using uppercase'''
            path = '\\\\iodine\\imaging_analysis\\People'
            self.assertEqual(path, get_proper_case_filename(path.upper()))
        
        def test_02_03_remote_lower(self):
            '''Test on Iodine using lowercase'''
            path = '\\\\iodine\\imaging_analysis\\People'
            self.assertEqual(path, get_proper_case_filename(path.lower()))
else:
    class TestGetProperCaseFilename(unittest.TestCase):
        '''Test the "get_proper_case_filename" function'''
        def test_01_01_pass_through(self):
            '''The function should pass an arbitraty string through unchanged'''
            path = '/usr/bin'
            self.assertEqual(path, get_proper_case_filename(path))
                         