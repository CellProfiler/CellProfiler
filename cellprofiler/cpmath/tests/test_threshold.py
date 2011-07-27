'''test_threshold - test the threshold module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision: 11024 $"

import numpy as np
import unittest

import cellprofiler.cpmath.threshold as T

class TestThreshold(unittest.TestCase):
    def test_01_01_negative_log_otsu(self):
        '''regression test of img-1466'''
        
        r = np.random.RandomState()
        r.seed(11)
        img = r.uniform(size=(10,10))
        img[0,0] = -1
        unmasked = T.get_otsu_threshold(img)
        masked = T.get_otsu_threshold(img, img >= 0)
        self.assertEqual(unmasked, masked)
        
if __name__=="__main__":
    unittest.main()
    