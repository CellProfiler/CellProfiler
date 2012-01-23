'''test_namesandtypes.py - test the NamesAndTypes module

'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Copyright (c) 2003-2009 Massachusetts Institute of Technology
#Copyright (c) 2009-2012 Broad Institute
#All rights reserved.
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

import numpy as np
import unittest

import cellprofiler.pipeline as cpp
import cellprofiler.modules.namesandtypes as N

M0, M1, M2, M3, M4, M5, M6 = ["MetadataKey%d" % i for i in range(7)]
C0, C1, C2, C3, C4, C5, C6 = ["Column%d" % i for i in range(7)]

def md(keys_and_counts):
    '''Generate metadata dictionaries for the given metadata shape
    
    keys_and_counts - a collection of metadata keys and the dimension of
                      their extent. For instance [(M0, 2), (M1, 3)] generates
                      six dictionaries with two unique values of M0 and
                      three for M1
    '''
    keys = [k for k, c in keys_and_counts]
    counts = np.array([c for k, c in keys_and_counts])
    divisors = np.hstack([[1], np.prod(counts[:-1])])
    
    return [dict([(k, "k" + str(int(i / d) % c))
                  for k, d, c in zip(keys, divisors, counts)])
            for i in range(np.prod(counts))]
                  

class TestNamesAndTypes(unittest.TestCase):
    def test_01_00_00_nothing(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = []
        n.column_names = []
        
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 0)
        
    def test_01_00_01_all(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_ALL
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("1", None, None, None, **{M0:"1"})]]
        n.column_names = [C0]
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 1)
        image_set = n.image_sets[0]
        self.assertEqual(len(image_set), 2)
        image_set_key, image_set_dictionary = image_set
        self.assertEqual(len(image_set_key), 1)
        self.assertEqual(image_set_key[0], "1")
        self.assertEqual(len(image_set_dictionary), 1)
        self.assertTrue(image_set_dictionary.has_key(C0))
        self.assertEqual(len(image_set_dictionary[C0]), 1)
        self.assertEqual(image_set_dictionary[C0][0].url, "1")
        
    def test_01_01_one(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("1", None, None, None, **{M0:"k1"})]]
        n.column_names = [C0]
        n.join.build("[{%s:%s}]" % (C0, M0))
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 1)
        image_set = n.image_sets[0]
        self.assertEqual(len(image_set), 2)
        image_set_key, image_set_dictionary = image_set
        self.assertEqual(len(image_set_key), 1)
        self.assertEqual(image_set_key[0], "1")
        self.assertEqual(len(image_set_dictionary), 1)
        self.assertTrue(image_set_dictionary.has_key(C0))
        self.assertEqual(len(image_set_dictionary[C0]), 1)
        self.assertEqual(image_set_dictionary[C0][0].url, "1")
        
    def test_01_02_match_one_same_key(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("1", None, None, None, **{M0:"k1"})],
             [cpp.ImagePlaneDetails("2", None, None, None, **{M0:"k1"})]]
        n.column_names = [C0, C1]
        n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M0))
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 1)
        image_set = n.image_sets[0]
        self.assertEqual(len(image_set), 2)
        image_set_key, image_set_dictionary = image_set
        self.assertEqual(len(image_set_key), 1)
        self.assertEqual(image_set_key[0], "k1")
        self.assertEqual(len(image_set_dictionary), 2)
        self.assertTrue(image_set_dictionary.has_key(C0))
        self.assertEqual(len(image_set_dictionary[C0]), 1)
        self.assertEqual(image_set_dictionary[C0][0].url, "1")
        self.assertTrue(image_set_dictionary.has_key(C1))
        self.assertEqual(len(image_set_dictionary[C1]), 1)
        self.assertEqual(image_set_dictionary[C1][0].url, "2")
        
    def test_01_03_match_one_different_key(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("1", None, None, None, **{M0:"k1"})],
             [cpp.ImagePlaneDetails("2", None, None, None, **{M1:"k1"})]]
        n.column_names = [C0, C1]
        n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M1))
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 1)
        image_set = n.image_sets[0]
        self.assertEqual(len(image_set), 2)
        image_set_key, image_set_dictionary = image_set
        self.assertEqual(len(image_set_key), 1)
        self.assertEqual(image_set_key[0], "k1")
        self.assertEqual(len(image_set), 2)
        self.assertTrue(image_set_dictionary.has_key(C0))
        self.assertEqual(len(image_set_dictionary[C0]), 1)
        self.assertEqual(image_set_dictionary[C0][0].url, "1")
        self.assertTrue(image_set_dictionary.has_key(C1))
        self.assertEqual(len(image_set_dictionary[C1]), 1)
        self.assertEqual(image_set_dictionary[C1][0].url, "2")
        
    def test_01_04_match_two_one_key(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("%s%d" % (C0, i), None, None, None, **m)
              for i, m in enumerate(md([(M0, 2)]))],
             [cpp.ImagePlaneDetails("%s%d" % (C1, i), None, None, None, **m)
                           for i, m in enumerate(md([(M1, 2)]))]]
        n.column_names = [C0, C1]
        n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M1))
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 2)
        for i, (image_set_keys, image_set) in enumerate(n.image_sets):
            self.assertEqual(len(image_set_keys), 1)
            self.assertEqual("k"+str(i), image_set_keys[0])
            for column_name in (C0, C1):
                self.assertTrue(image_set.has_key(column_name))
                self.assertEqual(len(image_set[column_name]), 1)
                self.assertEqual(image_set[column_name][0].url,
                                 "%s%d" % (column_name, i))
                
    def test_01_05_match_two_and_two(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("%s%s%s" % (C0, m[M0], m[M1]), None, None, None, **m)
              for i, m in enumerate(md([(M0, 2), (M1, 3)]))],
             [cpp.ImagePlaneDetails("%s%s%s" % (C1, m[M2], m[M3]), None, None, None, **m)
                           for i, m in enumerate(md([(M2, 2), (M3, 3)]))]]
        n.column_names = [C0, C1]
        n.join.build("[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]" % 
                     (C0, M0, C1, M2, C0, M1, C1, M3))
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 6)
        for i, (image_set_keys, image_set) in enumerate(n.image_sets):
            self.assertEqual(len(image_set_keys), 2)
            self.assertEqual("k"+str(i/3), image_set_keys[0])
            self.assertEqual("k"+str(i%3), image_set_keys[1])
            for column_name in (C0, C1):
                self.assertTrue(image_set.has_key(column_name))
                self.assertEqual(len(image_set[column_name]), 1)
                self.assertEqual(image_set[column_name][0].url,
                                 "%s%s%s" % (column_name, image_set_keys[0], image_set_keys[1]))
                
    def test_01_06_two_with_same_metadata(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("%s%s%s" % (C0, m[M0], m[M1]), None, None, None, **m)
              for i, m in enumerate(md([(M0, 2), (M1, 3)]))],
             [cpp.ImagePlaneDetails("%s%s%s" % (C1, m[M2], m[M3]), None, None, None, **m)
                           for i, m in enumerate(md([(M2, 2), (M3, 3)]))]]
        n.ipd_columns[0].append(cpp.ImagePlaneDetails("Bad", None, None,None, **{M0:"k1",M1:"k1"}))
        n.column_names = [C0, C1]
        n.join.build("[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]" % 
                     (C0, M0, C1, M2, C0, M1, C1, M3))
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 6)
        for i, (image_set_keys, image_set) in enumerate(n.image_sets):
            self.assertEqual(len(image_set_keys), 2)
            self.assertEqual("k"+str(i/3), image_set_keys[0])
            self.assertEqual("k"+str(i%3), image_set_keys[1])
            for column_name in (C0, C1):
                self.assertTrue(image_set.has_key(column_name))
                if image_set_keys == ("k1","k1") and column_name == C0:
                    self.assertEqual(len(image_set[C0]), 2)
                    self.assertTrue("Bad" in [ipd.url for ipd in image_set[C0]])
                else:
                    self.assertEqual(len(image_set[column_name]), 1)
                    self.assertEqual(image_set[column_name][0].url,
                                     "%s%s%s" % (column_name, image_set_keys[0], image_set_keys[1]))
        
    def test_01_07_one_against_all(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("One", None, None, None)],
             [cpp.ImagePlaneDetails("%s%d" % (C1, i), None, None, None, **m)
              for i, m in enumerate(md([(M0, 3)]))]]
        n.column_names = [C0, C1]
        n.join.build("[{'%s':None,'%s':'%s'}]" % (C0, C1, M0))
        n.make_image_sets()
        self.assertEqual(len(n.image_sets), 3)
        for i, (image_set_keys, image_set) in enumerate(n.image_sets):
            self.assertEqual(len(image_set_keys), 1)
            self.assertEqual("k%d" % i, image_set_keys[0])
            self.assertTrue(image_set.has_key(C0))
            self.assertEqual(len(image_set[C0]), 1)
            self.assertEqual(image_set[C0][0].url, "One")
            self.assertTrue(image_set.has_key(C1))
            self.assertTrue(len(image_set[C1]), 1)
            self.assertEqual(image_set[C1][0].url, "%s%d" % (C1, i))
            
    def test_02_08_some_against_all(self):
        #
        # Permute both the order of the columns and the order of joins
        #
        columns = { C0: [cpp.ImagePlaneDetails("%s%s" % (C0, m[M0]), None, None, None, **m)
                         for i, m in enumerate(md([(M0, 3)]))],
                    C1: [cpp.ImagePlaneDetails("%s%s%s" % (C0, m[M1], m[M2]), None, None, None, **m)
                         for i, m in enumerate(md([(M1, 3),(M2, 2)]))] }
        joins = [{C0:M0, C1:M1},{C0:None, C1:M2}]
        for cA, cB in ((C0, C1), (C1, C0)):
            for j0, j1 in ((0,1),(1,0)):
                n = N.NamesAndTypes()
                n.assignment_method.value = N.ASSIGN_RULES
                n.ipd_columns = [columns[cA], columns[cB]]
                n.column_names = [cA, cB]
                n.join.build(repr([joins[j0], joins[j1]]))
                n.make_image_sets()
                self.assertEqual(len(n.image_sets), 6)
                for i, (image_set_keys, image_set) in enumerate(n.image_sets):
                    if j0 == 0:
                        k0 = "k%d" % (i / 2)
                        k1 = "k%d" % (i % 2)
                    else:
                        k0 = "k%d" % (i % 3)
                        k1 = "k%d" % (i / 3)
                    k = [k0, k1]
                    self.assertEqual(image_set_keys[0], k[j0])
                    self.assertEqual(image_set_keys[1], k[j1])
                    self.assertEqual(len(image_set[C0]), 1)
                    self.assertEqual(image_set[C0][0].url, "%s%s" % (C0, k0))
                    self.assertEqual(len(image_set[C1]), 1)
                    self.assertEqual(image_set[C1][0].url, "%s%s%s" % (C0, k0, k1))
                    
                    