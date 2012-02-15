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
from cStringIO import StringIO
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
    def test_00_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120213205828
ModuleCount:3
HasImagePlaneDetails:True

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    :{"ShowFiltered"\x3A false}
    Filter based on rules:Yes
    Filter:or (extension does istif)

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:Yes
    Extraction method count:1
    Extraction method:Manual
    Source:From file name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})f(?P<Site>\x5B0-9\x5D{2})d(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Assignment method:Assign images matching rules
    Load as:Color image
    Image name:PI
    :\x5B{u\'Illum\'\x3A u\'Plate\', u\'DNA\'\x3A u\'Plate\', \'Cells\'\x3A u\'Plate\', u\'Actin\'\x3A u\'Plate\', u\'GFP\'\x3A u\'Plate\'}, {u\'Illum\'\x3A u\'Well\', u\'DNA\'\x3A u\'Well\', \'Cells\'\x3A u\'Well\', u\'Actin\'\x3A u\'Well\', u\'GFP\'\x3A u\'Well\'}, {u\'Illum\'\x3A u\'Site\', u\'DNA\'\x3A u\'Site\', \'Cells\'\x3A u\'Site\', u\'Actin\'\x3A u\'Site\', u\'GFP\'\x3A u\'Site\'}\x5D
    Assignments count:5
    Match this rule:or (metadata does ChannelNumber "0")
    Image name:DNA
    Objects name:Nuclei
    Load as:Grayscale image
    Match this rule:or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)
    Image name:Actin
    Objects name:Cells
    Load as:Color image
    Match this rule:or (metadata does ChannelNumber "2")
    Image name:GFP
    Objects name:Cells
    Load as:Mask
    Match this rule:or (metadata does ChannelNumber "2")
    Image name:Foo
    Objects name:Cells
    Load as:Objects
    Match this rule:or (metadata does ChannelNumber "2")
    Image name:Illum
    Objects name:Cells
    Load as:Illumination function

"Version":"1","PlaneCount":"5"
"URL","Series","Index","Channel","ColorFormat","SizeC","SizeT","SizeZ"
"file:///C:/trunk/ExampleImages/ExampleHT29/AS_09125_050116030001_D03f00d0.tif",,,,"monochrome","1","1","1"
"file:///C:/trunk/ExampleImages/ExampleHT29/AS_09125_050116030001_D03f00d1.tif",,,,"monochrome","1","1","1"
"file:///C:/trunk/ExampleImages/ExampleHT29/AS_09125_050116030001_D03f00d2.tif",,,,"monochrome","1","1","1"
"file:///C:/trunk/ExampleImages/ExampleHT29/ExampleHT29.cp",,,,,,,
"file:///C:/trunk/ExampleImages/ExampleHT29/k27IllumCorrControlv1.mat",,,,,,,
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, N.NamesAndTypes))
        self.assertEqual(module.assignment_method, N.ASSIGN_RULES)
        self.assertEqual(module.single_load_as_choice, N.LOAD_AS_COLOR_IMAGE)
        self.assertEqual(module.single_image_provider.value, "PI")
        self.assertEqual(module.assignments_count.value, 5)
        aa = module.assignments
        for assignment, rule, image_name, objects_name, load_as in (
            (aa[0], 'or (metadata does ChannelNumber "0")', "DNA", "Nuclei", N.LOAD_AS_GRAYSCALE_IMAGE),
            (aa[1], 'or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)', "Actin", "Cells", N.LOAD_AS_COLOR_IMAGE),
            (aa[2], 'or (metadata does ChannelNumber "2")', "GFP", "Cells", N.LOAD_AS_MASK),
            (aa[3], 'or (metadata does ChannelNumber "2")', "Foo", "Cells", N.LOAD_AS_OBJECTS),
            (aa[4], 'or (metadata does ChannelNumber "2")', "Illum", "Cells", N.LOAD_AS_ILLUMINATION_FUNCTION)):
            self.assertEqual(assignment.rule_filter.value, rule)
            self.assertEqual(assignment.image_name, image_name)
            self.assertEqual(assignment.object_name, objects_name)
            self.assertEqual(assignment.load_as_choice, load_as)
            
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
        self.assertEqual(image_set_key[0], 1)
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
        self.assertEqual(image_set_key[0], 1)
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
                    
                    