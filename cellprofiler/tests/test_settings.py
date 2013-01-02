'''test_settings.py - test the settings classes'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import unittest

import cellprofiler.settings as cps

class TestFilterSetting(unittest.TestCase):
    def test_01_01_simple(self):
        filters = [ cps.Filter.FilterPredicate("foo", "Foo", lambda a: a=="x", [])]
        f = cps.Filter("", filters, "foo")
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        
    def test_01_02_compound(self):
        f2 = cps.Filter.FilterPredicate("bar", "Bar", lambda: "y", [])
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a==b(), [f2])
        f = cps.Filter("", [f1], "foo bar")
        self.assertFalse(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))
        
    def test_01_03_literal(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a==b, 
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1], 'foo "x"')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        f = cps.Filter("", [f1], 'foo "y"')
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("x"))
        
    def test_01_04_escaped_literal(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a==b, 
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1], 'foo "\\\\"')
        self.assertTrue(f.evaluate("\\"))
        self.assertFalse(f.evaluate("/"))
        
    def test_01_05_literal_with_quote(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a==b, 
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1], 'foo "\\""')
        self.assertTrue(f.evaluate("\""))
        self.assertFalse(f.evaluate("/"))
        
    def test_01_06_parentheses(self):
        f1 = cps.Filter.FilterPredicate("eq", "Foo", lambda a, b: a==b, 
                                        [cps.Filter.LITERAL_PREDICATE])
        f2 = cps.Filter.FilterPredicate("ne", "Bar", lambda a, b: a!=b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1,f2], 'and (eq "x") (ne "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))
        
    def test_01_07_or(self):
        f1 = cps.Filter.FilterPredicate("eq", "Foo", lambda a, b: a==b, 
                                        [cps.Filter.LITERAL_PREDICATE])
        
        f = cps.Filter("", [f1], 'or (eq "x") (eq "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))
        
    def test_02_01_build_one(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a: a=="foo", [])
        f = cps.Filter("", [f1])
        f.build([f1])
        self.assertEqual(f.text, "foo")
        
    def test_02_02_build_literal(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a,b: a==b, 
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1])
        f.build([f1, "bar"])
        self.assertEqual(f.text, 'foo "bar"')
        
    def test_02_03_build_nested(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a,b: a==b, 
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1])
        f.build([cps.Filter.OR_PREDICATE, [f1, "bar"], [f1, u"baz"]])
        self.assertEqual(f.text, 'or (foo "bar") (foo "baz")')
        

        