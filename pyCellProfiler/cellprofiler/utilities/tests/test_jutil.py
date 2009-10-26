'''test_jutil.py - test the jutil module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import numpy as np
import unittest

import cellprofiler.utilities.jutil as J
jb = J.javabridge

env = jb.JB_Env()
env.create(['-Dfoo.bar=true']) # set a system property for testing

class TestJutil(unittest.TestCase):
    def test_01_01_to_string(self):
        jstring = env.new_string_utf("Hello, world")
        self.assertEqual(J.to_string(env, jstring), "Hello, world")
        
    def test_01_02_make_instance(self):
        jobject = J.make_instance(env, "java/lang/Object", "()V")
        self.assertTrue(J.to_string(env, jobject).startswith("java.lang.Object"))
        
    def test_01_03_call(self):
        jstring = env.new_string_utf("Hello, world")
        self.assertEqual(J.call(env, jstring, "charAt", "(I)C", 0), "H")
        
    def test_01_03_01_static_call(self):
        result = J.static_call(env, "Ljava/lang/String;", "valueOf", 
                               "(I)Ljava/lang/String;",123)
        self.assertEqual(result, "123")
        
    def test_01_04_make_method(self):
        class String(object):
            klass = env.find_class("java/lang/String")
            def __init__(self):
                self.o = env.new_string_utf("Hello, world")
            
            charAt = J.make_method(env, klass, "charAt", "(I)C", "My documentation")
        
        s = String()
        self.assertEqual(s.charAt.__doc__, "My documentation")
        self.assertEqual(s.charAt(0), "H")
    
    def test_01_05_get_static_field(self):
        klass = env.find_class("java/lang/Short")
        self.assertEqual(J.get_static_field(env, klass, "MAX_VALUE", "S"), 2**15 - 1)
    
    def test_01_06_get_enumeration_wrapper(self):
        properties = J.static_call(env, "java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        keys = J.call(env, properties, "keys", "()Ljava/util/Enumeration;")
        enum = J.get_enumeration_wrapper(env, keys)
        has_java_vm_name = False
        while(enum.hasMoreElements()):
            key = J.to_string(env, enum.nextElement())
            if key == "java.vm.name":
                has_java_vm_name = True
        self.assertTrue(has_java_vm_name)
        
    def test_01_07_get_dictionary_wrapper(self):
        properties = J.static_call(env, "java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = J.get_dictionary_wrapper(env, properties)
        self.assertTrue(d.size() > 10)
        self.assertFalse(d.isEmpty())
        keys = J.get_enumeration_wrapper(env, d.keys())
        values = J.get_enumeration_wrapper(env, d.elements())
        n_elems = d.size()
        for i in range(n_elems):
            self.assertTrue(keys.hasMoreElements())
            key = J.to_string(env, keys.nextElement())
            self.assertTrue(values.hasMoreElements())
            value = J.to_string(env, values.nextElement())
            self.assertEqual(J.to_string(env, d.get(key)), value)
        self.assertFalse(keys.hasMoreElements())
        self.assertFalse(values.hasMoreElements())
        
    def test_01_08_jenumeration_to_string_list(self):
        properties = J.static_call(env, "java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = J.get_dictionary_wrapper(env, properties)
        keys = J.jenumeration_to_string_list(env, d.keys())
        enum = J.get_enumeration_wrapper(env, d.keys())
        for i in range(d.size()):
            key = J.to_string(env, enum.nextElement())
            self.assertEqual(key, keys[i])
    
    def test_01_09_jdictionary_to_string_dictionary(self):
        properties = J.static_call(env, "java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = J.get_dictionary_wrapper(env, properties)
        pyd = J.jdictionary_to_string_dictionary(env, properties)
        keys = J.jenumeration_to_string_list(env, d.keys())
        for key in keys:
            value = J.to_string(env, d.get(key))
            self.assertEqual(pyd[key], value)
        
        
        