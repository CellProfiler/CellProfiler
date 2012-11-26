'''test_jutil.py - test the jutil module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import gc
import os
import numpy as np
import threading
import unittest
import sys

import cellprofiler.utilities.jutil as J
import bioformats # to start the VM
jb = J.javabridge

class TestJutil(unittest.TestCase):
    def setUp(self):
        self.env = J.attach()
    
    def tearDown(self):
        J.detach()
        
    def test_01_01_to_string(self):
        jstring = self.env.new_string_utf("Hello, world")
        self.assertEqual(J.to_string(jstring), "Hello, world")
        
    def test_01_02_make_instance(self):
        jobject = J.make_instance("java/lang/Object", "()V")
        self.assertTrue(J.to_string(jobject).startswith("java.lang.Object"))
        
    def test_01_03_call(self):
        jstring = self.env.new_string_utf("Hello, world")
        self.assertEqual(J.call(jstring, "charAt", "(I)C", 0), "H")
        
    def test_01_03_01_static_call(self):
        result = J.static_call("Ljava/lang/String;", "valueOf", 
                               "(I)Ljava/lang/String;",123)
        self.assertEqual(result, "123")
        
    def test_01_04_make_method(self):
        env = self.env
        class String(object):
            def __init__(self):
                self.o = env.new_string_utf("Hello, world")
                
            charAt = J.make_method("charAt", "(I)C", "My documentation")
            
        s = String()
        self.assertEqual(s.charAt.__doc__, "My documentation")
        self.assertEqual(s.charAt(0), "H")
    
    def test_01_05_get_static_field(self):
        klass = self.env.find_class("java/lang/Short")
        self.assertEqual(J.get_static_field(klass, "MAX_VALUE", "S"), 2**15 - 1)
    
    def test_01_06_get_enumeration_wrapper(self):
        properties = J.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        keys = J.call(properties, "keys", "()Ljava/util/Enumeration;")
        enum = J.get_enumeration_wrapper(keys)
        has_java_vm_name = False
        while(enum.hasMoreElements()):
            key = J.to_string(enum.nextElement())
            if key == "java.vm.name":
                has_java_vm_name = True
        self.assertTrue(has_java_vm_name)
        
    def test_01_07_get_dictionary_wrapper(self):
        properties = J.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = J.get_dictionary_wrapper(properties)
        self.assertTrue(d.size() > 10)
        self.assertFalse(d.isEmpty())
        keys = J.get_enumeration_wrapper(d.keys())
        values = J.get_enumeration_wrapper(d.elements())
        n_elems = d.size()
        for i in range(n_elems):
            self.assertTrue(keys.hasMoreElements())
            key = J.to_string(keys.nextElement())
            self.assertTrue(values.hasMoreElements())
            value = J.to_string(values.nextElement())
            self.assertEqual(J.to_string(d.get(key)), value)
        self.assertFalse(keys.hasMoreElements())
        self.assertFalse(values.hasMoreElements())
        
    def test_01_08_jenumeration_to_string_list(self):
        properties = J.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = J.get_dictionary_wrapper(properties)
        keys = J.jenumeration_to_string_list(d.keys())
        enum = J.get_enumeration_wrapper(d.keys())
        for i in range(d.size()):
            key = J.to_string(enum.nextElement())
            self.assertEqual(key, keys[i])
    
    def test_01_09_jdictionary_to_string_dictionary(self):
        properties = J.static_call("java/lang/System", "getProperties",
                                   "()Ljava/util/Properties;")
        d = J.get_dictionary_wrapper(properties)
        pyd = J.jdictionary_to_string_dictionary(properties)
        keys = J.jenumeration_to_string_list(d.keys())
        for key in keys:
            value = J.to_string(d.get(key))
            self.assertEqual(pyd[key], value)
            
    def test_01_10_make_new(self):
        env = self.env
        class MyClass:
            new_fn = J.make_new("java/lang/Object", '()V')
            def __init__(self):
                self.new_fn()
        my_instance = MyClass()
        
    def test_01_11_class_for_name(self):
        c = J.class_for_name('java.lang.String')
        name = J.call(c, 'getCanonicalName', '()Ljava/lang/String;')
        self.assertEqual(name, 'java.lang.String')
    
    def test_02_01_access_object_across_environments(self):
        #
        # Create an object in one environment, close the environment,
        # open a second environment, then use it and delete it.
        #
        env = self.env
        self.assertTrue(isinstance(env,J.javabridge.JB_Env))
        class MyInteger:
            new_fn = J.make_new("java/lang/Integer",'(I)V')
            def __init__(self, value):
                self.new_fn(value)
            intValue = J.make_method("intValue", '()I')
        my_value = 543
        my_integer=MyInteger(my_value)
        def run(my_integer = my_integer):
            env = J.attach()
            self.assertEqual(my_integer.intValue(),my_value)
            J.detach()
        t = threading.Thread(target = run)
        t.start()
        t.join()
        
    def test_02_02_delete_in_environment(self):
        env = self.env
        self.assertTrue(isinstance(env,J.javabridge.JB_Env))
        class MyInteger:
            new_fn = J.make_new("java/lang/Integer",'(I)V')
            def __init__(self, value):
                self.new_fn(value)
            intValue = J.make_method("intValue", '()I')
        my_value = 543
        my_integer=MyInteger(my_value)
        def run(my_integer = my_integer):
            env = J.attach()
            self.assertEqual(my_integer.intValue(),my_value)
            del my_integer
            J.detach()
        t = threading.Thread(target = run)
        t.start()
        t.join()
        
    def test_02_03_death_and_resurrection(self):
        '''Put an object into another in Java, delete it in Python and recover it'''
        
        np.random.seed(24)
        my_value = np.random.randint(0, 1000)
        jobj = J.make_instance("java/lang/Integer", "(I)V", my_value)
        integer_klass = self.env.find_class("java/lang/Integer")
        jcontainer = self.env.make_object_array(1, integer_klass)
        self.env.set_object_array_element(jcontainer, 0, jobj)
        del jobj
        gc.collect()
        jobjs = self.env.get_object_array_elements(jcontainer)
        jobj = jobjs[0]
        self.assertEqual(J.call(jobj, "intValue", "()I"), my_value)
    
    #def test_02_04_memory(self):
        #'''Make sure that memory is truly released when an object is dereferenced'''
        #env = self.env
        #self.assertTrue(isinstance(env,J.javabridge.JB_Env))
        #for i in range(25):
            #print "starting pass %d" % (i+1)
            #memory = np.random.uniform(size=1000*1000*10)
            #jarray = env.make_double_array(memory)
            #J.static_call("java/util/Arrays", "sort",
                          #"([D)V", jarray)
            #sorted_memory = env.get_double_array_elements(jarray)
            #np.testing.assert_almost_equal(sorted_memory[0], memory.min())
            #np.testing.assert_almost_equal(sorted_memory[-1], memory.max())
            #del memory
            #del sorted_memory
            #del jarray
            #gc.collect()
            
    def test_03_01_cw_from_class(self):
        '''Get a class wrapper from a class'''
        c = J.get_class_wrapper(J.make_instance('java/lang/Integer', '(I)V',
                                                14))
    
    def test_03_02_cw_from_string(self):
        '''Get a class wrapper from a string'''
        c = J.get_class_wrapper("java.lang.Number")
        
    def test_03_03_cw_get_classes(self):
        c = J.get_class_wrapper('java.lang.Number')
        classes = c.getClasses()
        self.assertEqual(len(J.get_env().get_object_array_elements(classes)), 0)
        
    def test_03_04_cw_get_annotation(self):
        c = J.get_class_wrapper('java.security.Identity')
        annotation = c.getAnnotation(J.class_for_name('java.lang.Deprecated'))
        self.assertTrue(annotation is not None)
    
    def test_03_05_cw_get_annotations(self):
        c = J.get_class_wrapper('java.security.Identity')
        annotations = c.getAnnotations()
        annotations = J.get_env().get_object_array_elements(annotations)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(J.to_string(annotations[0]),'@java.lang.Deprecated()')
        
    def test_03_06_cw_get_constructors(self):
        c = J.get_class_wrapper('java.lang.String')
        constructors = c.getConstructors()
        constructors = J.get_env().get_object_array_elements(constructors)
        self.assertEqual(len(constructors), 15)
        
    def test_03_07_cw_get_fields(self):
        c = J.get_class_wrapper('java.lang.String')
        fields = c.getFields()
        fields = J.get_env().get_object_array_elements(fields)
        self.assertEqual(len(fields), 1)
        self.assertEqual(J.call(fields[0], 'getName', '()Ljava/lang/String;'),
                         "CASE_INSENSITIVE_ORDER")
        
    def test_03_08_cw_get_field(self):
        c = J.get_class_wrapper('java.lang.String')
        field = c.getField('CASE_INSENSITIVE_ORDER')
        modifiers = J.call(field, 'getModifiers', '()I')
        static = J.get_static_field('java/lang/reflect/Modifier','STATIC','I')
        self.assertEqual((modifiers & static), static)
        
    def test_03_09_cw_get_method(self):
        sclass = J.class_for_name('java.lang.String')
        iclass = J.get_static_field('java/lang/Integer', 'TYPE', 
                                    'Ljava/lang/Class;')
        c = J.get_class_wrapper('java.lang.String')
        m = c.getMethod('charAt', [ iclass ])
        self.assertEqual(J.to_string(J.call(m, 'getReturnType', '()Ljava/lang/Class;')), 'char')
        m = c.getMethod('concat', [ sclass])
        self.assertEqual(J.to_string(J.call(m, 'getReturnType', '()Ljava/lang/Class;')), 
                         'class java.lang.String')
        
    def test_03_10_cw_get_methods(self):
        c = J.get_class_wrapper('java.lang.String')
        mmm = J.get_env().get_object_array_elements(c.getMethods())
        self.assertTrue(any([J.call(m, 'getName', '()Ljava/lang/String;') == 'concat'
                             for m in mmm]))
        
    def test_03_11_cw_get_constructor(self):
        c = J.get_class_wrapper('java.lang.String')
        sclass = J.class_for_name('java.lang.String')
        constructor = c.getConstructor([sclass])
        self.assertEqual(J.call(constructor, 'getName', '()Ljava/lang/String;'),
                         'java.lang.String')
        
    def test_04_01_field_get(self):
        c = J.get_class_wrapper('java.lang.Byte')
        f = J.get_field_wrapper(c.getField('MAX_VALUE'))
        v = f.get(None)
        self.assertEqual(J.to_string(v), '127')
        
    def test_04_02_field_name(self):
        c = J.get_class_wrapper('java.lang.Byte')
        f = J.get_field_wrapper(c.getField('MAX_VALUE'))
        self.assertEqual(f.getName(), 'MAX_VALUE')
        
    def test_04_03_field_type(self):
        c = J.get_class_wrapper('java.lang.Byte')
        f = J.get_field_wrapper(c.getField('MAX_VALUE'))
        t = f.getType()
        self.assertEqual(J.to_string(t), 'byte')
        
    def test_05_01_run_script(self):
        result = J.call(J.run_script("2+2"), "intValue", "()I")
        self.assertEqual(result, 4)
        
    def test_05_02_run_script_with_inputs(self):
        result = J.run_script("a+b", bindings_in={"a":2, "b":3})
        self.assertEqual(J.call(result, "intValue", "()I"), 5)
        
    def test_05_03_run_script_with_outputs(self):
        outputs = { "result": None}
        J.run_script("var result = 2+2;", bindings_out=outputs)
        self.assertEqual(J.call(outputs["result"], "intValue", "()I"), 4)
        
    def test_06_01_execute_asynch_main(self):
        J.execute_runnable_in_main_thread(J.run_script(
            "new java.lang.Runnable() { run:function() {}};"))
        
    def test_06_02_execute_synch_main(self):
        J.execute_runnable_in_main_thread(J.run_script(
            "new java.lang.Runnable() { run:function() {}};"), True)
        
    def test_06_03_call_main(self):
        c = J.run_script("""
        new java.util.concurrent.Callable() {
           call: function() { return 2+2; }};""")
        result = J.execute_callable_in_main_thread(c)
        self.assertEqual(J.call(result, "intValue", "()I"), 4)
        
if __name__=="__main__":
    unittest.main()
