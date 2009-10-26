'''test_javabridge.py - Test the java / python bridge

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

import cellprofiler.utilities.jutil
jb = cellprofiler.utilities.jutil.javabridge

env = jb.JB_Env()
env.create(['-Dfoo.bar=true']) # set a system property for testing
class TestJavabridge(unittest.TestCase):
    def test_01_01_version(self):
        major,minor = env.get_version()
        
    def test_01_02_find_class(self):
        string_class = env.find_class('java/lang/String')
        self.assertTrue(isinstance(string_class, jb.JB_Class))

    def test_01_03_new_string_utf(self):
        jstring = env.new_string_utf("Hello, world")
        self.assertTrue(isinstance(jstring, jb.JB_Object))
        
    def test_01_04_get_string_utf(self):
        jstring = env.new_string_utf("Hello, world")
        pstring = env.get_string_utf(jstring)
        self.assertEqual(pstring, "Hello, world")
        
    def test_01_05_get_object_class(self):
        jstring = env.new_string_utf("Hello, world")
        string_class = env.get_object_class(jstring)
        self.assertTrue(isinstance(string_class, jb.JB_Class))
        
    def test_01_06_new_global_ref(self):
        jstring = env.new_string_utf("Hello, world")
        ref = env.new_global_ref(jstring)
        pstring = env.get_string_utf(ref)
        self.assertEqual(pstring, "Hello, world")
    
    def test_01_07_delete_global_ref(self):
        jstring = env.new_string_utf("Hello, world")
        ref = env.new_global_ref(jstring)
        env.delete_global_ref(ref)
        #
        # Note: probably crashes the VM to do anything with the result
        #
        
    def test_01_08_delete_local_ref(self):
        jstring = env.new_string_utf("Hello, world")
        env.delete_local_ref(jstring)
        
    def test_01_09_get_method_id(self):
        klass = env.find_class("java/lang/String")
        method_id = env.get_method_id(klass,'charAt','(I)C')
        self.assertTrue(method_id is not None)
        
    def test_01_10_get_static_method_id(self):
        klass = env.find_class("java/lang/String")
        method_id = env.get_static_method_id(klass, 'copyValueOf','([C)Ljava/lang/String;')
        self.assertTrue(method_id is not None)
        
    def test_01_11_new_object(self):
        klass = env.find_class("java/lang/Byte")
        method_id = env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jbyte = env.new_object(klass, method_id, env.new_string_utf("55"))
        self.assertTrue(jbyte is not None)
    
    def test_01_12_get_static_field_id(self):
        klass = env.find_class("java/lang/Boolean")
        field_id = env.get_static_field_id(klass, "FALSE","Ljava/lang/Boolean;")
        self.assertTrue(field_id is not None)
        
    def test_01_13_get_byte_array_elements(self):
        pass # see test_03_09_call_method_array for test
    
    def test_01_14_get_object_array_elements(self):
        jstring = env.new_string_utf("Hello, world")
        klass = env.get_object_class(jstring)
        method_id = env.get_method_id(klass, 'split', '(Ljava/lang/String;)[Ljava/lang/String;')
        split = env.new_string_utf(", ")
        result = env.call_method(jstring, method_id, split)
        result = env.get_object_array_elements(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(env.get_string_utf(result[0]), "Hello")
        self.assertEqual(env.get_string_utf(result[1]), "world")
        
    def test_01_15_make_byte_array(self):
        array = np.array([ord(x) for x in "Hello, world"],np.uint8)
        jarray = env.make_byte_array(array)
        klass = env.find_class("java/lang/String")
        method_id = env.get_method_id(klass, '<init>', '([B)V')
        result = env.new_object(klass, method_id, jarray)
        self.assertEqual(env.get_string_utf(result), "Hello, world")
        
    def test_02_01_exception_did_not_occur(self):
        self.assertTrue(env.exception_occurred() is None)
        
    def test_02_02_exception_occurred(self):
        klass = env.find_class("java/lang/String")
        env.get_method_id(klass,'copyValueOf','([C)Ljava/lang/String;')
        x = env.exception_occurred()
        self.assertTrue(isinstance(x, jb.JB_Object))
        env.exception_describe()
        env.exception_clear()
        self.assertTrue(env.exception_occurred() is None)
        
    def test_03_01_call_method_char(self):
        jstring = env.new_string_utf("Hello, world")
        klass = env.get_object_class(jstring)
        method_id = env.get_method_id(klass, 'charAt', '(I)C')
        
        for i, c in enumerate("Hello, world"):
            self.assertEqual(c, env.call_method(jstring, method_id, i))
    
    def test_03_02_call_method_bool(self):
        jstring = env.new_string_utf("Hello, world")
        klass = env.get_object_class(jstring)
        method_id = env.get_method_id(klass, 'equals', '(Ljava/lang/Object;)Z')
        self.assertTrue(env.call_method(jstring, method_id, jstring))
        self.assertFalse(env.call_method(jstring, method_id, env.new_string_utf("Foo")))
        
    def test_03_03_call_method_byte(self):
        klass = env.find_class("java/lang/Byte")
        method_id = env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jbyte = env.new_object(klass, method_id, env.new_string_utf("55"))
        method_id = env.get_method_id(klass, 'byteValue','()B')
        self.assertEqual(env.call_method(jbyte, method_id), 55)
    
    def test_03_04_call_method_short(self):
        klass = env.find_class("java/lang/Short")
        method_id = env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jshort = env.new_object(klass, method_id, env.new_string_utf("55"))
        method_id = env.get_method_id(klass, 'shortValue','()S')
        self.assertEqual(env.call_method(jshort, method_id), 55)
    
    def test_03_05_call_method_int(self):
        klass = env.find_class("java/lang/Integer")
        method_id = env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jint = env.new_object(klass, method_id, env.new_string_utf("65537"))
        method_id = env.get_method_id(klass, 'intValue','()I')
        self.assertEqual(env.call_method(jint, method_id), 65537)
    
    def test_03_06_call_method_long(self):
        klass = env.find_class("java/lang/Long")
        method_id = env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jlong = env.new_object(klass, method_id, env.new_string_utf("4611686018427387904"))
        method_id = env.get_method_id(klass, 'longValue','()J')
        self.assertEqual(env.call_method(jlong, method_id), 4611686018427387904)
    
    def test_03_07_call_method_float(self):
        klass = env.find_class("java/lang/Float")
        method_id = env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jfloat = env.new_object(klass, method_id, env.new_string_utf("55.3"))
        method_id = env.get_method_id(klass, 'floatValue','()F')
        self.assertAlmostEqual(env.call_method(jfloat, method_id), 55.3,5)
    
    def test_03_08_call_method_double(self):
        klass = env.find_class("java/lang/Double")
        method_id = env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jdouble = env.new_object(klass, method_id, env.new_string_utf("-55.64"))
        method_id = env.get_method_id(klass, 'doubleValue','()D')
        self.assertAlmostEqual(env.call_method(jdouble, method_id), -55.64)
        
    def test_03_09_call_method_array(self):
        jstring = env.new_string_utf("Hello, world")
        klass = env.get_object_class(jstring)
        method_id = env.get_method_id(klass, 'getBytes', '()[B')
        result = env.call_method(jstring, method_id)
        self.assertTrue(isinstance(result, jb.JB_Object))
        a = env.get_byte_array_elements(result)
        self.assertEqual("Hello, world", a.tostring())
    
    def test_03_10_call_method_object(self):
        hello = env.new_string_utf("Hello, ")
        world = env.new_string_utf("world")
        klass = env.get_object_class(hello)
        method_id = env.get_method_id(klass, 'concat', '(Ljava/lang/String;)Ljava/lang/String;')
        result = env.call_method(hello, method_id, world)
        self.assertEqual("Hello, world", env.get_string_utf(result))
    
    def test_04_01_call_static_bool(self):
        klass = env.find_class("java/lang/Boolean")
        method_id = env.get_static_method_id(klass, "getBoolean",'(Ljava/lang/String;)Z')
        self.assertTrue(method_id is not None)
        self.assertFalse(env.call_static_method(klass, method_id, 
                                                env.new_string_utf("os.name")))
        self.assertTrue(env.call_static_method(klass, method_id,
                                               env.new_string_utf("foo.bar")))
        
    def test_04_02_call_static_byte(self):
        klass = env.find_class("java/lang/Byte")
        method_id = env.get_static_method_id(klass, "parseByte",'(Ljava/lang/String;)B')
        number = env.new_string_utf("55")
        self.assertEqual(env.call_static_method(klass, method_id, number), 55)
        
    def test_04_03_call_static_short(self):
        klass = env.find_class("java/lang/Short")
        method_id = env.get_static_method_id(klass, "parseShort",'(Ljava/lang/String;)S')
        number = env.new_string_utf("-55")
        self.assertEqual(env.call_static_method(klass, method_id, number), -55)
    
    def test_04_04_call_static_int(self):
        klass = env.find_class("java/lang/Integer")
        method_id = env.get_static_method_id(klass, "parseInt",'(Ljava/lang/String;)I')
        number = env.new_string_utf("55")
        self.assertEqual(env.call_static_method(klass, method_id, number), 55)
        
    def test_04_05_call_static_long(self):
        klass = env.find_class("java/lang/Long")
        method_id = env.get_static_method_id(klass, "parseLong",'(Ljava/lang/String;)J')
        number = env.new_string_utf("-55")
        self.assertEqual(env.call_static_method(klass, method_id, number), -55)
        
    def test_04_06_call_static_float(self):
        klass = env.find_class("java/lang/Float")
        method_id = env.get_static_method_id(klass, "parseFloat",'(Ljava/lang/String;)F')
        number = env.new_string_utf("-55.25")
        self.assertAlmostEqual(env.call_static_method(klass, method_id, number), -55.25)
        
    def test_04_07_call_static_double(self):
        klass = env.find_class("java/lang/Double")
        method_id = env.get_static_method_id(klass, "parseDouble",'(Ljava/lang/String;)D')
        number = env.new_string_utf("55.6")
        self.assertAlmostEqual(env.call_static_method(klass, method_id, number), 55.6)
        
    def test_04_08_call_static_object(self):
        klass = env.find_class("java/lang/String")
        method_id = env.get_static_method_id(klass, "valueOf",'(Z)Ljava/lang/String;')
        result = env.call_static_method(klass, method_id, True)
        self.assertEqual(env.get_string_utf(result), "true")
        
    def test_04_09_call_static_char(self):
        klass = env.find_class("java/lang/Character")
        method_id = env.get_static_method_id(klass, "toLowerCase", "(C)C")
        result = env.call_static_method(klass, method_id, "X")
        self.assertEqual(result, "x")
        
    def test_04_10_call_static_array(self):
        jstring = env.new_string_utf("Hello, world")
        klass = env.get_object_class(jstring)
        method_id = env.get_method_id(klass, "toCharArray","()[C")
        chars = env.call_method(jstring, method_id)
        method_id = env.get_static_method_id(klass, "copyValueOf","([C)Ljava/lang/String;")
        result = env.call_static_method(klass, method_id, chars)
        self.assertEqual(env.get_string_utf(result), "Hello, world")
    
    def test_05_01_get_static_object_field(self):
        klass = env.find_class("java/lang/Boolean")
        field_id = env.get_static_field_id(klass, "FALSE","Ljava/lang/Boolean;")
        result = env.get_static_object_field(klass, field_id)
        method_id = env.get_method_id(klass, "booleanValue","()Z")
        self.assertFalse(env.call_method(result, method_id))
        
    def test_05_02_get_static_boolean_field(self):
        pass # can't find any examples
    
    def test_05_03_get_static_byte_field(self):
        klass = env.find_class("java/io/ObjectStreamConstants")
        field_id = env.get_static_field_id(klass, "SC_EXTERNALIZABLE","B")
        result = env.get_static_byte_field(klass, field_id)
        self.assertEqual(result, 4)
        
    def test_05_04_get_static_short_field(self):
        klass = env.find_class("java/io/ObjectStreamConstants")
        field_id = env.get_static_field_id(klass, "STREAM_MAGIC","S")
        result = env.get_static_short_field(klass, field_id)
        self.assertEqual(result, -21267) # 0xaced see http://java.sun.com/javase/6/docs/platform/serialization/spec/protocol.html
    
    def test_05_05_get_static_int_field(self):
        klass = env.find_class("java/io/ObjectStreamConstants")
        field_id = env.get_static_field_id(klass, "PROTOCOL_VERSION_1","I")
        result = env.get_static_int_field(klass, field_id)
        self.assertEqual(result, 1)
        
    def test_05_06_get_static_long_field(self):
        klass = env.find_class("java/security/Key")
        field_id = env.get_static_field_id(klass, "serialVersionUID", "J")
        result = env.get_static_long_field(klass, field_id)
        self.assertEqual(result, 6603384152749567654l) # see http://java.sun.com/j2se/1.4.2/docs/api/constant-values.html#java.security.Key.serialVersionUID
    
    def test_05_07_get_static_float_field(self):
        klass = env.find_class("java/lang/Float")
        field_id = env.get_static_field_id(klass, "MAX_VALUE","F")
        result = env.get_static_float_field(klass, field_id)
        self.assertAlmostEqual(result, 3.4028234663852886 * 10.0**38)
        
    def test_05_08_get_static_double_field(self):
        klass = env.find_class("java/lang/Math")
        field_id = env.get_static_field_id(klass, "PI","D")
        result = env.get_static_double_field(klass, field_id)
        self.assertAlmostEqual(result, 3.141592653589793)
    
