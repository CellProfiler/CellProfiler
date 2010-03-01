'''test_javabridge.py - Test the java / python bridge

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import numpy as np
import unittest

import cellprofiler.utilities.jutil
import bioformats # needed to start the common vm
jb = cellprofiler.utilities.jutil.javabridge

class TestJavabridge(unittest.TestCase):
    def setUp(self):
        self.env = cellprofiler.utilities.jutil.attach()
    def tearDown(self):
        cellprofiler.utilities.jutil.detach()

    def test_01_01_version(self):
        major,minor = self.env.get_version()
        
    def test_01_02_find_class(self):
        string_class = self.env.find_class('java/lang/String')
        self.assertTrue(isinstance(string_class, jb.JB_Class))

    def test_01_03_new_string_utf(self):
        jstring = self.env.new_string_utf("Hello, world")
        self.assertTrue(isinstance(jstring, jb.JB_Object))
        
    def test_01_04_get_string_utf(self):
        jstring = self.env.new_string_utf("Hello, world")
        pstring = self.env.get_string_utf(jstring)
        self.assertEqual(pstring, "Hello, world")
        
    def test_01_05_get_object_class(self):
        jstring = self.env.new_string_utf("Hello, world")
        string_class = self.env.get_object_class(jstring)
        self.assertTrue(isinstance(string_class, jb.JB_Class))
        
    def test_01_06_deallocate_object(self):
        jstring = self.env.new_string_utf("Hello, world")
        del jstring
        
    def test_01_09_get_method_id(self):
        klass = self.env.find_class("java/lang/String")
        method_id = self.env.get_method_id(klass,'charAt','(I)C')
        self.assertTrue(method_id is not None)
        
    def test_01_10_get_static_method_id(self):
        klass = self.env.find_class("java/lang/String")
        method_id = self.env.get_static_method_id(klass, 'copyValueOf','([C)Ljava/lang/String;')
        self.assertTrue(method_id is not None)
        
    def test_01_11_new_object(self):
        klass = self.env.find_class("java/lang/Byte")
        method_id = self.env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jbyte = self.env.new_object(klass, method_id, self.env.new_string_utf("55"))
        self.assertTrue(jbyte is not None)
    
    def test_01_12_get_static_field_id(self):
        klass = self.env.find_class("java/lang/Boolean")
        field_id = self.env.get_static_field_id(klass, "FALSE","Ljava/lang/Boolean;")
        self.assertTrue(field_id is not None)
        
    def test_01_13_get_byte_array_elements(self):
        pass # see test_03_09_call_method_array for test
    
    def test_01_14_get_object_array_elements(self):
        jstring = self.env.new_string_utf("Hello, world")
        klass = self.env.get_object_class(jstring)
        method_id = self.env.get_method_id(klass, 'split', '(Ljava/lang/String;)[Ljava/lang/String;')
        split = self.env.new_string_utf(", ")
        result = self.env.call_method(jstring, method_id, split)
        result = self.env.get_object_array_elements(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.env.get_string_utf(result[0]), "Hello")
        self.assertEqual(self.env.get_string_utf(result[1]), "world")
        
    def test_01_15_make_byte_array(self):
        array = np.array([ord(x) for x in "Hello, world"],np.uint8)
        jarray = self.env.make_byte_array(array)
        klass = self.env.find_class("java/lang/String")
        method_id = self.env.get_method_id(klass, '<init>', '([B)V')
        result = self.env.new_object(klass, method_id, jarray)
        self.assertEqual(self.env.get_string_utf(result), "Hello, world")
        
    def test_02_01_exception_did_not_occur(self):
        self.assertTrue(self.env.exception_occurred() is None)
        
    def test_02_02_exception_occurred(self):
        klass = self.env.find_class("java/lang/String")
        self.env.get_method_id(klass,'copyValueOf','([C)Ljava/lang/String;')
        x = self.env.exception_occurred()
        self.assertTrue(isinstance(x, jb.JB_Object))
        self.env.exception_describe()
        self.env.exception_clear()
        self.assertTrue(self.env.exception_occurred() is None)
        
    def test_03_01_call_method_char(self):
        jstring = self.env.new_string_utf("Hello, world")
        klass = self.env.get_object_class(jstring)
        method_id = self.env.get_method_id(klass, 'charAt', '(I)C')
        
        for i, c in enumerate("Hello, world"):
            self.assertEqual(c, self.env.call_method(jstring, method_id, i))
    
    def test_03_02_call_method_bool(self):
        jstring = self.env.new_string_utf("Hello, world")
        klass = self.env.get_object_class(jstring)
        method_id = self.env.get_method_id(klass, 'equals', '(Ljava/lang/Object;)Z')
        self.assertTrue(self.env.call_method(jstring, method_id, jstring))
        self.assertFalse(self.env.call_method(jstring, method_id, self.env.new_string_utf("Foo")))
        
    def test_03_03_call_method_byte(self):
        klass = self.env.find_class("java/lang/Byte")
        method_id = self.env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jbyte = self.env.new_object(klass, method_id, self.env.new_string_utf("55"))
        method_id = self.env.get_method_id(klass, 'byteValue','()B')
        self.assertEqual(self.env.call_method(jbyte, method_id), 55)
    
    def test_03_04_call_method_short(self):
        klass = self.env.find_class("java/lang/Short")
        method_id = self.env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jshort = self.env.new_object(klass, method_id, self.env.new_string_utf("55"))
        method_id = self.env.get_method_id(klass, 'shortValue','()S')
        self.assertEqual(self.env.call_method(jshort, method_id), 55)
    
    def test_03_05_call_method_int(self):
        klass = self.env.find_class("java/lang/Integer")
        method_id = self.env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jint = self.env.new_object(klass, method_id, self.env.new_string_utf("65537"))
        method_id = self.env.get_method_id(klass, 'intValue','()I')
        self.assertEqual(self.env.call_method(jint, method_id), 65537)
    
    def test_03_06_call_method_long(self):
        klass = self.env.find_class("java/lang/Long")
        method_id = self.env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jlong = self.env.new_object(klass, method_id, self.env.new_string_utf("4611686018427387904"))
        method_id = self.env.get_method_id(klass, 'longValue','()J')
        self.assertEqual(self.env.call_method(jlong, method_id), 4611686018427387904)
    
    def test_03_07_call_method_float(self):
        klass = self.env.find_class("java/lang/Float")
        method_id = self.env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jfloat = self.env.new_object(klass, method_id, self.env.new_string_utf("55.3"))
        method_id = self.env.get_method_id(klass, 'floatValue','()F')
        self.assertAlmostEqual(self.env.call_method(jfloat, method_id), 55.3,5)
    
    def test_03_08_call_method_double(self):
        klass = self.env.find_class("java/lang/Double")
        method_id = self.env.get_method_id(klass, '<init>','(Ljava/lang/String;)V')
        jdouble = self.env.new_object(klass, method_id, self.env.new_string_utf("-55.64"))
        method_id = self.env.get_method_id(klass, 'doubleValue','()D')
        self.assertAlmostEqual(self.env.call_method(jdouble, method_id), -55.64)
        
    def test_03_09_call_method_array(self):
        jstring = self.env.new_string_utf("Hello, world")
        klass = self.env.get_object_class(jstring)
        method_id = self.env.get_method_id(klass, 'getBytes', '()[B')
        result = self.env.call_method(jstring, method_id)
        self.assertTrue(isinstance(result, jb.JB_Object))
        a = self.env.get_byte_array_elements(result)
        self.assertEqual("Hello, world", a.tostring())
    
    def test_03_10_call_method_object(self):
        hello = self.env.new_string_utf("Hello, ")
        world = self.env.new_string_utf("world")
        klass = self.env.get_object_class(hello)
        method_id = self.env.get_method_id(klass, 'concat', '(Ljava/lang/String;)Ljava/lang/String;')
        result = self.env.call_method(hello, method_id, world)
        self.assertEqual("Hello, world", self.env.get_string_utf(result))
    
    def test_04_01_call_static_bool(self):
        klass = self.env.find_class("java/lang/Boolean")
        method_id = self.env.get_static_method_id(klass, "getBoolean",'(Ljava/lang/String;)Z')
        self.assertTrue(method_id is not None)
        self.assertFalse(self.env.call_static_method(klass, method_id, 
                                                self.env.new_string_utf("os.name")))
        self.assertTrue(self.env.call_static_method(klass, method_id,
                                               self.env.new_string_utf("loci.bioformats.loaded")))
        
    def test_04_02_call_static_byte(self):
        klass = self.env.find_class("java/lang/Byte")
        method_id = self.env.get_static_method_id(klass, "parseByte",'(Ljava/lang/String;)B')
        number = self.env.new_string_utf("55")
        self.assertEqual(self.env.call_static_method(klass, method_id, number), 55)
        
    def test_04_03_call_static_short(self):
        klass = self.env.find_class("java/lang/Short")
        method_id = self.env.get_static_method_id(klass, "parseShort",'(Ljava/lang/String;)S')
        number = self.env.new_string_utf("-55")
        self.assertEqual(self.env.call_static_method(klass, method_id, number), -55)
    
    def test_04_04_call_static_int(self):
        klass = self.env.find_class("java/lang/Integer")
        method_id = self.env.get_static_method_id(klass, "parseInt",'(Ljava/lang/String;)I')
        number = self.env.new_string_utf("55")
        self.assertEqual(self.env.call_static_method(klass, method_id, number), 55)
        
    def test_04_05_call_static_long(self):
        klass = self.env.find_class("java/lang/Long")
        method_id = self.env.get_static_method_id(klass, "parseLong",'(Ljava/lang/String;)J')
        number = self.env.new_string_utf("-55")
        self.assertEqual(self.env.call_static_method(klass, method_id, number), -55)
        
    def test_04_06_call_static_float(self):
        klass = self.env.find_class("java/lang/Float")
        method_id = self.env.get_static_method_id(klass, "parseFloat",'(Ljava/lang/String;)F')
        number = self.env.new_string_utf("-55.25")
        self.assertAlmostEqual(self.env.call_static_method(klass, method_id, number), -55.25)
        
    def test_04_07_call_static_double(self):
        klass = self.env.find_class("java/lang/Double")
        method_id = self.env.get_static_method_id(klass, "parseDouble",'(Ljava/lang/String;)D')
        number = self.env.new_string_utf("55.6")
        self.assertAlmostEqual(self.env.call_static_method(klass, method_id, number), 55.6)
        
    def test_04_08_call_static_object(self):
        klass = self.env.find_class("java/lang/String")
        method_id = self.env.get_static_method_id(klass, "valueOf",'(Z)Ljava/lang/String;')
        result = self.env.call_static_method(klass, method_id, True)
        self.assertEqual(self.env.get_string_utf(result), "true")
        
    def test_04_09_call_static_char(self):
        klass = self.env.find_class("java/lang/Character")
        method_id = self.env.get_static_method_id(klass, "toLowerCase", "(C)C")
        result = self.env.call_static_method(klass, method_id, "X")
        self.assertEqual(result, "x")
        
    def test_04_10_call_static_array(self):
        jstring = self.env.new_string_utf("Hello, world")
        klass = self.env.get_object_class(jstring)
        method_id = self.env.get_method_id(klass, "toCharArray","()[C")
        chars = self.env.call_method(jstring, method_id)
        method_id = self.env.get_static_method_id(klass, "copyValueOf","([C)Ljava/lang/String;")
        result = self.env.call_static_method(klass, method_id, chars)
        self.assertEqual(self.env.get_string_utf(result), "Hello, world")
    
    def test_05_01_get_static_object_field(self):
        klass = self.env.find_class("java/lang/Boolean")
        field_id = self.env.get_static_field_id(klass, "FALSE","Ljava/lang/Boolean;")
        result = self.env.get_static_object_field(klass, field_id)
        method_id = self.env.get_method_id(klass, "booleanValue","()Z")
        self.assertFalse(self.env.call_method(result, method_id))
        
    def test_05_02_get_static_boolean_field(self):
        pass # can't find any examples
    
    def test_05_03_get_static_byte_field(self):
        klass = self.env.find_class("java/io/ObjectStreamConstants")
        field_id = self.env.get_static_field_id(klass, "SC_EXTERNALIZABLE","B")
        result = self.env.get_static_byte_field(klass, field_id)
        self.assertEqual(result, 4)
        
    def test_05_04_get_static_short_field(self):
        klass = self.env.find_class("java/io/ObjectStreamConstants")
        field_id = self.env.get_static_field_id(klass, "STREAM_MAGIC","S")
        result = self.env.get_static_short_field(klass, field_id)
        self.assertEqual(result, -21267) # 0xaced see http://java.sun.com/javase/6/docs/platform/serialization/spec/protocol.html
    
    def test_05_05_get_static_int_field(self):
        klass = self.env.find_class("java/io/ObjectStreamConstants")
        field_id = self.env.get_static_field_id(klass, "PROTOCOL_VERSION_1","I")
        result = self.env.get_static_int_field(klass, field_id)
        self.assertEqual(result, 1)
        
    def test_05_06_get_static_long_field(self):
        klass = self.env.find_class("java/security/Key")
        field_id = self.env.get_static_field_id(klass, "serialVersionUID", "J")
        result = self.env.get_static_long_field(klass, field_id)
        self.assertEqual(result, 6603384152749567654l) # see http://java.sun.com/j2se/1.4.2/docs/api/constant-values.html#java.security.Key.serialVersionUID
    
    def test_05_07_get_static_float_field(self):
        klass = self.env.find_class("java/lang/Float")
        field_id = self.env.get_static_field_id(klass, "MAX_VALUE","F")
        result = self.env.get_static_float_field(klass, field_id)
        self.assertAlmostEqual(result, 3.4028234663852886 * 10.0**38)
        
    def test_05_08_get_static_double_field(self):
        klass = self.env.find_class("java/lang/Math")
        field_id = self.env.get_static_field_id(klass, "PI","D")
        result = self.env.get_static_double_field(klass, field_id)
        self.assertAlmostEqual(result, 3.141592653589793)
    
