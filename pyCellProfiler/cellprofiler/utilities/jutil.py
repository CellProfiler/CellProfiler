'''jutil.py - utilities for the javabridge

jutil provides utility functions that can be used to wrap a Java class.
The strategy here is to create closure functions that can be used to build
something that looks classy. For instance:

def get_java_string_class(env):
     class JavaString(object):
         klass = javabridge.find_class('java/lang/String')
         def __init__(self, s):
             self.o = javabridge.new_string_utf(str(s))
         charAt = jutil.make_method(env, klass, 'charAt', '(I)C')
         compareTo = jutil.make_method(env, klass, 'compareTo',
                                       '(Ljava/lang/Object;)I')
         concat = jutil.make_method(env, klass, 'concat',
                                    '(Ljava/lang/String;)Ljava/lang/String;')
         ...
     return JavaString

It's important to duck-type your class by using "klass" to store the class
and self.o to store the Java object instance.
'''
__version__ = "$Revision: 1 %"

import sys
if sys.platform.startswith('win'):
    #
    # Try harder by looking for JAVA_HOME and in the registry
    #
    from setup import find_javahome
    import os
    java_home = find_javahome()
    jvm_dir = os.path.join(java_home,'bin','client')
    os.environ['PATH'] = os.environ['PATH'] +';'+jvm_dir
import javabridge
    

import re

class JavaError(ValueError):
    '''A procedural error caused by misuse of jutil'''
    def __init__(self, message=None):
        super(JavaError,self).__init__(message)
        
class JavaException(BaseException):
    '''Represents a Java exception thrown inside the JVM'''
    def __init__(self, env):
        '''Initialize by calling exception_occurred'''
        self.throwable = env.exception_occurred()
        if self.throwable is None:
            raise ValueError("Tried to create a JavaException but there was no current exception")
        env.exception_clear()
        message = call(env, self.throwable, 'getMessage', '()Ljava/lang/String;')
        if message is not None:
            message = env.get_string_utf(message)
            super(BaseException, self).__init__(message)

def call(env, o, method_name, sig, *args):
    '''Call a method on an object
    
    o - object in question
    method_name - name of method on object's class
    sig - calling signature
    '''
    klass = env.get_object_class(o)
    method_id = env.get_method_id(klass, method_name, sig)
    if method_id is None:
        raise JavaError('Could not find method name = "%s" '
                        'with signature = "%s' % (method_name, sig))
    result = env.call_method(o, method_id, *args)
    if env.exception_occurred() is not None:
        raise JavaException(env)
    ret_sig = sig[sig.find(')')+1:]
    return get_nice_result(env,result,ret_sig)
    
def make_method(env, klass, name, sig, doc='No documentation'):
    '''Return a class method for the given Java class
    
    env - a Java environment from javabridge
    klass - a Java class from javabridge.find_class or other
    sig - a calling signature. 
          See http://java.sun.com/j2se/1.4.2/docs/guide/jni/spec/types.htm
          An example: "(ILjava/lang/String;)I[" takes an integer and
          string as parameters and returns an array of integers.
          Cheat sheet:
          Z - boolean
          B - byte
          C - char
          S - short
          I - int
          J - long
          F - float
          D - double
          L - class (e.g. Lmy/class;)
          [ - array of (e.g. [B = byte array)
    '''
    method_id = env.get_method_id(klass, name, sig)
    if method_id is None:
        raise JavaError('Could not find method name = "%s" '
                        'with signature = "%s' % (name, sig))
    args_sig = split_sig(sig[1:sig.find(')')])
    ret_sig = sig[sig.find(')')+1:]
    def method(self, *args):
        nice_args = get_nice_args(env, args, args_sig)
        result = env.call_method(self.o, method_id, *nice_args)
        if env.exception_occurred() is not None:
            raise JavaException(env)
        if result is None:
            return
        return get_nice_result(env, result, ret_sig)
    method.__doc__ = doc
    return method

def get_static_field(env, klass, name, sig):
    '''Get the value for a static field on a class
    
    klass - the class or string name of class
    name - the name of the field
    sig - the signature, typically, 'I' or 'Ljava/lang/String;'
    '''
    if isinstance(klass, javabridge.JB_Object):
        # Get the object's class
        klass = env.get_object_class(klass)
    elif not isinstance(klass, javabridge.JB_Class):
        class_name = str(klass)
        klass = env.find_class(class_name)
        if klass is None:
            raise ValueError("Could not load class %s"%class_name)
    field_id = env.get_static_field_id(klass, name, sig)
    if sig == 'Z':
        return env.get_static_boolean_field(klass, field_id)
    elif sig == 'B':
        return env.get_static_byte_field(klass, field_id)
    elif sig == 'S':
        return env.get_static_short_field(klass, field_id)
    elif sig == 'I':
        return env.get_static_int_field(klass, field_id)
    elif sig == 'J':
        return env.get_static_long_field(klass, field_id)
    elif sig == 'F':
        return env.get_static_float_field(klass, field_id)
    elif sig == 'D':
        return env.get_static_double_field(klass, field_id)
    else:
        return get_nice_result(env.get_static_object_field(klass, field_id))
        
def split_sig(sig):
    '''Split a signature into its constituent arguments'''
    split = []
    orig_sig = sig
    while len(sig) > 0:
        match = re.match("\\[*(?:[ZBCSIJFD]|L[^;]+;)",sig)
        if match is None:
            raise ValueError("Invalid signature: %s"%orig_sig)
        split.append(match.group())
        sig=sig[match.end():]
    return split
        
def get_nice_args(env, args, sig):
    '''Convert arguments to Java types where appropriate
    
    returns a list of possibly converted arguments
    '''
    return [get_nice_arg(env, arg, subsig)
            for arg, subsig in zip(args, sig)]

def get_nice_arg(env, arg, sig):
    '''Convert an argument into a Java type when appropriate'''
    if sig[0] == 'L' and not (isinstance(arg, javabridge.JB_Object) or
                                isinstance(arg, javabridge.JB_Class)):
        #
        # Check for the standard packing of java objects into class instances
        #
        if hasattr(arg, "o"):
            return arg.o
    if (sig in ('Ljava/lang/String;','Ljava/lang/Object;') and not
         isinstance(arg, javabridge.JB_Object)):
        return env.new_string_utf(str(arg))
    return arg

def get_nice_result(env, result, sig):
    '''Convert a result that may be a java object into a string'''
    if sig == 'Ljava/lang/String;':
        return env.get_string_utf(result)
    elif sig == '[B':
        # Convert a byte array into a numpy array
        return env.get_byte_array_elements(result)
    return result

def to_string(env, jobject):
    '''Call the toString method on any object'''
    if not isinstance(jobject, javabridge.JB_Object):
        return str(jobject)
    return call(env, jobject, 'toString', '()Ljava/lang/String;')

def get_dictionary_wrapper(env, dictionary):
    '''Return a wrapper of java.util.Dictionary
    
    Given a JB_Object that implements java.util.Dictionary, return a wrapper
    that implements the class methods.
    '''
    class Dictionary(object):
        klass = env.get_object_class(dictionary)
        def __init__(self):
            self.o = dictionary
        size = make_method(env, klass, 'size', '()I',
                           'Returns the number of entries in this dictionary')
        isEmpty = make_method(env, klass, 'isEmpty', '()Z',
                              'Tests if this dictionary has no entries')
        keys = make_method(env, klass, 'keys', '()Ljava/util/Enumeration;',
                           'Returns an enumeration of keys in this dictionary')
        elements = make_method(env, klass, 'elements',
                               '()Ljava/util/Enumeration;',
                               'Returns an enumeration of elements in this dictionary')
        get = make_method(env, klass, 'get',
                          '(Ljava/lang/Object;)Ljava/lang/Object;',
                          'Return the value associated with a key or None if no value')
    return Dictionary()

def jdictionary_to_string_dictionary(env, hashtable):
    '''Convert a Java dictionary to a Python dictionary
    
    Convert each key and value in the Java dictionary to
    a string and construct a Python dictionary from the result.
    '''
    jhashtable = get_dictionary_wrapper(env, hashtable)
    jkeys = jhashtable.keys()
    keys = jenumeration_to_string_list(env, jkeys)
    result = {}
    for key in keys:
        result[key] = to_string(env, jhashtable.get(key))
    return result

def get_enumeration_wrapper(env, enumeration):
    '''Return a wrapper of java.util.Enumeration
    
    Given a JB_Object that implements java.util.Enumeration,
    return an object that wraps the class methods.
    '''
    class Enumeration(object):
        klass = env.get_object_class(enumeration)
        assert klass is not None
        def __init__(self):
            '''Call the init method with the JB_Object'''
            self.o = enumeration
        hasMoreElements = make_method(env, klass, 'hasMoreElements', '()Z',
                                      'Return true if the enumeration has more elements to retrieve')
        nextElement = make_method(env, klass, 'nextElement', 
                                  '()Ljava/lang/Object;')
    return Enumeration()
        
def jenumeration_to_string_list(env, enumeration):
    '''Convert a Java enumeration to a Python list of strings
    
    Convert each element in an enumeration to a string and store
    in a Python list.
    '''
    jenumeration = get_enumeration_wrapper(env, enumeration)
    result = []
    while jenumeration.hasMoreElements():
        result.append(to_string(env, jenumeration.nextElement()))
    return result

def make_new(env, klass, sig):
    '''Make a function that creates a new instance of the class
    
    A typical init function looks like this:
    new_fn = make_new(env, klass, '(I)V')
    def __init__(self, i):
        self.o = new_fn(i)
    
    It's important to store the object in self.o because it's expected to be
    there by make_method, etc.
    '''
    method_id = env.get_method_id(klass, '<init>', sig)
    if method_id is None:
        raise JavaError('Could not find constructor '
                        'with signature = "%s' % sig)
    args_sig = split_sig(sig[1:sig.find(')')])
    def constructor(self, *args):
        result = env.new_object(klass, method_id, 
                                *get_nice_args(env, args, args_sig))
        if env.exception_occurred() is not None:
            raise JavaException(env)
        self.o = result
        return result
    return constructor

def make_instance(env, class_name, sig, *args):
    '''Create an instance of a class
    
    class_name - name of class
    sig - signature of constructor
    args - arguments to constructor
    '''
    klass = env.find_class(class_name)
    if klass is None:
        raise ValueError("Could not find class %s"%class_name)
    method_id = env.get_method_id(klass, '<init>', sig)
    if method_id is None:
        raise JavaError('Could not find constructor with signature = %s'%sig)
    args_sig = split_sig(sig[1:sig.find(')')])
    result = env.new_object(klass, method_id, *get_nice_args(env, args, args_sig))
    if env.exception_occurred() is not None:
        raise JavaException(env)
    return result

def get_class_wrapper(env, obj):
    '''Return a wrapper for an object's class (e.g. for reflection)
    
    '''
    class_object = call(env, obj, 'getClass','()Ljava/lang/Class;')
    klass1 = env.get_object_class(class_object)
    class Klass(object):
        klass = klass1
        def __init__(self):
            self.o = class_object
        getClasses = make_method(env, klass, 'getClasses','()[Ljava/lang/Class;',
                                 'Returns an array containing Class objects representing all the public classes and interfaces that are members of the class represented by this Class object.')
        getConstructors = make_method(env, klass, 'getConstructors','()[Ljava/lang/reflect/Constructor;')
        getFields = make_method(env, klass, 'getFields','()[Ljava/lang/reflect/Field;')
        getField = make_method(env, klass, 'getField','(Ljava/lang/String;)Ljava/lang/reflect/Field;')
        getMethod = make_method(env, klass, 'getMethod','(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;')
        getMethods = make_method(env, klass, 'getMethods','()[Ljava/lang/reflect/Method;')
    return Klass()


    