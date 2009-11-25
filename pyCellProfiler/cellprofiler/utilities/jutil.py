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

import atexit
import gc
import threading
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
elif sys.platform == 'darwin':
    #
    # Put the jvm library on the path, hoping it is always in the same place
    #
    import os
    jvm_dir = '/System/Library/Frameworks/JavaVM.framework/Libraries'
    os.environ['PATH'] = os.environ['PATH'] + ':' + jvm_dir
elif sys.platform.startswith('linux'):
    #
    # The Broad's libjvm is here, but yours may be different if your
    # processor is not x86-64
    #
    from setup import find_javahome
    import os
    java_home = find_javahome()
    if java_home is None:
        raise RuntimeError("Could not find JAVA_HOME environment variable.")
    jvm_dir = os.path.join(java_home, 'jre','lib','amd64','server')
    os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':' + jvm_dir

import javabridge
import re

class JavaError(ValueError):
    '''A procedural error caused by misuse of jutil'''
    def __init__(self, message=None):
        super(JavaError,self).__init__(message)
        
class JavaException(Exception):
    '''Represents a Java exception thrown inside the JVM'''
    def __init__(self, throwable):
        '''Initialize by calling exception_occurred'''
        env = get_env()
        env.exception_describe()
        self.throwable = throwable
        try:
            if self.throwable is None:
                raise ValueError("Tried to create a JavaException but there was no current exception")
            #
            # The following has to be done by hand because the exception can't be
            # cleared at this point
            #
            klass = env.get_object_class(self.throwable)
            method_id = env.get_method_id(klass, 'getMessage', 
                                          '()Ljava/lang/String;')
            if method_id is not None:
                message = env.call_method(self.throwable, method_id)
                if message is not None:
                    message = env.get_string_utf(message)
                    super(BaseException, self).__init__(message)
        finally:
            env.exception_clear()

__vm = None
__wake_event = threading.Event()
__dead_event = threading.Event()
__thread_local_env = threading.local()
__kill = [False]
__dead_objects = []

def start_vm(args):
    '''Start the Java VM'''
    global __vm
    
    if __vm is not None:
        return
    start_event = threading.Event()
    pt = [] # holds the thread... eventually
    
    def start_thread():
        global __vm
        global __wake_event
        global __dead_event
        global __thread_local_env
        global __kill
        global __dead_objects
        
        __vm = javabridge.JB_VM()
        #
        # We get local copies here and bind them in a closure to guarantee
        # that they exist past atexit.
        #
        vm = __vm
        wake_event = __wake_event
        dead_event = __dead_event
        kill = __kill
        dead_objects = __dead_objects
        thread_local_env = __thread_local_env
        ptt = pt # needed to bind to pt inside exit_fn
        def exit_fn():
            if vm is not None:
                if getattr(thread_local_env,"env",None) is not None:
                    print "Must detach at exit"
                    detach()
                kill[0] = True
                wake_event.set()
                ptt[0].join()
        atexit.register(exit_fn)
        try:
            env = vm.create(args)
            __thread_local_env.env = env
        finally:
            start_event.set()
        wake_event.clear()
        while True:
            wake_event.wait()
            wake_event.clear()
            while(len(dead_objects)):
                dead_object = dead_objects.pop()
                if isinstance(dead_object, javabridge.JB_Object):
                    # Object may have been totally GC'ed
                    env.dealloc_jobject(dead_object)
            if kill[0]:
                break
        def null_defer_fn(jbo):
            '''Install a "do nothing" defer function in our env'''
            pass
        env.set_defer_fn(null_defer_fn)
        vm.destroy()
        __vm = None
        dead_event.set()
        
    t = threading.Thread(target=start_thread)
    pt.append(t)
    t.start()
    start_event.wait()

#
# We make kill_vm as a closure here to bind local copies of the global objects
#
def make_kill_vm():
    '''Kill the currently-running Java environment'''
    global __wake_event
    global __dead_event
    global __kill
    global __thread_local_env
    wake_event = __wake_event
    dead_event = __dead_event
    kill = __kill
    thread_local_env = __thread_local_env
    def kill_vm():
        global __vm
        if __vm is None:
            return
        if getattr(thread_local_env,"env",None) is not None:
            detach()
        kill[0] = True
        wake_event.set()
        dead_event.wait()
    return kill_vm

kill_vm = make_kill_vm()
    
def attach():
    '''Attach to the VM, receiving the thread's environment'''
    global __thread_local_env
    global __vm
    assert isinstance(__vm, javabridge.JB_VM)
    __thread_local_env.env = __vm.attach()
    return __thread_local_env.env
    
def get_env():
    '''Return the thread's environment
    
    Note: call start_vm() and attach() before calling this
    '''
    global __thread_local_env
    return __thread_local_env.env

def detach():
    '''Detach from the VM, releasing the thread's environment'''
    global __vm
    global __thread_local_env
    global __dead_objects
    global __wake_event
    global __kill
    
    env = __thread_local_env.env
    dead_objects = __dead_objects
    wake_event = __wake_event
    kill = __kill
    def defer_fn(jbo):
        '''Do deallocation on the JVM's thread after detach'''
        if not kill[0]:
            dead_objects.append(jbo)
            wake_event.set()
    env.set_defer_fn(defer_fn)
    __thread_local_env.env = None
    __vm.detach()

def call(o, method_name, sig, *args):
    '''Call a method on an object
    
    o - object in question
    method_name - name of method on object's class
    sig - calling signature
    '''
    env = get_env()
    klass = env.get_object_class(o)
    jexception = get_env().exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    method_id = env.get_method_id(klass, method_name, sig)
    jexception = get_env().exception_occurred()
    if method_id is None:
        if jexception is not None:
            raise JavaException(jexception)
        raise JavaError('Could not find method name = "%s" '
                        'with signature = "%s"' % (method_name, sig))
    args_sig = split_sig(sig[1:sig.find(')')])
    ret_sig = sig[sig.find(')')+1:]
    nice_args = get_nice_args(args, args_sig)
    result = env.call_method(o, method_id, *nice_args)
    x = env.exception_occurred()
    if x is not None:
        raise JavaException(x)
    return get_nice_result(result,ret_sig)

def static_call(class_name, method_name, sig, *args):
    '''Call a static method on a class
    
    class_name - name of the class, using slashes
    method_name - name of the static method
    sig - signature of the static method
    '''
    env = get_env()
    klass = env.find_class(class_name)
    if klass is None:
        raise JavaException(env, env.exception_occurred())
    
    method_id = env.get_static_method_id(klass, method_name, sig)
    if method_id is None:
        raise JavaError('Could not find method name = %s '
                        'with signature = %s' %(method_name, sig))
    args_sig = split_sig(sig[1:sig.find(')')])
    ret_sig = sig[sig.find(')')+1:]
    nice_args = get_nice_args(args, args_sig)
    result = env.call_static_method(klass, method_id,*nice_args)
    jexception = env.exception_occurred() 
    if jexception is not None:
        raise JavaException(env, jexception)
    return get_nice_result(result, ret_sig)

def make_method(name, sig, doc='No documentation'):
    '''Return a class method for the given Java class
    
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
    
    Note - this assumes that the JNI object is stored in self.o. Use like this:
    
    class Integer:
        new_fn = make_new("java/lang/Integer", "(I)V")
        def __init__(self, i):
            self.new_fn(i)
        intValue = make_method("intValue", "()I","Retrieve the integer value")
    i = Integer(435)
    if i.intValue() == 435:
        print "It worked"
    '''
    
    def method(self, *args):
        assert isinstance(self.o, javabridge.JB_Object)
        return call(self.o, name, sig, *args)
    method.__doc__ = doc
    return method

def get_static_field(klass, name, sig):
    '''Get the value for a static field on a class
    
    klass - the class or string name of class
    name - the name of the field
    sig - the signature, typically, 'I' or 'Ljava/lang/String;'
    '''
    env = get_env()
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
        
def get_nice_args(args, sig):
    '''Convert arguments to Java types where appropriate
    
    returns a list of possibly converted arguments
    '''
    return [get_nice_arg(arg, subsig)
            for arg, subsig in zip(args, sig)]

def get_nice_arg(arg, sig):
    '''Convert an argument into a Java type when appropriate'''
    env = get_env()
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

def get_nice_result(result, sig):
    '''Convert a result that may be a java object into a string'''
    env = get_env()
    if sig == 'Ljava/lang/String;':
        return env.get_string_utf(result)
    elif sig == '[B':
        # Convert a byte array into a numpy array
        return env.get_byte_array_elements(result)
    return result

def to_string(jobject):
    '''Call the toString method on any object'''
    env = get_env()
    if not isinstance(jobject, javabridge.JB_Object):
        return str(jobject)
    return call(jobject, 'toString', '()Ljava/lang/String;')

def get_dictionary_wrapper(dictionary):
    '''Return a wrapper of java.util.Dictionary
    
    Given a JB_Object that implements java.util.Dictionary, return a wrapper
    that implements the class methods.
    '''
    env = get_env()
    class Dictionary(object):
        def __init__(self):
            self.o = dictionary
        size = make_method('size', '()I',
                           'Returns the number of entries in this dictionary')
        isEmpty = make_method('isEmpty', '()Z',
                              'Tests if this dictionary has no entries')
        keys = make_method('keys', '()Ljava/util/Enumeration;',
                           'Returns an enumeration of keys in this dictionary')
        elements = make_method('elements',
                               '()Ljava/util/Enumeration;',
                               'Returns an enumeration of elements in this dictionary')
        get = make_method('get',
                          '(Ljava/lang/Object;)Ljava/lang/Object;',
                          'Return the value associated with a key or None if no value')
    return Dictionary()

def jdictionary_to_string_dictionary(hashtable):
    '''Convert a Java dictionary to a Python dictionary
    
    Convert each key and value in the Java dictionary to
    a string and construct a Python dictionary from the result.
    '''
    jhashtable = get_dictionary_wrapper(hashtable)
    jkeys = jhashtable.keys()
    keys = jenumeration_to_string_list(jkeys)
    result = {}
    for key in keys:
        result[key] = to_string(jhashtable.get(key))
    return result

def get_enumeration_wrapper(enumeration):
    '''Return a wrapper of java.util.Enumeration
    
    Given a JB_Object that implements java.util.Enumeration,
    return an object that wraps the class methods.
    '''
    env = get_env()
    class Enumeration(object):
        def __init__(self):
            '''Call the init method with the JB_Object'''
            self.o = enumeration
        hasMoreElements = make_method('hasMoreElements', '()Z',
                                      'Return true if the enumeration has more elements to retrieve')
        nextElement = make_method('nextElement', 
                                  '()Ljava/lang/Object;')
    return Enumeration()
        
def jenumeration_to_string_list(enumeration):
    '''Convert a Java enumeration to a Python list of strings
    
    Convert each element in an enumeration to a string and store
    in a Python list.
    '''
    jenumeration = get_enumeration_wrapper(enumeration)
    result = []
    while jenumeration.hasMoreElements():
        result.append(to_string(jenumeration.nextElement()))
    return result

def make_new(class_name, sig):
    '''Make a function that creates a new instance of the class
    
    A typical init function looks like this:
    new_fn = make_new("java/lang/Integer", '(I)V')
    def __init__(self, i):
        self.o = new_fn(i)
    
    It's important to store the object in self.o because it's expected to be
    there by make_method, etc.
    '''
    def constructor(self, *args):
        self.o = make_instance(class_name, sig, *args)
    return constructor

def make_instance(class_name, sig, *args):
    '''Create an instance of a class
    
    class_name - name of class
    sig - signature of constructor
    args - arguments to constructor
    '''
    args_sig = split_sig(sig[1:sig.find(')')])
    klass = get_env().find_class(class_name)
    jexception = get_env().exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    method_id = get_env().get_method_id(klass, '<init>', sig)
    jexception = get_env().exception_occurred()
    if method_id is None:
        if jexception is None:
            raise JavaError('Could not find constructor '
                            'with signature = "%s' % sig)
        else:
            raise JavaException(jexception)
    result = get_env().new_object(klass, method_id, 
                                  *get_nice_args(args, args_sig))
    jexception = get_env().exception_occurred() 
    if jexception is not None:
        raise JavaException(jexception)
    return result

def get_class_wrapper(obj):
    '''Return a wrapper for an object's class (e.g. for reflection)
    
    '''
    class_object = call(obj, 'getClass','()Ljava/lang/Class;')
    class Klass(object):
        def __init__(self):
            self.o = class_object
        getClasses = make_method('getClasses','()[Ljava/lang/Class;',
                                 'Returns an array containing Class objects representing all the public classes and interfaces that are members of the class represented by this Class object.')
        getConstructors = make_method('getConstructors','()[Ljava/lang/reflect/Constructor;')
        getFields = make_method('getFields','()[Ljava/lang/reflect/Field;')
        getField = make_method('getField','(Ljava/lang/String;)Ljava/lang/reflect/Field;')
        getMethod = make_method('getMethod','(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;')
        getMethods = make_method('getMethods','()[Ljava/lang/reflect/Method;')
    return Klass()


    
