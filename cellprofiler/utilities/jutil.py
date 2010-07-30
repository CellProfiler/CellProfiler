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

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision: 1 %"

import atexit
import gc
import numpy as np
import os
import threading
import subprocess
import sys

jvm_dir = None
if sys.platform.startswith('win'):
    #
    # Try harder by looking for JAVA_HOME and in the registry
    #
    from setup import find_javahome
    java_home = find_javahome()
    jvm_dir = None
    if java_home is not None:
        found_jvm = False
        for jre_home in (java_home, os.path.join(java_home, "jre")):
            for place_to_look in ('client','server'):
                jvm_dir = os.path.join(jre_home,'bin',place_to_look)
                if os.path.isfile(os.path.join(jvm_dir, "jvm.dll")):
                    os.environ['PATH'] = os.environ['PATH'] +';'+jvm_dir
                    found_jvm = True
                    break
            if found_jvm:
                break
        if not found_jvm:
            jvm_dir = None
            
elif sys.platform == 'darwin':
    #
    # Put the jvm library on the path, hoping it is always in the same place
    #
    jvm_dir = '/System/Library/Frameworks/JavaVM.framework/Libraries'
    os.environ['PATH'] = os.environ['PATH'] + ':' + jvm_dir
elif sys.platform.startswith('linux'):
    #
    # Run the findlibjvm program which uses java.library.path to
    # find the search path for the JVM.
    #
    import ctypes
    if hasattr(sys, 'frozen'):
        path = os.path.split(os.path.abspath(sys.argv[0]))[0]
        path = os.path.join(path, 'cellprofiler','utilities')
    else:
        path = os.path.split(__file__)[0]
    p = subprocess.Popen(["java","-cp", path, "findlibjvm"],
                         stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    jvm_dir = stdout.strip()
    ctypes.CDLL(os.path.join(jvm_dir, "libjvm.so"))

if jvm_dir is None:
    from cellprofiler.preferences \
         import get_report_jvm_error, set_report_jvm_error, get_headless
    from cellprofiler.preferences import set_has_reported_jvm_error
    
    if not get_headless():
        import wx

        app = wx.GetApp()
        if app is not None and get_report_jvm_error():
            dlg = wx.Dialog(wx.GetApp().GetTopWindow(),
                            title="Java not installed properly")
            sizer = wx.BoxSizer(wx.VERTICAL)
            dlg.SetSizer(sizer)
            text = wx.StaticText(dlg,-1, 
                                 "CellProfiler can't find Java on your computer.")
            text.Font = wx.Font(int(dlg.Font.GetPointSize()*5/4),
                                dlg.Font.GetFamily(),
                                dlg.Font.GetStyle(),
                                wx.FONTWEIGHT_BOLD)
            sizer.Add(text, 0, wx.ALIGN_LEFT | wx.ALL, 5)
            if java_home is None or sys.platform == "darwin":
                label = \
"""CellProfiler cannot find the location of Java on your computer.
CellProfiler can process images without Java, but uses the Bioformats Java
library to process images if Java is available.

You can download the Java runtime for your operating system at:
http://www.java.com/en/download/index.jsp
"""
            else:
                label = \
"""CellProfiler cannot find the location of Java on your computer.
CellProfiler can process images without Java, but uses the Bioformats Java
library to process images if Java is available.

Your computer may not be configured correctly. Your computer is configured
as if Java is installed in the directory, "%s", but the files that CellProfiler
needs do not seem to be installed there. Please see:

http://cellprofiler.org/wiki/index.php/Installing_Java

for more help.""" % java_home
                    
            sizer.Add(wx.StaticText(dlg, label = label), 
                      0, wx.EXPAND | wx.ALL, 5)
            report_ctrl = wx.CheckBox(
                dlg, label = "Don't show this message again.")
            report_ctrl.Value = False
            sizer.Add(report_ctrl, 0, wx.ALIGN_LEFT | wx.ALL, 5)
            buttons_sizer = wx.StdDialogButtonSizer()
            buttons_sizer.AddButton(wx.Button(dlg, wx.ID_OK))
            buttons_sizer.Realize()
            sizer.Add(buttons_sizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
            dlg.Fit()
            dlg.ShowModal()
            show_jvm_error_dlg = False
            if report_ctrl.Value:
                set_report_jvm_error(False)
            else:
                # Just for this run, turn off reporting
                set_has_reported_jvm_error()
            
    raise IOError("Can't find jvm directory")
    
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
                    super(JavaException, self).__init__(message)
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
    attach()

def print_all_stack_traces():
    thread_map = static_call("java/lang/Thread","getAllStackTraces",
                             "()Ljava/util/Map;")
    stack_traces = call(thread_map, "values","()Ljava/util/Collection;")
    sta = call(stack_traces, "toArray","()[Ljava/lang/Object;")
    stal = get_env().get_object_array_elements(sta)
    for stak in stal:
        stakes = get_env().get_object_array_elements(stak)
        for stake in stakes:
            print to_string(stake)
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
        while thread_local_env.attach_count > 1:
            detach()
        runtime = static_call("java/lang/Runtime","getRuntime",
                              "()Ljava/lang/Runtime;")
        call(runtime, "exit", "(I)V", 0)
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
    attach_count = getattr(__thread_local_env, "attach_count", 0)
    __thread_local_env.attach_count = attach_count + 1
    if attach_count == 0:
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
    
    assert __thread_local_env.attach_count > 0
    __thread_local_env.attach_count -= 1
    if __thread_local_env.attach_count > 0:
        return
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
    assert o is not None
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
        jexception = get_env().exception_occurred()
        raise JavaException(jexception)
    
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
        raise JavaException(jexception)
    return get_nice_result(result, ret_sig)

def make_method(name, sig, doc='No documentation'):
    '''Return a class method for the given Java class
    
    sig - a calling signature. 
          See http://java.sun.com/j2se/1.4.2/docs/guide/jni/spec/types.htm
          An example: "(ILjava/lang/String;)[I" takes an integer and
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
        return get_nice_result(env.get_static_object_field(klass, field_id),
                               sig)
        
def set_static_field(klass, name, sig, value):
    '''Set the value for a static field on a class
    
    klass - the class or string name of class
    name - the name of the field
    sig - the signature, typically, 'I' or 'Ljava/lang/String;'
    value - the value to set
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
        env.set_static_boolean_field(klass, field_id, value)
    elif sig == 'B':
        env.set_static_byte_field(klass, field_id, value)
    elif sig == 'C':
        assert len(str(value)) > 0
        env.set_static_char_field(klass, field_id, value)
    elif sig == 'S':
        env.set_static_short_field(klass, field_id, value)
    elif sig == 'I':
        env.set_static_int_field(klass, field_id, value)
    elif sig == 'J':
        env.set_static_long_field(klass, field_id, value)
    elif sig == 'F':
        env.get_static_float_field(klass, field_id, value)
    elif sig == 'D':
        env.set_static_double_field(klass, field_id, value)
    else:
        jobject = get_nice_arg(value, sig)
        env.set_static_object_field(klass, field_id, jobject)
        
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
    if sig == 'Ljava/lang/Integer;' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Integer', '(I)V', int(arg))
    if sig == 'Ljava/lang/Long' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Long', '(J)V', long(arg))
    if sig == 'Ljava/lang/Boolean;' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Boolean', '(Z)V', bool(arg))
    if isinstance(arg, np.ndarray):
        if sig == '[B':
            return env.make_byte_array(np.ascontiguousarray(arg.flatten(), np.uint8))
        elif sig == '[S':
            return env.make_short_array(np.ascontiguousarray(arg.flatten(), np.int16))
        elif sig == '[I':
            return env.make_int_array(np.ascontiguousarray(arg.flatten(), np.int32))
        elif sig == '[J':
            return env.make_long_array(np.ascontiguousarray(arg.flatten(), np.int64))
        elif sig == '[F':
            return env.make_float_array(np.ascontiguousarray(arg.flatten(), np.float32))
        elif sig == '[D':
            return env.make_double_array(np.ascontiguousarray(arg.flatten(), np.float64))
    elif (sig.startswith('L') and sig.endswith(';') and
          not isinstance(arg, (javabridge.JB_Object, javabridge.JB_Class))):
        #
        # Desperately try to make an instance of it with an integer constructor
        #
        if isinstance(arg, (int, long, bool)):
            return make_instance(sig[1:-1], '(I)V', int(arg))
        elif isinstance(arg, (str, unicode)):
            return make_instance(sig[1:-1], '(Ljava/lang/String;)V', arg)
    return arg

def get_nice_result(result, sig):
    '''Convert a result that may be a java object into a string'''
    env = get_env()
    if sig == 'Ljava/lang/String;':
        return env.get_string_utf(result)
    if sig == 'Ljava/lang/Integer;':
        return call(result, 'intValue', '()I')
    if sig == 'Ljava/lang/Long':
        return call(result, 'longValue', '()J')
    if sig == 'Ljava/lang/Boolean;':
        return call(result, 'booleanValue', '()Z')
    if sig == '[B':
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
        put = make_method('put',
                          '(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;',
                          'Associate a value with a key in the dictionary')
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
        getDeclaredMethod = make_method('getDeclaredMethod',
                                        '(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;')
    return Klass()

def attach_ext_env(env_address):
    '''Attach to an externally supplied Java environment
    
    env_address - the numeric address of the env memory pointer
    '''
    global __thread_local_env 
    env = javabridge.JB_Env()
    env.set_env(env_address)
    __thread_local_env.env = env
    
def make_run_dictionary(jobject_address):
    '''Support function for Py_RunString - jobject address -> globals / locals
    
    jobject_address - address of a Java Map of string to object
    '''
    jmap = get_env().make_jb_object(jobject_address)
    d = get_dictionary_wrapper(jmap)
    
    result = {}
    keys = jenumeration_to_string_list(d.keys())
    for key in keys:
        result[key] = d.get(key)
    return result
