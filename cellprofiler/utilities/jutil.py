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

The signatures are difficult, but you can cheat. The JSDK has a program,
'javap', that you can use to print out the signatures of everything
in a class.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

import codecs
import gc
import inspect
import logging
import numpy as np
import os
import threading
import traceback
import subprocess
import sys
import uuid

logger = logging.getLogger(__name__)

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
__main_thread_closures = []
__run_headless = False

RQCLS = "org/cellprofiler/runnablequeue/RunnableQueue"

class AtExit(object):
    '''AtExit runs a function as the main thread exits from the __main__ function
    
    We bind a reference to self to the main frame's locals. When
    the frame exits, "__del__" is called and the function runs. This is an
    alternative to "atexit" which only runs when all threads die.
    '''
    def __init__(self, fn):
        self.fn = fn
        stack = inspect.stack()
        for f, filename, lineno, module_name, code, index in stack:
            if (module_name == '<module>' and
                f.f_locals.get("__name__") == "__main__"):
                f.f_locals["X" + uuid.uuid4().hex] = self
                break
                
    def __del__(self):
        self.fn()
        
def start_vm(args, run_headless = False):
    '''Start the Java VM'''
    global __vm
    
    if __vm is not None:
        return
    start_event = threading.Event()
    pt = [] # holds the thread... eventually
    
    def start_thread(args=args, run_headless=run_headless):
        global __vm
        global __wake_event
        global __dead_event
        global __thread_local_env
        global __i_am_the_main_thread
        global __kill
        global __dead_objects
        global __main_thread_closures
        global __run_headless
        
        args = list(args)
        if run_headless:
            __run_headless = True
            args = args + [r"-Djava.awt.headless=true"]

        logger.debug("Creating JVM object")
        __thread_local_env.is_main_thread = True
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
        main_thread_closures = __main_thread_closures
        thread_local_env = __thread_local_env
        ptt = pt # needed to bind to pt inside exit_fn
        try:
            if hasattr(sys, 'frozen') and sys.platform != 'darwin':
                utils_path = os.path.join(
                    os.path.split(os.path.abspath(sys.argv[0]))[0],
                    "cellprofiler",
                    "utilities")
            else:
                utils_path = os.path.abspath(os.path.split(__file__)[0])
                if not os.path.isdir(utils_path):
                    # CPA's directory structure is this:
                    #
                    # CPAnalyst.app
                    #      Contents
                    #          MacOS
                    #              CPAnalyst
                    #          Resources
                    #
                    macos_path = os.path.abspath(os.path.split(sys.argv[0])[0])
                    contents_path = os.path.split(macos_path)[0]
                    root_path = os.path.join(contents_path, "Resources")
                    utils_path = os.path.join(root_path, "cellprofiler", "utilities")
            cp_args = [i for i, x in enumerate(args)
                       if x.startswith('-Djava.class.path=')]
            js_jar = os.path.join(utils_path, "js.jar")
            logger.debug("Loading javascript from %s" %js_jar)
            assert os.path.isfile(js_jar)
            if len(cp_args) > 0:
                arg_idx = cp_args[0]
                cp_arg = args[arg_idx]
                cp_arg = cp_arg + os.pathsep + js_jar
                args[arg_idx] = cp_arg
            else:
                cp_arg = "-Djava.class.path=" + js_jar
                arg_idx = -1
            if sys.platform == "darwin":
                logger.debug("Loading runnablequeue from %s" % utils_path)
                runnablequeue_jar = os.path.join(
                    utils_path, "runnablequeue-1.0.0.jar")
                assert os.path.exists(runnablequeue_jar)
                args[arg_idx] = cp_arg + os.pathsep + runnablequeue_jar
                
                logger.debug("Launching VM in non-python thread")
                vm.create_mac(args, RQCLS)
                logger.debug("Attaching to VM in monitor thread")
                env = vm.attach()
                __thread_local_env.env = env
            else:
                env = vm.create(args)
                __thread_local_env.env = env
        except:
            traceback.print_exc()
            logger.error("Failed to create Java VM")
            __vm = None
            return
        finally:
            logger.debug("Signalling caller")
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
            while(len(main_thread_closures)):
                main_thread_closures.pop()()
            if kill[0]:
                break
        def null_defer_fn(jbo):
            '''Install a "do nothing" defer function in our env'''
            pass
        if sys.platform == "darwin":
            #
            # Torpedo the main thread RunnableQueue
            #
            rqcls = env.find_class(RQCLS)
            stop_id = env.get_static_method_id(rqcls, "stop", "()V")
            env.call_static_method(rqcls, stop_id)
            env.set_defer_fn(null_defer_fn)
            vm.detach()
        else:
            env.set_defer_fn(null_defer_fn)
            vm.destroy()
        __vm = None
        dead_event.set()
        
    t = threading.Thread(target=start_thread)
    pt.append(t)
    t.start()
    start_event.wait()
    if __vm is None:
        raise RuntimeError("Failed to start Java VM")
    attach()
    AtExit(kill_vm)
    
def unwrap_javascript(o):
    '''Unwrap an object such as NativeJavaObject
    
    o - an object, possibly implementing org.mozilla.javascript.Wrapper
    
    return nice version
    '''
    if is_instance_of(o, "org/mozilla/javascript/Wrapper"):
        o = call(o, "unwrap", "()Ljava/lang/Object;")
    if not isinstance(o, javabridge.JB_Object):
        return o
    for class_name, method, signature in (
        ("java/lang/Boolean", "booleanValue", "()Z"),
        ("java/lang/Byte", "byteValue", "()B"),
        ("java/lang/Integer",  "intValue", "()I"),
        ("java/lang/Long", "longValue", "()L"),
        ("java/lang/Float", "floatValue", "()F"),
        ("java/lang/Double", "doubleValue", "()D")):
        if is_instance_of(o, class_name):
            return call(o, method, signature)
    return o
    
def run_script(script, bindings_in = {}, bindings_out = {}, 
               class_loader = None):
    '''Run a scripting language script
    
    script - script to run
    
    bindings_in - key / value pair of global name to Java object. The
                  engine scope is populated with variables given by the keys
                  and the variables are assigned the keys' values.
                  
    bindings_out - a dictionary of keys to be populated with values after
                   evaluation. For instance, bindings_out = dict(foo=None) to
                   get the value for the "foo" variable on output.
                   
    class_loader - class loader for scripting context
    
    Returns the object that is the result of the evaluation.
    '''
    context = static_call("org/mozilla/javascript/Context", "enter",
                          "()Lorg/mozilla/javascript/Context;")
    try :
        if class_loader is not None:
            call(context, "setApplicationClassLoader",
                 "(Ljava/lang/ClassLoader;)V",
                 class_loader)
        scope = make_instance("org/mozilla/javascript/ImporterTopLevel",
                              "(Lorg/mozilla/javascript/Context;)V",
                              context)
        for k, v in bindings_in.iteritems():
            call(scope, "put", 
                 "(Ljava/lang/String;Lorg/mozilla/javascript/Scriptable;"
                 "Ljava/lang/Object;)V", k, scope, v)
        result = call(context, "evaluateString",
             "(Lorg/mozilla/javascript/Scriptable;"
             "Ljava/lang/String;"
             "Ljava/lang/String;"
             "I"
             "Ljava/lang/Object;)"
             "Ljava/lang/Object;", 
             scope, script, "<java-python-bridge>", 0, None)
        result = unwrap_javascript(result)
        for k in list(bindings_out):
            bindings_out[k] = unwrap_javascript(call(
                scope, "get",
                "(Ljava/lang/String;"
                "Lorg/mozilla/javascript/Scriptable;)"
                "Ljava/lang/Object;", k, scope))
    finally:
        static_call("org/mozilla/javascript/Context", "exit", "()V")
    return result

def execute_runnable_in_main_thread(runnable, synchronous=False):
    '''Execute a runnable on the main thread
    
    runnable - a Java object implementing java.lang.Runnable
    
    synchronous - True if we should wait for the runnable to finish
    
    Hint: to make a runnable using scripting,
    
    return new java.lang.Runnable() {
      run: function() {
        <do something here>
      }
    };
    '''
    if sys.platform == "darwin":
        # Assumes that RunnableQueue has been deployed on the main thread
        if synchronous:
            future = make_instance(
                "java/util/concurrent/FutureTask",
                "(Ljava/lang/Runnable;Ljava/lang/Object;)V",
                runnable, None)
            execute_future_in_main_thread(future)
        else:
            static_call(RQCLS, "enqueue", "(Ljava/lang/Runnable;)V",
                        runnable)
    else:
        run_in_main_thread(
            lambda: call(runnable, "run", "()V"), synchronous)
            
def execute_future_in_main_thread(jfuture):
    '''Execute a class implementing Future in the main thread
    
    Synchronize with the return, running the event loop.
    '''
    # Portions of this were adapted from IPython/lib/inputhookwx.py
    #-----------------------------------------------------------------------------
    #  Copyright (C) 2008-2009  The IPython Development Team
    #
    #  Distributed under the terms of the BSD License.  The full license is in
    #  the file COPYING, distributed as par t of this software.
    #-----------------------------------------------------------------------------
    
    if sys.platform != "darwin":
        run_in_main_thread(lambda: call(jfuture, "run", "()V"))
        return call(jfuture, "get", "()Ljava/lang/Object;")
        
    import wx
    import time
    app = wx.GetApp()
    logger.debug("Enqueueing future on runnable queue")
    static_call(RQCLS, "enqueue", "(Ljava/lang/Runnable;)V", jfuture)
    if (app is None) or (not wx.Thread_IsMain()):
        logger.debug("Synchronizing without event loop")
        #
        # There could be a deadlock between the GIL being taken
        # by the execution of Future.get() and AWT needing WX to
        # run the event loop. Therefore, we poll before getting.
        #
        while not call(jfuture, "isDone", "()Z"):
            logger.debug("Future is not done")
            time.sleep(.1)
        return call(jfuture, "get", "()Ljava/lang/Object;")
    elif app.IsMainLoopRunning():
        evtloop = wx.EventLoop()
        logger.debug("Polling for future done within main loop")
        while not call(jfuture, "isDone", "()Z"):
            logger.debug("Future is not done")
            if evtloop.Pending():
                while evtloop.Pending():
                    logger.debug("Processing pending event")
                    evtloop.Dispatch()
            else:
                logger.debug("No pending wx event, run Dispatch anyway")
                evtloop.Dispatch()
            logger.debug("Sleeping")
            time.sleep(.1)
        
        logger.debug("Fetching future value")
        return call(jfuture, "get", "()Ljava/lang/Object;")
    else:
        logger.debug("Polling for future while running main loop")
        class EventLoopTimer(wx.Timer):
        
            def __init__(self, func):
                self.func = func
                wx.Timer.__init__(self)
        
            def Notify(self):
                self.func()
        
        class EventLoopRunner(object):
        
            def __init__(self, fn):
                self.fn = fn
                
            def Run(self, time):
                self.evtloop = wx.EventLoop()
                self.timer = EventLoopTimer(self.check_fn)
                self.timer.Start(time)
                self.evtloop.Run()
        
            def check_fn(self):
                if self.fn():
                    self.timer.Stop()
                    self.evtloop.Exit()
        event_loop_runner = EventLoopRunner(
            lambda: call(jfuture, "isDone", "()Z"))
        event_loop_runner.Run(time=10)
        return call(jfuture, "get", "()Ljava/lang/Object;")
        
        
def execute_callable_in_main_thread(jcallable):
    '''Execute a callable on the main thread, returning its value
    
    callable - a Java object implementing java.util.concurrent.Callable
    
    Returns the result of evaluating the callable's "call" method in the
    main thread.
    
    Hint: to make a callable using scripting,
    
    var my_import_scope = new JavaImporter(java.util.concurrent.Callable);
    with (my_import_scope) {
        return new Callable() {
            call: function {
                <do something that produces result>
                return result;
            }
        };
    '''
    if sys.platform == "darwin":
        # Assumes that RunnableQueue has been deployed on the main thread
        future = make_instance(
            "java/util/concurrent/FutureTask",
            "(Ljava/util/concurrent/Callable;)V",
            jcallable)
        return execute_future_in_main_thread(future)
    else:
        return run_in_main_thread(
            lambda: call(jcallable, "call", "()Ljava/lang/Object;"), 
            True)
    

def run_in_main_thread(closure, synchronous):
    '''Run a closure in the main Java thread
    
    closure - a callable object (eg lambda : print "hello, world")
    
    synchronous - True to wait for completion of execution
    '''
    global __main_thread_closures
    global __wake_event
    global __thread_local_env
    if (hasattr(__thread_local_env, "is_main_thread") and
        __thread_local_env.is_main_thread):
        return closure()
    
    if synchronous:
        done_event = threading.Event()
        done_event.clear()
        result = [None]
        exception = [None]
        def synchronous_closure():
            try:
                result[0] = closure()
            except Exception, e:
                logger.exception("Caught exception when executing closure")
                exception[0] = e
            done_event.set()
        __main_thread_closures.append(synchronous_closure)
        __wake_event.set()
        done_event.wait()
        if exception[0] is not None:
            raise exception[0]
        return result[0]
    else:
        __main_thread_closures.append(closure)
        __wake_event.set()
    
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
            
CLOSE_ALL_WINDOWS = """
        new java.lang.Runnable() { 
            run: function() {
                var all_frames = java.awt.Frame.getFrames();
                if (all_frames) {
                    for (idx in all_frames) {
                        java.lang.System.err.println("Disposing");
                        all_frames[idx].dispose();
                    }
                }
            }
        };"""

__awt_is_active = False
def activate_awt():
    '''Activate Java AWT by executing some trivial code'''
    global __awt_is_active
    if not __awt_is_active:
        execute_runnable_in_main_thread(run_script(
            """new java.lang.Runnable() {
                   run: function() {
                       java.awt.Color.BLACK.toString();
                   }
               };"""), True)
        __awt_is_active = True
        
def deactivate_awt():
    global __awt_is_active
    if __awt_is_active:
        r = run_script(CLOSE_ALL_WINDOWS)
        execute_runnable_in_main_thread(r, True)
        __awt_is_active = False
#
# We make kill_vm as a closure here to bind local copies of the global objects
#
def make_kill_vm():
    '''Kill the currently-running Java environment'''
    global __wake_event
    global __dead_event
    global __kill
    global __thread_local_env
    global __run_headless
    
    wake_event = __wake_event
    dead_event = __dead_event
    kill = __kill
    thread_local_env = __thread_local_env
    if not hasattr(thread_local_env, "attach_count"):
        thread_local_env.attach_count = 0
    def kill_vm():
        global __vm
        if __vm is None:
            return
        deactivate_awt()
        while getattr(thread_local_env, "attach_count", 0) > 0:
            detach()
        kill[0] = True
        wake_event.set()
        dead_event.wait()
    return kill_vm

'''Kill the currently-running Java environment

fn_poll_ui - if present, use this function to run the UI's event loop
             while waiting for the JVM to close AWT.
'''
kill_vm = make_kill_vm()
    
def attach():
    '''Attach to the VM, receiving the thread's environment'''
    global __thread_local_env
    global __vm
    assert isinstance(__vm, javabridge.JB_VM)
    attach_count = getattr(__thread_local_env, "attach_count", 0)
    __thread_local_env.attach_count = attach_count + 1
    if attach_count == 0:
        __thread_local_env.env = __vm.attach_as_daemon()
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

def is_instance_of(o, class_name):
    '''Return True if object is instance of class
    
    o - object in question
    class_name - class in question. Use slash form: java/lang/Object
    
    Note: returns False if o is not a java object (e.g. a string)
    '''
    if not isinstance(o, javabridge.JB_Object):
        return False
    env = get_env()
    klass = env.find_class(class_name)
    jexception = get_env().exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    result = env.is_instance_of(o, klass)
    jexception = get_env().exception_occurred()
    if jexception is not None:
        raise JavaException(jexception)
    return result
    
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
    return get_nice_result(result, ret_sig)

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
    if field_id is None:
        raise JavaError('Could not find field name = %s '
                        'with signature = %s' %(name, sig))
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
    is_java = (isinstance(arg, javabridge.JB_Object) or
               isinstance(arg, javabridge.JB_Class))
    if sig[0] == 'L' and not is_java:
        #
        # Check for the standard packing of java objects into class instances
        #
        if hasattr(arg, "o"):
            return arg.o
    #
    # If asking for an object, try converting basic types into Java-wraps
    # of Java basic types
    #
    if sig == 'Ljava/lang/Object;' and isinstance(arg, bool):
        return make_instance('java/lang/Boolean', '(Z)V', arg)
    if sig == 'Ljava/lang/Object;' and isinstance(arg, int):
        return make_instance('java/lang/Integer', '(I)V', arg)
    if sig == 'Ljava/lang/Object;' and isinstance(arg, long):
        return make_instance('java/lang/Long', '(J)V', arg)
    if sig == 'Ljava/lang/Object;' and isinstance(arg, float):
        return make_instance('java/lang/Double', '(D)V', arg)
    if (sig in ('Ljava/lang/String;','Ljava/lang/Object;') and not
         isinstance(arg, javabridge.JB_Object)):
        if isinstance(arg, unicode):
            arg, _ = codecs.utf_8_encode(arg)
        elif arg is None:
            return None
        else:
            arg = str(arg)
        return env.new_string_utf(arg)
    if sig == 'Ljava/lang/Integer;' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Integer', '(I)V', int(arg))
    if sig == 'Ljava/lang/Long' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Long', '(J)V', long(arg))
    if sig == 'Ljava/lang/Boolean;' and type(arg) in [int, long, bool]:
        return make_instance('java/lang/Boolean', '(Z)V', bool(arg))
    
    if isinstance(arg, np.ndarray):
        if sig == '[Z':
            return env.make_boolean_array(np.ascontiguousarray(arg.flatten(), np.bool8))
        elif sig == '[B':
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
    elif sig.startswith('L') and sig.endswith(';') and not is_java:
        #
        # Desperately try to make an instance of it with an integer constructor
        #
        if isinstance(arg, (int, long, bool)):
            return make_instance(sig[1:-1], '(I)V', int(arg))
        elif isinstance(arg, (str, unicode)):
            return make_instance(sig[1:-1], '(Ljava/lang/String;)V', arg)
    if sig.startswith('[L') and (not is_java) and hasattr(arg, '__iter__'):
        objs = [get_nice_arg(subarg, sig[1:]) for subarg in arg]
        k = env.find_class(sig[2:-1])
        a = env.make_object_array(len(objs), k)
        for i, obj in enumerate(objs):
            env.set_object_array_element(a, i, obj)
        return a
    return arg

def get_nice_result(result, sig):
    '''Convert a result that may be a java object into a string'''
    if result is None:
        return None
    env = get_env()
    if (sig == 'Ljava/lang/String;' or
        (sig == 'Ljava/lang/Object;' and 
         is_instance_of(result, "java/lang/String"))):
        return codecs.utf_8_decode(env.get_string_utf(result), 'replace')[0]
    if sig == 'Ljava/lang/Integer;':
        return call(result, 'intValue', '()I')
    if sig == 'Ljava/lang/Long':
        return call(result, 'longValue', '()J')
    if sig == 'Ljava/lang/Boolean;':
        return call(result, 'booleanValue', '()Z')
    if sig == '[B':
        # Convert a byte array into a numpy array
        return env.get_byte_array_elements(result)
    if isinstance(result, javabridge.JB_Object):
        #
        # Do longhand to prevent recursion
        #
        rklass = env.get_object_class(result)
        m = env.get_method_id(rklass, 'getClass', '()Ljava/lang/Class;')
        rclass = env.call_method(result, m)
        rkklass = env.get_object_class(rclass)
        m = env.get_method_id(rkklass, 'isPrimitive', '()Z')
        is_primitive = env.call_method(rclass, m)
        if is_primitive:
            rc = get_class_wrapper(rclass, True)
            classname = rc.getCanonicalName()
            if classname == 'boolean':
                return to_string(result) == 'true'
            elif classname in ('int', 'byte', 'short', 'long'):
                return int(to_string(result))
            elif classname in ('float', 'double'):
                return float(to_string(result))
            elif classname == 'char':
                return to_string(result)
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

def iterate_java(iterator):
    '''Make a Python iterator for a Java iterator
    
    usage:
    for x in iterate_java(foo):
        do_something_with(x)
    '''
    while(call(iterator, 'hasNext', '()Z')):
        yield call(iterator, 'next', '()Ljava/lang/Object;')
        
def iterate_collection(c):
    '''Make a Python iterator over the elements of a Java collection'''
    return iterate_java(call(c, "iterator", "()Ljava/util/Iterator;"))
        
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
    
    class_name - name of class in foo/bar/Baz form (not foo.bar.Baz)
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

def class_for_name(classname, ldr="system"):
    '''Return a java.lang.Class for the given name
    
    classname: the class name in dotted form, e.g. "java.lang.Class"
    '''
    if ldr == "system":
        ldr = static_call('java/lang/ClassLoader', 'getSystemClassLoader',
                          '()Ljava/lang/ClassLoader;')
    return static_call('java/lang/Class', 'forName', 
                       '(Ljava/lang/String;ZLjava/lang/ClassLoader;)'
                       'Ljava/lang/Class;', 
                       classname, True, ldr)

def get_class_wrapper(obj, is_class = False):
    '''Return a wrapper for an object's class (e.g. for reflection)
    
    '''
    if is_class:
        class_object = obj
    elif isinstance(obj, (str, unicode)):
        class_object = class_for_name(obj)
    else:
        class_object = call(obj, 'getClass','()Ljava/lang/Class;')
    class Klass(object):
        def __init__(self):
            self.o = class_object
        getAnnotation = make_method('getAnnotation',
                                    '(Ljava/lang/Class;)Ljava/lang/annotation/Annotation;',
                                    "Returns this element's annotation if present")
        getAnnotations = make_method('getAnnotations',
                                     '()[Ljava/lang/annotation/Annotation;')
        getCanonicalName = make_method('getCanonicalName',
                                       '()Ljava/lang/String;',
                                       'Returns the canonical name of the class')
        getClasses = make_method('getClasses','()[Ljava/lang/Class;',
                                 'Returns an array containing Class objects representing all the public classes and interfaces that are members of the class represented by this Class object.')
        getConstructor = make_method(
            'getConstructor', 
            '([Ljava/lang/Class;)Ljava/lang/reflect/Constructor;',
            'Return a constructor with the given signature')
        getConstructors = make_method('getConstructors','()[Ljava/lang/reflect/Constructor;')
        getFields = make_method('getFields','()[Ljava/lang/reflect/Field;')
        getField = make_method('getField','(Ljava/lang/String;)Ljava/lang/reflect/Field;')
        getMethod = make_method('getMethod','(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;')
        getMethods = make_method('getMethods','()[Ljava/lang/reflect/Method;')
        cast = make_method('cast', '(Ljava/lang/Object;)Ljava/lang/Object;',
                           'Throw an exception if object is not castable to this class')
        isPrimitive = make_method('isPrimitive', '()Z',
                                  'Return True if the class is a primitive such as boolean or int')
        newInstance = make_method('newInstance', '()Ljava/lang/Object;',
                                  'Make a new instance of the object with the default constructor')
    return Klass()

MOD_ABSTRACT  = 'ABSTRACT'
MOD_FINAL = 'FINAL'
MOD_INTERFACE = 'INTERFACE'
MOD_NATIVE = 'NATIVE'
MOD_PRIVATE = 'PRIVATE'
MOD_PROTECTED = 'PROTECTED'
MOD_PUBLIC = 'PUBLIC'
MOD_STATIC = 'STATIC'
MOD_STRICT = 'STRICT'
MOD_SYCHRONIZED = 'SYNCHRONIZED'
MOD_TRANSIENT = 'TRANSIENT'
MOD_VOLATILE = 'VOLATILE'
MOD_ALL = [MOD_ABSTRACT, MOD_FINAL, MOD_INTERFACE, MOD_NATIVE,
           MOD_PRIVATE, MOD_PROTECTED, MOD_PUBLIC, MOD_STATIC,
           MOD_STRICT, MOD_SYCHRONIZED, MOD_TRANSIENT, MOD_VOLATILE]

def get_modifier_flags(modifier_flags):
    '''Parse out the modifiers from the modifier flags from getModifiers'''
    result = []
    for mod in MOD_ALL:
        if modifier_flags & get_static_field('java/lang/reflect/Modifier',
                                             mod, 'I'):
            result.append(mod)
    return result

def get_field_wrapper(field):
    '''Return a wrapper for the java.lang.reflect.Field class'''
    class Field(object):
        def __init__(self):
            self.o = field
            
        get = make_method('get', '(Ljava/lang/Object;)Ljava/lang/Object;',
                          'Returns the value of the field represented by this '
                          'Field, on the specified object.')
        def getAnnotation(self, annotation_class):
            """Returns this element's annotation for the specified type
            
            annotation_class - find annotations of this class
            
            returns the annotation or None if not annotated"""
            
            if isinstance(annotation_class, (str, unicode)):
                annotation_class = class_for_name(annotation_class)
            return call(self.o, 'getAnnotation', 
                        '(Ljava/lang/Class;)Ljava/lang/annotation/Annotation;',
                        annotation_class)
        
        getBoolean = make_method('getBoolean', '(Ljava/lang/Object;)Z',
                                 'Read a boolean field from an object')
        getByte = make_method('getByte', '(Ljava/lang/Object;)B',
                              'Read a byte field from an object')
        getChar = make_method('getChar', '(Ljava/lang/Object;)C')
        getDouble = make_method('getDouble', '(Ljava/lang/Object;)D')
        getFloat = make_method('getFloat', '(Ljava/lang/Object;)F')
        getInt = make_method('getInt', '(Ljava/lang/Object;)I')
        getShort = make_method('getShort', '(Ljava/lang/Object;)S')
        getLong = make_method('getLong', '(Ljava/lang/Object;)J')
        getDeclaredAnnotations = make_method(
            'getDeclaredAnnotations',
            '()[Ljava/lang/annotation/Annotation;')
        getGenericType = make_method('getGenericType', 
                                     '()Ljava/lang/reflect/Type;')
        def getModifiers(self):
            return get_modifier_flags(call(self.o, 'getModifiers','()I'))
        getName = make_method('getName', '()Ljava/lang/String;')
        
        getType = make_method('getType', '()Ljava/lang/Class;')
        set = make_method('set', '(Ljava/lang/Object;Ljava/lang/Object;)V')
        setBoolean = make_method('setBoolean', '(Ljava/lang/Object;Z)V',
                                 'Set a boolean field in an object')
        setByte = make_method('setByte', '(Ljava/lang/Object;B)V',
                              'Set a byte field in an object')
        setChar = make_method('setChar', '(Ljava/lang/Object;C)V')
        setDouble = make_method('setDouble', '(Ljava/lang/Object;D)V')
        setFloat = make_method('setFloat', '(Ljava/lang/Object;F)V')
        setInt = make_method('setInt', '(Ljava/lang/Object;I)V')
        setShort = make_method('setShort', '(Ljava/lang/Object;S)V')
        setLong = make_method('setLong', '(Ljava/lang/Object;J)V')
    return Field()

def get_constructor_wrapper(obj):
    '''Get a wrapper for calling methods on the constructor object'''
    class Constructor(object):
        def __init__(self):
            self.o = obj
            
        getParameterTypes = make_method('getParameterTypes',
                                        '()[Ljava/lang/Class;',
                                        'Get the types of the constructor parameters')
        getName = make_method('getName', '()Ljava/lang/String;')
        newInstance = make_method('newInstance',
                                  '([Ljava/lang/Object;)Ljava/lang/Object;')
        getAnnotation = make_method('getAnnotation', 
                                    '()Ljava/lang/annotation/Annotation;')
        getModifiers = make_method('getModifiers', '()I')
    return Constructor()
        
def get_method_wrapper(obj):
    '''Get a wrapper for calling methods on the method object'''
    class Method(object):
        def __init__(self):
            self.o = obj
            
        getParameterTypes = make_method('getParameterTypes',
                                        '()[Ljava/lang/Class;',
                                        'Get the types of the constructor parameters')
        getName = make_method('getName', '()Ljava/lang/String;')
        invoke = make_method('invoke',
                             '(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;')
        getAnnotation = make_method('getAnnotation', 
                                    '()Ljava/lang/annotation/Annotation;')
        getModifiers = make_method('getModifiers', '()I')
    return Method()
        
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

if __name__=="__main__":
    import wx
    app = wx.PySimpleApp(False)
    frame = wx.Frame(None)
    frame.Sizer = wx.BoxSizer(wx.HORIZONTAL)
    start_button = wx.Button(frame, label="Start VM")
    frame.Sizer.Add(start_button, 1, wx.ALIGN_CENTER_HORIZONTAL)
    def fn_start(event):
        start_vm([])
        start_button.Enable(False)
    start_button.Bind(wx.EVT_BUTTON, fn_start)
    
    launch_button = wx.Button(frame, label="Launch AWT frame")
    frame.Sizer.Add(launch_button, 1, wx.ALIGN_CENTER_HORIZONTAL)
    
    def fn_launch_frame(event):
        execute_runnable_in_main_thread(run_script("""
        new java.lang.Runnable() {
            run: function() {
                with(JavaImporter(java.awt.Frame)) Frame().setVisible(true);
            }
        };"""))
    launch_button.Bind(wx.EVT_BUTTON, fn_launch_frame)
    
    stop_button = wx.Button(frame, label="Stop VM")
    frame.Sizer.Add(stop_button, 1, wx.ALIGN_CENTER_HORIZONTAL)
    def fn_stop(event):
        def do_kill_vm():
            attach()
            kill_vm()
            wx.CallAfter(stop_button.Enable, False)
        thread = threading.Thread(target=do_kill_vm)
        thread.start()
    stop_button.Bind(wx.EVT_BUTTON, fn_stop)
    frame.Layout()
    frame.Show()
    app.MainLoop()
        
    