'''macros.py - helper methods for finding and running macros'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#
__version__="$Revision$"

import sys

import bioformats
import cellprofiler.utilities.jutil as J

from cellprofiler.preferences import get_headless

def get_commands():
    '''Return a list of the available command strings'''
    hashtable = J.static_call('ij/Menus', 'getCommands',
                              '()Ljava/util/Hashtable;')
    if hashtable is None:
        #
        # This is a little bogus, but works - trick IJ into initializing
        #
        execute_command("pleaseignorethis")
        hashtable = J.static_call('ij/Menus', 'getCommands',
                                  '()Ljava/util/Hashtable;')
        if hashtable is None:
            return []
    keys = J.call(hashtable, "keys", "()Ljava/util/Enumeration;")
    keys = J.jenumeration_to_string_list(keys)
    values = J.call(hashtable, "values", "()Ljava/util/Collection;")
    values = [J.to_string(x) for x in J.iterate_java(
        J.call(values, 'iterator', "()Ljava/util/Iterator;"))]
    class CommandList(list):
        def __init__(self):
            super(CommandList, self).__init__(keys)
            self.values = values
    return CommandList()
        
def execute_command(command, options = None):
    '''Execute the named command within ImageJ'''
    if options is None:
        J.static_call("ij/IJ", "run", "(Ljava/lang/String;)V", command)
    else:
        J.static_call("ij/IJ", "run", 
                      "(Ljava/lang/String;Ljava/lang/String;)V",
                      command, options)
    
def execute_macro(macro_text):
    '''Execute a macro in ImageJ
    
    macro_text - the macro program to be run
    '''
    interp = J.make_instance("ij/macro/Interpreter","()V");
    J.execute_runnable_in_main_thread(J.run_script(
        """new java.lang.Runnable() {
        run: function() {
            interp.run(macro_text);
        }}""", dict(interp=interp, macro_text=macro_text)), synchronous=True)
    
def run_batch_macro(macro_text, imp):
    '''Run a macro in batch mode
    
    macro_text - the macro program to be run
    imp - an image plus to become the active image
    
    returns the image plus that was the active image at the end of the run
    '''
    script = """
    new java.util.concurrent.Callable() {
        call: function() {
             return interp.runBatchMacro(macro_text, imp);
        }
    };
    """
    interp = J.make_instance("ij/macro/Interpreter","()V");
    future = J.make_future_task(J.run_script(
        script, dict(interp=interp, macro_text=macro_text, imp=imp)))
    return J.execute_future_in_main_thread(future)
    
def get_user_loader():
    '''The class loader used to load user plugins'''
    return J.static_call("ij/IJ", "getClassLoader", "()Ljava/lang/ClassLoader;")

def get_plugin(classname):
    '''Return an instance of the named plugin'''
    if classname.startswith("ij."):
        cls = J.class_for_name(classname)
    else:
        cls = J.class_for_name(classname, get_user_loader())
    cls = J.get_class_wrapper(cls, True)
    constructor = J.get_constructor_wrapper(cls.getConstructor(None))
    return constructor.newInstance(None)

if __name__=="__main__":
    import sys
    J.attach()
    try:
        commands = get_commands()
        print "Commands: "
        for command in commands:
            print "\t" + command
        if len(sys.argv) == 2:
            execute_command(sys.argv[1])
        elif len(sys.argv) > 2:
            execute_command(sys.argv[1], sys.argv[2])
        
    finally:
        J.detach()
        J.kill_vm()
