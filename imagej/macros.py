'''macros.py - helper methods for finding and running macros'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2012 Broad Institute
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#
__version__="$Revision$"

import sys

import bioformats
import cellprofiler.utilities.jutil as J

def get_commands():
    '''Return a list of the available command strings'''
    script = """
    new java.util.concurrent.Callable() {
        call: function() {
           importClass(Packages.ij.Menus, Packages.ij.IJ);
           var hashtable=Menus.getCommands();
           if (hashtable==null) {
               IJ.run("pleaseignorethis");
               hashtable = Menus.getCommands();
           }
           return hashtable;
        }
    };
    """
    c = J.run_script(script, class_loader = get_user_loader())
    hashtable = J.execute_callable_in_main_thread(c)
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
    r = J.run_script("""
    new java.lang.Runnable() {
        run: function() { 
            importClass(Packages.ij.IJ, Packages.ij.ImageJ);
            var imagej = IJ.getInstance();
            if (imagej == null) {
                imagej = ImageJ();
            }
            imagej.setVisible(true);
            if (options==null) {
                IJ.run(command);
            } else {
                IJ.run(command, options);
            }
        }
    };""", 
         bindings_in = { 
             "command":command, 
             "options":options },
         class_loader=get_user_loader())
    J.execute_runnable_in_main_thread(r, True)
    
def execute_macro(macro_text):
    '''Execute a macro in ImageJ
    
    macro_text - the macro program to be run
    '''
    script = """
    new java.lang.Runnable() {
        run: function() {
            importClass(Packages.ij.IJ, Packages.ij.ImageJ);
            importClass(Packages.ij.macro.Interpreter);
            var imagej = IJ.getInstance();
            var interpreter = Interpreter();
            if (imagej == null) {
                imagej = ImageJ();
            }
            imagej.setVisible(true);
            interpreter.run(macro);
        }
    };"""
    runnable = J.run_script(script, 
                            bindings_in = { "macro":macro_text },
                            class_loader = get_user_loader())
    J.execute_runnable_in_main_thread(runnable, True)
    
def show_imagej():
    '''Show the ImageJ user interface'''
    r = J.run_script("""new java.lang.Runnable() {
        run: function() {
            var imageJ = Packages.ij.IJ.getInstance();
            if (imageJ == null) {
                imageJ = Packages.ij.ImageJ();
            }
            imageJ.setVisible(true);
            imageJ.toFront();
        }
    };""", class_loader=get_user_loader())
    J.execute_runnable_in_main_thread(r, True)
    
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
