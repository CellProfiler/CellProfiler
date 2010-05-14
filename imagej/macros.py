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

import bioformats
import cellprofiler.utilities.jutil as J

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
    return J.jenumeration_to_string_list(keys)
        
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
    macro_runner = J.make_instance("ij/macro/MacroRunner",
                                   "(Ljava/lang/String;)V",
                                   macro_text)
    J.call(macro_runner, "run", "()V")
    
def show_imagej():
    '''Show the ImageJ user interface'''
    ij_obj = J.static_call("ij/IJ", "getInstance", "()Lij/ImageJ;")
    if ij_obj is None:
        ij_obj = J.make_instance("ij/ImageJ", "()V")
    J.call(ij_obj, "setVisible", "(Z)V", True)
    J.call(ij_obj, "toFront", "()V")
    
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