"""macros.py - helper methods for finding and running macros"""

__version__ = "$Revision$"

import javabridge as J

from imagej.imageplus import get_imageplus_wrapper


def get_commands():
    """Return a list of the available command strings"""
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


def execute_command(command, options=None):
    """Execute the named command within ImageJ
    :param options:
    :param command:
    """
    if options is None:
        J.static_call("ij/IJ", "run", "(Ljava/lang/String;)V", command)
    else:
        J.static_call("ij/IJ", "run",
                      "(Ljava/lang/String;Ljava/lang/String;)V",
                      command, options)


def set_current_image(image_plus):
    """Put the given image on the top of the batch mode image stack

    image_plus - a wrapped imagePlus
    :param image_plus:
    """
    #
    # Make sure we are in batch mode prior to adding the image.
    # If not, the image just goes into the garbage.
    #
    J.static_call("ij/macro/Interpreter",
                  "setBatchMode",
                  "(Z)V", True)
    #
    # Remove the image, if it exists, from its current position
    # on the stack
    #
    J.static_call("ij/macro/Interpreter",
                  "removeBatchModeImage",
                  "(Lij/ImagePlus;)V", image_plus.o)
    J.static_call("ij/macro/Interpreter",
                  "addBatchModeImage",
                  "(Lij/ImagePlus;)V", image_plus.o)


def get_current_image():
    """Get the image from the top of the batch mode image stack

    returns None or a wrapped imagePlus
    """
    image_plus = J.static_call("ij/macro/Interpreter",
                               "getLastBatchModeImage",
                               "()Lij/ImagePlus;")
    if image_plus is not None:
        return get_imageplus_wrapper(image_plus)


def execute_macro(macro_text):
    """Execute a macro in ImageJ

    macro_text - the macro program to be run
    :param macro_text:
    """
    interp = J.make_instance("ij/macro/Interpreter", "()V")
    J.call(interp, "run", "(Ljava/lang/String;)V", macro_text)


def run_batch_macro(macro_text, imp):
    """Run a macro in batch mode

    macro_text - the macro program to be run
    imp - an image plus to become the active image

    returns the image plus that was the active image at the end of the run
    :param imp:
    :param macro_text:
    """
    script = """
    new java.util.concurrent.Callable() {
        call: function() {
             return interp.runBatchMacro(macro_text, imp);
        }
    };
    """
    interp = J.JClassWrapper("ij.macro.Interpreter")()
    return interp.runBatchMacro(macro_text, imp).o


def get_user_loader():
    """The class loader used to load user plugins"""
    return J.static_call("ij/IJ", "getClassLoader", "()Ljava/lang/ClassLoader;")


def get_plugin(classname):
    """Return an instance of the named plugin
    :param classname:
    """
    if classname.startswith("ij."):
        cls = J.class_for_name(classname)
    else:
        cls = J.class_for_name(classname, get_user_loader())
    cls = J.get_class_wrapper(cls, True)
    constructor = J.get_constructor_wrapper(cls.getConstructor(None))
    return constructor.newInstance(None)
