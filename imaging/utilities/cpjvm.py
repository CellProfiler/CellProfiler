"""cpjvm.py - CellProfiler-specific JVM utilities

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import javabridge
import bioformats
import logging
import os
import sys
import tempfile

import cellprofiler.preferences as cpprefs
from external_dependencies import get_cellprofiler_jars

logger = logging.getLogger(__name__)

def get_path_to_jars():
    '''Return the path to CellProfiler's jars directory'''
    if hasattr(sys, 'frozen') and sys.platform != 'darwin':
        # Starting path is base/CellProfiler - split off CellProfiler
        start_path = sys.argv[0]
        split_count = 1
    else:
        # Starting path is base/cellprofiler/utilities/cpjvm.py
        # Split 3 times.
        start_path = __file__
        split_count = 3
    root_path = os.path.abspath(start_path)
    for _ in range(split_count):
        root_path = os.path.split(root_path)[0]
        
    imagej_path = os.path.join(root_path, 'imagej','jars')
    return imagej_path

def get_patcher_args(class_path):
    '''Return the JVM args needed to patch ij1 classes
    
    ImageJ says:
    
    Please make sure that you initialize the LegacyService before using
    any ImageJ 1.x class. You can do that by adding this static initializer:
    
        static {
            LegacyInjector.preinit();
        }
    
    To debug this issue, start the JVM with the option:
    
    -javaagent:<path-to>/ij1-patcher-0.2.1.jar
    
    To enforce pre-initialization, start the JVM with the option:
    
    -javaagent:<path-to>/ij1-patcher-0.2.1.jar=init
    
    class_path - absolute path to all jars needed by ImageJ
    
    returns a sequence of arguments to add to the JVM args
    '''
    
    patchers = filter((lambda x:x.find("ij1-patcher") >=0), class_path)
    if len(patchers) > 0:
        patcher = patchers[0]
        return ["-javaagent:%s=init" % patcher]
    logger.warn("Did not find ij1-patcher.jar")
    return []

def get_jars():
    '''Get the final list of JAR files passed to javabridge'''
    imagej_path = get_path_to_jars()
    if hasattr(sys, 'frozen'):
        jar_files = [
            jar_filename
            for jar_filename in os.listdir(imagej_path)
            if jar_filename.lower().endswith(".jar")]
        sort_dict = { "cellprofiler-java.jar": -1}
        jdcp = os.path.join(
            imagej_path, "cellprofiler-java-dependencies-classpath.txt")
        if os.path.isfile(jdcp):
            with open(jdcp, "r") as fd:
                jars = fd.readline().split(os.pathsep)
                sort_dict.update(dict([
                    (os.path.split(j)[-1], i) for i, j in enumerate(jars)]))
        def sort_fn(a, b):
            aa,bb = [(sort_dict.get(x, sys.maxint), x)
                     for x in a, b]
            return cmp(aa, bb)
        jar_files = sorted(jar_files, cmp = sort_fn)
    else:
        jar_files = get_cellprofiler_jars()
    jar_files = [os.path.join(imagej_path, f)  for f in jar_files]
    class_path = javabridge.JARS + jar_files
    
    if os.environ.has_key("CLASSPATH"):
        class_path += os.environ["CLASSPATH"].split(os.pathsep)
        logging.debug(
            "Adding Java class path from environment variable, ""CLASSPATH""")
        logging.debug("    CLASSPATH="+os.environ["CLASSPATH"])
        
    plugin_directory = cpprefs.get_ij_plugin_directory()
    if (plugin_directory is not None and 
        os.path.isdir(plugin_directory)):
        logger.debug("Using %s as imagej plugin directory" % plugin_directory)
        #
        # Add the plugin directory to pick up .class files in a directory
        # hierarchy.
        #
        class_path.append(plugin_directory)
        logger.debug("Adding %s to class path" % plugin_directory)
        #
        # Add any .jar files in the directory
        #
        for jarfile in os.listdir(plugin_directory):
            jarpath = os.path.join(plugin_directory, jarfile)
            if jarfile.lower().endswith(".jar"):
                logger.debug("Adding %s to class path" % jarpath)
                class_path.append(jarpath)
            else:
                logger.debug("Skipping %s" % jarpath)
    else:
        logger.info("Plugin directory doesn't point to valid folder: "
                    + plugin_directory)
        
    if sys.platform.startswith("win") and not hasattr(sys, 'frozen'):
        # Have to find tools.jar
        from javabridge.locate import find_jdk
        jdk_path = find_jdk()
        if jdk_path is not None:
            tools_jar = os.path.join(jdk_path, "lib","tools.jar")
            class_path.append(tools_jar)
        else:
            logger.warning("Failed to find tools.jar")
    return class_path
    
def cp_start_vm():
    '''Start CellProfiler's JVM via Javabridge
    
    JVM parameters are harvested from preferences and
    the environment variables:
    
    CP_JDWP_PORT - port # for debugging Java within the JVM
    cpprefs.get_awt_headless() - controls java.awt.headless to prevent
        awt from being invoked
    '''
    
    args = ["-Dloci.bioformats.loaded=true",
            "-Dlogback.configurationFile=logback.xml",
            "-Djava.util.prefs.PreferencesFactory="+
            "org.cellprofiler.headlesspreferences.HeadlessPreferencesFactory"]

    class_path = get_jars()
    args += get_patcher_args(class_path)
    awt_headless = cpprefs.get_awt_headless()
    if awt_headless:
        logger.debug("JVM will be started with AWT in headless mode")
        args.append("-Djava.awt.headless=true")
        
    heap_size = str(cpprefs.get_jvm_heap_mb())+"m"
    if os.environ.has_key("CP_JDWP_PORT"):
        args.append(
            ("-agentlib:jdwp=transport=dt_socket,address=127.0.0.1:%s"
             ",server=y,suspend=n") % os.environ["CP_JDWP_PORT"])
    javabridge.start_vm(args=args,
                        class_path=class_path,
                        max_heap_size = heap_size)
    #
    # Enable Bio-Formats directory cacheing
    #
    try:
        c_location = javabridge.JClassWrapper("loci.common.Location")
        c_location.cacheDirectoryListings(True)
        logger.debug("Enabled Bio-formats directory cacheing")
    except:
        logger.warning("Bioformats version does not support directory cacheing")
    #
    # Monkey-patch bioformats.formatreader.get_class_list to add
    # the classes we added to loci.formats.in
    #
    import bioformats.formatreader
    
    old_get_class_list = bioformats.formatreader.get_class_list
    
    def get_class_list():
        "Return a wrapped instance of loci.formats.ClassList"
        
        env = javabridge.get_env()
        class_list = old_get_class_list()
        rais_classname = 'loci/common/RandomAccessInputStream'
        #
        # Move any class to the back that thinks it can read garbage
        #
        fd, path = tempfile.mkstemp(suffix=".garbage",
                                    dir=cpprefs.get_temporary_directory())
        stream = None
        try:
            os.write(fd, "This is not an image file")
            os.fsync(fd)
            stream = javabridge.make_instance(
                rais_classname, '(Ljava/lang/String;)V', path)
            problem_classes = []
            for klass in env.get_object_array_elements(class_list.get_classes()):
                try:
                    instance =  javabridge.call(
                        klass, "newInstance", "()Ljava/lang/Object;")
                    can_read_garbage = javabridge.call(
                        instance, "isThisType",
                        "(L%s;)Z" % rais_classname, stream)
                    if can_read_garbage:
                        problem_classes.append(klass)
                        class_list.remove_class(klass)
                except:
                    logger.info("Caught exception in %s.isThisType",
                                javabridge.to_string(klass))
        finally:
            os.close(fd)
            javabridge.call(stream, "close", "()V")
            os.remove(path)
                    
        for classname in ("loci.formats.in.FlowSightReader", 
                          "loci.formats.in.IM3Reader"):
            try:
                klass = javabridge.class_for_name(classname)
                class_list.add_class(klass)
            except:
                logger.warn("Could not find Bio-formats reader %s" % classname,
                            exc_info=1)
        for klass in problem_classes:
            class_list.add_class(klass)
        return class_list
    bioformats.formatreader.get_class_list = get_class_list

def cp_stop_vm(kill=True):
    '''Shut down the Java VM

    Check for headlessness and the state of ImageJ and take
    whatever action is needed to stop AWT and the JVM.
    '''
    if not cpprefs.get_awt_headless():
        from imagej.imagej2 import allow_quit, the_imagej_context
        ij1 = javabridge.JClassWrapper("ij.IJ").getInstance()
        if ij1 is not None and ij1.isVisible():
            #
            # If legacy ImageJ is visible, then we have
            # to pick it apart in order to get around
            # the "Fiji won't quit" bug:
            #
            # http://fiji.sc/2014-07-11_-_Fiji_won%27t_quit!
            # http://fiji.sc/bugzilla/show_bug.cgi?id=805
            # https://github.com/fiji/fiji/issues/94
            # https://bugs.openjdk.java.net/browse/JDK-8061307
            #
            # The following code was needed in order to get around
            # the JDK problem on Red Hat 6 OpenJDK 1.7.0_79
            #
            env = javabridge.get_env()
            #
            # Use the window manager to find all of the ImagePlus windows.
            # Tell them that it's OK to close w/o prompt.
            # Close them all.
            #
            wm = javabridge.JClassWrapper("ij.WindowManager")
            for i in range(1, wm.getImageCount()+1):
                wm.getImage(i).changes = False
            wm.closeAllWindows()
            #
            # Pick apart and destroy IJ1's menus
            #
            menubar = ij1.getMenuBar()
            if menubar is not None:
                while menubar.getMenuCount() > 0:
                    menubar.remove(menubar.getMenu(0))
            #
            # Mercilessly pick apart all other windows
            # This throws the exception described in JDK-8061307
            #
            try:
                ij1.removeAll()
            except:
                logger.info("Caught expected AWT exception", exc_info=1)
        if the_imagej_context is not None:
            #
            # Tell the app service that it's OK to quit without prompt
            #
            allow_quit()
            app = javabridge.JWrapper(the_imagej_context.getService(
                    "org.scijava.app.AppService")).getApp()
            app.quit()
        if ij1 is not None:
            #
            # Yes, the proper way to get ImageJ to quit is
            # to start it.
            #
            ij1.run()
        #
        # Account for a "bug" in javabridge.1.0.11 that doesn't shelter
        # the caller of deactivate_awt() from the OpenJDK bug.
        #
        if javabridge.__version__ < "1.0.12":
            javabridge.jutil.CLOSE_ALL_WINDOWS = """
                new java.lang.Runnable() {
                    run: function() {
                        var all_frames = java.awt.Frame.getFrames();
                        if (all_frames) {
                            for (idx in all_frames) {
                                try {
                                    all_frames[idx].dispose();
                                } catch (err) {
                                }
                            }
                        }
                    }
               }"""
        javabridge.jutil.deactivate_awt()
    if kill:
        javabridge.kill_vm()
