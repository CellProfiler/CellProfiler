"""cpjvm.py - CellProfiler-specific JVM utilities

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import javabridge
import bioformats
import logging
import os
import sys

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
        if sys.platform.startswith("win"):
            patcher = patchers[0].replace(os.path.sep, os.path.altsep)
        else:
            patcher = patchers[0]
        return ["-javaagent:%s=init" % patcher]
    logger.warn("Did not find ij1-patcher.jar")
    return []
    
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

    imagej_path = get_path_to_jars()
    if hasattr(sys, 'frozen'):
        jar_files = [
            jar_filename
            for jar_filename in os.listdir(imagej_path)
            if jar_filename.lower().endswith(".jar")]
        def sort_fn(a, b):
            aa,bb = [(0 if x.startswith("cellprofiler-java") else 1, x)
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
        