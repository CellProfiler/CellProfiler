'''Bioformats package - wrapper for loci.bioformats java code

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

import logging
import os
import re
import cellprofiler.utilities.jutil as jutil
import urllib
from cellprofiler.preferences import \
     get_headless, get_awt_headless, get_ij_plugin_directory, get_jvm_heap_mb
import sys
from external_dependencies import get_cellprofiler_jars

logger = logging.getLogger("bioformats")

# See http://www.openmicroscopy.org/site/support/bio-formats5/supported-formats.html
READABLE_FORMATS = ('al3d', 'am', 'amiramesh', 'apl', 'arf', 'avi', 'bmp', 
                    'c01', 'cfg', 'cxd', 'dat', 'dcm', 'dicom', 'dm3', 'dv', 
                    'eps', 'epsi', 'fits', 'flex', 'fli', 'gel', 'gif', 'grey', 
                    'hdr', 'html', 'hx', 'ics', 'ids', 'img', 'ims', 'ipl', 
                    'ipm', 'ipw', 'jp2', 'jpeg', 'jpg', 'l2d', 'labels', 'lei', 
                    'lif', 'liff', 'lim', 'lsm', 'mdb', 'mnc', 'mng', 'mov', 
                    'mrc', 'mrw', 'mtb', 'naf', 'nd', 'nd2', 'nef', 'nhdr', 
                    'nrrd', 'obsep', 'oib', 'oif', 'ome', 'ome.tiff', 'pcx', 
                    'pgm', 'pic', 'pict', 'png', 'ps', 'psd', 'r3d', 'raw', 
                    'scn', 'sdt', 'seq', 'sld', 'stk', 'svs', 'tif', 'tiff', 
                    'tnb', 'txt', 'vws', 'xdce', 'xml', 'xv', 'xys', 'zvi')
WRITABLE_FORMATS = ('avi', 'eps', 'epsi', 'ics', 'ids', 'jp2', 'jpeg', 'jpg', 
                    'mov', 'ome', 'ome.tiff', 'png', 'ps', 'tif', 'tiff')

def start_cellprofiler_jvm():
    '''Start the Java VM with arguments appropriate for CellProfiler'''
    global logger
    
    if hasattr(sys, 'frozen'):
        if sys.platform != 'darwin':
            root_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
            bioformats_path = os.path.join(root_path, 'bioformats')
        else:
            bioformats_path = os.path.abspath(os.path.split(__file__)[0])
            root_path = os.path.split(bioformats_path)[0]
        imagej_path = os.path.join(root_path, 'imagej','jars')
        def sort_fn(a, b):
            aa,bb = [(0 if x.startswith("cellprofiler-java") else 1, x)
                     for x in a, b]
            return cmp(aa, bb)
        jar_files = [
            jar_filename
            for jar_filename in os.listdir(imagej_path)
            if jar_filename.lower().endswith(".jar")]
        jar_files = sorted(jar_files, cmp = sort_fn)
    else:
        bioformats_path = os.path.abspath(os.path.split(__file__)[0])
        root_path = os.path.split(bioformats_path)[0]
        jar_files = get_cellprofiler_jars()
        imagej_path = os.path.join(root_path, 'imagej','jars')
    
    class_path = os.pathsep.join(
        [os.path.join(imagej_path, jar_file) for jar_file in jar_files])
    
    if os.environ.has_key("CLASSPATH"):
        class_path += os.pathsep + os.environ["CLASSPATH"]
        
    plugin_directory = get_ij_plugin_directory()
    logger.debug("Using %s as imagej plugin directory" % plugin_directory)
    if (plugin_directory is not None and 
        os.path.isdir(plugin_directory)):
        #
        # Add the plugin directory to pick up .class files in a directory
        # hierarchy.
        #
        class_path += os.pathsep + plugin_directory
        logger.debug("Adding %s to class path" % plugin_directory)
        #
        # Add any .jar files in the directory
        #
        for jarfile in os.listdir(plugin_directory):
            jarpath = os.path.join(plugin_directory, jarfile)
            if jarfile.lower().endswith(".jar"):
                logger.debug("Adding %s to class path" % jarpath)
                class_path += os.pathsep + jarpath
            else:
                logger.debug("Skipping %s" % jarpath)
    else:
        logger.info("Plugin directory doesn't point to valid folder: " + plugin_directory)
        
    if sys.platform.startswith("win") and not hasattr(sys, 'frozen'):
        # Have to find tools.jar
        from cellprofiler.utilities.setup import find_jdk
        jdk_path = find_jdk()
        if jdk_path is not None:
            tools_jar = os.path.join(jdk_path, "lib","tools.jar")
            class_path += os.pathsep + tools_jar
        else:
            logger.warning("Failed to find tools.jar")
    
    jvm_arg = "%dm" % get_jvm_heap_mb()
        
    args = [r"-Djava.class.path="+class_path,
            r"-Dloci.bioformats.loaded=true",
            #r"-verbose:class",
            #r"-verbose:jni",
            r"-Dlogback.configurationFile=logback.xml",
            r"-Xmx%s" % jvm_arg]

    if plugin_directory is not None and os.path.isdir(plugin_directory):
        # For IJ1 compatibility
        args += [r"-Dplugins.dir=%s" % plugin_directory]
    
    # In headless mode, we have to avoid changing the Java preferences.
    # 
    # Aside from that, we need to prevent ImageJ from exiting and from
    # displaying the updater dialog - at least temporarily, we do that
    # through preferences. We use the HeadlessPreferencesFactory to
    # limit the scope of the changes to this process - otherwise we'd
    # turn off updating for the machine.
    #
    # TODO: ImageJ is implementing a pluggable mechanism to control the
    #       quit process. We can also contribute a pluggable mechanism
    #       that gives us more control over the updater.
    #
    args += [
        "-Djava.util.prefs.PreferencesFactory="
        "org.cellprofiler.headlesspreferences.HeadlessPreferencesFactory"]
    run_headless = get_awt_headless()
        
    logger.debug("JVM arguments: " + " ".join(args))
    jutil.start_vm(args, run_headless)
    logger.debug("Java virtual machine started.")
    jutil.attach()
    try:
        jutil.static_call("loci/common/Location",
                          "cacheDirectoryListings",
                          "(Z)V", True)
    except:
        logger.warning("Bioformats version does not support directory cacheing")
start_cellprofiler_jvm()
from formatreader import load_using_bioformats

if __name__ == "__main__":
    # Handy-dandy PyShell for exploring BioFormats / Rhino / ImageJ
    import wx.py.PyCrust
    
    wx.py.PyCrust.main()
    jutil.kill_vm()
