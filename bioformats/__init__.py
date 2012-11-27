'''Bioformats package - wrapper for loci.bioformats java code

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

import logging
import os
import re
import cellprofiler.utilities.jutil as jutil
from cellprofiler.preferences import get_headless, get_ij_plugin_directory
from cellprofiler.preferences import get_ij_version, IJ_1, IJ_2
import sys

logger = logging.getLogger("bioformats")

# See http://www.loci.wisc.edu/software/bio-formats
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

USE_IJ2 = None
def start_cellprofiler_jvm():
    '''Start the Java VM with arguments appropriate for CellProfiler'''
    global USE_IJ2
    global logger
    
    if hasattr(sys, 'frozen') and sys.platform != 'darwin':
        root_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
    else:
        root_path = os.path.abspath(os.path.split(__file__)[0])
        root_path = os.path.split(root_path)[0]
    path = os.path.join(root_path, 'bioformats')
    imagej_path = os.path.join(root_path, 'imagej')
    loci_jar = os.path.join(path, "loci_tools.jar")
    ij2_jar = os.path.join(imagej_path, "imagej-2.0-SNAPSHOT-all.jar")
    ij_jar = os.path.join(imagej_path, "ij.jar")
    imglib_jar = os.path.join(imagej_path, "imglib.jar")
    javacl_jar = os.path.join(imagej_path, "javacl-1.0-beta-4-shaded.jar")
    USE_IJ2 = get_ij_version() == IJ_2
    if os.path.exists(ij2_jar) and USE_IJ2:
        class_path = os.pathsep.join((loci_jar, ij2_jar))
        USE_IJ2 = True
    else:
        USE_IJ2 = False
        class_path = os.pathsep.join((loci_jar, ij_jar, imglib_jar, 
                                      javacl_jar))
    if os.environ.has_key("CLASSPATH"):
        class_path += os.pathsep + os.environ["CLASSPATH"]
        
    if sys.platform.startswith("win") and not hasattr(sys, 'frozen'):
        # Have to find tools.jar
        from cellprofiler.utilities.setup import find_jdk
        jdk_path = find_jdk()
        if jdk_path is not None:
            tools_jar = os.path.join(jdk_path, "lib","tools.jar")
            class_path += os.pathsep + tools_jar
        else:
            logger.warning("Failed to find tools.jar")
    
    jvm_arg = [x.groups()[0] for x in [
        re.match('--jvm-heap-size=([0-9]+[gGkKmM])', y) for y in sys.argv]
               if x is not None]
    if len(jvm_arg) > 0:
        jvm_arg = jvm_arg[0]
    else:
        jvm_arg = "512m"
        
    args = [r"-Djava.class.path="+class_path,
            r"-Dloci.bioformats.loaded=true",
            #r"-verbose:class",
            #r"-verbose:jni",
            r"-Xmx%s" % jvm_arg]
    if get_ij_plugin_directory() is not None:
        args.append("-Dplugins.dir="+get_ij_plugin_directory())
    
    #
    # Get the log4j logger setup from a file in the bioformats directory
    # if such a file exists.
    #
    log4j_properties = os.path.join(path, "log4j.properties")
    if os.path.exists(log4j_properties):
        log4j_properties = "file:/"+log4j_properties.replace(os.path.sep, "/")
        args += [r"-Dlog4j.configuration="+log4j_properties]
        init_logger = False
    else:
        init_logger = True
        
    run_headless = (get_headless() and 
                    not os.environ.has_key("CELLPROFILER_USE_XVFB"))
        
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
    
    #
    # Start the log4j logger to avoid error messages.
    #
    if init_logger:
        try:
            jutil.static_call("org/apache/log4j/BasicConfigurator",
                              "configure", "()V")
            log4j_logger = jutil.static_call("org/apache/log4j/Logger",
                                             "getRootLogger",
                                             "()Lorg/apache/log4j/Logger;")
            warn_level = jutil.get_static_field("org/apache/log4j/Level","WARN",
                                                "Lorg/apache/log4j/Level;")
            jutil.call(log4j_logger, "setLevel", "(Lorg/apache/log4j/Level;)V", 
                       warn_level)
            del logger
            del warn_level
        except:
            logger.error("Failed to initialize log4j\n", exc_info=True)
    if not run_headless:
        jutil.activate_awt()
start_cellprofiler_jvm()
from formatreader import load_using_bioformats

if __name__ == "__main__":
    # Handy-dandy PyShell for exploring BioFormats / Rhino / ImageJ
    import wx.py.PyCrust
    
    wx.py.PyCrust.main()
    J.kill_vm()