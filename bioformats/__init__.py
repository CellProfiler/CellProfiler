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

if hasattr(sys, 'frozen') and sys.platform != 'darwin':
    __root_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
else:
    __root_path = os.path.abspath(os.path.split(__file__)[0])
    __root_path = os.path.split(__root_path)[0]
__path = os.path.join(__root_path, 'bioformats')
__imagej_path = os.path.join(__root_path, 'imagej')
__ij2_jar = os.path.join(__imagej_path, "imagej-2.0-SNAPSHOT-all.jar")
__ij_jar = os.path.join(__imagej_path, "ij.jar")
__imglib_jar = os.path.join(__imagej_path, "imglib.jar")
__javacl_jar = os.path.join(__imagej_path, "javacl-1.0-beta-4-shaded.jar")
__precompiled_headless_jar = os.path.join(__imagej_path, "precompiled_headless.jar")
USE_IJ2 = get_ij_version() == IJ_2
if os.path.exists(__ij2_jar) and USE_IJ2:
    __class_path = __ij2_jar
    USE_IJ2 = True
else:
    USE_IJ2 = False
    __class_path = os.pathsep.join((__ij_jar, __imglib_jar, __javacl_jar))
    if sys.platform == "darwin":
        # Start ImageJ headless
        # precompiled_headless.jar contains substitute classes for running
        # headless.
        #
        __class_path = os.pathsep.join((__precompiled_headless_jar, __class_path))
if os.environ.has_key("CLASSPATH"):
    __class_path += os.pathsep + os.environ["CLASSPATH"]
    
if sys.platform.startswith("win") and not hasattr(sys, 'frozen'):
    # Have to find tools.jar
    from cellprofiler.utilities.setup import find_jdk
    jdk_path = find_jdk()
    if jdk_path is not None:
        __tools_jar = os.path.join(jdk_path, "lib","tools.jar")
        __class_path += os.pathsep + __tools_jar
    else:
        logger.warning("Failed to find tools.jar")

jvm_arg = [x.groups()[0] for x in [
    re.match('--jvm-heap-size=([0-9]+[gGkKmM])', y) for y in sys.argv]
           if x is not None]
if len(jvm_arg) > 0:
    jvm_arg = jvm_arg[0]
else:
    jvm_arg = "512m"
    
__args = [r"-Djava.class.path="+__class_path,
          #r"-Djava.ext.dirs=%s"%__path,
          r"-Dloci.bioformats.loaded=true",
          #r"-verbose:class",
          #r"-verbose:jni",
          r"-Xmx%s" % jvm_arg]
if get_ij_plugin_directory() is not None:
    __args.append("-Dplugins.dir="+get_ij_plugin_directory())

#
# Get the log4j logger setup from a file in the bioformats directory
# if such a file exists.
#
__log4j_properties = os.path.join(__path, "log4j.properties")
if os.path.exists(__log4j_properties):
    __log4j_properties = "file:/"+__log4j_properties.replace(os.path.sep, "/")
    __args += [r"-Dlog4j.configuration="+__log4j_properties]
    __init_logger = False
else:
    __init_logger = True
    
if ((get_headless() and not os.environ.has_key("CELLPROFILER_USE_XVFB"))
    or sys.platform=="darwin"):
    __args += [ r"-Djava.awt.headless=true" ]

logger.debug("JVM arguments: " + " ".join(__args))
jutil.start_vm(__args)
logger.debug("Java virtual machine started.")
    
