'''Bioformats package - wrapper for loci.bioformats java code

'''
__version__ = "$Revision$"

import os
import cellprofiler.utilities.jutil as jutil
from cellprofiler.preferences import get_headless
import sys
import traceback

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

if hasattr(sys, 'frozen'):
    __path = os.path.split(os.path.abspath(sys.argv[0]))[0]
    __path = os.path.join(__path, 'bioformats')
else:
    __path = os.path.abspath(os.path.split(__file__)[0])
__loci_jar = os.path.join(__path, "loci_tools.jar")
__class_path = __loci_jar
if os.environ.has_key("CLASSPATH"):
    __class_path += os.pathsep + os.environ["CLASSPATH"]
    
__args = [r"-Djava.class.path="+__class_path,
          #r"-Djava.ext.dirs=%s"%__path,
          r"-Dloci.bioformats.loaded=true",
          #r"-verbose:class",
          #r"-verbose:jni",
          r"-Xmx512m"]

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
    
if get_headless() or sys.platform=="darwin":
    __args += [ r"-Djava.awt.headless=true" ]

jutil.start_vm(__args)

if get_headless() or sys.platform=="darwin":
    jutil.attach()
    jutil.static_call("java/lang/System", "setProperty", '(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;', "java.awt.headless", "TRUE")
    jutil.detach()
#
# Start the log4j logger to avoid error messages.
#
if __init_logger:
    jutil.attach()
    try:
        jutil.static_call("org/apache/log4j/BasicConfigurator",
                          "configure", "()V")
        logger = jutil.static_call("org/apache/log4j/Logger",
                                   "getRootLogger",
                                   "()Lorg/apache/log4j/Logger;")
        warn_level = jutil.get_static_field("org/apache/log4j/Level","WARN",
                                            "Lorg/apache/log4j/Level;")
        jutil.call(logger, "setLevel", "(Lorg/apache/log4j/Level;)V", 
                   warn_level)
        del logger
        del warn_level
    except:
        sys.stderr.write("Failed to initialize log4j\n")
        traceback.print_exc()
    finally:
        jutil.detach()
