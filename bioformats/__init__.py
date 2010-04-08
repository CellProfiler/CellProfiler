'''Bioformats package - wrapper for loci.bioformats java code

'''
__version__ = "$Revision$"

import os
import cellprofiler.utilities.jutil as jutil
import cellprofiler.utilities.javabridge as javabridge
from cellprofiler.preferences import get_headless
import sys

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
          r"-Djava.ext.dirs=%s"%__path,
          r"-Dloci.bioformats.loaded=true",
          #r"-verbose:class",
          #r"-verbose:jni",
          r"-Xmx512m"]

if get_headless() or sys.platform=="darwin":
    __args += [ r"-Djava.awt.headless=true" ]

jutil.start_vm(__args)
