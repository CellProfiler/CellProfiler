'''Bioformats package - wrapper for loci.bioformats java code

'''
__version__ = "$Revision: 1$"

import os
import cellprofiler.utilities.jutil as jutil
import cellprofiler.utilities.javabridge as javabridge
from cellprofiler.preferences import get_headless
import sys

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

if get_headless():
    __args += [ r"-Djava.awt.headless=true" ]

jutil.start_vm(__args)
