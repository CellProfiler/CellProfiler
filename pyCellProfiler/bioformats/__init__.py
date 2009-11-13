'''Bioformats package - wrapper for loci.bioformats java code

'''
__version__ = "$Revision: 1$"

import os
import cellprofiler.utilities.jutil as jutil
import cellprofiler.utilities.javabridge as javabridge

__path = os.path.abspath(os.path.split(__file__)[0])
__loci_jar = os.path.join(__path, "loci_tools.jar")

jutil.start_vm(["-Djava.class.path="+__loci_jar,
                "-Djava.awt.headless=true",
                "-Dloci.bioformats.loaded=true"])
