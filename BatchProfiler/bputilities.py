# Utility functions for Batch Profiler
import os
import sys
from cStringIO import StringIO

def get_batch_data_version_and_githash(batch_filename):
    """Get the commit's GIT hash stored in the batch file's pipeline
    
    batch_filename - Batch_data.h5 file to look at
    
    returns the GIT hash as stored in the file.
    """
    import cellprofiler.measurements as cpmeas
    import cellprofiler.pipeline as cpp
    
    m = cpmeas.load_measurements(batch_filename, mode="r")
    pipeline_txt = m[cpmeas.EXPERIMENT, cpp.M_PIPELINE]
    pipeline = cpp.Pipeline()
    version, git_hash = pipeline.loadtxt(StringIO(pipeline_txt))
    return version, git_hash

def get_cellprofiler_location(batch_filename):
    '''Get the location of the CellProfiler source to use'''
    
    version, git_hash = get_batch_data_version_and_githash(batch_filename)
    path = os.path.join(os.environ["CPCLUSTER"], )
        

