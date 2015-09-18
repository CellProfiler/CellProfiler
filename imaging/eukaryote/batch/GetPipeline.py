#!/usr/bin/env ./batchprofiler.sh
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
#
# Extract the pipeline from a batch data file
#
import cgitb
cgitb.enable()
import cgi

import os, sys
import bputilities
from bpformdata import *
import RunBatch
from cellprofiler.utilities.hdf5_dict import HDF5Dict
M_USER_PIPELINE = "Pipeline_UserPipeline"
M_PIPELINE = "Pipeline_Pipeline"
F_BATCH_DATA_H5 = "Batch_data.h5"
EXPERIMENT = "Experiment"

with bputilities.CellProfilerContext():
    batch_id = BATCHPROFILER_DEFAULTS[BATCH_ID]
    my_batch = RunBatch.BPBatch.select(batch_id)
    path = RunBatch.batch_data_file_path(my_batch)
    h = HDF5Dict(path, mode="r")
    if M_USER_PIPELINE in h.second_level_names(EXPERIMENT):
        feature = M_USER_PIPELINE
    else:
        feature = M_PIPELINE
    pipeline_text = h[EXPERIMENT, feature, 0][0].decode("unicode_escape")
    h.close()
    del h
    if isinstance(pipeline_text, unicode):
        pipeline_text = pipeline_text.encode("us-ascii")
print "Content-Type: text/plain"
print "Content-Length: %d" % len(pipeline_text)
print "Content-Disposition: attachment; filename=\"%d.cppipe\"" % batch_id
print
sys.stdout.write(pipeline_text)
sys.stdout.flush()


