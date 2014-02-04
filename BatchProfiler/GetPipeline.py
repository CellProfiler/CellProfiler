#!/usr/bin/env ./python-2.6.sh
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
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
import cellprofiler.preferences
cellprofiler.preferences.set_headless()

import os, sys
import RunBatch
from cellprofiler.measurements import Measurements, EXPERIMENT
from cellprofiler.pipeline import M_PIPELINE
try:
    from cellprofiler.pipeline import M_USER_PIPELINE
except:
    M_USER_PIPELINE = "Pipeline_UserPipeline"
from cellprofiler.modules.createbatchfiles import F_BATCH_DATA_H5

form = cgi.FieldStorage()
batch_id = int(form["batch_id"].value)
my_batch = RunBatch.LoadBatch(batch_id)
path = os.path.join(my_batch["data_dir"], F_BATCH_DATA_H5)
m = Measurements(filename=path, mode="r")
if m.has_feature(EXPERIMENT, M_USER_PIPELINE):
    feature = M_USER_PIPELINE
else:
    feature = M_PIPELINE
pipeline_text = m[EXPERIMENT, feature]
if isinstance(pipeline_text, unicode):
    pipeline_text = pipeline_text.encode("us-ascii")
print "Content-Type: text/plain"
print "Content-Length: %d" % len(pipeline_text)
print "Content-Disposition: attachment; filename=\"%d.cppipe\"" % batch_id
print
sys.stdout.write(pipeline_text)

