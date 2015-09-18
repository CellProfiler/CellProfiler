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
# View a batch from the database, with some options to re-execute it
#
import cgitb
cgitb.enable()
from bpformdata import *
import RunBatch
import StyleSheet
import cgi
import os

def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

def delete_run(my_batch, my_run):
    if delete_action in (A_DELETE_ALL, A_DELETE_TEXT):
        for bat in RunBatch.BPBatchArrayTask.select_by_run(my_run):
            remove_if_exists(RunBatch.batch_array_task_text_file_path(bat))
            remove_if_exists(RunBatch.batch_array_task_err_file_path(bat))

    if delete_action in (A_DELETE_ALL, A_DELETE_OUTPUT):
        remove_if_exists(RunBatch.run_out_file_path(my_batch, my_run))
    
delete_action = BATCHPROFILER_DEFAULTS[K_DELETE_ACTION]
run_id = BATCHPROFILER_DEFAULTS[RUN_ID]
batch_id = BATCHPROFILER_DEFAULTS[BATCH_ID]
if run_id is not None and delete_action is not None:
    my_run = RunBatch.BPRun.select(run_id)
    my_batch = my_run.batch
    delete_run(my_batch, my_run)
elif batch_id is not None:
    my_batch = RunBatch.BPBatch.select(batch_id)
    for my_run in my_batch.select_runs():
        delete_run(my_batch, my_run)
    
    
url = "ViewBatch.py?batch_id=%d"%(my_batch.batch_id)
print "Content-Type: text/html"
print
print "<html><head>"
print "<meta http-equiv='refresh' content='0; URL=%(url)s' />"%(globals())
print "</head>"
print "<body>This page should be redirected to <a href='%(url)s'/>%(url)s</a></body>"%(globals())
print "</html>"
