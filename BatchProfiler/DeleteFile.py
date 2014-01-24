#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/python-2.6.sh
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
# View a batch from the database, with some options to re-execute it
#
import cgitb
cgitb.enable()
import RunBatch
import StyleSheet
import cgi
import os
import os.path

def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

form = cgi.FieldStorage()
delete_action = form["delete_action"].value
if form.has_key("run_id"):
    run_id = int(form["run_id"].value)
    my_batch,my_run = RunBatch.LoadRun(run_id)
    if ((delete_action.upper() == "ALL") or (delete_action.upper() == "TEXT")):
        remove_if_exists(RunBatch.RunTextFilePath(my_batch, my_run))

    if (delete_action.upper() == "ALL") or (delete_action.upper() == "OUTPUT"):
        remove_if_exists(RunBatch.RunOutFilePath(my_batch, my_run))

    if (delete_action.upper() == "ALL") or (delete_action.upper() == "DONE"):
        remove_if_exists(RunBatch.RunDoneFilePath(my_batch, my_run))
elif form.has_key("batch_id"):
    batch_id = int(form["batch_id"].value)
    my_batch = RunBatch.LoadBatch(batch_id)
    for my_run in my_batch["runs"]:
        if ((delete_action.upper() == "ALL") or (delete_action.upper() == "TEXT")):
            remove_if_exists(RunBatch.RunTextFilePath(my_batch, my_run))

        if (delete_action.upper() == "ALL") or (delete_action.upper() == "OUTPUT"):
            remove_if_exists(RunBatch.RunOutFilePath(my_batch, my_run))

        if (delete_action.upper() == "ALL") or (delete_action.upper() == "DONE"):
            remove_if_exists(RunBatch.RunDoneFilePath(my_batch, my_run))
    
    
url = "ViewBatch.py?batch_id=%(batch_id)d"%(my_batch)
print "Content-Type: text/html"
print
print "<html><head>"
print "<meta http-equiv='refresh' content='0; URL=%(url)s' />"%(globals())
print "</head>"
print "<body>This page should be redirected to <a href='%(url)s'/>%(url)s</a></body>"%(globals())
print "</html>"
try:
    import cellprofiler.utilities.jutil as jutil
    jutil.kill_vm()
except:
    import traceback
    traceback.print_exc()
