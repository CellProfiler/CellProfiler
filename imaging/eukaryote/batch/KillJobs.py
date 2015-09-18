#!/usr/bin/env ./batchprofiler.sh
#
# Kill all jobs in a batch
#
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
print "Content-Type: text/html\r"
print "\r"
import cgitb
cgitb.enable()
import RunBatch
import bputilities
from bpformdata import *
import cgi
import os
import subprocess
import sys

job_id = BATCHPROFILER_VARIABLES[JOB_ID]
task_id = BATCHPROFILER_VARIABLES[TASK_ID]
batch_id = BATCHPROFILER_VARIABLES[BATCH_ID]
if job_id is not None:
    job = RunBatch.BPJob.select_by_job_id(job_id)
    if job is None:
        if task_id is None:
            bputilities.kill_job(job_id)
        else:
            bputilities.kill_tasks(job_id, [task_id])
    elif task_id is None:
        RunBatch.kill_job(job)
    else:
        task = RunBatch.BPJobTask.select_by_task_id(job, task_id)
        RunBatch.kill_task(task)
    print"""
    <html><head><title>Job %(job_id)d killed</title></head>
    <body>Job %(job_id)d killed
    </body>
    </html>
"""%(globals())
elif batch_id is not None:
    RunBatch.kill_batch(batch_id)
    
    url = "ViewBatch.py?batch_id=%d"%(batch_id)
    print "<html><head>"
    print "<meta http-equiv='refresh' content='0; URL=%(url)s' />"%(globals())
    print "</head>"
    print "<body>This page should be redirected to <a href='%(url)s'/>%(url)s</a></body>"%(globals())
    print "</html>"
else:
    print """<html><head><title>Kill jobs</title></head>
    <body>
    <h1>Kill jobs started by the imageweb webserver</h1>
    <form action='KillJobs.py' method='POST'>
    Job ID:<input type='text' name='job_id' />
    <input type='submit' value='Kill'/>
    </form>
    """
    script = """#!/bin/sh
set -v
if [ -e "$HOME/.batchprofiler.sh" ]; then
. "$HOME/.batchprofiler.sh"
fi
set +v
qstat
"""
    scriptfile = bputilities.make_temp_script(script)
    try:
        output = subprocess.check_output([scriptfile])
    finally:
        os.remove(scriptfile)
    result = []
    lines = output.split("\n")
    header = lines[0]
    columns = [i for i in range(len(header))
               if i == 0 or (header[i-1].isspace() and not header[i].isspace())]
    columns.append(len(header))
    rows = [[line[columns[i]:columns[i+1]].strip() 
             for i in range(len(columns)-1)]
            for line in lines]
    header = rows[0]
    body = rows[2:]
    print """
    <h2>Jobs on imageweb</h2>
    <table>
    """
    print "<tr>%s</tr>"%("".join([
        '<th>%s</th>'%field for field in header]))
    for fields in body:
        try:
            job_id = int(fields[0])
        except:
            continue
        fields[0] = """
        <form action='KillJobs.py' method='POST'>
        Job ID: %d 
        <input type='hidden' name='job_id' value='%d'/>
        <input type='submit' value='Kill'/>
        </form>""" % (job_id, job_id)
        
        print "<tr><td>%s</td></tr>" % "</td><td>".join(fields)
    """
    </table>
    </body>
    """
sys.stdout.close()
bputilities.shutdown_cellprofiler()