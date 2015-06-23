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
import subprocess
import sys

job_id = BATCHPROFILER_VARIABLES[JOB_ID]
batch_id = BATCHPROFILER_VARIABLES[BATCH_ID]
if job_id is not None:
    job = RunBatch.BPJob.select(job_id)
    if job is None:
        bputilities.kill_job(job_id)
    else:
        RunBatch.kill_job(job)
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
    p = subprocess.Popen(["bash"],stdin = subprocess.PIPE,
                         stdout=subprocess.PIPE)
    listing = p.communicate(
        ". /broad/software/scripts/useuse;reuse GridEngine8;qstat\n")[0]
    listing_lines = listing.split('\n')
    header = listing_lines[0]
    columns = [header.find(x) for x in header.split(' ') if len(x)]
    columns.append(1000)
    body = listing_lines[2:]
    print """
    <h2>Jobs on imageweb</h2>
    <table>
    """
    print "<tr>%s</tr>"%("".join(['<th>%s</th>'%(header[columns[i]:columns[i+1]])
                                  for i in range(len(columns)-1)]))
    for line in body:
        print "<tr>%s</tr>"%("".join(['<td>%s</td>'%(line[columns[i]:columns[i+1]])
                                      for i in range(len(columns)-1)]))
    """
    </table>
    </body>
    """
bputilities.shutdown_cellprofiler()