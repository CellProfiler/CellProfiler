#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/python-2.6.sh
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
# Kill all jobs in a batch
#
import cgitb
cgitb.enable()
import RunBatch
import cgi
import os

form = cgi.FieldStorage()
batch_id = int(form["batch_id"].value)
my_batch = RunBatch.LoadBatch(batch_id)
status_dir = "%(data_dir)s/status/"%(my_batch)
txt_output_dir = "%(data_dir)s/txt_output/"%(my_batch)
if os.path.exists(status_dir):
    os.chmod(status_dir,0777)
if os.path.exists(txt_output_dir):
    os.chmod(txt_output_dir,0777)
    
for run in my_batch["runs"]:
    for path in [RunBatch.RunTextFilePath(my_batch,run),
                 RunBatch.RunDoneFilePath(my_batch,run),
                 RunBatch.RunOutFilePath(my_batch,run)]:
        if os.path.exists(path):
            os.chmod(path,0644)

url = "ViewBatch.py?batch_id=%(batch_id)d"%(my_batch)
print "Content-Type: text/html"
print
print "<html><head>"
print "<meta http-equiv='refresh' content='0; URL=%(url)s' />"%(globals())
print "</head>"
print "<body>This page should be redirected to <a href='%(url)s'/>%(url)s</a></body>"%(globals())
print "</html>"
