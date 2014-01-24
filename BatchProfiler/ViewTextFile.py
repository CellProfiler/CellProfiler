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
# View the text file that is produced by bsub during batch processing
#
import cgitb
cgitb.enable()
import RunBatch
import cgi
import os
import os.path
import stat

if cgi.FieldStorage().has_key("run_id"):
    run_id = int(cgi.FieldStorage()["run_id"].value)
    my_batch,my_run = RunBatch.LoadRun(run_id)
    text_file_path = RunBatch.RunTextFilePath(my_batch,my_run);
else:
    text_file_path = cgi.FieldStorage()["file_name"].value
#
# This is a temporary work-around because files get created
# with the wrong permissions
#
if (os.stat(text_file_path)[0] & stat.S_IREAD) == 0:
    os.chmod(text_file_path,0644)
text_file = open(text_file_path,"r")
print "Content-Type: text/plain"
print
print text_file.read()
text_file.close()

