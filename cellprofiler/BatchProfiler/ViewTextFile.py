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
# View the text file that is produced by bsub during batch processing
#
import cgitb
cgitb.enable()
from bpformdata import *
import RunBatch
import os
import stat
import sys

def main():
    batch_array_id = BATCHPROFILER_VARIABLES[BATCH_ARRAY_ID]
    task_id = BATCHPROFILER_VARIABLES[TASK_ID]

    if batch_array_id is not None and task_id is not None:
        do_it(batch_array_id, task_id)
    else:
        show_help()
        
def do_it(batch_array_id, task_id):
    batch_array = RunBatch.BPBatchArray.select(batch_array_id)
    bat = RunBatch.BPBatchArrayTask.select_by_batch_array_and_task_id(
        batch_array, task_id)
    if bat is None:
        show_help()
        return
    run = bat.run
    my_batch = run.batch
    file_type = BATCHPROFILER_VARIABLES[FILE_TYPE]
    if file_type == FT_TEXT_FILE:
        show_file(RunBatch.batch_array_task_text_file_path(bat))
    elif file_type == FT_ERR_FILE:
        show_file(RunBatch.batch_array_task_err_file_path(bat))
    elif file_type == FT_OUT_FILE:
        download_attachment(
            RunBatch.run_out_file(my_batch, my_run),
            RunBatch.run_out_file_path(my_batch, my_run))
    else:
        show_help("Unknown file type: %s" % file_type)

def show_file(path):
    text_file = open(path,"r")
    print "Content-Type: text/plain"
    print
    print text_file.read()
    text_file.close()
    
def download_attachment(filename, path):
    length = os.stat(path).st_size
    print "Content-Type: application/octet-stream"
    print 'Content-Disposition: attachment; filename="%s"' % filename
    print 'Content-Length: %d' % length
    print
    read_so_far = 0
    with open(path, "rb") as fd:
        while read_so_far < length:
            data = fd.read()
            if len(data) == 0:
                break
            read_so_far += len(data)
            sys.stdout.write(data)

def show_help(message = None):
    import yattag
    doc, tag, text = yattag.Doc().tagtext()
    assert isinstance(doc, yattag.Doc)
    with tag("html"):
        with tag("head"):
            with tag("title"):
                text("BatchProfiler: view text file")
        with tag("body"):
            with tag("div"):
                text("This page is designed to be used by ")
                with tag("a", href="ViewBatch.py"):
                    text("View Batch")
                if message is not None:
                    text("The page reports the following error: \"%s\"" %
                         message)
                text(". Most likely you have reached this page in error. ")
                text("If you feel that's not the case, you can contact your ")
                text("sysadmin and if you are the sysadmin, you can post to ")
                text("the ")
                with tag("a", href="http://cellprofiler.org/forum"):
                    text("CellProfiler forum")
                text(".")
    print "Content-Type: text/html"
    print
    print yattag.indent(doc.getvalue())

main()
