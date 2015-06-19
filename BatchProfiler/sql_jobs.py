#!/usr/bin/env ./batchprofiler.sh
"""sql_jobs.py - launch and monitor remote jobs that submit SQL

This file is a command-line script for running jobs and a set of functions
for running the script on the cluster. To get help on running from the
command-line:

sql_jobs.py --help

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import re
import os
import subprocess
import sys
import RunBatch

def run_sql_file(batch_id, sql_filename):
    """Use the mysql command line to run the given SQL script
    
    batch_id    - sql_file is associated with this batch
    sql_path    - path and file of the sql script
    queue       - run job on this queue
    email       - who to email when done
    returns the RunBatch.BPJob
    """
    batch = RunBatch.BPBatch()
    batch.select(batch_id)
    run = RunBatch.BPSQLRun.select_by_sql_filename(batch, sql_filename)
    if run is None:
        sql_path = os.path.join(batch.data_dir, sql_filename)
        cmd = "%s -b %d -i %s"%(__file__, batch_id, sql_path)
        run = RunBatch.BPSQLRun.create(batch, sql_filename, cmd)
    return RunBatch.run_one(batch, run, cwd=os.path.dirname(__file__))

def sql_file_job_and_status(batch_id, sql_file):
    """Return the latest job ID associated with the batch and sql path
    
    batch_id - batch id associated with the submission
    sql_path - path to the sql script submitted
    
    returns latest job or None if not submitted
    """
    batch = RunBatch.BPBatch()
    batch.select(batch_id)
    run = RunBatch.BPSQLRun.select_by_sql_filename(sql_file)
    result = run.select_jobs()
    if len(result) == 0:
        return None, None, None
    return run, result[0][0], result[0][1]
    
if __name__ == "__main__":
    import optparse
    sys.path.append(os.path.dirname(__file__))
    from bpformdata import \
         BATCHPROFILER_MYSQL_DATABASE, BATCHPROFILER_MYSQL_HOST, \
         BATCHPROFILER_MYSQL_PASSWORD, BATCHPROFILER_MYSQL_PORT, \
         BATCHPROFILER_MYSQL_USER

    parser = optparse.OptionParser()
    parser.add_option("-i","--input-sql-script",
                      dest="sql_script",
                      help="The SQL script to run on the server")
    parser.add_option("-b","--batch-id",
                      dest="batch_id",
                      help="The batch ID of the batch being run")
    options,args = parser.parse_args()
    path, filename = os.path.split(options.sql_script)
    if len(path):
        os.chdir(path)
    script_fd = open(filename,"r")
    cmd = ["mysql",
           "-A",
           "-B",
           "--column-names=0",
           "--local-infile=1"]
    if BATCHPROFILER_MYSQL_USER is not None:
        cmd += ["-u", BATCHPROFILER_MYSQL_USER]
    if BATCHPROFILER_MYSQL_PASSWORD is not None:
        cmd += ["-p", BATCHPROFILER_MYSQL_PASSWORD]
    if BATCHPROFILER_MYSQL_HOST is not None:
        cmd += ["-h", BATCHPROFILER_MYSQL_HOST]
    if BATCHPROFILER_MYSQL_DATABASE is not None:
        cmd += ["-D", BATCHPROFILER_MYSQL_DATABASE]
    if BATCHPROFILER_MYSQL_PORT is not None:
        cmd += ["-P", BATCHPROFILER_MYSQL_PORT]
    p = subprocess.Popen(cmd,
                         stdin=script_fd,
                         stdout=sys.stdout,
                         stderr=sys.stderr,
                         cwd = path if len(path) == 0 else None)
    p.communicate()
    script_fd.close()
