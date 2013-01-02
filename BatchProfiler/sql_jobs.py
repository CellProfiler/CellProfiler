#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/development/python-2.6.sh
"""sql_jobs.py - launch and monitor remote jobs that submit SQL

This file is a command-line script for running jobs and a set of functions
for running the script on the cluster. To get help on running from the
command-line:

sql_jobs.py --help

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import re
import os
import subprocess
import sys

python_path = os.path.split(__file__)[0]
sys.path.append(python_path)
from RunBatch import connection, GetJobStatus

def run_sql_file(batch_id, sql_path, stdout_path, queue="hour", project="imaging"):
    """Use the mysql command line to run the given SQL script
    
    batch_id    - sql_file is associated with this batch
    sql_path    - path and file of the sql script
    stdout_path - redirect output here
    queue       - run job on this queue
    
    returns bsub job id
    """
    
    # We start the job in a suspended state in order to enter
    # the job status into the database before the job starts
    #
    if os.path.isfile(stdout_path):
        os.remove(stdout_path)
    #
    # DK_ROOT is needed to point at the dotkit which supplies the "use"
    # function to the shell
    #
    # The "use" function is needed to configure Python 2.6 so we can
    # use MySQLDB
    #
    # We set PYTHON_EGG_CACHE to the web-server's egg cache in order
    # to get a writeable directory
    #
    if not os.environ.has_key("DK_ROOT"):
        os.environ["DK_ROOT"] = "/broad/tools/dotkit"
    remote_cmd = ("source /broad/tools/dotkit/ksh/.dk_init;"
                  "use Python-2.6;"
                  "export PYTHON_EGG_CACHE=/imaging/analysis/People/imageweb/python_egg_cache;"
                  "python %s -b %d -i %s"%(__file__,batch_id, sql_path))
    cmd=[". /broad/lsf/conf/profile.lsf;",
         "bsub",
         "-H",
         "-q","%(queue)s"%(locals()),
         "-g","/imaging/batch/%(batch_id)d"%(locals()),
         "-M","500000",
         "-P","%(project)s"%(locals()),
         "-L","/bin/bash",
         "-o",stdout_path,
         """ "%s" """%(remote_cmd)
         ]
    cmd = " ".join(cmd)
    p=os.popen(cmd)
    output=p.read()
    exit_code=p.close()
    job=None
    if output:
        match = re.search(r'<([0-9]+)>',output)
        if len(match.groups()) > 0:
            job=int(match.groups()[0])
        else:
            raise RuntimeError("Failed to start job. Output is as follows: %s"%(output))
    else:
        raise RuntimeError("Failed to start job: No output from bsub")
    cursor = connection.cursor()
    statement = """insert into sql_job
                   (job_id, batch_id, sql_file)
                   values (%(job)d, %(batch_id)d, '%(sql_path)s')"""%(locals())
    cursor.execute(statement)
    cursor.close()
    cmd = ["bresume", "%d"%(job)]
    cmd = " ".join(cmd)
    p=os.popen(". /broad/lsf/conf/profile.lsf;"+cmd,'r')
    output=p.read()
    exit_code=p.close()
    return job

def sql_file_job_id(batch_id, sql_path):
    """Return the latest job ID associated with the batch and sql path
    
    batch_id - batch id associated with the submission
    sql_path - path to the sql script submitted
    
    returns latest job ID or None if not submitted
    """ 
    statement = """select job_id, start_timestamp, stop_timestamp
                     from sql_job where batch_id = %(batch_id)d
                      and sql_path = '%(sql_path)s
                    order by start_timestamp desc
                    limit 1"""
    cursor = connection.cursor()
    try:
        statement = """select job_id, start_timestamp, stop_timestamp
                         from sql_job where batch_id = %(batch_id)d
                          and sql_file like '%%%(sql_path)s'
                        order by start_timestamp desc
                        limit 1"""%(locals())
        rowcount = cursor.execute(statement)
        if rowcount == 0:
            return None
        job_id, start_timestamp, stop_timestamp = cursor.fetchone()
    finally:
        cursor.close()
    return job_id
        
def sql_job_status(job_id):
    """Return the job status for the last submission attempt of this sql script
    
    job_id - the job_id, for instance returned by sql_file_job_id
    
    returns either the result of bjobs for the latest job submitted or
    "DONE" if the job is marked as done in the database.
    Raises an exception if no job is found.
    """
    cursor = connection.cursor()
    try:
        statement = """select stop_timestamp
                         from sql_job where job_id = %(job_id)d"""%(locals())
        rowcount = cursor.execute(statement)
        if rowcount == 0:
            raise ValueError("Can't find job with job_id = %(job_id)d in the database"%(locals()))
        (stop_timestamp,) = cursor.fetchone()
    finally:
        cursor.close()
    
    if not stop_timestamp is None:
        return "DONE"
    
    status = GetJobStatus(job_id)
    if status is None:
        # Job didn't report to database and isn't visible
        return "UNKNOWN"
    if not status.has_key("STAT"):
        return "UNKNOWN"
    return status["STAT"]

def sql_job_run_time(job_id):
    """Return a time-delta of running time of the given job
    
    job_id - job id, for instance as returned by sql_file_job_id
    
    returns a datetime.timedelta of the time between start and stop
    or None if no stop timestamp is recorded in the database
    """
    cursor = connection.cursor()
    try:
        statement = """select start_timestamp, stop_timestamp
                         from sql_job where job_id = %(job_id)d"""%(locals())
        rowcount = cursor.execute(statement)
        if rowcount == 0:
            raise ValueError("Can't find job with job_id = %(job_id)d in the database"%(locals()))
        start_timestamp, stop_timestamp = cursor.fetchone()
    finally:
        cursor.close()
    if stop_timestamp is None:
        return None
    return stop_timestamp - start_timestamp
    
if __name__ == "__main__":
    import optparse
    from RunBatch import batchprofiler_db, batchprofiler_host
    from RunBatch import batchprofiler_password, batchprofiler_user

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
    p = subprocess.Popen(["mysql",
                          "-A",
                          "-B",
                          "--column-names=0",
                          "--local-infile=1",
                          "-u",batchprofiler_user,
                          "--password=%s"%(batchprofiler_password),
                          "-h",batchprofiler_host,
                          "-D",batchprofiler_db],
                          stdin=script_fd,
                          stdout=sys.stdout)
    p.communicate()
    script_fd.close()
    cursor = connection.cursor()
    cursor.execute("""update sql_job 
                         set stop_timestamp=current_timestamp
                       where batch_id = %s
                         and sql_file = '%s'
                         and stop_timestamp is null"""%(options.batch_id, options.sql_script))
    cursor.close()
    os.remove(filename)
