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
# Functions for running a batch or a single run from the database
#
import MySQLdb
import subprocess
import os
import re
import stat

import bputilities
from bpformdata import BATCHPROFILER_MYSQL_HOST, \
     BATCHPROFILER_MYSQL_PORT, \
     BATCHPROFILER_MYSQL_USER, \
     BATCHPROFILER_MYSQL_PASSWORD, \
     BATCHPROFILER_MYSQL_DATABASE, \
     BATCHPROFILER_DEFAULTS, URL, PREFIX

JS_SUBMITTED = "SUBMITTED"
JS_RUNNING = "RUNNING"
JS_ERROR = "ERROR"
JS_DONE = "DONE"
JS_ABORTED = "ABORTED"

connect_params = { "user": BATCHPROFILER_MYSQL_USER,
                   "db": BATCHPROFILER_MYSQL_DATABASE }
for k, v in (("host", BATCHPROFILER_MYSQL_HOST),
             ("port", BATCHPROFILER_MYSQL_PORT),
             ("passwd", BATCHPROFILER_MYSQL_PASSWORD)):
    if v is not None:
        connect_params[k] = v
        
connection = MySQLdb.Connect(**connect_params)

class bpcursor(object):
    '''Wrapper for connection's cursor'''
    def __init__(self):
        self.cursor = connection.cursor()
        
    def __enter__(self):
        return self.cursor
    
    def __exit__(self, exctype, value, traceback):
        if exctype is None:
            connection.commit()
        else:
            connection.rollback()
        self.cursor.close()

class BPBatch(object):
    '''Data structure encapsulating overall batch parameters'''
        
    def create(self, email, data_dir, queue, batch_size, 
               write_data, timeout, cpcluster, project, memory_limit,
               priority, runs):
        '''Create a batch in the database
        
        email - mail address of owner
        data_dir - directory holding the Batch_data.h5 file
        queue - submit to this queue
        batch_size - # of image sets / batch
        write_data - 1 to write the measurements file
        timeout (obsolete)
        cpcluster - root directory of CellProfiler
        project - who to charge
        memory_limit - # of mb reserved on cluster node
        priority - priority of job
        runs - a sequence of (start / last / group name) for the runs
               for this batch.
        '''
        self.email = email
        self.data_dir = data_dir
        self.queue = queue
        self.batch_size = batch_size
        self.write_data = write_data
        self.timeout = timeout
        self.cpcluster = cpcluster
        self.project = project
        self.memory_limit = memory_limit
        with bpcursor() as cursor:
            cmd = """
            insert into batch (batch_id, email, data_dir, queue, batch_size, 
                               write_data, timeout, cpcluster, project, memory_limit,
                               priority)
            values (null,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            bindings = [locals()[x] for x in (
                'email', 'data_dir', 'queue','batch_size','write_data','timeout',
                'cpcluster','project','memory_limit', 'priority')]
            cursor.execute(cmd, bindings)
            cursor.execute("select last_insert_id()")
            self.batch_id = cursor.fetchone()[0]
            cmd = """
            insert into run (batch_id, bstart, bend, bgroup)
            values (%d, %%d, %%d, %%s)
            """ % self.batch_id
            cursor.executemany(cmd, runs)
        
    def select(self, batch_id):
        '''Select a batch from the database'''
        self.batch_id = batch_id
        with bpcursor() as cursor:
            cmd = """
            select email, data_dir, queue, batch_size, write_data, timeout,
                   cpcluster, project, memory_limit, priority from batch
                   where batch_id = %d
            """ % self.batch_id
            cursor.execute(cmd)
            self.email, self.data_dir, self.queue, self.batch_size, \
                self.write_data, self.timeout, self.cpcluster, self.project, \
                self.memory_limit, self.priority = cursor.fetchone()
        
    def select_runs(self):
        '''Select the associated runs from the database
        
        Returns a list of BPRun records
        '''
        with bpcursor() as cursor:
            cmd = """
            select run_id, bstart, bend, bgroup from run where batch_id = %d
            """ % self.batch_id
            result = []
            for run_id, bstart, bend, bgroup in cursor:
                result.append(BPRun(self.batch_id, run_id, bstart, bend, bgroup))
        return result
    
    def select_jobs(self, by_status=None, by_run=None):
        '''Get jobs with one of the given statuses
        
        args - the statuses to fetch
        
        returns a sequence of run, job, status tuples
        '''
        cmd = """
        select rjs.run_id, rjs.bstart, rjs.bend, rjs.bgroup, rjs.job_id, 
               js.status
        from (select r.run_id as run_id, r.bstart as bstart, r.bend as bend, 
              r.bgroup as bgroup, js.job_id as job_id, max(js.created) as when
              from run r join job_status js on r.run_id = js.run_id
              where r.batch_id = %d
              group by run_id, job_id) rjs 
        join job_status js 
        on rjs.run_id = js.run_id and rjs.job_id = js.job_id 
                                  and rjs.created = js.created
        """ % self.batch_id
        clauses = []
        if by_statuses is not None:
            clauses.append("status in ( '%s' )" % ("','".join(args)))
        if by_run is not None:
            clauses.append("run_id = %d" % by_run)
        if len(clauses) > 0:
            cmd += "        where " + " and ".join(clauses)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            result = []
            for run_id, bstart, bend, bgroup, job_id, status in cursor:
                run = BPRun(self.batch_id, run_id, bstart, bend, bgroup)
                job = BPJob(run_id, job_id)
                result.append(run, job, status)
        return result
              
        
class BPRun(object):
    def __init__(self, batch_id, run_id, bstart, bend, bgroup=None):
        self.batch_id = batch_id
        self.run_id = run_id
        self.bstart = bstart
        self.bend = bend
        self.bgroup = bgroup
        
    def get_command_line_params(self):
        '''Return the CP parameters that select the desired image set range'''
        return ["-f", self.start, "-l", self.last, "-N", self.get_job_name()]
    
    def get_job_name(self):
        '''The name to give a job for this run
        
        e.g. "qsub -N <job-name>"
        '''
        return "CellProfiler.batch%d.%dto%d" % \
               (self.batch_id, self.bstart, self.bend)
    

class BPJob(object):
    '''A job dispatched on the cluster'''
    def __init__(self, run_id, job_id):
        self.run_id = run_id
        self.job_id = job_id
        
    def create(self):
        '''Write the job to the database'''
        with bpcursor() as cursor:
            cursor.execute("insert into job (job_id, run_id) values (%d, %d)",
                           [self.job_id, self.run_id])
            cursor.execute(
                "insert into job_status (job_id, run_id, status) "
                "values (%d, %d, %s)",
                [self.job_id, self.run_id, JS_SUBMITTED])
    
    def update_status(self, status):
        '''Insert a new status into the database for this job'''
        with bpcursor() as cursor:
            cursor.execute(
                "insert into job_status (job_id, status) values (%d, %s)",
                [self.job_id, status])

def run_all(batch_id):
    """Submit jobs for all imagesets in the batch
    
    Load the batch with the given batch_id from the database
    and run each using CPCluster and bsub
    """
    my_batch = BPBatch()
    my_batch.select(batch_id)
    txt_output = os.path.join(my_batch.data_dir, "txt_output")
    scripts = os.path.join(my_batch.data_dir, "scripts")
    if not os.path.exists(txt_output):
        os.mkdir(txt_output)
    if not os.path.exists(scripts):
        os.mkdir(scripts)
        os.chmod(scripts, stat.S_IWUSR | stat.S_IREAD)
    response = []
    for run in my_batch.select_runs():
        run_response = run_one(my_batch, run)
        response.append(run_response)
    return response

def run_one(my_batch, run):
    assert isinstance(my_batch, BPBatch)
    assert isinstance(run, BPRun)
    txt_output = os.path.join(my_batch.data_dir, "txt_output")
    script = """"#!/bin/sh
export RUN_ID=%d
""" % run.run_id
    #
    # This is a REST PUT to JobStatus.py to create the job record
    #
    script += 'curl -v -H "Content-type: application/json" -X PUT '
    script += '--data "{\\"action\\":\\"create\\",\\"job_id\\":$JOB_ID,'
    script +='\\"run_id\\":%d}" ' % run.run_id
    script += '%s/JobStatus.py\n' % BATCHPROFILER_DEFAULTS[URL]
    #
    # CD to the CellProfiler root directory
    #
    script += 'cd %s\n' % my_batch.cpcluster
    #
    # Source cpenv.sh to prepare to run Python
    #
    script += '. %s\n' % os.path.join(PREFIX, "cpenv.sh")
    #
    # Run CellProfiler
    #
    script += 'xvfb-run python CellProfiler.py -c -r -b --do-not-fetch '
    script += '-p "%s" ' % os.path.join(my_batch.data_dir, "Batch_data.h5")
    script += '-f %d -l %d ' % (run.bstart, run.bend)
    script += '-o %s' % my_batch.data_dir
    if my_batch.write_data:
            script += os.path.join(my_batch.data_dir, 
                                   "%d_to_%d.h5" % (run.bstart, run.bend))
    script += "\n"
    #
    # Figure out the status from the error code
    #
    script += "if [ $? == 0 ]; then JOB_STATUS=%s;" % JS_GOOD
    script += "else JOB_STATUS=%s; " % JS_ERROR
    #
    # Set the status based on the result from CellProfiler
    # Use CURL again
    #
    script += 'curl -v -H "Content-type: application/json" -X PUT '
    script += '--data "{\\"action\\":\\"update\\",\\"job_id\\":$JOB_ID,'
    script +='\\"run_id\\":%d\\"status\\":\\"$JOB_STATUS\\"}" ' % run.run_id
    script += '%s/JobStatus.py\n' % BATCHPROFILER_DEFAULTS[URL]
    bputilities.run_on_tgt_os(
        script, 
        my_batch.project,
        run.get_job_name(),
        my_batch.queue,
        os.path.join(txt_output, "%d_to_%d.txt" % run.bstart, run.bend))

def kill_one(run):
    batch = BPBatch()
    batch.select(run.batch_id)
    jobs = batch.select_jobs(by_status = [JS_RUNNING], by_run=run.run_id)
    bputilities.kill_jobs([job.job_id for job in jobs])

def kill_batch(batch_id):
    batch = BPBatch()
    batch.select(batch_id)
    jobs = batch.select_jobs(by_status = [JS_RUNNING])
    bputilities.kill_jobs([job.job_id for job in jobs])

def run_text_file(run):
    """Return the name of the text file created by bsub
    
    Return the name of the text file created by bsub
    run - instance of BPRun
    """
    assert isinstance(run, BPRun)
    return "%d_to_%d.txt"%(run.bstart, run.bend)

def run_text_file_path(batch, run):
    """Return the path to the text file created by bsub
    
    batch - the BPBatch
    run - the BPRun
    """
    return os.path.join(batch.data_dir, run_text_file(run))

def GetCPUTime(batch, run):
    try:
        text_file = open(RunTextFilePath(batch, run),"r")
        text = text_file.read()
        text_file.close()
        match = re.compile(".*\s+CPU time\s+:\s+([0-9.]+)\s+sec",
                           re.DOTALL).search(text)
        return float(match.group(1))
    except:
        return

