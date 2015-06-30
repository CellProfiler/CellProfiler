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
'''As a variable, incomplete is anything but "DONE"'''
JS_INCOMPLETE = "INCOMPLETE"
'''As a variable, all statuses'''
JS_ALL = "ALL"
INCOMPLETE_STATUSES = (JS_SUBMITTED, JS_RUNNING, JS_ERROR, JS_ABORTED)

'''The run type for a CellProfiler run'''
RT_CELLPROFILER = "CellProfiler"

'''The run type for a SQL job'''
RT_SQL = "SQL"

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
            rb_cmd = """
            insert into run_base (batch_id, run_type, command)
            values (%d, '%s', %%s)
            """ % (self.batch_id, RT_CELLPROFILER)
            rc_cmd = """
            insert into run_cellprofiler (run_id, bstart, bend)
            values (last_insert_id(), %s, %s)
            """
            for f, l in runs:
                command = cellprofiler_command(self, f, l)
                cursor.execute(rb_cmd, [command])
                cursor.execute(rc_cmd, [f, l])
                
    @staticmethod
    def select_batch_count():
        '''Return the # of rows in the batch table'''
        with bpcursor() as cursor:
            cursor.execute("select count('x') from batch")
            return cursor.fetchone()[0]
        
    @staticmethod
    def select_batch_range(start, count, desc=True):
        '''Select a range of batches from the batch table
        
        start - one-based index of the first row in the table to return
        count - max # of rows to return
        desc - True to order by largest batch ID first, False for first
        '''
        desc_kwd = "desc" if desc else ""
        cmd = """select batch_id, email, data_dir, queue, batch_size, 
                        write_data, timeout, cpcluster, project, 
                        memory_limit, priority
                   from (select bb.*, @rownum := @rownum + 1 as rank
                           from batch bb, (select @rownum := 0) r
                       order by bb.batch_id %s) b
                 where rank between %d and %d""" % \
             (desc_kwd, start, start+count-1)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            batches = []
            for batch_id, email, data_dir, queue, batch_size, write_data, \
                timeout, cpcluster, project, memory_limit, priority in \
                cursor.fetchall():
                batch = BPBatch()
                batch.batch_id = batch_id
                batch.email = email
                batch.data_dir = data_dir
                batch.queue = queue
                batch.batch_size = batch_size
                batch.write_data = write_data,
                batch.timeout = timeout
                batch.cpcluster = cpcluster
                batch.project = project
                batch.memory_limit = memory_limit
                batch.priority = priority
                batches.append(batch)
            return batches
                                                    
        
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
            select run_id, bstart, bend, command from run where batch_id = %d
            """ % self.batch_id
            cursor.execute(cmd)
            result = []
            for run_id, bstart, bend, command in cursor:
                result.append(BPRun(self.batch_id, run_id, bstart, bend, command))
        return result
    
    def select_job_count(self):
        '''Return the # of jobs with links to the batch through the run tbl'''
        cmd = """select count('x') from run r where r.batch_id = %d
                    and exists (select 'x' from job_status js
                                 where js.run_id = r.run_id)""" % self.batch_id
        with bpcursor() as cursor:
            cursor.execute(cmd)
            return cursor.fetchone()[0]
        
    def select_job_count_group_by_state(self):
        '''Return a dictionary of state and # jobs in that state'''
        cmd = """
        select count('x') as job_count, js.status
        from (select r.run_id as run_id, r.bstart as bstart, r.bend as bend, 
              r.command as command, js.job_id as job_id, 
              max(js.created) as js_created, j.created as j_created
              from run r join job_status js on r.run_id = js.run_id
              join job j on j.run_id = js.run_id and j.job_id = js.job_id
              where r.batch_id = %d
              group by r.run_id, js.job_id) rjs
        join (select r.run_id as run_id, max(j.created) as created
                from run r join job j on r.run_id = j.run_id group by j.run_id) j
        on j.run_id = rjs.run_id and j.created = rjs.j_created
        join job_status js 
        on rjs.run_id = js.run_id and rjs.job_id = js.job_id 
                                  and rjs.js_created = js.created
        group by js.status
        """ % self.batch_id
        with bpcursor() as cursor:
            cursor.execute(cmd)
            return dict([(status, count) for count, status in cursor])
        
    def select_jobs(self, by_status=None, by_run=None, page_size = None,
                    first_item = None):
        '''Get jobs with one of the given statuses
        
        by_status - a sequence of status names to search for (default = all)
        by_run - a run ID to search for (Default = all runs
        page_size - return at most this many items (default all)
        first_item - the one-based index of the first item on the page
                     (default first)
        
        returns a sequence of run, job, status tuples
        '''
        cmd = """
        select rjs.run_id, rjs.bstart, rjs.bend, rjs.command, rjs.job_id, 
               js.status, @rownum:=@rownum+1 as rank
        from (select r.run_id as run_id, r.bstart as bstart, r.bend as bend, 
              r.command as command, js.job_id as job_id, 
              max(js.created) as js_created, j.created as j_created
              from run r join job_status js on r.run_id = js.run_id
              join job j on j.run_id = js.run_id and j.job_id = js.job_id
              where r.batch_id = %d
              group by r.run_id, js.job_id) rjs
        join (select r.run_id as run_id, max(j.created) as created
                from run r join job j on r.run_id = j.run_id group by j.run_id) j
        on j.run_id = rjs.run_id and j.created = rjs.j_created
        join job_status js 
        on rjs.run_id = js.run_id and rjs.job_id = js.job_id 
                                  and rjs.js_created = js.created
        join (select @rownum:=0) as ranktbl
        """ % self.batch_id
        clauses = []
        if by_status is not None:
            clauses.append("js.status in ( '%s' )" % ("','".join(by_status)))
        if by_run is not None:
            clauses.append("rjs.run_id = %d" % by_run)
        if len(clauses) > 0:
            cmd += "        where " + " and ".join(clauses)
        if first_item is not None and page_size is not None:
            cmd = "select * from (%s) cmd where cmd.rank between %d and %d" % (
                cmd, first_item, first_item + page_size - 1)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            result = []
            for run_id, bstart, bend, command, job_id, status, rank in cursor:
                run = BPRun(self.batch_id, run_id, bstart, bend, command)
                job = BPJob(run_id, job_id)
                result.append((run, job, status))
        return result
    
    @property
    def wants_measurements_file(self):
        return int(self.write_data) == 1
              
        
class BPRunBase(object):
    def __init__(self, batch_id, run_id, command):
        self.batch_id = batch_id
        self.run_id = run_id
        self.command = command
        # Should we put . $PREFIX/cpenv.sh into the remote script?
        self.source_cpenv = None
        
    def get_job_name(self):
        '''The name to give a job for this run
        
        e.g. "qsub -N <job-name>"
        '''
        return "CellProfiler.batch%d.%s" % \
               (self.batch_id, self.get_file_name())
    
    def get_file_name(self):
        raise NotImplemented("Use BPRun or BPSQLRun")
    
    @staticmethod
    def select(run_id):
        '''Select a BPRun or BPSQLRun given a run_id
        
        '''
        with bpcursor() as cursor:
            cmd = """select rb.run_type 
            from run_base rb where run_id=%d""" % run_id
            cursor.execute(cmd)
            run_type = cursor.fetchone()[0]
        if run_type == RT_SQL:
            return BPSQLRun.select_by_run_id(run_id)
        return BPRun.select(run_id)
    
    def select_jobs(self, by_status = None):
        cmd = """
            select js.job_id, js.status
            from job_status js
            join job j on js.job_id = j.job_id and js.run_id = j.run_id
            where js.created in
                (select max(js2.created) from job_status js2
                  where js2.run_id = %d
               group by js2.job_id)
              and j.created in 
                (select max(j2.created) from job j2 where j2.run_id = %d)
              and j.run_id = %d
            """ % (self.run_id, self.run_id, self.run_id)
        clauses = []
        if by_status is not None:
            cmd += " and status in ( '%s' )" % ("','".join(args))
        with bpcursor() as cursor:
            cursor.execute(cmd)
            result = []
            for job_id, status in cursor:
                job = BPJob(self.run_id, job_id)
                result.append((job, status))
        return result

class BPRun(BPRunBase):
    def __init__(self, batch_id, run_id, bstart, bend, command):
        super(self.__class__, self).__init__(batch_id, run_id, command)
        self.bstart = bstart
        self.bend = bend
        self.source_cpenv = True
        
    @staticmethod
    def select(run_id):
        cmd = ("select r.batch_id, r.bstart, r.bend, r.command "
               "from run r where run_id = %d" % run_id)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            batch_id, bstart, bend, cmd = cursor.fetchone()
            return BPRun(int(batch_id), run_id, int(bstart), int(bend), cmd)
            
    def get_file_name(self):
        return "%d_to_%d" % (self.bstart, self.bend)

class BPSQLRun(BPRunBase):
    def __init__(self, batch_id, run_id, sql_filename, command):
        super(self.__class__, self).__init__(batch_id, run_id, command)
        self.sql_filename = sql_filename
        self.source_cpenv = False
        
    @staticmethod
    def create(batch, sql_filename, command):
        with bpcursor() as cursor:
            cursor.execute("""
            insert into run_base (batch_id, run_type, command)
            values (%s, %s, %s)""", [batch.batch_id, RT_SQL, command])
            run_id = cursor.lastrowid
            cursor.execute("""
            insert into run_sql (run_id, sql_filename)
            values(last_insert_id(), %s)""", [sql_filename])
            return BPSQLRun(batch.batch_id, run_id, sql_filename, command)
        
    @staticmethod
    def select_by_run_id(run_id):
        with bpcursor() as cursor:
            cursor.execute("""
            select rb.batch_id, rb.command, rs.sql_filename
            from run_base rb join run_sql rs on rb.run_id = rs.run_id
            where rb.run_id = %s and rb.run_type = 'SQL'""", [run_id])
            if cursor.rowcount == 0:
                return None
            batch_id, command, sql_filename = cursor.fetchone()
            return BPSQLRun(int(batch_id), run_id, sql_filename, command)
        
    @staticmethod    
    def select_by_sql_filename(batch, sql_filename):
        with bpcursor() as cursor:
            cursor.execute("""
            select rb.run_id, rb.command
            from run_base rb join run_sql rs on rb.run_id = rs.run_id
            where rs.sql_filename = %s 
              and rb.run_type = 'SQL'
              and rb.batch_id = %s""", [sql_filename, batch.batch_id])
            if cursor.rowcount == 0:
                return None
            run_id, command = cursor.fetchone()
            return BPSQLRun(batch.batch_id, int(run_id), sql_filename, command)
        
    def get_file_name(self):
        fnbase = os.path.splitext(self.sql_filename)[0]
        return "SQL_%s" % fnbase
    
    
class BPJob(object):
    '''A job dispatched on the cluster'''
    def __init__(self, run_id, job_id):
        self.run_id = run_id
        self.job_id = job_id
        
    def create(self, status = JS_SUBMITTED):
        '''Write the job to the database'''
        with bpcursor() as cursor:
            cursor.execute(
                "select count('x') from job j where j.job_id=%s and j.run_id=%s",
                [str(self.job_id), str(self.run_id)])
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "insert into job (job_id, run_id) values (%s, %s)",
                    [str(self.job_id), str(self.run_id)])
            cursor.execute(
                "insert into job_status (job_id, run_id, status) "
                "values (%s, %s, %s)",
                [str(self.job_id), str(self.run_id), status])
    
    def update_status(self, status):
        '''Insert a new status into the database for this job'''
        with bpcursor() as cursor:
            cursor.execute(
                "insert into job_status (run_id, job_id, status) "
                "values (%s, %s, %s)",
                [str(self.run_id), str(self.job_id), status])
            
    @staticmethod
    def select(job_id):
        '''Find a job in the database'''
        with bpcursor() as cursor:
            cursor.execute(
                "select j.run_id from job j "
                "where j.job_id = %s order by created desc limit 1",
                [str(job_id)])
            result = cursor.fetchall()
            if len(result) == 0:
                return
            return BPJob(result[0][0], job_id)

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

def run_one(my_batch, run, cwd = None):
    '''Run the command associated with a batch/run
    
    my_batch - the batch
    run - the run. run.command is the command to run
    cwd - the working directory for the command. Defaults to my_batch.cpcluster
    '''
    assert isinstance(my_batch, BPBatch)
    assert isinstance(run, BPRunBase)
    txt_output = text_file_directory(my_batch)
    if not os.path.exists(txt_output):
        os.mkdir(txt_output)
        os.chmod(txt_output, 
                 stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH |
                 stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    script_dir = script_file_directory(my_batch)
    if not os.path.exists(script_dir):
        os.mkdir(script_dir)
        os.chmod(script_dir, 
                 stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH |
                 stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    script = """#!/bin/sh
export RUN_ID=%d
""" % run.run_id
    #
    # A work-around if HOME has been defined differently on the host
    #
    script += """
if [ ! -z "$SGE_O_HOME" ]; then
    export HOME="$SGE_O_HOME"
    echo "Set home to $HOME"
fi
"""
    #
    # This is a REST PUT to JobStatus.py to create the job record
    #
    script += """curl -v -H "Content-type: application/json" -X PUT \\
--data '{"action":"create","job_id":'$JOB_ID',"run_id":%d,"status":"RUNNING"}'\\
 %s/JobStatus.py\n""" % (run.run_id, BATCHPROFILER_DEFAULTS[URL])
    #
    # CD to the CellProfiler root directory
    #
    if cwd is None:
        cwd = my_batch.cpcluster
    script += 'cd %s\n' % cwd
    #
    # Source cpenv.sh to prepare to run Python
    #
    if run.source_cpenv:
        script += '. %s\n' % os.path.join(PREFIX, "bin", "cpenv.sh")
    #
    # set +e allows the command to error-out without ending this script.
    #        This lets us capture the error status.
    #
    script += "set +e\n"
    #
    # Run CellProfiler
    #
    script += run.command +"\n"
    #
    # Figure out the status from the error code
    #
    script += "if [ $? == 0 ]; then\n"
    script += "JOB_STATUS=%s\n" % JS_DONE
    script += "else\n JOB_STATUS=%s\n " % JS_ERROR
    script += "fi\n"
    #
    # Go back to erroring-out
    #
    script += "set -e\n"
    #
    # Set the status based on the result from CellProfiler
    # Use CURL again
    #
    script += """curl -v -H "Content-type: application/json" -X PUT """
    script += """--data '{"action":"update","job_id":'$JOB_ID',"""
    script += """"run_id":%d,"status":"'$JOB_STATUS'"}' """ % run.run_id
    script += '%s/JobStatus.py\n' % BATCHPROFILER_DEFAULTS[URL]
    script_filename = script_file_path(my_batch, run)
    with open(script_filename, "w") as fd:
        fd.write(script)
    os.chmod(script_filename,stat.S_IWUSR |
             stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH |
             stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    job_id = bputilities.run_on_tgt_os(
        script = script, 
        group_name = my_batch.project,
        job_name = run.get_job_name(),
        queue_name = my_batch.queue,
        priority = my_batch.priority,
        output = run_text_file_path(my_batch, run),
        err_output = run_err_file_path(my_batch, run))
    job = BPJob(run.run_id, job_id)
    job.create()
    return job

def cellprofiler_command(my_batch, bstart, bend):
    script = 'xvfb-run python CellProfiler.py -c -r -b --do-not-fetch '
    script += '-p "%s" ' % os.path.join(my_batch.data_dir, "Batch_data.h5")
    script += '-f %d -l %d ' % (bstart, bend)
    script += '-o "%s"' % my_batch.data_dir
    if my_batch.write_data:
            script += " " + run_out_file_path(my_batch, bstart = bstart, bend=bend)
    script += "\n"
    return script

def kill_one(run):
    batch = BPBatch()
    batch.select(run.batch_id)
    jobs = batch.select_jobs(by_status = [JS_SUBMITTED, JS_RUNNING], 
                             by_run=run.run_id)
    bputilities.kill_jobs([job.job_id for run, job, status in jobs])
    for job in jobs:
        job.update_status(JS_ABORTED)
        
def kill_job(job):
    bputilities.kill_jobs([job.job_id])
    job.update_status(JS_ABORTED)

def kill_batch(batch_id):
    batch = BPBatch()
    batch.select(batch_id)
    jobs = batch.select_jobs(by_status = [JS_SUBMITTED, JS_RUNNING])
    bputilities.kill_jobs([job.job_id for run, job, status in jobs])
    for run, job, status in jobs:
        job.update_status(JS_ABORTED)
        

def run_text_file(run):
    """Return the name of the text file created by bsub
    
    Return the name of the text file created by bsub
    run - instance of BPRun
    """
    assert isinstance(run, BPRunBase)
    return "%s.txt" % run.get_file_name()

def run_err_file(run):
    """Return the name of the stderr output created by bsub
    
    run - instance of BPRun
    """
    assert isinstance(run, BPRunBase)
    return "%s.err.txt" % run.get_file_name()

def text_file_directory(batch):
    return os.path.join(batch.data_dir, "txt_output")

def script_file_directory(batch):
    return os.path.join(batch.data_dir, "job_scripts")

def batch_script_file(script_file):
    '''The name of the SQL script file modded to pull in all of the .CSV files

    script_file - the name of the original file
    '''
    return "batch_%s" % script_file

def batch_script_directory(batch):
    '''The directory housing the modded SQL files
    
    batch - batch in question
    script_file - the name of the original file
    
    Note: this can't be in batch.data_dir because
          it would be automagically scanned and
          picked up by sql_jobs
    '''
    return os.path.join(batch.data_dir, "sql_scripts")
    
def batch_script_path(batch, script_file):
    return os.path.join(batch_script_directory(batch),
                        batch_script_file(script_file))

def script_file_path(batch, run):
    return os.path.join(script_file_directory(batch), 
                        "run_%s.sh" % run.get_file_name())

def run_text_file_path(batch, run):
    """Return the path to the text file created by bsub
    
    batch - the BPBatch
    run - the BPRun
    """
    return os.path.join(text_file_directory(batch), run_text_file(run))

def run_err_file_path(batch, run):
    return os.path.join(text_file_directory(batch), run_err_file(run))

def batch_data_file_path(batch):
    '''Return the path to Batch_data.h5 for this batch'''
    return os.path.join(batch.data_dir, "Batch_data.h5")

def run_out_file(batch, run):
    return run.get_file_name() + ".h5"

def run_out_file_path(batch, run=None, bstart=None, bend=None):
    if run is not None:
        bstart = run.bstart
        bend = run.bend
    return os.path.join(batch.data_dir, 
                        "%d_to_%d.h5" % (bstart, bend))

def GetCPUTime(batch, run):
    '''Get the CPU time in seconds for the completion time of the last job
    
    batch - the batch being queried
    run - the job's last run
    '''
    assert isinstance(batch, BPBatch)
    assert isinstance(run, BPRunBase)
    with bpcursor() as cursor:
        cmd = """
select unix_timestamp(js2.created)-unix_timestamp(js1.created) as cputime
from (select max(j.created) as created from job j where j.run_id=%s) as jc
join job j on j.created=jc.created
join job_status js1 on j.job_id = js1.job_id and j.run_id = js1.run_id
join job_status js2 on j.job_id = js2.job_id and j.run_id = js2.run_id
where js1.status=%s and js2.status in (%s, %s) and j.run_id=%s
"""
        cursor.execute(cmd, [run.run_id, JS_RUNNING, JS_DONE, JS_ERROR, run.run_id])
        return cursor.fetchone()[0]
