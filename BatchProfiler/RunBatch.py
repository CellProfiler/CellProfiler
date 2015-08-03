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
     BATCHPROFILER_DEFAULTS, URL, PREFIX, K_ACTION, K_HOST_NAME, K_STATUS, \
     K_WANTS_XVFB, A_CREATE, A_UPDATE, JOB_ID, RUN_ID, BATCH_ARRAY_ID, TASK_ID

JS_USE_REST = False

TASK_ID_START = 1
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
    def __init__(self, batch_id, email, data_dir, queue, batch_size,
                 write_data, timeout, cpcluster, project, memory_limit,
                 priority):
        self.batch_id = batch_id
        self.email = email
        self.data_dir = data_dir
        self.queue = queue
        self.batch_size = batch_size
        self.write_data = write_data
        self.timeout = timeout
        self.cpcluster = cpcluster
        self.project = project
        self.memory_limit = memory_limit
        self.priority = priority
        
    @staticmethod
    def create(cursor, email, data_dir, queue, batch_size, 
               write_data, timeout, cpcluster, project, memory_limit,
               priority):
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
        '''
        cmd = """
        insert into batch (batch_id, email, data_dir, queue, batch_size, 
            write_data, timeout, cpcluster, project, memory_limit,
            priority)
        values (null,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        bindings = [locals()[x] for x in (
                'email', 'data_dir', 'queue','batch_size','write_data','timeout',
                'cpcluster','project','memory_limit', 'priority')]
        cursor.execute(cmd, bindings)
        batch_id = cursor.lastrowid
        return BPBatch(batch_id = batch_id,
                       email = email,
                       data_dir = data_dir,
                       queue = queue,
                       batch_size = batch_size,
                       write_data = write_data,
                       timeout = timeout,
                       cpcluster = cpcluster,
                       project = project,
                       memory_limit = memory_limit,
                       priority = priority)
                
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
                                                    
    @staticmethod    
    def select(batch_id):
        '''Select a batch from the database'''
        with bpcursor() as cursor:
            cmd = """
            select email, data_dir, queue, batch_size, write_data, timeout,
                   cpcluster, project, memory_limit, priority from batch
                   where batch_id = %d
            """ % batch_id
            cursor.execute(cmd)
            email, data_dir, queue, batch_size, \
                write_data, timeout, cpcluster, project, \
                memory_limit, priority  = cursor.fetchone()
            batch = BPBatch(batch_id, email, data_dir, queue, batch_size,
                write_data, timeout, cpcluster, project,
                memory_limit, priority)
            return batch
        
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
                result.append(BPRun(self, run_id, bstart, bend, command))
        return result
    
    def select_task_count(self, run_type):
        '''Return the # of jobs with links to the batch through the run tbl'''
        cmd = """
        select count('x') 
          from run_job_status r 
         where r.batch_id = %d and r.run_type='%s'""" % (self.batch_id, run_type)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            return cursor.fetchone()[0]
        
    def select_task_count_by_state(self, run_type, state):
        '''Return the # of jobs in a particular state'''
        cmd = """
        select count('x') from run_job_status r
         where r.batch_id = %d and r.run_type='%s' and r.status='%s'
        """ % (self.batch_id, run_type, state)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            return cursor.fetchone()[0]
        
    def select_task_count_group_by_state(self, run_type):
        '''Return a dictionary of state and # jobs in that state'''
        cmd = """
        select count('x') as job_count, rjs.status
        from run_job_status rjs where batch_id = %d and run_type = '%s'
        group by rjs.status
        """ % (self.batch_id, run_type)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            return dict([(status, count) for count, status in cursor])
        
    def select_tasks(self, page_size = None, first_item = None, state=None):
        '''Get tasks for cellprofiler run records
        
        page_size - return at most this many items (default all)
        first_item - the one-based index of the first item on the page
                     (default first)
        state - one of the states, for instance "RUNNING"
        
        returns a sequence of BPTaskStatus records
        '''
        cmd = """
        select rjs.run_id, rjs.bstart, rjs.bend, rjs.command, 
               rjs.batch_array_id, rjs.batch_array_task_id, rjs.task_id,
               rjs.job_record_id, rjs.job_id, rjs.job_created,
               rjs.job_task_id, rjs.task_status_id, rjs.status, 
               rjs.status_updated, @rownum:=@rownum+1 as rank
        from run_job_status rjs
        join (select @rownum:=0) as ranktbl
        where rjs.batch_id = %d
        """ % self.batch_id
        if state is not None:
            cmd += " and rjs.status = '%s'" % state
        cmd += "\n         order by rjs.bstart"

        if first_item is not None and page_size is not None:
            cmd = "select * from (%s) cmd where cmd.rank between %d and %d" % (
                cmd, first_item, first_item + page_size - 1)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            runs = {}
            batch_arrays = {}
            jobs = {}
            result = []
            for run_id, bstart, bend, command, \
               batch_array_id, batch_array_task_id, task_id, \
               job_record_id, job_id, job_created, \
               job_task_id, task_status_id, status, \
               status_updated, rank in cursor:
                if not run_id in runs:
                    runs[run_id] = BPRun(self, run_id, bstart, bend, command)
                if not batch_array_id in batch_arrays:
                    batch_arrays[batch_array_id] = \
                        BPBatchArray(self, batch_array_id)
                if not job_record_id in jobs:
                    jobs[job_record_id] = \
                        BPJob(job_record_id, job_id, 
                              batch_arrays[batch_array_id], job_created)
                batch_array_task = BPBatchArrayTask(
                    batch_array_task_id, batch_arrays[batch_array_id],
                    runs[run_id], task_id)
                job_task = BPJobTask(job_task_id, jobs[job_record_id],
                                     batch_array_task)
                result.append(BPJobTaskStatus(
                    task_status_id, job_task, status, status_updated))
        return result
    
    def select_queued_tasks(self):
        '''Get tasks that are in the running or submitted states
        
        returns a sequence of BPTask records
        '''
        return self.select_tasks_by_state((JS_SUBMITTED, JS_RUNNING))
    
    def select_tasks_by_state(self, states):
        cmd = """
        select rjs.run_id, rjs.run_type, rjs.bstart, rjs.bend, 
               rjs.sql_filename, rjs.command, 
               rjs.batch_array_id, rjs.batch_array_task_id, rjs.task_id,
               rjs.job_record_id, rjs.job_id, rjs.job_created,
               rjs.job_task_id
        from run_job_status rjs
        where rjs.batch_id = %d and rjs.status in ('%s')
        order by rjs.bstart
        """ % (self.batch_id, "','".join(states))
        with bpcursor() as cursor:
            cursor.execute(cmd)
            runs = {}
            batch_arrays = {}
            jobs = {}
            result = []
            for run_id, run_type, bstart, bend, sql_filename, command, \
               batch_array_id, batch_array_task_id, task_id, \
               job_record_id, job_id, created, \
               job_task_id in cursor:
                if not run_id in runs:
                    runs[run_id] = BPRun(self, run_id, bstart, bend, command)
                if not batch_array_id in batch_arrays:
                    batch_arrays[batch_array_id] = \
                        BPBatchArray(self, batch_array_id)
                if not job_record_id in jobs:
                    jobs[job_record_id] = \
                        BPJob(job_record_id, job_id, 
                              batch_arrays[batch_array_id], created)
                batch_array_task = BPBatchArrayTask(
                    batch_array_task_id, batch_arrays[batch_array_id],
                    runs[run_id], task_id)
                job_task = BPJobTask(job_task_id, jobs[job_record_id],
                                     batch_array_task)
                result.append(job_task)
        return result

    @property
    def wants_measurements_file(self):
        return int(self.write_data) == 1
              
        
class BPRunBase(object):
    def __init__(self, batch, run_id, command):
        self.batch = batch
        self.run_id = run_id
        self.command = command
        # Should we put . $PREFIX/cpenv.sh into the remote script?
        self.source_cpenv = None
        
    def get_file_name(self):
        raise NotImplemented("Use BPRun or BPSQLRun")
    
    @staticmethod
    def select(run_id, batch = None):
        '''Select a BPRun or BPSQLRun given a run_id
        
        '''
        with bpcursor() as cursor:
            cmd = """
            select batch_id, run_type, command, bstart, bend, sql_filename 
              from run where run_id=%d""" % run_id
            cursor.execute(cmd)
            batch_id, run_type, command, bstart, bend, sql_filename =\
                cursor.fetchone()
            if batch is None:
                batch = BPBatch.select(batch_id)
        if run_type == RT_SQL:
            return BPSQLRun(batch, run_id, sql_filename, command)
        return BPRun(batch, run_id, bstart, bend, command)
    

class BPRun(BPRunBase):
    def __init__(self, batch, run_id, bstart, bend, command):
        super(self.__class__, self).__init__(batch, run_id, command)
        self.bstart = bstart
        self.bend = bend
        self.source_cpenv = True
        self.wants_xvfb = True
        
    @staticmethod
    def create(cursor, batch, bstart, bend, command):
        cmd = """
        insert into run_base (batch_id, run_type, command)
        values (%s, %s, %s)
        """
        cursor.execute(cmd, [str(batch.batch_id), RT_CELLPROFILER, command])
        run_id = cursor.lastrowid
        cmd = """
        insert into run_cellprofiler (run_id, bstart, bend)
        values (%s, %s, %s)"""
        cursor.execute(cmd, [str(run_id), bstart, bend])
        return BPRun(batch, run_id, bstart, bend, command)
    
    def get_file_name(self):
        return "%d_to_%d" % (self.bstart, self.bend)

class BPSQLRun(BPRunBase):
    def __init__(self, batch, run_id, sql_filename, command):
        super(self.__class__, self).__init__(batch, run_id, command)
        self.sql_filename = sql_filename
        self.source_cpenv = False
        self.wants_xvfb = False
        
    @staticmethod
    def create(cursor, batch, sql_filename, command):
        cursor.execute("""
        insert into run_base (batch_id, run_type, command)
        values (%s, %s, %s)""", [str(batch.batch_id), RT_SQL, command])
        run_id = cursor.lastrowid
        cursor.execute("""
        insert into run_sql (run_id, sql_filename)
        values(last_insert_id(), %s)""", [sql_filename])
        return BPSQLRun(batch, run_id, sql_filename, command)
        
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
            return BPSQLRun(batch, int(run_id), sql_filename, command)
        
    def get_file_name(self):
        fnbase = os.path.splitext(self.sql_filename)[0]
        return "SQL_%s" % fnbase
    
class BPBatchArray(object):
    def __init__(self, batch, batch_array_id):
        self.batch = batch
        self.batch_array_id = batch_array_id

    def get_job_name(self):
        '''The name to give a job for this batch array
        
        e.g. "qsub -N <job-name>"
        '''
        return "CellProfiler.batch%d.%d" % \
               (self.batch.batch_id, self.batch_array_id)
    
        
    @staticmethod
    def select(batch_array_id):
        with bpcursor() as cursor:
            cmd = """
            select batch_id from batch_array 
             where batch_array_id=%d""" % batch_array_id
            cursor.execute(cmd)
            if cursor.rowcount == 0:
                return None
            batch_id = cursor.fetchone()[0]
        return BPBatchArray(BPBatch.select(batch_id), batch_array_id)
    
    @staticmethod
    def create(cursor, batch):
        cmd = "insert into batch_array (batch_id) values (%s)"
        cursor.execute(cmd, [str(batch.batch_id)])
        return BPBatchArray(batch, int(cursor.lastrowid))
        
class BPBatchArrayTask(object):
    def __init__(self, batch_array_task_id, batch_array, run, task_id):
        self.batch_array_task_id = batch_array_task_id
        self.batch_array = batch_array
        self.run = run
        self.task_id = task_id
        
    @staticmethod
    def create(cursor, batch_array, run, task_id):
        cmd = """
        insert into batch_array_task (batch_array_id, task_id, run_id)
        values (%s, %s, %s)
        """
        cursor.execute(cmd, [str(batch_array.batch_array_id),
                             str(task_id), str(run.run_id)])
        batch_array_task_id = cursor.lastrowid
        return BPBatchArrayTask(batch_array_task_id, batch_array, run, task_id)
    
    @staticmethod
    def select_by_batch_array_and_task_id(batch_array, task_id):
        with bpcursor() as cursor:
            cmd = """
            select bat.batch_array_task_id, bat.run_id,
              r.run_type, r.command, r.bstart, r.bend, r.sql_filename
              from batch_array_task bat
              join run r on r.run_id = bat.run_id
             where bat.batch_array_id = %d and bat.task_id = %d
            """ % (batch_array.batch_array_id, task_id)
            cursor.execute(cmd)
            batch_array_task_id, run_id, run_type, command, bstart, bend,\
                sql_filename = cursor.fetchone()
            if run_type == RT_SQL:
                run = BPSQLRun(
                    batch_array.batch, int(run_id), sql_filename, command)
            else:
                run =  BPRun(
                    batch_array.batch, int(run_id), int(bstart), int(bend),
                    command)
            return BPBatchArrayTask(
                batch_array_task_id, batch_array, run, task_id)
        
    @staticmethod
    def select_by_run(run):
        with bpcursor() as cursor:
            cmd = """
            select batch_array_id, batch_array_task_id, task_id
              from batch_array_task
             where run_id = %d
            """ % run.run_id
            cursor.execute(cmd)
            bats = []
            bas = {}
            for batch_array_id, batch_array_task_id, task_id in cursor:
                if batch_array_id not in bas:
                    bas[batch_array_id] = \
                        BPBatchArray(run.batch, batch_array_id)
                bats.append(BPBatchArrayTask(
                    batch_array_task_id, bas[batch_array_id], run, task_id))
            return bats

class BPJobTask(object):
    def __init__(self, job_task_id, job, batch_array_task):
        self.job_task_id = job_task_id
        self.job = job
        self.batch_array_task = batch_array_task
        
    @staticmethod
    def create(cursor, job, batch_array_task):
        cmd = """
        insert into job_task (job_record_id, batch_array_task_id)
        values (%s, %s)
        """
        cursor.execute(cmd, [job.job_record_id, batch_array_task.batch_array_task_id])
        return BPJobTask(cursor.lastrowid, job, batch_array_task)
    
    @staticmethod
    def select_by_task_id(job, task_id):
        cmd = """
            select rjs.run_id, rjs.run_type, rjs.command, rjs.bstart, rjs.bend,
                   rjs.sql_filename, rjs.batch_array_task_id, 
                   rjs.job_task_id
             from run_job_status rjs
            where rjs.task_id = %d and rjs.job_record_id = %d
            """ % (task_id, job.job_record_id)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            if cursor.rowcount == 0:
                return
            run_id, run_type, command, bstart, bend, sql_filename, \
                batch_array_task_id, job_task_id = cursor.fetchone()
        if run_type == RT_SQL:
            run = BPSQLRun(
                job.batch_array.batch, run_id, sql_filename, command)
        else:
            run = BPRun(job.batch_array.batch, run_id, bstart, bend, command)
        bat = BPBatchArrayTask(batch_array_task_id, job.batch_array, run,
                               task_id)
        return BPJobTask(job_task_id, job, bat)
    
    @staticmethod
    def select_by_run(run):
        '''Find the submitted or running tasks, if any, for a run'''
        cmd = """
            select rjs.batch_array_id, rjs.batch_array_task_id, rjs.task_id,
                   rjs.job_record_id, rjs.job_id, rjs.job_created,
                   rjs.job_task_id
             from run_job_status rjs
            where rjs.run_id = %s and (rjs.status = %s or rjs.status = %s)
            """
        result = []
        with bpcursor() as cursor:
            cursor.execute(cmd, (run.run_id, JS_SUBMITTED, JS_RUNNING))
            batch_arrays = {}
            for batch_array_id, batch_array_task_id, task_id, \
                job_record_id, job_id, created, job_task_id in cursor:
                if batch_array_id not in batch_arrays:
                    batch_arrays[batch_array_id] = BPBatchArray(
                        run.batch, batch_array_id)
                bat = BPBatchArrayTask(
                    batch_array_task_id, batch_array, run, task_id)
                job = BPJob(job_record_id, job_id, batch_array, created)
                result.append(BPJobTask(job_task_id, job, bat))
            return result
               
        
class BPJobTaskStatus(object):
    def __init__(self, job_task_status_id, job_task, status, created):
        self.task_status_id = job_task_status_id
        self.job_task = job_task
        self.status = status
        self.created = created
        
    @staticmethod
    def create(cursor, job_task, status):
        cmd = """
        insert into task_status (job_task_id, status)
        values (%s, %s)
        """
        cursor.execute(cmd, [str(job_task.job_task_id), status])
        task_status_id = cursor.lastrowid
       
        cursor.execute(
            "select created from task_status where task_status_id=%d" %
            task_status_id)
        return BPJobTaskStatus(task_status_id, job_task, status,
                               cursor.fetchone()[0])
    
    @staticmethod
    def select_by_job_task(job_task):
        cmd = """
        select task_status_id, status, status_updated from run_job_status
         where job_task_id = %d""" % job_task.job_task_id
        with bpcursor() as cursor:
            cursor.execute(cmd)
            if cursor.rowcount == 0:
                return None
            task_status_id, status, created = cursor.fetchone()
            return BPJobTaskStatus(task_status_id, job_task, status, created)
        
class BPJob(object):
    '''A job dispatched on the cluster'''
    def __init__(self, job_record_id, job_id, batch_array, created):
        self.job_record_id = job_record_id
        self.job_id = job_id
        self.batch_array = batch_array
        self.created = created
        
    @staticmethod
    def select_by_job_id(job_id):
        cmd = """
        select job_record_id, batch_array_id, created from job j
        where job_id = %d
          and not exists
              (select 'x' from job j2 
                where j2.job_id = j.job_id and j2.created > j.created)
                """ % job_id
        with bpcursor() as cursor:
            cursor.execute(cmd)
            if cursor.rowcount == 0:
                return None
            job_record_id, batch_array_id, created = cursor.fetchone()
        batch_array = BPBatchArray.select(batch_array_id)
        return BPJob(job_record_id, job_id, batch_array, created)
    
    @staticmethod
    def select(batch_array, job_id):
        cmd = """
        select job_record_id, created from job j
        where job_id = %d and batch_array_id = %d""" % \
            (job_id, batch_array.batch_array_id)
        with bpcursor() as cursor:
            cursor.execute(cmd)
            if cursor.rowcount == 0:
                return None
            job_record_id, created = cursor.fetchone()
            return BPJob(job_record_id, job_id, batch_array, created)
        
        
    @staticmethod
    def create(cursor, batch_array, job_id):
        '''Write the job to the database
        
        '''
        cmd = """
        insert into job (job_id, batch_array_id)
        select %s, %s from dual
         where not exists 
              (select 'x' from job where job_id = %s and batch_array_id = %s)
        """
        cursor.execute(
            cmd, [str(job_id), str(batch_array.batch_array_id)] * 2)
        cursor.execute(
            """select job_record_id, created 
                 from job where job_id=%s and batch_array_id=%s""",
            (str(job_id), str(batch_array.batch_array_id)))
        job_record_id, created = cursor.fetchone()
        return BPJob(job_record_id, job_id, batch_array, created)
    
    def select_queued_tasks(self):
        '''Select tasks in the submitted or running states'''
        cmd = """
        select batch_array_task_id, task_id, 
               run_id, run_type, command, bstart, bend, sql_filename,
               job_task_id
          from run_job_status 
         where job_record_id = %s
           and (status = %s or status = %s)"""
        with bpcursor() as cursor:
            cursor.execute(
                cmd, (str(self.job_record_id), JS_SUBMITTED, JS_RUNNING))
            result = []
            for batch_array_task_id, task_id, \
                run_id, run_type, command, bstart, bend, sql_filename,\
                job_task_id in cursor:
                if run_type == RT_SQL:
                    run = BPSQLRun(
                        self.batch_array.batch, run_id, sql_filename, command)
                else:
                    run = BPRun(
                        self.batch_array.batch, run_id, bstart, bend, command)
                bat = BPBatchArrayTask(batch_array_task_id, self.batch_array,
                                       run, task_id)
                result.append(BPJobTask(job_task_id, self, bat))
            return result

class BPTaskHost(object):
    def __init__(self, job_task, host_name, xvfb_server):
        self.job_task = job_task
        self.host_name = host_name
        self.xvfb_server = xvfb_server
    
    @staticmethod    
    def create(cursor, job_task, host_name, wants_xvfb):
        if wants_xvfb:
            cmd = """
            insert into task_host (job_task_id, hostname, xvfb_server)
            select %d, '%s', ifnull(max(jh.xvfb_server)+1, 99)
              from task_host jh
              join task_status ts 
                on ts.job_task_id = jh.job_task_id
             where jh.hostname = '%s'
               and ts.status = '%s'
               and not exists
                   (select 'x' 
                      from task_status ts1
                     where ts1.created > ts.created
                       and ts1.job_task_id = ts.job_task_id)
            """ % (job_task.job_task_id, host_name, host_name, JS_RUNNING)
        else:
            cmd = """
            insert into task_host (job_task_id, hostname)
            values (%d, %s)""" % (job_task.job_task_id, host_name)
        cursor.execute(cmd)
        if wants_xvfb:
            cursor.execute(
                """select xvfb_server from task_host
                where job_task_id = %s""", (job_task.job_task_id,))
            return BPTaskHost(job_task, host_name, cursor.fetchone()[0])
        else:
            return BPTaskHost(job_task, host_name, None)
            
def run(batch, runs, cwd = None):
    """Submit a job array for the given runs
    
    """
    txt_output = text_file_directory(batch)
    scripts = script_file_directory(batch)
    if not os.path.exists(txt_output):
        os.mkdir(txt_output)
        os.chmod(txt_output, 0777)
    if not os.path.exists(scripts):
        os.mkdir(scripts)
        os.chmod(scripts, 0755)
    with bpcursor() as cursor:
        bats = []
        batch_array = BPBatchArray.create(cursor, batch)
        for idx, run in enumerate(runs):
            task_id = idx+TASK_ID_START
            bats.append(
                BPBatchArrayTask.create(cursor, batch_array, run, task_id))
        
    script = "#!/bin/sh\n"
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
    # Source cpenv.sh
    #
    script += ". %s\n" % os.path.join(PREFIX, "bin", "cpenv.sh")
    if JS_USE_REST:
        #
        # This is a REST PUT to JobStatus.py to create the job record
        #
        data = "{"+ (",".join(['"%s":"%s"' % (k, v) for k, v in (
            (K_ACTION, A_UPDATE), (K_STATUS, JS_RUNNING))])) 
        data += ',"%s":\'$JOB_ID\'' % JOB_ID
        data += ',"%s":%d' % (BATCH_ARRAY_ID, batch_array.batch_array_id)
        data += ',"%s":\'$SGE_TASK_ID\'' % TASK_ID
        data += ',"%s":"\'$HOSTNAME\'"}' % (K_HOST_NAME)
        script += """BATCHPROFILER_COMMAND=`curl -s """
        script += """-H "Content-type: application/json" -X PUT """
        script += """--data '%s' %s/JobStatus.py`\n""" % (
            data, BATCHPROFILER_DEFAULTS[URL])
    else:
        script += """
if [ -e $HOME/.batchprofiler.sh ]; then
    . $HOME/.batchprofiler.sh
fi
"""
        script += "BATCHPROFILER_COMMAND="
        script += "`PYTHONPATH=%s:$PYTHONPATH " % os.path.dirname(__file__)
        script += "python -c 'from JobStatus import update_status;"
        script += "print update_status(%d, '$JOB_ID', '$SGE_TASK_ID', " % \
            batch_array.batch_array_id
        script += "\"%s\", \"'$HOSTNAME'\")'`\n" % JS_RUNNING
                        
    #
    # CD to the CellProfiler root directory
    #
    if cwd is None:
        cwd = batch.cpcluster
    script += 'cd %s\n' % cwd
    #
    # set +e allows the command to error-out without ending this script.
    #        This lets us capture the error status.
    #
    script += "set +e\n"
    #
    # Run CellProfiler
    #
    script += "echo 'Running '$BATCHPROFILER_COMMAND' on '$HOSTNAME\n"
    script += "echo $BATCHPROFILER_COMMAND| bash\n"
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
    if JS_USE_REST:
        #
        # Set the status based on the result from CellProfiler
        # Use CURL again
        #
        script += """curl -s -H "Content-type: application/json" -X PUT """
        script += """--data '{"action":"update","%s":'$JOB_ID',""" % JOB_ID
        script += """"%s":%d,""" % (BATCH_ARRAY_ID, batch_array.batch_array_id)
        script += """"%s":'$SGE_TASK_ID',""" % TASK_ID
        script += """"%s":"'$JOB_STATUS'"}' """ % K_STATUS
        script += '%s/JobStatus.py\n' % BATCHPROFILER_DEFAULTS[URL]
    else:
        script += "PYTHONPATH=%s:$PYTHONPATH " % os.path.dirname(__file__)
        script += "python -c 'from JobStatus import update_status;"
        script += "print update_status(%d, '$JOB_ID', '$SGE_TASK_ID', " % \
            batch_array.batch_array_id
        script += "\"'$JOB_STATUS'\", \"'$HOSTNAME'\")'\n"
        
    script_filename = batch_array_script_file_path(batch_array)
    with open(script_filename, "w") as fd:
        fd.write(script)
    os.chmod(script_filename,stat.S_IWUSR |
             stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH |
             stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    job_id = bputilities.run_on_tgt_os(
        script = script, 
        group_name = batch.project,
        job_name = batch_array.get_job_name(),
        queue_name = batch.queue,
        priority = batch.priority,
        cwd = cwd,
        output = batch_array_text_file_path(batch_array),
        err_output = batch_array_err_file_path(batch_array),
        task_range = slice(TASK_ID_START, TASK_ID_START+len(bats)))
    tasks = []
    with bpcursor() as cursor:
        job = BPJob.create(cursor, batch_array, job_id)
        for bat in bats:
            task = BPJobTask.create(cursor, job, bat)
            tasks.append(task)
            BPJobTaskStatus.create(cursor, task, JS_SUBMITTED)
    return tasks

def cellprofiler_command(my_batch, bstart, bend):
    script = 'python CellProfiler.py -c -r -b --do-not-fetch '
    script += '-p "%s" ' % os.path.join(my_batch.data_dir, "Batch_data.h5")
    script += '-f %d -l %d ' % (bstart, bend)
    script += '-o "%s"' % my_batch.data_dir
    if my_batch.write_data:
            script += " " + run_out_file_path(my_batch, bstart = bstart, bend=bend)
    script += "\n"
    return script

def kill_run(run):
    tasks = BPJobTask.select_by_run(run)
    jobs = {}
    for task in tasks:
        if task.job.job_id not in jobs:
            jobs[task.job.job_id] = []
        jobs[task.job.job_id].append(task.batch_array_task.task_id)
    for job_id, task_ids in jobs.items():
        bputilities.kill_tasks(job_id, task_ids)
    with bpcursor() as cursor:
        for task in tasks:
            BPTaskStatus.create(cursor, task, JS_ABORTED)
            
def kill_task(task):
    bputilities.kill_task(task.job.job_id, task.batch_array_task.task_id)
    with bpcursor() as cursor:
        BPTaskStatus.create(cursor, task, JS_ABORTED)
    
def kill_job(job):
    bputilities.kill_jobs([job.job_id])
    with bpcursor() as cursor:
        for task in job.select_queued_tasks():
            BPJobTaskStatus.create(cursor, task, JS_ABORTED)

def kill_batch(batch_id):
    batch = BPBatch.select(batch_id)
    tasks = batch.select_queued_tasks()
    jobs = {}
    for task in tasks:
        if task.job.job_id not in jobs:
            jobs[task.job.job_id] = task.job
    bputilities.kill_jobs(jobs.keys())
    with bpcursor() as cursor:
        for task in tasks:
            BPJobTaskStatus.create(cursor, task, JS_ABORTED)

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

def batch_array_script_file_path(batch_array):
    return os.path.join(script_file_directory(batch_array.batch),
                        "job-array.%d.sh" % batch_array.batch_array_id)

def batch_array_text_file_path(batch_array):
    return os.path.join(text_file_directory(batch_array.batch),
                        "run.%d.\\$TASK_ID.txt" % batch_array.batch_array_id)

def batch_array_err_file_path(batch_array):
    return os.path.join(text_file_directory(batch_array.batch),
                        "run.%d.\\$TASK_ID.err" % batch_array.batch_array_id)

def batch_array_task_text_file_path(task):
    path = batch_array_text_file_path(task.batch_array)
    return path.replace("\\$TASK_ID", str(task.task_id))
    
def batch_array_task_err_file_path(task):
    path = batch_array_err_file_path(task.batch_array)
    return path.replace("\\$TASK_ID", str(task.task_id))

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
