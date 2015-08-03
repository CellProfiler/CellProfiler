#!/usr/bin/env ./batchprofiler.sh
#
# Start a batch operation from a web page
#
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2015 Broad Institute
# All rights reserved.
#
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

################
#
# REST interface for job status
#
################


from MySQLdb import OperationalError
import os
import sys
import time
import traceback

from RunBatch import BPJob, BPJobTask, BPTaskHost, BPJobTaskStatus, \
     BPBatchArray, \
     JS_SUBMITTED, JS_RUNNING, bpcursor


'''Wait a maximum of 60 seconds for the job to be placed in the database'''
JOB_QUERY_TIMEOUT = 60

'''# of seconds to pause between attempts'''
JOB_QUERY_PAUSE = .25

def update_status(batch_array_id, job_id, task_id, status, host_name):
    batch_array = BPBatchArray.select(batch_array_id)
    timeout = time.time() + JOB_QUERY_TIMEOUT
    while time.time() < timeout:
        job = BPJob.select(batch_array, job_id)
        if job is not None:
            break
        time.sleep(JOB_QUERY_PAUSE)
    task = BPJobTask.select_by_task_id(job, task_id)
    run = task.batch_array_task.run
    for attempt in range(10):
        try:
            with bpcursor() as cursor:
                BPJobTaskStatus.create(cursor, task, status)
                cmd = run.command
                if status == JS_RUNNING:
                    if host_name is not None:
                        task_host = BPTaskHost.create(
                            cursor, task, host_name,
                            run.wants_xvfb)
                        if run.wants_xvfb:
                            cmd = 'xvfb-run -n %d %s' % (
                                task_host.xvfb_server, run.command)
                    return cmd
                else:
                    return "OK"
                break
        except OperationalError as e:
            traceback.print_exc()
    else:
        raise

if __name__ == "__main__":
    import cgitb
    cgitb.enable()
    import json
    from bpformdata import REQUEST_METHOD, RM_PUT, K_ACTION, A_CREATE, A_READ, \
         A_UPDATE, A_DELETE, JOB_ID, RUN_ID, BATCH_ARRAY_ID, TASK_ID, \
         K_STATUS, K_HOST_NAME, K_WANTS_XVFB    

    if REQUEST_METHOD == RM_PUT:
        data = json.load(sys.stdin)
        action = data[K_ACTION]
        job_id = int(data[JOB_ID])
        batch_array_id = int(data[BATCH_ARRAY_ID])
        if TASK_ID in data and data[TASK_ID] is not None:
            task_id = int(data[TASK_ID])
        host_name = data.get(K_HOST_NAME, None)
        wants_xvfb = data.get(K_WANTS_XVFB, False)
        status = data.get(K_STATUS, JS_SUBMITTED)
            
        if action == A_CREATE:
            batch_array = BPBatchArray.select(batch_array_id)
            with bpcursor() as cursor:
                job = BPJob.create(cursor, batch_array, job_id)
            print "Content-Type: text/plain"
            print
            print "OK"
        elif action == A_UPDATE:
            cmd = update_status(batch_array_id, job_id, task_id, status, host_name)
            print "Content-Type: text/plain"
            print
            print cmd
            
        else:
            raise NotImplementedError("Unsupported action: %s" % action)
    else:
        raise NotImplementedError("Unsupported http method: %s" % REQUEST_METHOD)

