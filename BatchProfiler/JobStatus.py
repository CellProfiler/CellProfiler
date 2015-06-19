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

import cgitb
cgitb.enable()
import json
import os
import sys
from RunBatch import BPJob, JS_SUBMITTED
from bpformdata import REQUEST_METHOD, RM_PUT

K_ACTION = "action"
A_CREATE = "create"
A_READ = "read"
A_UPDATE = "update"
A_DELETE = "delete"

K_JOB_ID = "job_id"
K_RUN_ID = "run_id"
K_STATUS = "status"

if REQUEST_METHOD == RM_PUT:
    data = json.load(sys.stdin)
    action = data[K_ACTION]
    job_id = int(data[K_JOB_ID])
    run_id = int(data[K_RUN_ID])
    status = data.get(K_STATUS, JS_SUBMITTED)
    job = BPJob(run_id, job_id)
    if action == A_CREATE:
        job.create(status)
        print "Content-Type: text/plain"
        print
        print "OK"
    elif action == A_UPDATE:
        job.update_status(status)
        print "Content-Type: text/plain"
        print
        print "OK"
    else:
        raise NotImplementedError("Unsupported action: %s" % action)
else:
    raise NotImplementedError("Unsupported http method: %s" % REQUEST_METHOD)

