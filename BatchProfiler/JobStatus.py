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
from RunBatch import BPJob, JS_SUBMITTED, JS_RUNNING
from bpformdata import REQUEST_METHOD, RM_PUT, K_ACTION, A_CREATE, A_READ, \
     A_UPDATE, A_DELETE, JOB_ID, RUN_ID, K_STATUS, K_HOST_NAME, K_WANTS_XVFB

K_JOB_ID = JOB_ID
K_RUN_ID = RUN_ID

if REQUEST_METHOD == RM_PUT:
    data = json.load(sys.stdin)
    action = data[K_ACTION]
    job_id = int(data[K_JOB_ID])
    run_id = int(data[K_RUN_ID])
    host_name = data.get(K_HOST_NAME, None)
    wants_xvfb = data.get(K_WANTS_XVFB, False)
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
        if status == JS_RUNNING and host_name is not None:
            xvfb_server = job.create_job_host(host_name, wants_xvfb)
            print xvfb_server
        else:
            print "OK"
    else:
        raise NotImplementedError("Unsupported action: %s" % action)
else:
    raise NotImplementedError("Unsupported http method: %s" % REQUEST_METHOD)

