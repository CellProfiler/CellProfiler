#!/usr/bin/env ./batchprofiler.sh
#
# REST service to build CellProfiler
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

import cgitb
cgitb.enable()
import sys
import cgi
import json
import os

from bputilities import build_cellprofiler, get_version_and_githash, is_built
from bpformdata import RM_GET, RM_PUT, REQUEST_METHOD, REVISION, GIT_HASH, \
     DATETIME_VERSION, IS_BUILT, QUEUE, PROJECT, EMAIL, BATCHPROFILER_DEFAULTS

query_revision = BATCHPROFILER_DEFAULTS[REVISION]
def do_get():
    if query_revision is not None:
        datetime_version, git_hash = get_version_and_githash(query_revision)
        buildstatus = is_built(version=datetime_version, git_hash=git_hash)
        print "Content-Type: application/json\r"
        print "\r"
        print json.dumps(
            { GIT_HASH: git_hash, 
              DATETIME_VERSION: datetime_version,
              IS_BUILT: buildstatus
              })
        return
    else:
        print "Content-Type: text/plain\r"
        print "\r"
        print """API for BuildCellProfiler.py

GET /?revision=<revision> HTTP/1.1

Look up the GIT hash and datetime version of a revision.

<revision> - a GIT treeish reference (tag, GIT hash, branch)

returns a JSON-encoded dictionary: 
{
    %(GIT_HASH)s:"<git-hash>",
    %(DATETIME_VERSION)s:"<datetime-version>",
    %(IS_BUILT)s:<true/false>,
}
<git-hash> - the full GIT hash of the reference 
<datetime-version> - the UTC time of the checkin in the format YYYYMMDDHHMMSS.

The value for %(IS_BUILT)s is true if the version of CellProfiler has been
built.

PUT /?revision=<revision> HTTP/1.1

(optional JSON dictionary = 
 {%(EMAIL)s:<email>:%(QUEUE)s:<queue>:%(PROJECT)s:<project>})

<email> - send user email when built
<queue> - run build on this queue
<project> - charge to this project

returns the same JSON-encoded dictionary as for GET
""" % globals()
        
def do_put():
    datetime_version, git_hash = get_version_and_githash(query_revision)
    buildstatus = is_built(version=datetime_version, git_hash=git_hash)
    if not buildstatus:
        try:
            options = json.load(data)
            assert isinstance(options, dict)
        except:
            options = {}
        build_cellprofiler(version = datetime_version,
                           git_hash = git_hash,
                           queue_name = options.get(QUEUE, None),
                           group_name = options.get(PROJECT, None),
                           email_address = options.get(EMAIL, None))
    
    print "Content-Type: application/json\r"
    print "\r"
    print json.dumps(
        { GIT_HASH: git_hash, 
          DATETIME_VERSION: datetime_version,
          IS_BUILT: buildstatus
          })

if REQUEST_METHOD == RM_GET:
    do_get()
elif REQUEST_METHOD == RM_PUT:
    do_put()
else:
    raise NotImplementedError(
        "Request method %s not implemented" % REQUEST_METHOD)

