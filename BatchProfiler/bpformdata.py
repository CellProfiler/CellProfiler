'''bpformdata.py - Form data constants for Batch Profiler

'''
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

import cgi
import os

######
#
# Refrain from importing CellProfiler
# in order to speed up process load time
#
######

# Form keys

DATA_DIR = "data_dir"
EMAIL = "email"
QUEUE = "queue"
PROJECT = "project"
PRIORITY = "priority"
WRITE_DATA = "write_data"
BATCH_SIZE = "batch_size"
MEMORY_LIMIT = "memory_limit"
REVISION = "revision"
SUBMIT_BATCH = "submit_batch"
URL = "url"
BATCH_ID = "batch_id"
RUN_ID = "run_id"
JOB_ID = "job_id"
#
# keys for types of data
#
GIT_HASH = "git_hash"
DATETIME_VERSION = "datetime_version"
'''True if the version of CellProfiler has been successfully built'''
IS_BUILT = "is_built"
#
# Integer variables
#
BP_INTEGER_VARIABLES = (PRIORITY, BATCH_SIZE, BATCH_ID, RUN_ID, JOB_ID)

#
# Actions for DeleteFile
#
K_DELETE_ACTION = "delete_action"
A_DELETE_ALL = "ALL"
A_DELETE_TEXT = "TEXT"
A_DELETE_OUTPUT = "OUTPUT"

# Environment variables
#
# From the webserver
#
SCRIPT_URI_KEY = "SCRIPT_URI"
SERVER_NAME_KEY = "SERVER_NAME"
#
# These should be "export" defined by ~/.batchprofiler.sh
#
'''Environment variable that stores DNS name of MySQL host (optional)'''
E_BATCHPROFILER_MYSQL_HOST="BATCHPROFILER_MYSQL_HOST"
BATCHPROFILER_MYSQL_HOST = os.environ.get(E_BATCHPROFILER_MYSQL_HOST)
'''Environment variable that stores the port # (optional)'''
E_BATCHPROFILER_MYSQL_PORT = "BATCHPROFILER_MYSQL_PORT"
BATCHPROFILER_MYSQL_PORT = os.environ.get(E_BATCHPROFILER_MYSQL_PORT)

'''Environment variable that stores the MySQL user name'''
E_BATCHPROFILER_MYSQL_USER = "BATCHPROFILER_MYSQL_USER"
BATCHPROFILER_MYSQL_USER = os.environ.get(E_BATCHPROFILER_MYSQL_USER)

'''Environment variable that stores the MySQL user's password (optional)'''
E_BATCHPROFILER_MYSQL_PASSWORD = "BATCHPROFILER_MYSQL_PASSWORD"
BATCHPROFILER_MYSQL_PASSWORD = os.environ.get(E_BATCHPROFILER_MYSQL_PASSWORD)

'''The environment variable that stores the MySQL database name (optional)'''
E_BATCHPROFILER_MYSQL_DATABASE = "BATCHPROFILER_MYSQL_DATABASE"
BATCHPROFILER_MYSQL_DATABASE = os.environ.get(
    E_BATCHPROFILER_MYSQL_DATABASE, "batchprofiler")

'''The environment variable for the maximum allowed amount of memory for a node'''
E_BATCHPROFILER_MAX_MEMORY_LIMIT = "BATCHPROFILER_MAX_MEMORY_LIMIT"
MAX_MEMORY_LIMIT = float(os.environ.get(E_BATCHPROFILER_MAX_MEMORY_LIMIT, 
                                        256000))

'''The environment variable for the minimum allowed amount of memory for a node'''
E_BATCHPROFILER_MIN_MEMORY_LIMIT = "BATCHPROFILER_MIN_MEMORY_LIMIT"
MIN_MEMORY_LIMIT = float(os.environ.get(E_BATCHPROFILER_MIN_MEMORY_LIMIT, 
                                        1000))

try:
    '''The location of the CellProfiler build'''
    PREFIX = os.environ["PREFIX"]
except:
    raise ValueError("Environment variable, ""PREFIX"", is not defined. Please run scripts with variables defined in cpenv.sh")

'''The checkout location for CellProfiler versions

Defaults to $PREFIX/checkouts

The naming convention for the root directory of CellProfiler is 
$BATCHPROFILER_CPSRC/<version>_<githash>
where <version> is the UTC time of the commit in the format,
YYYYMMDDHHMMSS
'''
E_BATCHPROFILER_CPCHECKOUT = "BATCHPROFILER_CPCHECKOUT"
BATCHPROFILER_CPCHECKOUT = os.environ.get(
    E_BATCHPROFILER_CPCHECKOUT, os.path.join(PREFIX, "checkouts"))

RM_GET = "GET"
RM_PUT = "PUT"
RM_POST = "POST"

REQUEST_METHOD = os.environ.get("REQUEST_METHOD", RM_GET)
if REQUEST_METHOD != RM_POST:
    import urlparse
    QUERY_STRING = os.environ.get("QUERY_STRING")
    if QUERY_STRING is None:
        QUERY_DICT = {}
    else:
        QUERY_DICT = urlparse.parse_qs(QUERY_STRING)
else:
    field_storage = cgi.FieldStorage()
#
# Needed for subprocesses
#
try:
    LC_ALL = os.environ["LC_ALL"]
except:
    LC_ALL = "en_us.UTF-8"

def __get_defaults():
    url = os.environ.get("SCRIPT_URI")
    if url is None:
        server_name = os.environ.get(SERVER_NAME_KEY, "localhost")
        if server_name is None:
            kv = [(URL, "")]
        else:
            kv = [(URL, "http://%s/cgi-bin" % server_name)]
    else:
        kv = [(URL, str(url.rsplit("/", 1)[0]))]
    def lookup_default(key):
        ekey = "BATCHPROFILER_"+key.upper()
        if REQUEST_METHOD == RM_POST:
            value = field_storage.getvalue(key, os.environ.get(ekey, None))
        else:
            value = QUERY_DICT.get(key, [os.environ.get(ekey, None)])[0]
        if key in BP_INTEGER_VARIABLES and value is not None:
            return int(value)
        return value
    
    kv += [(k, lookup_default(k)) for k in 
           (DATA_DIR, EMAIL, QUEUE, PROJECT, PRIORITY, WRITE_DATA, BATCH_SIZE,
            MEMORY_LIMIT, REVISION, SUBMIT_BATCH, K_DELETE_ACTION, BATCH_ID,
            RUN_ID, JOB_ID) 
           if lookup_default(k) is not None]
    return dict(kv)
            
BATCHPROFILER_DEFAULTS = __get_defaults()

__all__ = filter(
    (lambda k: k not in (cgi.__name__, os.__name__, BP_INTEGER_VARIABLES)), 
    globals().keys())