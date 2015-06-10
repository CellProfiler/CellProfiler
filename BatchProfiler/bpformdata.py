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
BATCHPROFILER_MYSQL_USER = os.environ[E_BATCHPROFILER_MYSQL_USER]

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
RM_GET = "GET"
RM_PUT = "PUT"
RM_POST = "POST"

REQUEST_METHOD = os.environ.get("REQUEST_METHOD", RM_GET)
#
# Predefined by the environment
#
try:
    PREFIX = os.environ["PREFIX"]
except:
    raise ValueError("Environment variable, ""PREFIX"", is not defined. Please run scripts with variables defined in cpenv.sh")
#
# Needed for subprocesses
#
try:
    LC_ALL = os.environ["LC_ALL"]
except:
    LC_ALL = "en_us.UTF-8"

def __get_defaults():
    form_data = cgi.FieldStorage()
    url = os.environ.get("SCRIPT_URI")
    if url is None:
        server_name = os.environ.get(SERVER_NAME)
        if server_name is None:
            kv = [(URL, "")]
        else:
            kv = [(URL, "http://%s/cgi-bin" % server_name)]
    else:
        kv = [(URL, str(url.rsplit("/", 1)[0]))]
    def lookup_default(key):
        ekey = "BATCHPROFILER_"+key
        return form_data.getvalue(key, os.environ.get(ekey))
    
    kv += [(k, lookup_default(k)) for k in 
           (DATA_DIR, EMAIL, QUEUE, PROJECT, PRIORITY, WRITE_DATA, BATCH_SIZE,
            MEMORY_LIMIT, REVISION, SUBMIT_BATCH) if lookup_default(k) is not None]
    return dict(kv)
            
BATCHPROFILER_DEFAULTS = __get_defaults()