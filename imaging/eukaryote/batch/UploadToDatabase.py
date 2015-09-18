#!/usr/bin/env ./batchprofiler.sh
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
# Upload .CSV files to the database
#
import cgitb
cgitb.enable()
import RunBatch
from bpformdata import *
import cgi
import re
import os
import os.path
import subprocess

import sql_jobs

#
# TODO: Move the logic that collects the load statements into sql_jobs
#
# TODO: use yattag to build the HTML
#

batch_id = BATCHPROFILER_VARIABLES[BATCH_ID]
sql_script = BATCHPROFILER_VARIABLES[SQL_SCRIPT]
output_file = BATCHPROFILER_VARIABLES[OUTPUT_FILE]
queue = BATCHPROFILER_VARIABLES[QUEUE]
my_batch = RunBatch.BPBatch()
my_batch.select(batch_id)

re_load_line = re.compile("'(.+?)[0-9]+_[0-9]+_(Image|Object).CSV'\sREPLACE\sINTO\sTABLE\s([A-Za-z0-9_]+)\s")
re_ignore_line = re.compile("SHOW WARNINGS;")
table_lines = []
image_prefix = None
object_prefix = None
sql_script_file = open(my_batch.data_dir+os.sep+sql_script,"r")
in_table_defs = True
try:
    for line in sql_script_file:
        match = re_load_line.search(line)
        if match:
            in_table_defs = False
            if match.groups(1)[1] == 'Image':
                image_table = match.groups(1)[2]
                image_prefix = match.groups(1)[0]
            else :
                object_table = match.groups(1)[2]
                object_prefix = match.groups(1)[0]
        elif (not re_ignore_line.search(line)) and in_table_defs:
            table_lines.append(line)
finally:
    sql_script_file.close()    

re_file_name = re.compile("^(.+?)[0-9]+_[0-9]+_(Image|Object).CSV$")
image_files = []
object_files = []
for file_name in os.listdir(my_batch.data_dir):
    match = re_file_name.search(file_name)
    if match:
        if (image_prefix and match.groups(1)[0] == image_prefix 
            and match.groups(1)[1] == 'Image'):
            image_files.append(file_name)
        elif (object_prefix and 
              match.groups(1)[0] == object_prefix and 
              match.groups(1)[1] == 'Object'):
            object_files.append(file_name)

batch_script_file = RunBatch.batch_script_file(sql_script)
batch_script_dir = RunBatch.batch_script_directory(my_batch)
if not os.path.isdir(batch_script_dir):
    os.makedirs(batch_script_dir)
batch_script_path = RunBatch.batch_script_path(my_batch, sql_script)
sql_script_file = open(batch_script_path, "w")
try:
    sql_script_file.writelines(table_lines)
    for file_name in image_files:
        path_name = os.path.join(my_batch.data_dir, file_name)
        sql_script_file.write("""SELECT 'Loading %(file_name)s into %(image_table)s';"""%(globals()))
        sql_script_file.write("""LOAD DATA LOCAL INFILE '%(path_name)s' REPLACE INTO TABLE %(image_table)s FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"';
SHOW WARNINGS;
"""%(globals()))
    for file_name in object_files:
        path_name = os.path.join(my_batch.data_dir, file_name)
        sql_script_file.write("""SELECT 'Loading %(file_name)s into %(object_table)s';"""%(globals()))
        sql_script_file.write("""LOAD DATA LOCAL INFILE '%(path_name)s' REPLACE INTO TABLE %(object_table)s FIELDS TERMINATED BY ',';
SHOW WARNINGS;
"""%(globals()))
finally:
    sql_script_file.close()

print """Content-type: text/html

<html><head>
<title>Upload to database</title>
</head>
<script type="text/javascript">
function toggleVerboseText() 
{
    make_visible = too_verbose.style.display == 'none';
    too_verbose.style.display = make_visible?'block':'none';
    verbose_placeholder.style.display = make_visible?'none':'block';
    show_hide.value=make_visible?'Hide most':'Show all';
}
</script>
<body>
<h1>Database script file</h1>
"""
sql_script_file=open(batch_script_path,"r")
lines = sql_script_file.readlines()
sql_script_file.close()
line_count = len(lines)
if line_count > 10:
    print """<input id="show_hide" type="button" onclick="toggleVerboseText()" value="Show all" /><br/>"""
print """<tt>"""
for line, index in zip(lines,range(line_count)):
    if line_count > 10 and index == 3:
        print "<div id='verbose_placeholder' style='display:block'>...</div>"
        print "<div id='too_verbose' style='display:none'>"
    print "<div style='whitespace:nowrap'>%s</div>"%(line)
    if line_count > 10 and index == line_count-4:
        print "</div>"
print "</tt>"
job = sql_jobs.run_sql_file(batch_id, batch_script_file)
    
print "<h2>SQL script submitted to cluster as job # %s"%(job.job_id)
print "</body>"
print "</html>"
