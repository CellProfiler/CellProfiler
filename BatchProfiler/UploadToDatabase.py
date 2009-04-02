#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/development/python-2.6.sh
#
# Upload .CSV files to the database
#
import cgitb
cgitb.enable()
import RunBatch
import cgi
import re
import os
import os.path
import subprocess

import sql_jobs

form = cgi.FieldStorage()
batch_id = int(form["batch_id"].value)
sql_script = form["sql_script"].value
output_file = form["output_file"].value
queue = (form.has_key("queue") and form["queue"].value) or None
my_batch = RunBatch.LoadBatch(batch_id)

re_load_line = re.compile("'(.+?)[0-9]+_[0-9]+_(image|object).CSV'\sREPLACE\sINTO\sTABLE\s([A-Za-z0-9_]+)\s")
re_ignore_line = re.compile("SHOW WARNINGS;")
table_lines = []
sql_script_file = open(my_batch["data_dir"]+os.sep+sql_script,"r")
try:
    for line in sql_script_file:
        match = re_load_line.search(line)
        if match:
            if match.groups(1)[1] == 'image':
                image_table = match.groups(1)[2]
                image_prefix = match.groups(1)[0]
            else :
                object_table = match.groups(1)[2]
                object_prefix = match.groups(1)[0]
        elif not re_ignore_line.search(line):
            table_lines.append(line)
finally:
    sql_script_file.close()    

re_file_name = re.compile("^(.+?)[0-9]+_[0-9]+_(image|object).CSV$")
image_files = []
object_files = []
for file_name in os.listdir(my_batch["data_dir"]):
    match = re_file_name.search(file_name)
    if match:
        if match.groups(1)[0] == image_prefix and match.groups(1)[1] == 'image':
            image_files.append(file_name)
        elif match.groups(1)[0] == object_prefix and match.groups(1)[1] == 'object':
            object_files.append(file_name)

batch_script = my_batch["data_dir"]+os.sep+"batch_"+sql_script
batch_script = os.path.abspath(batch_script)
sql_script_file = open(batch_script,"w")
try:
    sql_script_file.writelines(table_lines)
    for file_name in image_files:
        sql_script_file.write("""SELECT 'Loading %(file_name)s into %(image_table)s';"""%(globals()))
        sql_script_file.write("""LOAD DATA LOCAL INFILE '%(file_name)s' REPLACE INTO TABLE %(image_table)s FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"';
SHOW WARNINGS;
"""%(globals()))
    for file_name in object_files:
        sql_script_file.write("""SELECT 'Loading %(file_name)s into %(object_table)s';"""%(globals()))
        sql_script_file.write("""LOAD DATA LOCAL INFILE '%(file_name)s' REPLACE INTO TABLE %(object_table)s FIELDS TERMINATED BY ',';
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
sql_script_file=open(batch_script,"r")
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
if queue is None:
    job_id = sql_jobs.run_sql_file(batch_id, batch_script, output_file)
else:
    job_id = sql_jobs.run_sql_file(batch_id, batch_script, output_file, queue)
    
print "<h2>SQL script submitted to cluster as job # %s"%(job_id)
print "</body>"
print "</html>"
