#!/broad/tools/apps/Python-2.5.2/bin/python
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

form = cgi.FieldStorage()
batch_id = int(form["batch_id"].value)
sql_script = form["sql_script"].value
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
sql_script_file = open(batch_script,"w")
try:
    sql_script_file.writelines(table_lines)
    for file_name in image_files:
        sql_script_file.write("""LOAD DATA LOCAL INFILE '%(file_name)s' REPLACE INTO TABLE %(image_table)s FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"';
SHOW WARNINGS;
"""%(globals()))
    for file_name in object_files:
        sql_script_file.write("""LOAD DATA LOCAL INFILE '%(file_name)s' REPLACE INTO TABLE %(object_table)s FIELDS TERMINATED BY ',';
SHOW WARNINGS;
"""%(globals()))
finally:
    sql_script_file.close()

print "Content-Type: text/html"
print
print "<html><head>"
print "<title>Upload to database</title>"
print "</head>"
print "<body>"
print "<h1>Database script file</h1>"
print "<tt>"
sql_script_file=open(batch_script,"r")
for line in sql_script_file:
    print "<div style='whitespace:nowrap'>%s</div>"%(line)
print "</tt>"
sql_script_file.close()
print "<h1>Results of database upload</h1>"
sql_script_file=open(batch_script,"r")
old_dir = os.curdir
os.chdir(my_batch["data_dir"])
pipe = subprocess.Popen(["mysql","-h","imgdb01","-A","-ucpadmin","-pcPus3r","--local-infile=1"],stdin=sql_script_file,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#pipe=os.popen("mysql -h imgdb01 -A -u cpadmin -p --local-infile=1 < %s"%(batch_script),"r")
stdout,stderr = pipe.communicate()
print "<tt>"
for line in stdout.split('\n'):
    print "<div style='whitespace:nowrap'>%s</div>"%(line)
for line in stderr.split('\n'):
    print "<div style='whitespace:nowrap'>%s</div>"%(line)
print "</tt>"
sql_script_file.close()
os.chdir(old_dir)
print "</body>"
print "</html>"
os.unlink(batch_script)
