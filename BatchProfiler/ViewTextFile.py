#!/broad/tools/apps/Python-2.5.2/bin/python
#
# View the text file that is produced by bsub during batch processing
#
import cgitb
cgitb.enable()
import RunBatch
import cgi
import os
import os.path

run_id = int(cgi.FieldStorage()["run_id"].value)
my_batch,my_run = RunBatch.LoadRun(run_id)
text_file = open(RunBatch.RunTextFilePath(my_batch,my_run),"r")
print "Content-Type: text/plain"
print
print text_file.read()
text_file.close()

