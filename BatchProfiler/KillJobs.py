#!/broad/tools/apps/Python-2.5.2/bin/python
#
# Kill all jobs in a batch
#
import cgitb
cgitb.enable()
import RunBatch
import cgi

form = cgi.FieldStorage()
batch_id = int(form["batch_id"].value)
my_batch = RunBatch.LoadBatch(batch_id)
for run in my_batch["runs"]:
    RunBatch.KillOne(run)

url = "ViewBatch.py?batch_id=%(batch_id)d"%(my_batch)
print "Content-Type: text/html"
print
print "<html><head>"
print "<meta http-equiv='refresh' content='0; URL=%(url)s' />"%(globals())
print "</head>"
print "<body>This page should be redirected to <a href='%(url)s'/>%(url)s</a></body>"%(globals())
print "</html>"
