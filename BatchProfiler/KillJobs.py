#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/python-2.6.sh
#
# Kill all jobs in a batch
#
print "Content-Type: text/html"
print
import cgitb
cgitb.enable()
import RunBatch
import cgi
import subprocess
import sys

form = cgi.FieldStorage()
if form.has_key("job_id"):
    import subprocess
    job_id = int(form["job_id"].value)
    run = {"job_id":job_id}
    RunBatch.KillOne(run)
    print"""
    <html><head><title>Job %(job_id)d killed</title></head>
    <body>Job %(job_id)d killed
    </body>
    </html>
"""%(globals())
elif form.has_key("batch_id"):
    batch_id = int(form["batch_id"].value)
    my_batch = RunBatch.LoadBatch(batch_id)
    for run in my_batch["runs"]:
        RunBatch.KillOne(run)
    
    url = "ViewBatch.py?batch_id=%(batch_id)d"%(my_batch)
    print "<html><head>"
    print "<meta http-equiv='refresh' content='0; URL=%(url)s' />"%(globals())
    print "</head>"
    print "<body>This page should be redirected to <a href='%(url)s'/>%(url)s</a></body>"%(globals())
    print "</html>"
else:
    print """<html><head><title>Kill jobs</title></head>
    <body>
    <h1>Kill jobs started by the imageweb webserver</h1>
    <form action='KillJobs.py' method='POST'>
    Job ID:<input type='text' name='job_id' />
    <input type='submit' value='Kill'/>
    </form>
    """
    p = subprocess.Popen(["bash"],stdin = subprocess.PIPE,
                         stdout=subprocess.PIPE)
    listing = p.communicate(". /broad/lsf/conf/profile.lsf;bjobs\n")[0]
    listing_lines = listing.split('\n')
    header = listing_lines[0]
    columns = [header.find(x) for x in header.split(' ') if len(x)]
    columns.append(1000)
    body = listing_lines[1:]
    print """
    <h2>Jobs on imageweb</h2>
    <table>
    """
    print "<tr>%s</tr>"%("".join(['<th>%s</th>'%(header[columns[i]:columns[i+1]])
                                  for i in range(len(columns)-1)]))
    for line in body:
        print "<tr>%s</tr>"%("".join(['<td>%s</td>'%(line[columns[i]:columns[i+1]])
                                      for i in range(len(columns)-1)]))
    """
    </table>
    </body>
    """
try:
    import cellprofiler.utilities.jutil as jutil
    jutil.kill_vm()
except:
    import traceback
    traceback.print_exc()
