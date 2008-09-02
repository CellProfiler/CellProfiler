#!/broad/tools/apps/Python-2.5.2/bin/python
#
# View a batch from the database, with some options to re-execute it
#
import cgitb
cgitb.enable()
import RunBatch
import cgi
import os
import os.path

print "Content-Type: text/html"
print

batch_id = int(cgi.FieldStorage()["batch_id"].value)
my_batch = RunBatch.LoadBatch(batch_id)
if cgi.FieldStorage().has_key("submit_run"):
    for run in my_batch["runs"]:
        if run["run_id"] == int(cgi.FieldStorage()["submit_run"].value):
            result = RunBatch.RunOne(my_batch,run)
            print "<html><head><title>Batch # %d resubmitted</title>"%(batch_id)
            print "<style type='text/css'>"
            print """
        table.run_table {
            border-spacing: 0px;
            border-collapse: collapse;
        }
        table.run_table td {
            text-align: left;
            vertical-align: baseline;
            padding: 0.1em 0.5em;
            border: 1px solid #666666;
        }
        """
            print "</style></head>"
            print "<body>"
            print "Your batch # %d has been resubmitted."%(batch_id)
            print "<table class='run-table'>"
            for key in result:
                print "<tr><td><b>%s:</b></td><td style='white-space:nowrap'>%s</td></tr>"%(key,result[key])
            print "</table>"
            print "</body></html>"
else:
    print "<html>"
    print "<head>"
    print "<title>View batch # %d</title>"%(batch_id)
    print "<style type='text/css'>"
    print """
table.run_table {
    border-spacing: 0px;
    border-collapse: collapse;
}
table.run_table th {
    text-align: left;
    font-weight: normal;
    padding: 0.1em 0.5em;
    border: 1px solid #666666;
}
table.run_table td {
    text-align: right;
    padding: 0.1em 0.5em;
    border: 1px solid #666666;
}
table.run_table thead th {
    text-align: center;
}
"""
    print "</style>"
    print "</head>"
    print "<body>"
    print "<h1>View batch # %d</h1>"%(batch_id)
    print "<table class='run_table'>"
    print "<thead><tr>"
    print "<td>Start</td><td>End</td><td>Job #</td><td>Status</td><td>Text output file</td><td>Results file</td>"
    print "</tr></thead>"
    for run in my_batch["runs"]:
        x = my_batch.copy()
        x.update(run)
        text_file = "%(data_dir)s/txt_output/%(start)d_to_%(end)d.txt"%(x)
        done_file = "%(data_dir)s/status/Batch_%(start)d_to_%(end)d_DONE.mat"%(x)
        out_file = "%(data_dir)s/status/Batch_%(start)d_to_%(end)d_OUT.mat"%(x)
        if os.name == "nt":
            done_file=done_file.replace("/imaging/analysis","//iodine/imaging_analysis")
            done_file=done_file.replace("/",os.sep)
        print "<tr>"
        print "<td>%(start)d</td><td>%(end)d</td><td>%(job_id)s</td>"%(x)
        if os.path.isfile(done_file):
            print "<td style='color:green'>Complete</td>"
        else:
            job_status = RunBatch.GetJobStatus(run["job_id"])
            stat = "Unknown"
            if job_status:
                stat=job_status["STAT"]
            print """
<td style='color:red'>%s<br/>
    <form>
    <input type='button' value='Resubmit' onclick='parent.location="ViewBatch.py?batch_id=%d&amp;submit_run=%d"'/>
    </form>
</td>"""%(stat,batch_id,run["run_id"])
        print "<td>%(text_file)s</td><td>%(out_file)s</td>"%(locals())
        print "</tr>"
    print "</table>"
    print "</body>"
