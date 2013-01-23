#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/python-2.6.sh
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
#
# View a batch from the database, with some options to re-execute it
#
import cgitb
cgitb.enable()
import RunBatch
import sql_jobs
import StyleSheet
import cgi
import os
import os.path

print "Content-Type: text/html"
print

form = cgi.FieldStorage()
batch_id = int(form["batch_id"].value)
my_batch = RunBatch.LoadBatch(batch_id)
jobs_by_state = {}

job_ids = [run["job_id"] for run in my_batch["runs"]]
job_dictionary = RunBatch.GetJobStatus(job_ids)
           
for run in my_batch["runs"]:
    stat  = "Unknown"
    if os.path.isfile(RunBatch.RunDoneFilePath(my_batch,run)):
        try:
            fd = open(RunBatch.RunDoneFilePath(my_batch, run), "r")
            stat = fd.readline().strip()
            if stat.startswith("Done"):
                stat = "Complete"
            fd.close()
        except:
            stat = "Unknown"
    elif run["job_id"]==None:
        pass
    else :
        job_status = job_dictionary[run["job_id"]]
        if job_status and job_status.has_key("STAT"):
            stat = job_status["STAT"]
    run["status"]=stat;
    if jobs_by_state.has_key(stat):
        jobs_by_state[stat].append(run)
    else:
        jobs_by_state[stat] = [run]

if form.has_key("submit_run"):
    print "<html><head><title>Batch # %d resubmitted</title>"%(batch_id)
    StyleSheet.PrintStyleSheet()
    print "</head>"
    print "<body>"
    submit_run = form["submit_run"].value
    run_id = submit_run.isdigit() and int(submit_run)
    print "Your batch # %d has been resubmitted."%(batch_id)
    print "<table class='run-table'>"
    for run in my_batch["runs"]:
        if ((run["run_id"] == run_id) or
            (submit_run == 'Incomplete' and run["status"] != "Complete") or
            (run["status"] == submit_run) or
            (submit_run == 'All')):
            result = RunBatch.RunOne(my_batch,run)
            for key in result:
                print "<tr><td><b>%s:</b></td><td style='white-space:nowrap'>%s</td></tr>"%(key,result[key])
    print "</table>"
    print "</body></html>"
else:
    print "<html>"
    print "<head>"
    print "<title>View batch # %d</title>"%(batch_id)
    StyleSheet.PrintStyleSheet()
    print "</head>"
    print "<body>"
    print "<div>"
    print "<h1>View batch # %d</h1>"%(batch_id)
    print "</div>"
    print "<div style='padding:5px'>"
    #
    # The summary table
    #
    print "<div style='position:relative;float:left'>"
    print "<table class='run_table'>"
    print "<thead>"
    print "<tr><th style='align:center' colspan='3'>Job Summary</th></tr>"
    print "<tr><th>Status</th><th>Count</th><th>Resubmit jobs</th></tr>"
    print "</thead>"
    print "<tbody>"
    for state in jobs_by_state:
        print "<tr><th>%s</th><td>%d</td>"%(state,len(jobs_by_state[state]))
        print """
        <td><form action='ViewBatch.py' method='POST' target='ResubmitWindow'>
            <input type='hidden' name='batch_id' value='%d' />
            <input type='hidden' name='submit_run' value='%s' />
            <input type='submit' value='Resubmit %s'
           onclick='return confirm("Do you really want to resubmit all %s batches?")' />
        </form></td>
        """%(my_batch["batch_id"], state, state, state.lower())
        print "</tr>"
    print "</tbody>"
    print "</table>"
    print "</div>"
    #
    # Kill button
    print "<div style='position:relative; float:left; padding:2px'>"
    print "<div style='position:relative; float:top'>"
    print "<table><tr>"
    print "<td><form action='KillJobs.py'>"
    print "<input type='hidden' name='batch_id' value='%(batch_id)d'/>"%(my_batch)
    print """<input type='submit' 
                     value='Kill all incomplete jobs' 
                     onclick='return confirm("Do you really want to kill all jobs?")' />"""
    print "</form></td>"
    #
    # Resubmit buttons
    #
    print """
    <td><form action='ViewBatch.py' method='POST' target='ResubmitWindow'>
    <input type='hidden' name='batch_id' value='%(batch_id)d' />
    <input type='hidden' name='submit_run' value='All' />
    <input type='submit' value='Resubmit all'
           onclick='return confirm("Do you really want to resubmit all batches?")' />
    </form></td>
    <td><form action='ViewBatch.py' method='POST' target='ResubmitWindow' style='position:relative; float:left'>
    <input type='hidden' name='batch_id' value='%(batch_id)d' />
    <input type='hidden' name='submit_run' value='Incomplete' />
    <input type='submit' value='Resubmit incomplete'
           title='Resubmit jobs for batches that have not successfully completed'
           onclick='return confirm("Do you really want to resubmit incomplete batches?")' />
    </form></td></tr></table>
"""%(my_batch)
    print "</div>"

    #
    # Fix permissions
    #
    print "<div style='position:relative; float:top'>"
    print "<form action='FixPermissions.py'>"
    print "<input type='hidden' name='batch_id' value='%(batch_id)d'/>"%(my_batch)
    print """<input type='submit' 
                     value='Fix file permissions' />"""
    print "</form>"
    print "</div>"
    print "</div>"
    #
    # Upload to database table
    #
    sql_files = []
    for filename in os.listdir(my_batch["data_dir"]):
        if filename.upper().endswith(".SQL"):
            sql_files.append(filename)
    if len(sql_files):
        print "<div style='clear:both; padding-top:10px'>"
        print "<h2>Database scripts</h2>"
        print "<table class='run_table'><tr><th>Script file</th><th>Action</th><th>Last job id</th><th>Status</th><th>Run time</th><th>Output</th></tr>"
        for filename in sql_files:
            if filename.startswith('batch_'):
                continue
            job_id = sql_jobs.sql_file_job_id(batch_id, filename)
            print "<tr><td>%s</td>"%(filename)
            print "<td>"
            run_button = True
            output_file = filename[:-3]+'out'
            output_path = os.path.join(my_batch["data_dir"],output_file)
            if not job_id is None:
                status = sql_jobs.sql_job_status(job_id)
                if status in ('PEND','PSUSP','RUN'):
                    # A kill button for jobs that are killable
                    print "<form action ='KillJobs.py' method='POST' target='KillJob'>"
                    print "<input type='hidden' name='job_id' value='%s' />"%(job_id)
                    print """<input type='submit' value='Kill'
                                    onclick='confirm("Are you sure you want to kill the database upload?")' />"""
                    run_button = False
            if run_button:
                print "<form action='UploadToDatabase.py' method='POST'>"
                print "<input type='hidden' name='sql_script' value='%s' />"%(filename)
                print "<input type='hidden' name='output_file' value='%s' />"%(output_path)
                print "<input type='hidden' name='batch_id' value='%(batch_id)d'/>"%(my_batch)
                print "<span style='white-space:nowrap'>"
                print """<input type='submit'
                                 value='Run'
                                 onclick='confirm("Are you sure you want to upload to the database using %(filename)s?")' />"""%(globals())
                print "&nbsp;Queue:<select name='queue'>"
                print "<option value='hour'>Hour</option>"
                print "<option value='week'>Week</option>"
                print "<option value='priority'>Priority</option>"
                print "</select>"
                print "</span>"
                print "</form>"
            print "</td>"
            if job_id is None:
                print "<td colspan='4'>not run</td>"
            else:
                run_time = sql_jobs.sql_job_run_time(job_id)
                print "<td>%s</td><td>%s</td>"%(job_id,status)
                if run_time is None:
                    print "<td>-</td>"
                else:
                    print "<td>%d sec</td>"%(run_time.seconds)
                if status in ('DONE','EXIT'):
                    print "<td><a href='ViewTextFile.py?file_name=%(output_path)s'>%(output_file)s</a></td>"%(globals())
                elif status == 'RUN':
                    print "<td><a href='BPeek.py?job_id=%(job_id)d'>job output</a></td>"%(globals())
                else:
                    print "<td>-</td>"
            print "</tr>"
        print "</table>"
        print "</div>"
    #
    # The big table
    #
    print "<div style='clear:both; padding-top:10px'>"
    print "<table class='run_table'>"
    print "<thead><tr>"
    print "<th>Start</th><th>End</th><th>Job #</th><th>Status</th><th>Text output file</th><th>Results file</th>"
    print "<th><div>"
    print "<div style='position:relative; float=top'>Delete files</div>"
    print """<div style='position:relative; float=top'>
    <form action='DeleteFile.py' method='POST'>
    <input type='hidden' name='batch_id' value='%(batch_id)d'/>
    <input type='submit' 
                         name='delete_action' 
                         value='Text'   
                         title='Delete all text files' 
                         onclick='return confirm("Do you really want to delete all text files for this batch?")'/>
    <input type='submit' 
                         name='delete_action' 
                         value='Output'   
                         title='Delete output files' 
                         onclick='return confirm("Do you really want to delete all output files for this batch?")'/>
    <input type='submit' 
                         name='delete_action' 
                         value='Done'   
                         title='Delete done files' 
                         onclick='return confirm("Do you really want to delete all done files for this batch?")'/>
    <input type='submit' 
                         name='delete_action' 
                         value='All'   
                         title='Delete all files for this batch' 
                         onclick='return confirm("Do you really want to delete all files for this batch?")'/>
    </form>
    </div>"""%(my_batch)
    print "</div></th>"
    print "</tr></thead>"
    print "<tbody>"
    for run in my_batch["runs"]:
        x = my_batch.copy()
        x.update(run)
        x["text_file"] = RunBatch.RunTextFile(run)
        x["text_path"] = RunBatch.RunTextFilePath(my_batch,run)
        x["done_file"] = RunBatch.RunDoneFile(run)
        x["done_path"]=  RunBatch.RunDoneFilePath(my_batch,run)
        x["out_file"] =  RunBatch.RunOutFile(run)
        x["out_path"] =  RunBatch.RunOutFilePath(my_batch,run)
        print "<tr>"
        print "<td>%(start)d</td><td>%(end)d</td><td>%(job_id)s</td>"%(x)
        if run["status"] == "Complete":
            cpu = RunBatch.GetCPUTime(my_batch,run)
            print "<td style='color:green'>"
            if cpu:
                print "Complete (%.2f sec)"%(cpu)
            else:
                print "Complete"
            stat = "Complete"
            print "</td>"
        else:
            print """
<td style='color:red'>%s<br/>
    <form action='ViewBatch.py' method='POST' target='ResubmitWindow'>
    <input type='hidden' name='batch_id' value='%d' />
    <input type='hidden' name='submit_run' value='%d' />
    <input type='submit' value='Resubmit' />
    </form>
</td>"""%(run["status"],batch_id,run["run_id"])
        if jobs_by_state.has_key(stat):
            jobs_by_state[stat].append(run)
        else :
            jobs_by_state[stat] = [ run ]
        print "<td>"
        if os.path.isfile(x["text_path"]):
            print "<a href='ViewTextFile.py?run_id=%(run_id)d' title='%(text_path)s'>%(text_file)s</a>"%(x)
        elif run["status"]=='RUN':
            print "<a href='BPeek.py?job_id=%(job_id)d'>Peek</a>"%(x)
        else :
            print "<span title='Text file not yet available'>%(text_file)s</span>"%(x)
        print "</td>"
        print "<td title='%(out_path)s'>%(out_file)s</td>"%(x)
        #
        # This cell contains the form that deletes things
        #
        print "<td><form action='DeleteFile.py' method='POST'>"
        print "<input type='hidden' name='run_id' value='%(run_id)d' />"%(x)
        print """<input type='submit' 
                         name='delete_action' 
                         value='Text'   
                         title='Delete file %(text_path)s' 
                         onclick='return confirm("Do you really want to delete %(text_file)s?")'/>"""%(x)
        print """<input type='submit' 
                         name='delete_action' 
                         value='Output'   
                         title='Delete file %(out_path)s' 
                         onclick='return confirm("Do you really want to delete %(out_file)s?")'/>"""%(x)
        print """<input type='submit' 
                         name='delete_action' 
                         value='Done'   
                         title='Delete file %(done_path)s' 
                         onclick='return confirm("Do you really want to delete %(done_file)s?")'/>"""%(x)
        print """<input type='submit' 
                         name='delete_action' 
                         value='All'   
                         title='Delete all files for this run' 
                         onclick='return confirm("Do you really want to delete %(text_file)s, %(out_file)s and %(done_file)s?")'/>"""%(x)
        print "</form></td>"
        print "</tr>"
    print "</tbody>"
    print "</table>"
    print "</div>"
    print "</body>"
    print "</html>"
try:
    import cellprofiler.utilities.jutil as jutil
    jutil.kill_vm()
except:
    import traceback
    traceback.print_exc()
