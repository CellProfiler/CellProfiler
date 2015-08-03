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
# View a batch from the database, with some options to re-execute it
#
import cgitb
cgitb.enable()
import RunBatch
from bpformdata import *
import bputilities
import sql_jobs
import StyleSheet
import cgi
import math
import os
import os.path
import yattag

RESUBMIT="Resubmit"

resubmit_action = \
    'return confirm("Do you really want to resubmit %s batches?")'

print "Content-Type: text/html"
print

class ViewBatchDoc(object):
    def __init__(self):
        defaults = BATCHPROFILER_DEFAULTS.copy()
        del defaults[SUBMIT_RUN]
        del defaults[RUN_ID]
        del defaults[JOB_ID]
        del defaults[BATCH_ARRAY_ID]
        del defaults[TASK_ID]
        del defaults[SQL_SCRIPT]
        del defaults[OUTPUT_FILE]
        del defaults[K_DELETE_ACTION]
        del defaults[FILE_TYPE]
        del defaults[K_STATUS]
        if defaults[PAGE_SIZE] is None:
            defaults[PAGE_SIZE] = 25
        self.doc, self.tag, self.text = yattag.Doc(defaults).tagtext()
        assert isinstance(self.doc, yattag.Doc)
        self.batch_id = BATCHPROFILER_VARIABLES[BATCH_ID]
        self.run_id = BATCHPROFILER_VARIABLES[RUN_ID]
        self.submit_run = BATCHPROFILER_VARIABLES[SUBMIT_RUN]
        self.status = BATCHPROFILER_VARIABLES[K_STATUS]
        
    def build(self):        
        if self.batch_id is None:
            self.build_batch_list()
            return
        self.read_batch()
        if self.submit_run is not None:
            self.build_submit_run()
        else:
            self.build_batch()
    
    def read_batch(self):
        self.my_batch = RunBatch.BPBatch.select(self.batch_id)
        self.jobs_by_state = self.my_batch.select_task_count_group_by_state(
            RunBatch.RT_CELLPROFILER)
        page_size = BATCHPROFILER_DEFAULTS[PAGE_SIZE] or 25
        first_item = BATCHPROFILER_DEFAULTS[FIRST_ITEM] or 1
        self.tasks = self.my_batch.select_tasks(
            page_size = page_size,
            first_item = first_item,
            state = self.status)
        def comparator(a, b):
            run_a = a.job_task.batch_array_task.run
            run_b = b.job_task.batch_array_task.run
            return cmp(run_a.bstart, run_b.bstart)
        self.tasks.sort(cmp=comparator)

    def build_batch_list(self):
        page_size = BATCHPROFILER_DEFAULTS[PAGE_SIZE] or 25
        first_item = BATCHPROFILER_DEFAULTS[FIRST_ITEM] or 1
        count = RunBatch.BPBatch.select_batch_count()
        batches = RunBatch.BPBatch.select_batch_range(first_item, page_size)
        if len(batches) == 0:
            first_item = 1
            batches = RunBatch.BPBatch.select_batch_range(first_item, page_size)
        with self.tag("html"):
            with self.tag("head"):
                with self.tag("title"):
                    self.text("Batch Profiler: View Batches")
                with self.tag("style"):
                    self.doc.asis(StyleSheet.BATCHPROFILER_STYLE)
            with self.tag("body"):
                with self.tag("h1"):
                    self.text("Batches %d to %d" % (batches[-1].batch_id,
                                                    batches[0].batch_id))
                with self.tag("div"):
                    with self.tag("table", klass="run_table"):
                        with self.tag("tr"):
                            for caption in (
                                "Batch ID", "Data directory", "Project", "Email"):
                                with self.tag("th"):
                                    self.text(caption)
                        for batch in batches:
                            with self.tag("tr"):
                                with self.tag("td"):
                                    url = "ViewBatch.py?%s=%d" % \
                                        (BATCH_ID, batch.batch_id)
                                    with self.tag("a", href=url):
                                        self.text(str(batch.batch_id))
                                for field in batch.data_dir, batch.project,\
                                    batch.email:
                                    with self.tag("td"):
                                        self.text(field)
                with self.tag("div"):
                    page_starts = []
                    if first_item > page_size:
                        page_starts.append((first_item - page_size, "Prev"))
                    page_starts += \
                        [(i, str(i)) for i in range(1, count+1, page_size)]
                    if first_item + page_size < count:
                        page_starts += [(first_item + page_size, "Next")]
                    for item, label in page_starts:
                        url = "ViewBatch.py?%s=%d&%s=%d" % (
                            FIRST_ITEM, item, PAGE_SIZE, page_size)
                        style = "padding-left: 1em;padding-right: 1em"
                        with self.tag("a", href=url, style=style):
                            self.text(label)
                            
    def build_submit_run(self):
        title = "Batch # %d resubmitted"%(self.my_batch.batch_id)
        with self.tag("html"):
            with self.tag("head"):
                with self.tag("title"):
                    self.text(title)
                with self.tag("style"):
                    self.doc.asis(StyleSheet.BATCHPROFILER_STYLE)
            with self.tag("body"):
                with self.tag("h1"):
                    self.text(title)
                with self.tag("div"):
                    self.text("Your batch # %d has been resubmitted." %
                              self.my_batch.batch_id)
                with self.tag("table", klass="run-table"):
                    with self.tag("tr"):
                        with self.tag("th"):
                            self.text("First")
                        with self.tag("th"):
                            self.text("Last")
                        with self.tag("th"):
                            self.text("Job #")
                        with self.tag("th"):
                            self.text("Task #")
                    kwds = {}
                    run_id = BATCHPROFILER_VARIABLES[RUN_ID]
                    submit_run = BATCHPROFILER_VARIABLES[SUBMIT_RUN]
                    if submit_run == RESUBMIT or submit_run is None:
                        runs = batch.select_runs()
                    if run_id is not None:
                        runs = [RunBatch.BPRun.select(run_id)]
                    else:
                        if submit_run == RunBatch.JS_INCOMPLETE:
                            statuses = RunBatch.INCOMPLETE_STATUSES
                        else:
                            statuses = [submit_run]
                        tasks = self.my_batch.select_tasks_by_state(statuses)
                        runs = [task.batch_array_task.run for task in tasks]
                    tasks = RunBatch.run(self.my_batch, runs)
                    for task in tasks:
                        assert isinstance(task, RunBatch.BPJobTask)
                        run = task.batch_array_task.run
                        assert isinstance(run, RunBatch.BPRun)
                        with self.tag("tr"):
                            with self.tag("td"):
                                self.text(str(run.bstart))
                            with self.tag("td"):
                                self.text(str(run.bend))
                            with self.tag("td"):
                                self.text(str(task.job.job_id))
                            with self.tag("td"):
                                self.text(str(task.batch_array_task.task_id))
    def build_batch(self):
        title = "View batch # %d" % self.batch_id
        with self.tag("html"):
            with self.tag("head"):
                with self.tag("title"):
                    self.text(title)
                with self.tag("style"):
                    self.doc.asis(StyleSheet.BATCHPROFILER_STYLE)
                self.build_scripts()
            with self.tag("h1"):
                self.text(title)
            with self.tag("div", style="padding:5px"):
                self.build_summary_table()
                self.build_summary_buttons()
                self.build_database_scripts()
            with self.tag("div", style='clear:both; padding-top:10px'):
                self.build_job_table()
            with self.tag("div", style='clear:both; padding-top:10px'):
                self.build_footer()
                
                
    def build_scripts(self):
        fix_permissions = """
function fix_permissions() {
    var button = document.getElementById("fix_permissions_button")
    var oldInnerText = button.innerText;
    button.innerText = "Fixing...";
    button.disabled = true;
    var xmlhttp = new XMLHttpRequest();
    var params = "%s=%d";
    xmlhttp.onreadystatechange=function() {
        if (xmlhttp.readyState == 4) {
            if ((xmlhttp.status >= 200)  && (xmlhttp.status < 300)) {
                var result=JSON.parse(xmlhttp.responseText);
                alert("Changed " + Object.keys(result).length + " permissions");
            } else {
                alert("Failed to change permissions");
            }
            button.innerText = oldInnerText;
            button.disabled = false;
        }
    }
    xmlhttp.open(
        "POST",
        "FixPermissions.py", true);
    xmlhttp.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xmlhttp.setRequestHeader("Content-Length", params.length);
    xmlhttp.send(params);
}""" % (BATCH_ID, self.batch_id)
        with self.tag("script", language="javascript"):
            self.doc.asis(fix_permissions)
                
    def build_summary_table(self):
        confirm_action =\
            'return confirm("Do you really want to resubmit all %s batches?")'
        with self.tag("div", style='position:relative;float:left'):
            with self.tag("table", klass='run_table'):
                with self.tag("thead"):
                    with self.tag("tr"):
                        with self.tag("th", style='align:center', colspan='3'):
                            self.text("Job Summary")
                    with self.tag("tr"):
                        for caption in "Status", "Count", "Resubmit jobs":
                            with self.tag("th"):
                                self.text(caption)
                with self.tag("tbody"):
                    for state in self.jobs_by_state:
                        visible_state = state.lower()
                        with self.tag("tr"):
                            with self.tag("td"):
                                vburl = "ViewBatch.py?%s=%d&%s=%s" % (
                                    BATCH_ID, self.batch_id, K_STATUS, state)
                                with self.tag("a", href = vburl):
                                    self.text(state)
                            with self.tag("td"):
                                self.text(str(self.jobs_by_state[state]))
                            with self.tag("td"):
                                with self.tag("form", 
                                              action = "ViewBatch.py",
                                              method = "POST",
                                              target = "ResubmitWindow"):
                                    self.doc.input(type='hidden',
                                                   name=BATCH_ID)
                                    self.doc.input(type='hidden',
                                                   name=SUBMIT_RUN,
                                                   value=state)
                                    self.doc.stag(
                                        "input",
                                        type='submit',
                                        value='Resubmit %s' %visible_state,
                                        onclick = confirm_action % visible_state)
                    #
                    # The "All" summary row is a place to put things that
                    # affect all runs.
                    #
                    with self.tag("tr"):
                        with self.tag("td"):
                            with self.tag("a", href="ViewBatch.py?%s=%d" %
                                          (BATCH_ID, self.batch_id)):
                                self.text("All")
                        with self.tag("td"):
                            self.text(str(sum(self.jobs_by_state.values())))
                        with self.tag("td"):
                            with self.tag("form", 
                                          action = "ViewBatch.py",
                                          method = "Post",
                                          target = "ResubmitWindow"):
                                self.doc.input(
                                    type='hidden', name=BATCH_ID)
                                self.doc.input(type='hidden',
                                               name=SUBMIT_RUN,
                                               value=RunBatch.JS_ALL)
                                self.doc.stag(
                                    "input",
                                    type='submit',
                                    name="Resubmit_button",
                                    value="Resubmit all",
                                    onclick=resubmit_action % "all")
                            
    def build_summary_buttons(self):
        kill_action = 'return confirm("Do you really want to kill all jobs?")'
        with self.tag("div", style='position:relative; float:left; padding:2px'):
            with self.tag("div", style='position:relative; float:top'):
                with self.tag("table"):
                    with self.tag("tr"):
                        with self.tag("td"):
                            with self.tag("form", action='KillJobs.py'):
                                self.doc.input(type='hidden', name=BATCH_ID)
                                self.doc.stag(
                                    "input",
                                    type='submit',
                                    name='kill_jobs_button',
                                    value='Kill all incomplete jobs',
                                    onclick = kill_action)
                            for status, label, text in (
                                (RunBatch.JS_INCOMPLETE, "Resubmit incomplete",
                                 "all incomplete"),):
                                with self.tag("td"):
                                    with self.tag("form", 
                                                  action = "ViewBatch.py",
                                                  method = "Post",
                                                  target = "ResubmitWindow"):
                                        self.doc.input(
                                            type='hidden', name=BATCH_ID)
                                        self.doc.input(type='hidden',
                                                       name=SUBMIT_RUN,
                                                       value=status)
                                        self.doc.stag(
                                            "input",
                                            type='submit',
                                            name="Resubmit_button",
                                            value=label,
                                            onclick=resubmit_action % text)
                    with self.tag("tr"):
                        with self.tag("td", style="text-align:left"):
                            with self.tag("button", type="button",
                                          id="fix_permissions_button",
                                          onclick="fix_permissions()"):
                                self.text("Fix file permissions")
                        with self.tag("td", style="text-align:left"):
                            with self.tag("form", action='GetPipeline.py'):
                                self.doc.input(type="hidden", name=BATCH_ID)
                                self.doc.stag("input", type="submit",
                                               value="Download pipeline")
    def build_database_scripts(self):
        sql_files = []
        kill_db_action = 'confirm("Are you sure you want to kill the database upload?")'
        upload_action = 'confirm("Are you sure you want to upload to the database using %s?")'
        for filename in os.listdir(self.my_batch.data_dir):
            if filename.upper().endswith(".SQL"):
                sql_files.append(filename)
        if len(sql_files) == 0:
            return
            
        with self.tag("div", style='clear:both; padding-top:10px'):
            with self.tag("h2"):
                self.text("Database scripts")
            with self.tag("table", klass="run_table"):
                with self.tag("tr"):
                    for caption in ("Script file", "Action", "Last job id", 
                                    "Status", "Output"):
                        with self.tag("th"):
                            self.text(caption)
                for filename in sql_files:
                    if filename.startswith('batch_'):
                        continue
                    run, task, task_status = sql_jobs.sql_file_job_and_status(
                        self.my_batch.batch_id, 
                        RunBatch.batch_script_file(filename))
                    status = None if task_status is None else task_status.status
                    with self.tag("tr"):
                        with self.tag("td"):
                            self.text(filename)
                        with self.tag("td", style="text-align:left"):
                            run_button = True
                            output_file = filename[:-3]+'out'
                            output_path = os.path.join(
                                self.my_batch.data_dir, output_file)
                            if task is not None and task_status.status in (
                                RunBatch.JS_RUNNING, RunBatch.JS_SUBMITTED):
                                #
                                # A kill button for jobs that are killable
                                with self.tag("div"):
                                    with self.tag("form", 
                                              action="KillJobs.py",
                                              method="POST",
                                              target="KillJob"):
                                        self.doc.input(type='hidden', 
                                                       name=JOB_ID,
                                                       value=str(job.job_id))
                                        self.doc.stag(
                                            "input",
                                            type='submit', value="Kill",
                                            name="Kill_db_button",
                                            onclick=kill_db_action)
                            with self.tag("div"):
                                with self.tag("form",
                                              action = "UploadToDatabase.py",
                                              method = "POST"):
                                    self.doc.input(
                                        type='hidden',
                                        name=SQL_SCRIPT,
                                        value=filename)
                                    self.doc.input(type='hidden', 
                                                   name=OUTPUT_FILE,
                                                   value = output_path)
                                    self.doc.input(type='hidden',
                                                   name=BATCH_ID)
                                    with self.tag(
                                        "span", style='white-space:nowrap'):
                                        self.doc.stag(
                                            "input",
                                            type="submit",
                                            value="Run",
                                            onclick=upload_action % filename)
                                        self.doc.asis("&nbsp;Queue:")
                                        with self.doc.select(name=QUEUE):
                                            for queue_name in\
                                                bputilities.get_queues():
                                                with self.doc.option(
                                                    value=queue_name):
                                                    self.text(queue_name)
                        if task is None:
                            with self.tag("td", colspan="4"):
                                self.text("not run")
                        else:
                            with self.tag("td"):
                                self.text(str(task.job.job_id))
                            with self.tag("td"):
                                if status != RunBatch.JS_DONE:
                                    self.text(status)
                                else:
                                    run_time = \
                                        task_status.created - task.job.created
                                    self.text("Complete(%.2f sec)" % 
                                              run_time.total_seconds())
                            self.build_text_file_table_cell(task)
                            
    def build_job_table(self):
        with self.tag("div"):
            with self.tag("table", klass="run_table"):
                self.build_job_table_head()
                self.build_job_table_body()
    
    def build_job_table_head(self):
        with self.tag("thead"):
            with self.tag("tr"):
                for caption in "Start", "End", "Job #", "Task #", "Status", \
                    "Text output file":
                    with self.tag("th"):
                        self.text(caption)
                if self.my_batch.wants_measurements_file:
                    with self.tag("th"):
                        self.text("Results file")
                with self.tag("th"):
                    with self.tag("div"):
                        with self.tag(
                            "div", style='position:relative; float=top'):
                            self.text("Delete files")
                            with self.tag(
                                "div", style='position:relative; float=top'):
                                with self.tag("form", 
                                              action="DeleteFile.py",
                                              method="POST"):
                                    self.doc.input(type="hidden",
                                                   name=BATCH_ID)
                                    self.doc.stag(
                                        "input",
                                        type="submit",
                                        name=K_DELETE_ACTION,
                                        value=A_DELETE_TEXT,
                                        title='Delete all text files',
                                        onclick='return confirm("Do you really want to delete all text files for this batch?")')
                                    self.doc.stag(
                                        "input",
                                        type="submit",
                                        name=K_DELETE_ACTION,
                                        value=A_DELETE_OUTPUT,
                                        title="Delete output files",
                                        onclick='return confirm("Do you really want to delete all output files for this batch?")')
                                    self.doc.stag(
                                        "input",
                                        type="submit",
                                        name=K_DELETE_ACTION,
                                        value=A_DELETE_ALL,
                                        title='Delete all files for this batch',
                                        onclick='return confirm("Do you really want to delete all files for this batch?")')
    def build_job_table_body(self):
        with self.tag("tbody"):
            for task_status in self.tasks:
                assert isinstance(task_status, RunBatch.BPJobTaskStatus)
                task = task_status.job_task
                status = task_status.status
                assert isinstance(task, RunBatch.BPJobTask)
                run = task.batch_array_task.run
                assert isinstance(run, RunBatch.BPRun)
                job = task.job
                assert isinstance(job, RunBatch.BPJob)
                out_file = RunBatch.run_out_file(self.my_batch, run)
                out_path = RunBatch.run_out_file_path(self.my_batch, run)
                with self.tag("tr"):
                    with self.tag("td"):
                        self.text(str(run.bstart))
                    with self.tag("td"):
                        self.text(str(run.bend))
                    with self.tag("td"):
                        self.text(str(job.job_id))
                    with self.tag("td"):
                        self.text(str(task.batch_array_task.task_id))
                    if status == RunBatch.JS_DONE:
                        with self.tag("td"):
                            cpu = task_status.created - job.created
                            self.text("Complete (%.2f sec)" %
                                      (cpu.total_seconds()))
                    else:
                        with self.tag("td", style='color:red'):
                            with self.tag("div"):
                                self.text(status.lower().capitalize())
                            with self.tag("div"):
                                with self.tag("form",
                                              action="ViewBatch.py",
                                              method="POST",
                                              target="ResubmitWindow"):
                                    self.doc.input(
                                        type="hidden", name=BATCH_ID)
                                    self.doc.input(
                                        type="hidden",
                                        name=RUN_ID,
                                        value=str(run.run_id))
                                    self.doc.stag(
                                        "input",
                                        type='submit',
                                        name=SUBMIT_RUN,
                                        value=RESUBMIT)
                    self.build_text_file_table_cell(task)
                    if self.my_batch.wants_measurements_file:
                        with self.tag("td"):
                            if os.path.isfile(out_path):
                                with self.tag(
                                    "a", 
                                    href='ViewTextFile.py?run_id=%d&%s=%s' % 
                                    (run.run_id, FILE_TYPE, FT_OUT_FILE),
                                    title=out_path):
                                    self.text(out_file)
                            else:
                                with self.tag("span", 
                                         title='Output file not available'):
                                    self.text(out_file)
                    #
                    # This cell contains the form that deletes things
                    #
                    with self.tag("td"):
                        with self.tag("form", action='DeleteFile.py', method='POST'):
                            self.doc.input(
                                type="hidden", name=RUN_ID,
                                value = str(run.run_id))
                            text_file = os.path.basename(
                                RunBatch.batch_array_task_text_file_path(
                                    task.batch_array_task))
                            for action, filename in (
                                (A_DELETE_TEXT, text_file),
                                (A_DELETE_OUTPUT, RunBatch.run_out_file(self.my_batch, run)),
                                (A_DELETE_ALL, "all files for this run")):
                                self.doc.stag(
                                    "input",
                                    type="submit", name=K_DELETE_ACTION,
                                    value=action,
                                    title='Delete file %s' % filename, 
                                    onclick='return confirm("Do you really want'
                                    'to delete %s?")' % filename)
    def build_footer(self):
        '''Build the footer for scrolling through the pages'''
        page_size = BATCHPROFILER_DEFAULTS[PAGE_SIZE] or 25
        first_item = BATCHPROFILER_VARIABLES[FIRST_ITEM] or 1
        if self.status is None:
            count = self.my_batch.select_task_count(RunBatch.RT_CELLPROFILER)
        else:
            count = self.my_batch.select_task_count_by_state(
                RunBatch.RT_CELLPROFILER, self.status)
        with self.tag("table"):
            with self.tag("tr"):
                with self.tag("td"):
                    with self.tag("form", action="ViewBatch.py"):
                        self.doc.input(type="hidden", name=BATCH_ID)
                        self.doc.text("Jobs / page:")
                        self.doc.input(type="text", 
                                       name=PAGE_SIZE, 
                                       id="input_%s" % PAGE_SIZE)
                        self.doc.stag("input", type="submit", value="Set")
                if count > page_size:
                    links = []
                    if first_item > page_size:
                        links.append((first_item, "Previous"))
                    links += [(i, str(i)) for i in range(1, count+1, page_size)]
                    if first_item + page_size <= count:
                        links += [(first_item + page_size, "Next")]
                    for item, label in links:
                        query = [(BATCH_ID, str(self.my_batch.batch_id)),
                                 (FIRST_ITEM, str(item)),
                                 (PAGE_SIZE, str(page_size))]
                        if self.status is not None:
                            query.append((K_STATUS, self.status))
                        url = "ViewBatch.py?" + "&".join([
                            "%s=%s" % (k, v) for k, v in query])
                        with self.tag("td", style="padding-left: 2pt"):
                            if item == first_item and label == str(item):
                                self.text(str(item))
                            else:
                                with self.tag(
                                    "a", 
                                    href=url,
                                    style="text-align: center;"
                                    "text-decoration: none;display: block;"): 
                                    self.text(label)

    def build_text_file_table_cell(self, task):
        '''Build a table cell containing links to the stdout/err output'''
        bat = task.batch_array_task
        text_path = RunBatch.batch_array_task_text_file_path(bat)
        text_file = os.path.basename(text_path)
        err_path = RunBatch.batch_array_task_err_file_path(bat)
        err_file = os.path.basename(err_path)
        with self.tag("td", style="text-align: left"):
            for ft, path, filename in (
                (FT_TEXT_FILE, text_path, text_file),
                (FT_ERR_FILE, err_path, err_file)):
                with self.tag("div"):
                    if os.path.isfile(path):
                        with self.tag(
                            "a", style="text-align: left",
                            href='ViewTextFile.py?%s=%d&%s=%d&%s=%s' % 
                            (BATCH_ARRAY_ID, bat.batch_array.batch_array_id,
                             TASK_ID, bat.task_id, FILE_TYPE, ft),
                            title=path):
                            self.text(filename)
                    else:
                        with self.tag(
                            "span", 
                            style="text-align: left",
                            title='Text file not available'):
                            self.text(filename)

    def render(self):
        print StyleSheet.BATCHPROFILER_DOCTYPE
        print yattag.indent(self.doc.getvalue())
        
with bputilities.CellProfilerContext():
    doc = ViewBatchDoc()
    doc.build()
doc.render()