#!/usr/bin/env ./batchprofiler.sh
#
# Start a batch operation from a web page
#
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

import cgitb
cgitb.enable()
print "Content-Type: text/html\r"
print "\r"
import sys
import cgi
import os
import numpy as np
import traceback
import urllib
import cellprofiler.preferences
cellprofiler.preferences.set_headless()
from cellprofiler.modules.createbatchfiles import F_BATCH_DATA_H5
from bputilities import *
from bpformdata import *
import RunBatch
from StyleSheet import BATCHPROFILER_DOCTYPE
import email.message
import email.mime.text
import socket
from cStringIO import StringIO
import yattag

BUILD_CPCLUSTER_BUTTON = "build_cpcluster"
HOW_TO_USE_BATCHPROFILER_URL = "http://dev.broadinstitute.org/imaging/privatewiki/index.php/How_To_Use_BatchProfiler"
myself = os.path.split(
    os.environ.get(SCRIPT_FILENAME_KEY,
                   "/batchprofiler/cgi-bin/NewBatch.py"))[1]
class NewBatchDoc(object):
    def __init__(self):
        defaults = BATCHPROFILER_DEFAULTS.copy()
        del defaults[SUBMIT_BATCH]
        self.doc, self.tag, self.text = yattag.Doc(defaults=defaults).tagtext()
        assert isinstance(self.doc, yattag.Doc)
        self.__has_image_sets = None
        self.__inputs_validated = None
        self.__cpcluster_needs_building = None
        self.no_image_sets_reason = "???"
        self.__submit_batch_pressed = \
            BATCHPROFILER_DEFAULTS.get(SUBMIT_BATCH, False) == "yes"
    
    def wants_batch_submission(self):
        if self.__submit_batch_pressed and self.has_image_sets:
            return self.inputs_validated
        return False
    
    def build(self):
        if self.wants_batch_submission():
            self.build_submit_batch()
        else:
            self.build_normal()
            
    def build_normal(self):
        self.build_head()
        self.build_body()
        
    def build_head(self):
        with self.tag("head"):
            with self.tag("title"):
                self.text("CellProfiler 2.0 Batch Submission")
            with self.tag("script", language="JavaScript"):
                self.doc.asis("""
function go_to_key(key, data_dir) {
    url='%(myself)s';
    add_char = "?";
    all_k = new Array("%(EMAIL)s","%(QUEUE)s","%(PRIORITY)s",
                      "%(PROJECT)s","%(BATCH_SIZE)s","%(MEMORY_LIMIT)s",
                      "%(REVISION)s");
    for (k in all_k) {
        v = document.getElementById('input_'+all_k[k]);
        url = url+add_char+all_k[k]+'='+escape(v.value);
        add_char = "&";
    }
    url = url + add_char + "%(DATA_DIR)s="+escape(data_dir);
    v = document.getElementById('input_write_data');
    if (v.checked)
    {
        url = url + add_char + "write_data=yes";
    } else {
        url = url + add_char + "write_data=no";
    }
parent.location = url+"#input_"+key;
}
                """ % globals())
                if self.has_image_sets and self.batch_git_hash is not None:
                    self.doc.asis(("""
function use_githash() {
    v = document.getElementById("input_%(REVISION)s");
    v.value = "%%s";
}""" % globals()) % self.batch_git_hash)
                self.doc.asis("""
function build_cellprofiler() {
    var v = document.getElementById("input_%(REVISION)s");
    var email = document.getElementById("input_%(EMAIL)s");
    var button = document.getElementById("%(BUILD_CPCLUSTER_BUTTON)s");
    var revision = v.value;
    button.innerText = "Building...";
    button.disabled = true;
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange=function() {
        if (xmlhttp.readyState == 4) {
            if ((xmlhttp.status >= 200)  && (xmlhttp.status < 300)) {
                var result=JSON.parse(xmlhttp.responseText);
                if (result['%(IS_BUILT)s']) {
                    alert(revision + " is already built");
                } else {
                    alert("Building " +revision);
                }
            } else {
                alert("Failed to build " + revision);
            }
            button.innerText = "Build CellProfiler"
            button.disabled = false;
        }
    }
    xmlhttp.open(
        "PUT",
        "BuildCellProfiler.py?%(REVISION)s="+encodeURI(revision), true);
    xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xmlhttp.send(JSON.stringify({ %(EMAIL)s:email }));
}
""" % globals())
                
    def build_body(self):
        with self.tag("body"):
            with self.tag("h1"):
                self.text("CellProfiler 2.1.1+ Batch Submission")
            if self.__submit_batch_pressed:
                with self.tag("div"):
                    self.text("Submit batch pressed")
            self.build_introduction()
            self.build_form()
            #self.build_diagnostics()

    def build_diagnostics(self):
        try:
            messages = (
                "Has image sets " if self.has_image_sets else self.no_image_sets_reason,
                "Inputs validated" if self.inputs_validated else self.invalid_reason,
                "Has groups" if self.has_groups else "No groups")
            for message in messages:
                with self.tag("div"):
                    self.text(message)
        except:
            with self.tag("div"):
                self.text(traceback.format_exc())
                
        if self.has_image_sets:
            with self.tag("div"):
                if self.batch_git_hash is None:
                    self.text(self.no_batch_version_reason)
                else:
                    self.text("Batch_data version: %s, git_hash: %s" %
                              (self.batch_datetime_version,
                               self.batch_git_hash))
        
        with self.tag("table"):
            for k, v in BATCHPROFILER_DEFAULTS.iteritems():
                with self.tag("tr"):
                    for value in k, v:
                        with self.tag("td"):
                            self.text(str(value))
                            
    def build_introduction(self):
        with self.tag("div"):
            with self.tag("p"):
                self.text(
                    "This webpage will let you submit a Batch_data.h5 "
                    "file produced by CellProfiler 2.1.1 or later to "
                    "the cluster.")
            with self.tag("p"):
                self.text("""
For details on the settings below and for general help on submitting a 
CellProfiler job to the LSF, please see this """)
                with self.tag("a", href=HOW_TO_USE_BATCHPROFILER_URL):
                    self.text("page")
                self.text(".")
            with self.tag("p"):
                self.text("""
Submit a %(F_BATCH_DATA_H5)s file created by CellProfiler 2.0. You need to
specify the default output folder, which should contain your Batch_data file
for the pipeline. In addition, there are some parameters that tailor how the
batch is run.""" % globals())
                
    def build_form(self):
        with self.tag("form", action=myself, method="POST"):
            self.doc.input(type="hidden", name=SUBMIT_BATCH, value="yes")
            with self.tag("table", style='white-space=nowrap'):
                for label, name, kwds, fn in (
                    ("E-mail", EMAIL, dict(size="40"), None),
                    ("Queue", QUEUE, None, self.build_queue_choices),
                    ("Priority", PRIORITY, {}, None),
                    ("Project", PROJECT, {}, None),
                    ("Batch size", BATCH_SIZE, {}, None),
                    ("Memory limit", MEMORY_LIMIT, {}, None),
                    ("Write data", WRITE_DATA, {}, self.build_write_data), 
                    ("Revision", REVISION, {}, self.build_revision)):
                    id = "input_%s" % name
                    with self.tag("tr"):
                        with self.tag("th"):
                            self.text("%s:" % label)
                        with self.tag("td"):
                            if fn is None:
                                self.doc.input(
                                    type="text",
                                    id=id,
                                    name=name, **kwds)
                            else:
                                fn(id=id, name=name)
            self.show_directory(DATA_DIR, "Batch data file")
            if self.has_image_sets:
                with self.tag("div"):
                    self.doc.stag("input", 
                                  type="submit", 
                                  name="input_submit_batch",
                                  value="Submit batch")
                    if self.__submit_batch_pressed and not self.inputs_validated:
                        self.text("Invalid inputs: %s" % self.invalid_reason)
                with self.tag("div"):
                    if self.has_groups:
                        with self.tag("h2"):
                            self.text("Groups")
                        with self.tag("table"):
                            with self.tag("tr"):
                                with self.tag("th"):
                                    self.text("Group #")
                                with self.tag("th"):
                                    self.text("# of image sets")
                            for group_number, group_count in self.group_counts:
                                with self.tag("tr"):
                                    with self.tag("td"):
                                        self.text(str(group_number))
                                    with self.tag("td"):
                                        self.text(str(group_count))
                    else:
                        self.text("Batch_data.h5 has %d image sets" %
                                  len(self.image_numbers))
            else:
                with self.tag("div"):
                    self.text(self.no_image_sets_reason)
                                
    def build_queue_choices(self, id, name):
        with self.doc.select(id=id, name=name):
            for queue in get_queues():
                with self.doc.option(value=queue):
                    self.text(queue)
                
    def build_write_data(self, id, name):
        kwds = {}
        if BATCHPROFILER_DEFAULTS.has_key(WRITE_DATA) and\
           BATCHPROFILER_DEFAULTS[WRITE_DATA] == "yes":
            kwds["checked"] = "yes"
        self.doc.input(type="checkbox", id=id, name=name, **kwds)
        
    def build_revision(self, id, name):
        self.doc.input(type="text", id=id, name=name)
        if self.has_image_sets and self.batch_git_hash is not None:
            with self.tag("button",
                     type="button",
                     onclick="use_githash()"):
                self.text(
                    "Use %s / %s (from Batch_data.h5)" % 
                    (self.batch_git_hash[:8], self.batch_datetime_version))
        with self.tag("button",
                      type="button",
                      id = BUILD_CPCLUSTER_BUTTON,
                      onclick="build_cellprofiler()"):
            self.text("Build CellProfiler")
                    
    def render(self):
        print BATCHPROFILER_DOCTYPE
        print yattag.indent(self.doc.getvalue())
        
    @property
    def has_image_sets(self):
        '''True if the data directory has a good batch data file'''
        if self.__has_image_sets is not None:
            return self.__has_image_sets
        self.__has_image_sets = False
        data_dir = BATCHPROFILER_DEFAULTS[DATA_DIR]
        if not os.path.isdir(data_dir):
            self.no_image_sets_reason = \
                "The %s directory is missing." % data_dir
            return False
        batch_file = os.path.join(data_dir, F_BATCH_DATA_H5)
        if not os.path.isfile(batch_file):
            self.no_image_sets_reason = \
                "The batch data file, %s, does not exist." % batch_file
            return False
        try:
            self.image_numbers = get_batch_image_numbers(batch_file)
            result = get_batch_groups(batch_file)
            if result is None:
                self.has_groups = False
            else:
                self.has_groups = True
                self.group_numbers, self.group_indexes = result
                group_counts = np.bincount(self.group_numbers)
                self.group_counts = [
                    (i, count) for i, count in enumerate(group_counts)
                    if count > 0]
        except:
            self.no_image_sets_reason = \
                "Failed to read batch file %s.\n" % batch_file
            self.no_image_sets_reason += traceback.format_exc()
            return False
        try:
            self.batch_datetime_version, self.batch_git_hash = \
                get_batch_data_version_and_githash(batch_file)
        except:
            import bputilities
            self.batch_datetime_version = self.batch_git_hash = None
            self.no_batch_version_reason = traceback.format_exc()
            
        self.__has_image_sets = True
        return True
    
    @property
    def inputs_validated(self):
        '''True if the inputs for job submission are OK'''
        if self.__inputs_validated is not None:
            return self.__inputs_validated
        self.__inputs_validated = False
        if not self.has_image_sets:
            self.offending_input = DATA_DIR
            self.invalid_reason = self.no_image_sets_reason
            return False
        try:
            int(BATCHPROFILER_DEFAULTS[BATCH_SIZE])
        except:
            self.offending_input = BATCH_SIZE
            self.invalid_reason = "Batch size must be a number"
        try:
            memory_limit = float(BATCHPROFILER_DEFAULTS[MEMORY_LIMIT])
            if memory_limit < MIN_MEMORY_LIMIT or\
               memory_limit > MAX_MEMORY_LIMIT:
                self.offending_input = MEMORY_LIMIT
                self.invalid_reason = \
                    "The memory limit must be between %f and %f GB" % \
                    (MIN_MEMORY_LIMIT, MAX_MEMORY_LIMIT)
                return False
        except:
            self.offending_input = MEMORY_LIMIT
            self.invalid_reason = \
                "The memory limit must be a number between %f and %f GB." %\
                (MIN_MEMORY_LIMIT, MAX_MEMORY_LIMIT)
            return False
        revision = BATCHPROFILER_DEFAULTS[REVISION]
        try:
            self.datetime_version, self.git_hash = \
                get_version_and_githash(revision)
            self.cpcluster = get_cellprofiler_location(
                git_hash = self.git_hash, version = self.datetime_version)
            if not os.path.isdir(os.path.join(self.cpcluster, ".git")):
                self.__cpcluster_needs_building = True
                self.offending_input = BUILD_CPCLUSTER_BUTTON
                self.invalid_reason = \
                    "%s must be compiled" % revision
            else:
                self.__cpcluster_needs_building = False
        except:
            self.offending_input = REVISION
            self.invalid_reason = \
                "Can't find CellProfiler revision, %s.\n" % revision
            self.invalid_reason += traceback.format_exc()
            return False
        # TO DO more validation
        self.__inputs_validated = True
        return True
    
    @property
    def cpcluster_needs_building(self):
        self.inputs_validated
        return self.__cpcluster_needs_building
    
    def output_directory_link(self, key, path):
        directory = os.path.split(path)[1]
        if len(directory) == 0:
            directory = path
        with self.doc.tag(
            "a", 
            href='javascript:go_to_key("%s","%s")' % (key, path)):
            self.text(directory)
        
    def recursive_show_directory(self, key, head, tail):
        '''Recursively show the directory tree'''
        with self.doc.tag("ul"):
            with self.doc.tag("li"):
                self.output_directory_link(key, head)
            if len(tail) == 0:
                with self.doc.tag("ul"):
                    for filename in sorted(os.listdir(head)):
                        fn_path = os.path.join(head, filename)
                        if os.path.isdir(fn_path):
                            with self.doc.tag("li"):
                                self.output_directory_link(key, fn_path)
            else:
                next_head = os.path.join(head, tail[0])
                self.recursive_show_directory(key, next_head, tail[1:])
                
    def show_directory(self, key, title):
        '''Show the directory structure for the variable given by the key
    
        key - key into form_data
        title - the user-visible title of the field
    
        returns the current value of the key
        '''
        path = BATCHPROFILER_DEFAULTS[key]
        data_dir_js = "document.getElementById('input_%s').value" % key
        go_to_key = "javascript:go_to_key('%s',%s) " % (key, data_dir_js)

        with self.tag("div", id="%s_div" % key):
            with self.tag("div"):
                with self.tag("label", 
                              **{ "for":'input_%s' % key}):
                    self.text("%s" % title) 
                self.doc.input(type='text',
                               size='40',
                               id='input_%s' % key,
                               name=key)
                with self.tag("button",
                              type='button',
                              onclick=go_to_key):
                    self.doc.text("Browse...")
            parts = []
            head = path
            while True:
                head, tail = os.path.split(head)
                if len(head) == 0:
                    head = tail
                    break
                elif len(tail) == 0:
                    break
                parts.insert(0, tail)
            self.recursive_show_directory(
                key, head, parts)

        return path

    def build_submit_batch(self):
        '''Build the webpage for handling a submitted batch
        
        Also do the batch submission
        '''
        batch_size = int(BATCHPROFILER_DEFAULTS[BATCH_SIZE])
        if self.has_groups:
            first_last = np.hstack(
                [[True], 
                self.group_numbers[1:] != self.group_numbers[:-1], 
                [True]])
            gn = self.group_numbers[first_last[:-1]]
            first = self.image_numbers[first_last[:-1]]
            last = self.image_numbers[first_last[1:]]
        else:
            first = self.image_numbers[::batch_size]
            last = self.image_numbers[(batch_size-1)::batch_size]
            if len(last) < len(first):
                last = np.hstack([last, self.image_numbers[-1]])
        batch = RunBatch.BPBatch()
        runs = [(f, l) for f, l in zip(first, last)]
        #
        # Put it in the database
        #
        write_data = 1 if BATCHPROFILER_VARIABLES[WRITE_DATA] is not None else 0
        batch.create(
            email = BATCHPROFILER_DEFAULTS[EMAIL],
            data_dir = BATCHPROFILER_DEFAULTS[DATA_DIR],
            queue = BATCHPROFILER_DEFAULTS[QUEUE],
            batch_size = BATCHPROFILER_DEFAULTS[BATCH_SIZE],
            write_data = write_data,
            timeout = 60,
            cpcluster = self.cpcluster,
            project = BATCHPROFILER_DEFAULTS[PROJECT],
            memory_limit = BATCHPROFILER_DEFAULTS[MEMORY_LIMIT],
            priority = BATCHPROFILER_DEFAULTS[PRIORITY],
            runs = runs)
        RunBatch.run_all(batch.batch_id)
        vb_url = "ViewBatch.py?batch_id=%d" % batch.batch_id
        self.send_batch_submission_email(batch, vb_url)
        job_list = batch.select_jobs()
        with self.tag("head"):
            with self.tag("title"):
                self.text("Batch #%d" % batch.batch_id)
            with self.tag("style", type="text/css"):
                self.doc.asis("""
                    table {
                        border-spacing: 0px;
                        border-collapse: collapse;
                    }
                    td {
                        text-align: left;
                        vertical-align: baseline;
                        padding: 0.1em 0.5em;
                        border: 1px solid #666666;
                    }""")
        with self.tag("body"):
            with self.tag("h1"):
                self.text("Results for batch #")
                with self.tag("a", href=vb_url):
                    self.text(str(batch.batch_id))
            with self.tag("table"):
                with self.tag("thead"):
                    with self.tag("tr"):
                        with self.tag("th"):
                            self.text("First image set")
                        with self.tag("th"):
                            self.text("Last image set")
                        with self.tag("th"):
                            self.text("job #")
                for run, job, status in job_list:
                    assert isinstance(run, RunBatch.BPRun)
                    assert isinstance(job, RunBatch.BPJob)
                    with self.tag("tr"):
                        with self.tag("td"):
                            self.text(str(run.bstart))
                        with self.tag("td"):
                            self.text(str(run.bend))
                        with self.tag("td"):
                            self.text(str(job.job_id))

    def send_batch_submission_email(self, batch, vb_url):
        doc, tag, text = yattag.Doc().tagtext()
        with tag("html"):
            with tag("head"):
                with tag("title"):
                    text("Batch # %d" % batch.batch_id)
                with tag("style", type="text/css"):
                    doc.asis("""
table {
    border-spacing: 0px;
    border-collapse: collapse;
}
td {
    text-align: left;
    vertical-align: baseline;
    padding: 0.1em 0.5em;
    border: 1px solid #666666;
}
""")
            with tag("body"):
                with tag("h1"):
                    text("Results for batch # ")
                    with tag("a", href = vb_url):
                        text(str(batch.batch_id))
                with tag("div"):
                    text("Data Directory: %s" % 
                         BATCHPROFILER_DEFAULTS[DATA_DIR])
        email_text = yattag.indent(doc.getvalue())
        send_html_mail(recipient=BATCHPROFILER_DEFAULTS[EMAIL],
                       subject = "Batch %d submitted"%(batch.batch_id),
                       html=email_text)

with CellProfilerContext():    
    doc = NewBatchDoc()
    doc.build()
doc.render()
