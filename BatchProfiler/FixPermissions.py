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
# Kill all jobs in a batch
#
import RunBatch
from bpformdata import *
import cgi
import os
import json
import stat
import yattag

def maybe_chmod(path, mode, d):
    '''Change the mode of a file if it exists and needs changing
    
    path - path to file or directory
    mode - the new mode
    d - add the key/value of path:mode to the dictionary if the mode was changed
    '''
    if os.path.exists(path) and \
       stat.S_IMODE(os.stat(directory).st_mode) != mode:
        os.chmod(directory, mode)
        what_i_did[directory] = mode
    
def handle_post():
    batch_id = BATCHPROFILER_DEFAULTS[BATCH_ID]
    my_batch = RunBatch.BPBatch()
    my_batch.select(batch_id)
    txt_output_dir = RunBatch.text_file_directory(my_batch)
    job_script_dir = RunBatch.script_file_directory(my_batch)
    what_i_did = {}
    maybe_chmod(txt_output_dir, 0777, what_i_did)
    maybe_chmod(job_script_dir, 0777, what_i_did)
        
    for run in my_batch.select_runs():
        for path in [RunBatch.run_text_file_path(my_batch, run),
                     RunBatch.run_out_file_path(my_batch, run),
                     RunBatch.run_err_file_path(my_batch, run)]:
            maybe_chmod(path, 0644, what_i_did)
    
    print "Content-Type: application/json"
    print
    print json.dumps(what_i_did)

'''Javascript for fixPermissions function

Assumes that batch ID is in the element
with the name, "input_batch_id".

button = button that was pressed
'''
FIX_PERMISSIONS_AJAX_JAVASCRIPT = '''
function fix_permissions(button) {
    var batch_id_elem = document.getElementById("input_%(BATCH_ID)s");
    var batch_id = batch_id_elem.value;
    var oldInnerText = button.innerText;
    button.innerText = "Fixing permissions for batch " + batch_id;
    button.disabled = true
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange=function() {
        if (xmlhttp.readyState == 4) {
            if (xmlhttp.status == 200) {
                var result = JSON.parse(xmlhttp.responseText);
                var count = Object.keys(result).length
                alert("Changed permissions for "+count+" file(s) / folder(s)");
            } else {
                alert("Failed to change permissions.\n" + xmlhttp.responseText);
            }
            button.innerText = oldInnerText;
            button.disabled = false;
        }
    }
    xmlhttp.open(
        "POST",
        "FixPermissions.py", true);
    xmlhttp.setRequestHeader(
        "Content-Type", "application/x-www-form-urlencoded;charset=UTF-8");
    xmlhttp.send("%(BATCH_ID)s=" + batch_id);
}
''' % globals()

TITLE = "BatchProfiler: fix file permissions"
def handle_get():
    '''Display a form for fixing the permissions'''
    batch_id_id = "input_%s" % BATCH_ID
    button_id = "button_%s" % BATCH_ID
    fix_permissions_action = \
        "fix_permissions(document.getElementById('%s'));" % button_id
    doc, tag, text = yattag.Doc().tagtext()
    assert isinstance(doc, yattag.Doc)
    with tag("html"):
        with tag("head"):
            with tag("title"):
                text(TITLE)
            with tag("script", language="JavaScript"):
                doc.asis(FIX_PERMISSIONS_AJAX_JAVASCRIPT)
        with tag("body"):
            with tag("h1"):
                text(TITLE)
            with tag("div"):
                text("""
This webpage fixes permission problems for the files and folders in your batch
that were created outside BatchProfiler's control. It will grant read
permission for the job script folder, the text output folder, the text and 
error output files and the measurements file.""")
            with tag("div"):
                with tag("label", **{ "for":batch_id_id }):
                    text("Batch ID")
                doc.input(name=BATCH_ID, id=batch_id_id, type="text")
                with tag("button", id=button_id,
                         onclick=fix_permissions_action):
                    text("Fix permissions")
    print "Content-Type: text/html"
    print
    print '<!DOCTYPE html PUBLIC ' \
          '"-//W3C//DTD XHTML 1.0 Transitional//EN"' \
          '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
    print yattag.indent(doc.getvalue())
    
if __name__ == "__main__":
    import cgitb
    cgitb.enable()

    if REQUEST_METHOD == RM_GET:
        handle_get()
    elif REQUEST_METHOD == RM_POST:
        handle_post()
    else:
        raise ValueError("Unhandled request method: %s" % REQUEST_METHOD)
        
    
            
