#!/usr/bin/env ./python-2.6.sh
#
# Start a batch operation from a web page
#
import cgitb
cgitb.enable()
print "Content-Type: text/html\r"
print "\r"
import sys
import cgi
import os
import traceback
import urllib
import cellprofiler.preferences
cellprofiler.preferences.set_headless()
import cellprofiler.modules
import cellprofiler.pipeline as cpp
from cellprofiler.modules.createbatchfiles import F_BATCH_DATA, CreateBatchFiles
import RunBatch

form_data = cgi.FieldStorage()
myself = os.path.split(__file__)[1]
if len(myself) == 0:
    myself = __file__
    
def show_directory(key, title, default, hidden_vars):
    '''Show the directory structure for the variable given by the key
    
    key - key into form_data
    title - the user-visible title of the field
    default - default value for the field
    
    returns the current value of the key
    '''
    if form_data.has_key(key):
        value = form_data[key].value
    else:
        value = default
    
    paths = []
    path = value
    hvs = hidden_vars_inputs(hidden_vars)
    while True:
        head,tail = os.path.split(path)
        if len(tail) == 0:
            paths.insert(0,(head, path))
            break
        paths.insert(0,(tail, path))
        path = head

    print '''<div id="%(key)s_div">
    <div>
        <label for='input_%(key)s'>%(title)s:&nbsp;</label><input type='text' 
                          size='40'
                          id='input_%(key)s'
                          name='%(key)s' 
                          value='%(value)s'/>
        <input type='button' value='Browse...' 
            onclick="javascript:go_to_key('%(key)s')"/>
    </div>
    '''%(locals())
    for dirname, dirpath in paths:
        all_keys = dict(hidden_vars)
        all_keys[key] = dirpath
        url = "%s?%s"%(myself, urllib.urlencode(all_keys))
        print '''<ul><li><a href='%(url)s'>%(dirname)s</a></li>'''%(locals())
    filenames = [(filename, os.path.join(value, filename))
                 for filename in os.listdir(value)
                 if os.path.isdir(os.path.join(value, filename))]
    filenames.sort()
    if len(filenames):
        print '''<ul>'''
        for dirname, dirpath in filenames:
            all_keys = dict(hidden_vars)
            all_keys[key] = dirpath
            url = "%s?%s"%(myself, urllib.urlencode(all_keys))
            print '''<li><a href='%(url)s'>%(dirname)s</a></li>'''%(locals())
        print '''</ul>'''
    print ''.join(['</ul>']*len(paths))
    print '''</div>
'''
    return value

def hidden_vars_inputs(hidden_vars):
    '''Create hidden input elements for each key in hidden_vars'''
    s = ''
    for key in hidden_vars.keys():
        s+= '''<input type='hidden' name='%s' value='%s'/>'''%(key,hidden_vars[key])
    return s

def lookup(key, default):
    if form_data.has_key(key):
        return form_data[key].value
    else:
        return default

def minus_key(d, key):
    d = dict(d)
    del d[key]
    return d

keys = { 'data_dir':lookup('data_dir', '/imaging/analysis'),
         'email':lookup('email', 'user@broadinstitute.org'),
         'queue':lookup('queue', 'hour'),
         'project':lookup('project','imaging'),
         'priority':lookup('priority','50'),
         'write_data':lookup('write_data','no'),
         'batch_size':lookup('batch_size','10'),
         'memory_limit':lookup('memory_limit','2000'),
         'timeout':lookup('timeout','30'),
         'revision':lookup('revision','8009'),
         'url':myself
         }

batch_file = os.path.join(keys['data_dir'], F_BATCH_DATA)
grouping_keys = None
error_message = None
if os.path.exists(batch_file):
    pipeline = cpp.Pipeline()
    print "<span style='visibility:hidden'>"
    try:
        had_problem = [False]
        def error_callback(event, caller):
            if (isinstance(event, cpp.LoadExceptionEvent) or
                isinstance(event, cpp.RunExceptionEvent)):
                sys.stderr.write("Handling exception: %s\n"%str(event))
                sys.stderr.write(traceback.format_exc())
                had_problem = [True]
        pipeline.add_listener(error_callback)
        pipeline.load(batch_file)
        if had_problem[0]:
            raise RuntimeError("Failed to load batch file")
        image_set_list = pipeline.prepare_run(None)
        if had_problem[0]:
            raise RuntimeError("Failed to prepare batch file")
        grouping_keys, groups = pipeline.get_groupings(image_set_list)
        svn_revision = None
        for module in pipeline.modules():
            if isinstance(module,CreateBatchFiles):
                svn_revision = module.revision.value
                break
    except:
        error_message = "Failed to open %s\n%s" % (batch_file, traceback.format_exc())
        error_message = error_message.replace("\n","<br/>")
    print "</span>"
    
if (form_data.has_key('submit_batch') and 
    form_data['submit_batch'].value == 'yes' and
    grouping_keys is not None):
    #
    # Submit the batch according to the directions
    #
    batch = {
        "email":         form_data["email"].value,
        "queue":         form_data["queue"].value,
        "priority":      int(form_data["priority"].value) if form_data.has_key("priority") else 50,
        "project":       form_data["project"].value if form_data.has_key("project") else 'imaging',
        "data_dir":      form_data["data_dir"].value,
        "write_data":    (form_data.has_key("write_data") and 1) or 0,
        "batch_size":    int(form_data["batch_size"].value),
        "memory_limit":  float(form_data["memory_limit"].value) if form_data.has_key("memory_limt") else 2000,
        "timeout":       float(form_data["timeout"].value),
        "cpcluster":     "CellProfiler_2_0:/imaging/analysis/CPCluster/CellProfiler-2.0/%s"%form_data["revision"].value,
        "batch_file":    batch_file,
        "runs":          []
    }
    if len(grouping_keys):
        for grouping, image_numbers in groups:
            start = min(image_numbers)
            end = max(image_numbers)
            status_file_name = ("%s/status/Batch_%d_to_%d_DONE.mat"%
                                (batch["data_dir"], start, end))
            run = { "start": start,
                    "end": end,
                    "group": grouping,
                    "status_file_name":status_file_name}
            batch["runs"].append(run)
    else:
        batch_size = 10
        if form_data.has_key("batch_size"):
            batch_size = int(form_data["batch_size"].value)
        for i in range(1,image_set_list.count()+1,batch_size):
            start = i
            end = min(start + batch_size -1,image_set_list.count())
            status_file_name = ("%s/status/Batch_%d_to_%d_DONE.mat"%
                                (batch["data_dir"], start, end))
            run = { "start": start,
                    "end": end,
                    "group": None,
                    "status_file_name":status_file_name}
            batch["runs"].append(run)
    print '''<html>
    <head><title>Batch # (batch_id)d</title>
    <style type='text/css'>
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
</style></head>
</head>
<body>'''%(locals())
    batch_id = RunBatch.CreateBatchRun(batch)
    results = RunBatch.RunAll(batch_id)
    print '''
<h1>Results for batch # <a href='ViewBatch.py?batch_id=%(batch_id)d'>%(batch_id)d</a></h1>
<table>
<thead><tr><th>First image set</th><th>Last image set</th>'''%(locals())
    for key in grouping_keys:
        print '<th>%s</th>'%key
        
    print '<th>job #</th></tr></thead>'
    
    for i,result in enumerate(results):
        print "<tr><td>%(start)d</td><td>%(end)d</td>"%(result)
        for key in grouping_keys:
            print "<td>%s</td>"% groups[i][0][key]
        print "<td>%(job)d</td></tr>"%(result)
    print "</table></body></html>"
    sys.exit()
    
print '''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html lang="en-US" xml:lang="en-US" xmlns="http://www.w3.org/1999/xhtml">
    <head>
         <title>CellProfiler 2.0 Batch submission</title>
         <script language='JavaScript'>
function go_to_key(key) {
    url='%(myself)s'
    add_char = "?"
    all_k = new Array("data_dir","email","queue","priority",
                        "project","batch_size","memory_limit",
                        "timeout","revision")
    for (k in all_k) {
        v = document.getElementById('input_'+all_k[k])
        url = url+add_char+all_k[k]+'='+escape(v.value)
        add_char = "&"
    }
    v = document.getElementById('input_write_data')
    if (v.checked)
    {
        url = url + add_char + "write_data=yes"
    } else {
        url = url + add_char + "write_data=no"
    }
parent.location = url+"#input_"+key
}
         </script>
    </head>
    <body>
    <H1>CellProfiler 2.0 Batch submission</H1>
    <div>
    Submit a %(F_BATCH_DATA)s file created by CellProfiler 2.0. You need to
    specify the default output folder, which should contain your
    Batch_data file and the default input folder for the pipeline. In
    addition, there are some parameters that tailor how the batch is run.
    </div>
    '''%(globals())

print '''<form action='%(url)s'>
<input type='hidden' name='submit_batch' value='yes'/>
<table style='white-space=nowrap'>
<tr><th>E-mail:</th>
<td><input type='text' size="40" id='input_email' name='email' value='%(email)s'/></td></tr>
<tr><th>Queue:</th>
<td><select id='input_queue' name='queue'>
'''%(keys)
for queue in ('hour', 'week', 'broad','short','long','hugemem','preview','priority'):
    selected = 'selected="selected"' if queue == keys['queue'] else ''
    print '''<option value='%(queue)s' %(selected)s>%(queue)s</option>'''%(locals())

print '''</select></td></tr>'''
keys_plus = keys.copy()
keys_plus["write_data_checked"] = "" if keys["write_data"] == "no" else 'checked="yes"'
print '''
<tr><th>Priority:</th>
<td><input type='text' id='input_priority' name='priority' value='%(priority)s'/></td></tr>
<tr><th>Project:</th>
<td><input type='text' id='input_project' name='project' value='%(project)s'/></td></tr>
<tr><th>Batch size:</th>
<td><input type='text' id='input_batch_size' name='batch_size' value='%(batch_size)s'/></td></tr>
<tr><th>Memory limit:</th>
<td><input type='text' id='input_memory_limit' name='memory_limit' value='%(memory_limit)s'/></td></tr>
<tr><th>Write data:</th>
<td><input type='checkbox' id='input_write_data' name='write_data' value='yes' %(write_data_checked)s/></td></tr>
<tr><th>Timeout:&nbsp;</th>
<td><input type='text' id='input_timeout' name='timeout' value='%(timeout)s'/></td></tr>
'''%(keys_plus)
print '''<tr><th>SVN revision:</th><td><select name='revision' id='input_revision'>'''
vroot = '/imaging/analysis/CPCluster/CellProfiler-2.0'
vdirs = list(os.listdir(vroot))
vdirs.sort()
for filename in vdirs:
    vpath = os.path.join(vroot, filename)
    if not os.path.isdir(vpath):
        continue
    print '''<option %s>%s</option>'''%('selected="selected"' if filename == keys['revision'] else '',filename)
print '''</select> (at /imaging/analysis/CPCluster/CellProfiler-2.0/)</td></tr></table>'''
show_directory('data_dir','Data output directory',keys['data_dir'], 
               minus_key(keys,'data_dir'))
if grouping_keys is not None:
    print '''<div><input type='submit' value='Submit batch'/></div>'''
    if len(grouping_keys):
        print '<h2>Groups</h2>'
        print '<table><tr>'
        for key in grouping_keys:
            print '<th>%s</th>'%key
        print '<th># of image sets</th></tr>'
        for group in groups:
            print '<tr>'
            for key in grouping_keys:
                print '<td>%s</td>'%group[0][key]
            print '<td>%d</td></tr>'%len(group[1])
        print '</table>'
    else:
        print '<div>Batch_data.mat has %d image sets</div>'%image_set_list.count()
        if svn_revision is not None:
            print '<div>It was saved using CellProfiler SVN revision # %d</div>'%(svn_revision)
elif error_message is not None:
    print error_message
else:
    print 'Directory does not contain a Batch_data.mat file'
print '</form>'
print '</body></html>'
try:
    import cellprofiler.utilities.jutil as jutil
    jutil.kill_vm()
except:
    import traceback
    traceback.print_exc()
