#!/usr/bin/env ./python-2.6.sh
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
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
import urllib

import cellprofiler.pipeline as cpp
import cellprofiler.preferences

cellprofiler.preferences.set_headless()
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
    
print '''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html lang="en-US" xml:lang="en-US" xmlns="http://www.w3.org/1999/xhtml">
    <head>
         <title>CellProfiler 2.0 Batch submission</title>
         <script language='JavaScript'>
function go_to_key(key) {
    url='%(myself)s'
    add_char = "?"
    all_k = new Array("output_dir","image_dir","email","queue",
                        "project","batch_size","memory_limit")
    for (k in all_k) {
        v = document.getElementById('input_'+all_k[k]).value
        url = url+add_char+all_k[k]+'='+escape(v)
        add_char = "&"
    }
parent.location = url+"#input_"+key
}
         </script>
    </head>
    <body>
    <H1>CellProfiler 2.0 Batch submission</H1>
    <div>
    Submit a Batch_data.mat file created by CellProfiler 2.0. You need to
    specify the Default Output Folder, which should contain your
    Batch_data file and the default input folder for the pipeline. In
    addition, there are some parameters that tailor how the batch is run.
    </div>
    '''%(globals())

keys = { 'output_dir':lookup('output_dir', '/imaging/analysis'),
         'image_dir':lookup('image_dir', '/imaging/analysis'),
         'email':lookup('email', 'user@broadinstitute.org'),
         'queue':lookup('queue', 'broad'),
         'project':lookup('project','imaging'),
         'write_data':lookup('write_data','no'),
         'batch_size':lookup('batch_size','10'),
         'memory_limit':lookup('memory_limit','2000')
         }
print '''<form action='SubmitBatch.py'>
<div style='white-space=nowrap'><label for='input_email'>E-mail:&nbsp;</label>
<input type='text' size="40" id='input_email' name='email' value='%(email)s'/></div>

<div style='white-space=nowrap'><label for='input_queue'>Queue:&nbsp;</label>
<select id='input_queue' name='queue'>
'''%(keys)
for queue in ('broad','short','long','hugemem','preview','priority'):
    selected = 'selected="selected"' if queue == keys['queue'] else ''
    print '''<option value='%(queue)s' %(selected)s>%(queue)s</option>'''%(locals())

print '''</select></div>'''
print '''<div style='white-space=nowrap'><label for='input_email'>Project:&nbsp;</label>
<input type='text' id='input_project' name='project' value='%(project)s'/></div>

<div style='white-space=nowrap'><label for='input_email'>Batch size:&nbsp;</label>
<input type='text' id='input_batch_size' name='batch_size' value='%(batch_size)s'/></div>

<div style='white-space=nowrap'><label for='input_email'>Memory limit:&nbsp;</label>
<input type='text' id='input_memory_limit' name='memory_limit' value='%(memory_limit)s'/></div>
'''%(keys)
batch_file = os.path.join(keys['output_dir'], 'Batch_data.mat')
grouping_keys = None
if os.path.exists(batch_file):
    pipeline = cpp.Pipeline()
    try:
        pipeline.load(batch_file)
        image_set_list = pipeline.prepare_run(None)
        grouping_keys, groups = pipeline.get_groupings(image_set_list)
    except:
        print "Failed to open %s"%batch_file
show_directory('output_dir','Default output directory',keys['output_dir'], 
               minus_key(keys,'output_dir'))
show_directory('image_dir','Default image directory', keys['image_dir'], 
               minus_key(keys,'image_dir'))
if grouping_keys is not None:
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
        print 'Batch_data.mat has %d image sets'%image_set_list.count()
print '</form>'
print '</body></html>'
