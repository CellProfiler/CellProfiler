#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/python-2.6.sh
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
# Run bpeek on the indicated job #
#
import cgitb
cgitb.enable()
import subprocess
import cgi
import os
import os.path
import stat

job_id = int(cgi.FieldStorage()["job_id"].value)
print "Content-Type: text/plain\r"
print "\r"
p=subprocess.Popen(['bash'],stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
x=p.communicate('. /broad/lsf/conf/profile.lsf;bpeek %d\n'%(job_id))
print x[0]
