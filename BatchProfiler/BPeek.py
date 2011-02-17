#!/usr/bin/env /imaging/analysis/People/imageweb/batchprofiler/cgi-bin/python-2.6.sh
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
