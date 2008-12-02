#!/broad/tools/apps/Python-2.5.2/bin/python
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
