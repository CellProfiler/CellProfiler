#!/usr/bin/env ./python-2.6.sh
#
# Determine whether a directory exists
#
import cgitb
cgitb.enable()
print "Content-Type: text/plain\r"
print "\r"
import sys
import cgi
import os

form_data = cgi.FieldStorage()
path = form_data["path"].value
if os.path.exists(path):
    print "OK"
else:
    print "Failure"
