#!/usr/bin/env ./python-2.6.sh
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
