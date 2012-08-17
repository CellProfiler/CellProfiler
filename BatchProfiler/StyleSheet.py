"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
# This module prints out the standardized stylesheet
# for the batchprofiler web pages
#

def PrintStyleSheet():
    print "<style type='text/css'>"
    print """
table.run_table {
    border-spacing: 0px;
    border-collapse: collapse;
}
table.run_table th {
    text-align: left;
    font-weight: bold;
    padding: 0.1em 0.5em;
    border: 1px solid #666666;
}
table.run_table td {
    text-align: right;
    padding: 0.1em 0.5em;
    border: 1px solid #666666;
}
table.run_table thead th {
    text-align: center;
}
"""
    print "</style>"
