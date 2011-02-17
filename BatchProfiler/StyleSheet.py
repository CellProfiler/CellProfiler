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
