# This module prints out the standardized stylesheet
# for the batchprofiler web pages
#
BATCHPROFILER_DOCTYPE = '<!DOCTYPE html PUBLIC ' \
              '"-//W3C//DTD XHTML 1.0 Transitional//EN"' \
              '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'

BATCHPROFILER_STYLE = """
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
div.error_message {
    color: red;
    font-style: italic;
}
"""

def PrintStyleSheet():
    print "<style type='text/css'>"
    print BATCHPROFILER_STYLE
    print "</style>"