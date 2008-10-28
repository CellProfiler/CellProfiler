#!/broad/tools/apps/Python-2.5.2/bin/python
# This is a CGI script that begins to start a batch on the clusters
# Functionality:
#   Parse the posted fields:
#      email
#      data_dir
#      queue
#      batch_size: 
#      write_data: "y" or "n"
#      timeout
#   Create a batch run # and store the details in the DB
#   Send the user an email with the URL of the job
#   Start BatchRunner.py in the background with the supplied info
import cgitb
cgitb.enable()
print "Content-Type: text/html\r"
print "\r"
import cgi
import os
import re
import socket
import smtplib
import email.message
import email.mime.text
import subprocess
import RunBatch
import traceback
from scipy.io.mio import loadmat

# # # # # # # # # #
#
# Constants
#
# # # # # # # # # #

batch_url = "http://%s/batchprofiler/cgi-bin/ViewBatch.py"%(socket.gethostname())
SENDMAIL="/usr/sbin/sendmail"

def PartitionBatches(my_batch):
    """Partition the image sets in the batch_info into runs
    
    Returns an array of runs where each run is a dictionary composed of
    start = index of first image set in run
    end   = index of last image set in run
    status_file_name = where the status file will be stored
    """
    result=[]
    dd=my_batch["data_dir"]
    for start in range(1, my_batch["num_sets"] + 1, my_batch["batch_size"]):
        end=start+my_batch["batch_size"]-1
        status_file_name = "%(dd)s/status/Batch_%(start)d_to_%(end)d_DONE.mat"%(locals())
        result.append({ 
            "start":start, 
            "end":min(start+my_batch["batch_size"]-1,my_batch["num_sets"]),
            "status_file_name":status_file_name})
        
    my_batch["runs"]=result
    return result

def CheckParameters(my_batch):
    """Check the parameters of the batch for consistency and correctness
    
    Make sure that the data directory is writeable.
    Make sure that txt_output and status are writeable
    Make sure that the cpcluster directory exists
    """
    if not os.path.exists(my_batch["data_dir"]):
        return 'The data directory, %(data_dir)s, must exist'%(my_batch)
    if not os.path.exists(my_batch["batch_file"]):
        return 'The batch file, "%(batch_file)s", must exist'%(my_batch)
    if not os.access(my_batch["batch_file"],os.R_OK):
        return 'The batch file, "%(batch_file)s", must be readable'%(my_batch)
    txt_output =  os.sep.join(['%(data_dir)s'%(my_batch),'txt_output'])
    status = os.sep.join(['%(data_dir)s'%(my_batch),'status'])
    writeable= {
        "data directory":my_batch["data_dir"],
        "CPCluster directory":my_batch["cpcluster"]
    }
    if os.path.exists(txt_output):
        writeable["command output"]=txt_output
        text_output_files=os.listdir(txt_output)
        for tof in text_output_files:
            match = re.match("^([0-9]+)_to_([0-9]+)\\.txt$",tof)
            if match:
                (start,last)=match.groups()
                writeable["image set %s to %s command output file"%(start,last)]=os.sep.join([txt_output,tof])
    if os.path.exists(status):
        writeable["Matlab output"]=status
        status_files=os.listdir(status)
        for tof in status_files:
            match = re.match("^Batch_([0-9]+)_to_([0-9]+)_DONE\\.txt$",tof)
            if match:
                (start,last)=match.groups()
                writeable["image set %d to %d Matlab status file"%(start,last)]=os.sep.join([status,tof])
            match = re.match("^Batch_([0-9]+)_to_([0-9]+)_OUT\\.txt$",tof)
            if match:
                (start,last)=match.groups()
                writeable["image set %s to %s Matlab data file"%(start,last)]=os.sep.join([status,tof])

    for dir_key in writeable:
        if not os.access(writeable[dir_key],os.W_OK):
            return 'The %s, "%s", must be writeable'%(dir_key,writeable[dir_key])
    if not os.path.exists(txt_output):
        os.mkdir(txt_output)
        os.chmod(txt_output,0666)
    if not os.path.exists(status):
        os.mkdir(status)
        os.chmod(status,0666)

def SendMail(recipient,body):
    if os.name != 'nt':
        pipe=os.popen("%s -t"%(SENDMAIL),"w")
        pipe.write("To: %s\n"%(recipient))
        pipe.write("Subject: Batch %d submitted\n"%(batch_id))
        pipe.write("Content-Type: text/html\n")
        pipe.write("\n")
        pipe.write(body)
        pipe.write("\n")
        pipe.close()

    return

# # # # # # # # # # #
#
# Main script
#
# # # # # # # # # # #

form_data=cgi.FieldStorage()
if form_data.has_key("data_dir"):
    try:
        if not form_data.has_key("email"):
            raise RuntimeError("The e-mail field isn't filled in.")
        imaging_analysis=(os.name=='nt' and "//iodine/imaging_analysis") or "/imaging/analysis"
        batch_file="%s/Batch_data.mat"%(form_data["data_dir"].value)
        batch_file=batch_file.replace("/imaging/analysis",imaging_analysis)
        batch_file=batch_file.replace("/",os.sep)
        CPCluster='/imaging/analysis/CPCluster/%s'%(form_data["cpcluster"].value)
        my_batch = {
            "email":         form_data["email"].value,
            "queue":         form_data["queue"].value,
            "data_dir":      form_data["data_dir"].value,
            "write_data":    (form_data["write_data"].value.upper()=="Y" and 1) or 0,
            "batch_size":    int(form_data["batch_size"].value),
            "timeout":       float(form_data["timeout"].value),
            "cpcluster":     CPCluster,
            "batch_file":    batch_file
            }
        error = CheckParameters(my_batch)
        if error:
            exception = RuntimeError()
            exception.message = error
            raise exception
        batch_info = loadmat(batch_file)
        my_batch["num_sets"] = batch_info['handles'].Current.NumberOfImageSets
        runs = PartitionBatches(my_batch)
        batch_id = RunBatch.CreateBatchRun(my_batch)
        results = RunBatch.RunAll(batch_id)
        text=[]
        text.append("<html>")
        text.append("<head><title>Batch # %d</title>"%(batch_id))
        text.append("<style type='text/css'>")
        text.append("""
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
    """)
        text.append("</style></head>")
        text.append("</head>")
        text.append("<body>")
        text.append("<h1>Results for batch # <a href='%s?batch_id=%d'>%d</a></h1>"%(batch_url,batch_id,batch_id))
        text.append("<table>")
        text.append("<th><tr><td>First image set</td><td>Last image set</td><td>job #</td></tr></th>")
        for result in results:
            text.append("<tr><td>%(start)d</td><td>%(end)d</td><td>%(job)d</td></tr>"%(result))
        text.append("</table>")
        text.append("</body>")
        text.append("</html>")
        body= '\n'.join(text)
        SendMail(my_batch["email"],body)
        print body
    except RuntimeError,e:
        print "<html><body>"
        print "<h1 style='color:red'>Unable to process job</h1>"
        print e.message
        print "</body></html>"
else:
    print "<html>"
    print "<head><link href='/RunNewBatch.html' rel='Alternate' /></head>"
    print "</html>"

