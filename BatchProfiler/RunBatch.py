#
# Functions for running a batch or a single run from the database
#
import MySQLdb
import subprocess
import os
import re

connection = MySQLdb.Connect(host="imgdb01", db="batchprofiler", user="cpadmin", passwd="cPus3r")

def SetEnvironment(my_batch):
    orig_PATH = os.environ['PATH']
    orig_LD_LIBRARY_PATH = os.environ['LD_LIBRARY_PATH']
    mcr_root = '/imaging/analysis/CPCluster/MCR/v78'
    jre_file = open('/imaging/analysis/CPCluster/MCR/v78/sys/java/jre/glnxa64/jre.cfg')
    jre_version = jre_file.readline().replace('\n','').replace('\r','')
    jre_file.close()
    os.environ['LD_LIBRARY_PATH']=':'.join([
        orig_LD_LIBRARY_PATH,
        '%(mcr_root)s/sys/java/jre/glnxa64/jre%(jre_version)s/lib/amd64/native_threads'%(locals()),
        '%(mcr_root)s/sys/java/jre/glnxa64/jre%(jre_version)s/lib/amd64/server'%(locals()),
        '%(mcr_root)s/sys/java/jre/glnxa64/jre%(jre_version)s/lib/amd64/client'%(locals()),
        '%(mcr_root)s/sys/java/jre/glnxa64/jre%(jre_version)s/lib/amd64'%(locals())])
    os.environ['XAPPLRESDIR']='%(mcr_root)s/X11/app-defaults'%(locals())
    os.environ['MCR_CACHE_ROOT']='%(cpcluster)s'%(my_batch)
    os.environ['PATH']=orig_PATH.replace('/home/radon01/ljosa/software/x86_64/bin:','').replace('/home/radon01/ljosa/bin:','')
    return orig_PATH, orig_LD_LIBRARY_PATH

def RestoreEnvironment(vals):
    os.environ['PATH'] = vals[0]
    os.environ['LD_LIBRARY_PATH'] = vals[1]

def CreateBatchRun(my_batch):
    """Create a batch ID in the database
    
    Create a batch ID in the database, store the details for the
    run there.
    Returns: batch_id created in database
    """
    batch_id = CreateBatchRecord(my_batch)
    for run in my_batch["runs"]:
        CreateRunRecord(batch_id,run)
    
    return batch_id

def CreateBatchRecord(my_batch):
    """Create a record in batchprofiler.batch
    
    Uses global variables, recipient, data_dir, queue, batch_size, write_data, timeout and connection
    Returns: batch_id
    """
    cursor = connection.cursor()
    cmd = """
    insert into batch (batch_id, email, data_dir, queue, batch_size, write_data, timeout,cpcluster)
    values (null,'%(email)s','%(data_dir)s','%(queue)s',%(batch_size)d,%(write_data)d,%(timeout)f,'%(cpcluster)s')"""%(my_batch)
    cursor.execute(cmd)
    cursor = connection.cursor()
    cursor.execute("select last_insert_id()")
    my_batch["batch_id"]=cursor.fetchone()[0]
    cursor.close()
    return my_batch["batch_id"]

def CreateRunRecord(batch_id, run):
    """Create a run record within the current batch
    
    """
    run["batch_id"]=batch_id
    cursor = connection.cursor()
    sql="""
    insert into run (run_id, batch_id, bstart,bend,status_file_name)
    values (null,%(batch_id)d,%(start)d,%(end)d,'%(status_file_name)s')"""%(run)
    cursor.execute(sql)
    cursor.close()
    cursor = connection.cursor()
    cursor.execute("select last_insert_id()")
    run["run_id"] = cursor.fetchone()[0]
    cursor.close()
    return

def CreateJobRecord(run_id, job_id):
    """Create a job record in the batchprofiler database
    
    Create a job record with the given run_id and job_id
    """
    cursor = connection.cursor()
    sql="""
    insert into job (job_id, run_id) values ('%s','%s')"""%(job_id,run_id)
    cursor.execute(sql)
    cursor.close()

def LoadBatch(batch_id):
    """Load a batch from the database
    
    Return the batch with the given batch ID and the associated runs
    """
    sql = "select email,data_dir,queue,batch_size,write_data,timeout,cpcluster from batch where batch_id=%d"%(batch_id)
    cursor = connection.cursor()
    cursor.execute(sql)
    row = cursor.fetchone()
    cursor.close()
    my_batch= {
        "batch_id":   batch_id,
        "email":      row[0],
        "data_dir":   row[1],
        "queue":      row[2],
        "batch_size": int(row[3]),
        "write_data": int(row[4]),
        "timeout":    float(row[5]),
        "cpcluster":  row[6]
        }
    cursor = connection.cursor()
    cursor.execute("""
select run_id,batch_id,bstart,bend,status_file_name,
       (select max(job_id) from job j where j.run_id = r.run_id)
  from run r where batch_id=%d"""%(batch_id));
    runs = []
    my_batch["runs"] = runs
    for row in cursor.fetchall():
        runs.append({
            "run_id":           int(row[0]),
            "batch_id":         int(row[1]),
            "start":            int(row[2]),
            "end":              int(row[3]),
            "status_file_name": row[4],
            "job_id":           (row[5] and int(row[5]))
        })
    cursor.close()
    return my_batch

def LoadRun(run_id):
    """Load a batch and a run from the database
    
    Return a batch dictionary and run dictionary with the associated run id
    """
    sql = "select batch_id from run where run_id=%d"%(run_id)
    cursor = connection.cursor()
    cursor.execute(sql)
    row = cursor.fetchone()
    cursor.close()
    batch_id = int(row[0])
    my_batch = LoadBatch(batch_id)
    for run in my_batch["runs"]:
        if run["run_id"] == run_id:
            return (my_batch,run)


def RunAll(batch_id):
    """Submit jobs for all imagesets in the batch
    
    Load the batch with the given batch_id from the database
    and run each using CPCluster and bsub
    """
    my_batch = LoadBatch(batch_id)
    response = []
    for run in my_batch["runs"]:
        run_response = RunOne(my_batch,run)
        response.append(run_response)
    return response

def RunOne(my_batch,run):
    x=my_batch.copy()
    x.update(run)
    x["write_data_yes"]=(my_batch["write_data"]!=0 and "yes") or "no"
    cmd=["bsub",
         "-q","%(queue)s"%(x),
         "-o","%(data_dir)s/txt_output/%(start)s_to_%(end)s.txt"%(x),
         "%(cpcluster)s/CPCluster.py"%(x),
         "%(data_dir)s/Batch_data.mat"%(x),
         "%(start)d"%(x),
         "%(end)d"%(x),
         "%(data_dir)s/status"%(x),
         "Batch_",
         "%(write_data_yes)s"%(x),
         "%(timeout)d"%(x)]
    cmd = ' '.join(cmd)
    old_environ = SetEnvironment(my_batch)
    p=os.popen(". /broad/lsf/conf/profile.lsf;"+cmd,'r')
    output=p.read()
    exit_code=p.close()
    RestoreEnvironment(old_environ)
    job=None
    if output:
        match = re.search(r'<([0-9]+)>',output)
        if len(match.groups()) > 0:
            job=int(match.groups()[0])
            CreateJobRecord(run["run_id"],job)
            run["job_id"]=job
    result = {
        "start":run["start"],
        "end":run["end"],
        "command":cmd,
        "exit_code":exit_code,
        "output":output,
        "job":job
        }
        
    return result

def KillOne(run):
    p=os.popen(". /broad/lsf/conf/profile.lsf;bkill %(job_id)d"%(run),"r")
    

def RunTextFile(run):
    """Return the name of the text file created by bsub
    
    Return the name of the text file created by bsub
    run - the dictionary for a run as created by LoadBatch
    """
    return "%(start)d_to_%(end)d.txt"%(run)

def RunTextFilePath(batch, run):
    """Return the path to the text file created by bsub
    
    batch - the dictionary for a batch, as created by LoadBatch
    run - the dictionary for a run as created by LoadBatch
    """
    return "%(data_dir)s/txt_output/"%(batch)+RunTextFile(run)

def RunDoneFile(run):
    """Return the name of the DONE.mat file
    
    Return the name of the file generated by the batch script when the job
    has finished.
    run - a dictionary for a run as created by LoadBatch
    """
    return "Batch_%(start)d_to_%(end)d_DONE.mat"%(run)

def RunDoneFilePath(batch,run):
    """Return the path to the DONE.mat file
    
    Return the path to the file generated by the batch script when the job
    has finished.
    batch - the dictionary for a batch as created by LoadBatch
    run - the dictionary for a run as created by LoadBatch
    """
    return "%(data_dir)s/status/"%(batch)+RunDoneFile(run)

def RunOutFile(run):
    """Return the name of the OUT.mat file
    
    Return the name of the data output file generated by running a batch.
    
    run - a dictionary for a run as created by LoadBatch
    """
    return "Batch_%(start)d_to_%(end)d_OUT.mat"%(run)

def RunOutFilePath(batch,run):
    """Return the path to the OUT.mat file
    
    Return the path to the data output file generated by running a batch.
    
    batch - the dictionary for a batch as created by LoadBatch
    run - the dictionary for a run as created by LoadBatch
    """
    return "%(data_dir)s/status/"%(batch)+RunOutFile(run)

def GetJobStatus(job_id):
    p=os.popen(". /broad/lsf/conf/profile.lsf;bjobs %d"%(job_id),"r")
    fields=p.readline()
    if not re.match("Job <[0-9]+> is not found",fields):
        fields = fields.split()
    else:
        p.close()
        return
    values=p.readline().split()
    p.close()
    result = {}
    for i in range(len(fields)):
        result[fields[i]]=values[i]
    return result

def GetCPUTime(batch, run):
    try:
        text_file = open(RunTextFilePath(batch, run),"r")
        text = text_file.read()
        text_file.close()
        match = re.compile("\s+CPU time\s+:\s+([0-9.]+)\s+sec").search(text)
        return float(match.group(1))
    except:
        return

