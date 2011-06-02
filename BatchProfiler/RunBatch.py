#
# Functions for running a batch or a single run from the database
#
import MySQLdb
import subprocess
import os
import re

batchprofiler_host = "imgdb02"
batchprofiler_db = "batchprofiler"
batchprofiler_user = "cpadmin"
batchprofiler_password = "cPus3r"
connection = MySQLdb.Connect(host = batchprofiler_host,
                             db=batchprofiler_db,
                             user=batchprofiler_user,
                             passwd=batchprofiler_password)

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
    insert into batch (batch_id, email, data_dir, queue, batch_size, 
                       write_data, timeout, cpcluster, project, memory_limit,
                       priority)
    values (null,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    bindings = [my_batch[x] for x in (
        'email', 'data_dir', 'queue','batch_size','write_data','timeout',
        'cpcluster','project','memory_limit', 'priority')]
    cursor.execute(cmd, bindings)
    cursor = connection.cursor()
    cursor.execute("select last_insert_id()")
    my_batch["batch_id"]=cursor.fetchone()[0]
    cursor.close()
    return my_batch["batch_id"]

def encode_group_string(x):
    '''Escape the characters '=' and ',' in a group key or value'''
    x = str(x)
    return x.replace('\\','\\\\').replace('=','\\=').replace(',','\\,')

def decode_group_string(x):
    '''Decode an encoded string'''
    return x.replace('\\,',',').replace('\\=','=').replace('\\\\','\\')

def pack_group(run):
    '''Convert the 'group' field into the database field'''
    if not run.has_key("group"):
        run["group_or_null"] = None
        return
    if run["group"] is None:
        run["group_or_null"] = None
        return
    run["group_or_null"] = ','.join(['='.join([encode_group_string(x) 
                                               for x in item])
                                     for item in run["group"].items()])

def unpack_group(run):
    if run["group_or_null"] is None:
        run["group"] = None
    else:
        kvs = re.split('(?<!\\\\),',run["group_or_null"])
        kvpairs = [[decode_group_string(x) for x in re.split('(?<!\\\\)=',kv)]
                   for kv in kvs]
        run["group"] = dict(kvpairs)

def CreateRunRecord(batch_id, run):
    """Create a run record within the current batch
    
    """
    run["batch_id"]=batch_id
    pack_group(run)
    cursor = connection.cursor()
    if run["group_or_null"] is None:
        sql = ("insert into run "
               "(run_id, batch_id, bstart, bend, bgroup, status_file_name) "
               "values (null, %s, %s, %s, null, %s)")
        binding_names = ('batch_id', 'start', 'end', 'status_file_name')
    else:
        sql= ("insert into run "
              "(run_id, batch_id, bstart,bend,bgroup, status_file_name)"
              "values (null,%s,%s,%s,%s,%s)")
        binding_names =  ('batch_id','start','end','group_or_null',
                          'status_file_name')
    bindings = [run[x] for x in binding_names]

    cursor.execute(sql, bindings)
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
    sql = ("select email, data_dir, queue, batch_size, write_data, timeout,"
           "cpcluster, project, memory_limit, priority "
           "from batch where batch_id=%d") % (batch_id)
    cursor = connection.cursor()
    cursor.execute(sql)
    row = cursor.fetchone()
    cursor.close()
    my_batch= {
        "batch_id":     batch_id,
        "email":        row[0],
        "data_dir":     row[1],
        "queue":        row[2],
        "batch_size":   int(row[3]),
        "write_data":   int(row[4]),
        "timeout":      float(row[5]),
        "cpcluster":    row[6],
        "project":      row[7],
        "memory_limit": float(row[8]),
        "priority": int(row[9])
        }
    cursor = connection.cursor()
    cursor.execute("""
select run_id,batch_id,bstart,bend,bgroup,status_file_name,
       (select max(job_id) from job j where j.run_id = r.run_id)
  from run r where batch_id=%d"""%(batch_id));
    runs = []
    my_batch["runs"] = runs
    for row in cursor.fetchall():
        run = {
            "run_id":           int(row[0]),
            "batch_id":         int(row[1]),
            "start":            int(row[2]),
            "end":              int(row[3]),
            "group_or_null":    row[4],
            "status_file_name": row[5],
            "job_id":           (row[6] and int(row[6]))
        }
        unpack_group(run)
        runs.append(run)
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
    txt_output = os.path.join(my_batch["data_dir"],"txt_output")
    status = os.path.join(my_batch["data_dir"],"status")
    if not os.path.exists(txt_output):
        os.mkdir(txt_output)
    if not os.path.exists(status):
        os.mkdir(status)
    response = []
    for run in my_batch["runs"]:
        run_response = RunOne(my_batch,run)
        response.append(run_response)
    return response

def IsPythonBatch(my_batch):
    return my_batch["cpcluster"].startswith("CellProfiler_2_0")

def PythonDir(my_batch):
    cpcluster = my_batch["cpcluster"]
    if cpcluster.find(':') == -1:
        return os.curdir
    return cpcluster[cpcluster.find(':')+1:]

def RunOne(my_batch,run):
    x=my_batch.copy()
    x.update(run)
    if IsPythonBatch(my_batch):
        return RunOne_2_0(x, run)
    else:
        return RunOne_1_0(x, run)

def RunOne_1_0(x, run):
    x["write_data_yes"]=(my_batch["write_data"]!=0 and "yes") or "no"
    x["memory_limit_gb"]=max(1,int(my_batch["memory_limit"]/1000))
    x["memory_limit_gb2"]=x["memory_limit_gb"]*2
    cmd=["bsub",
         "-q","%(queue)s"%(x),
         "-M","%(memory_limit_gb2)d"%(x),
         "-R",'"rusage[mem=%(memory_limit_gb)d]"'%(x),
         "-P","%(project)s"%(x),
         "-g","/imaging/batch/%(batch_id)d"%(x),
         "-J","/imaging/batch/%(batch_id)d/%(start)s_to_%(end)s"%(x),
         "-o","%(data_dir)s/txt_output/%(start)s_to_%(end)s.txt"%(x),
         "-sp","%(priority)d" % x,
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
    p=os.popen(". /broad/lsf/conf/profile.lsf;umask 2;"+cmd,'r')
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

def RunOne_2_0(x, run):
    '''Run one batch in pyCP'''
    x["write_data_yes"]=(x["write_data"]!=0 and "yes") or "no"
    x["memory_limit_gb"]=max(1,int(x["memory_limit"]/1000))
    x["memory_limit_gb2"]=x["memory_limit_gb"]*2
    x["done_file"] = RunDoneFilePath(x, run)
    python_dir = PythonDir(x)
    select = "select[ostype=CENT5.5]"
    try:
        version = int(os.path.split(python_dir)[1])
        if version < 9970:
            # Pre-centos: use PC6000
            select="select[model=PC6000]"
    except:
        pass
    os.environ["CELLPROFILER_USE_XVFB"] = "1"
    os.environ["DISPLAY"] = "1"
    cmd=["bsub",
         "-q","%(queue)s"%(x),
         "-sp","%(priority)d" % x,
         "-M","%(memory_limit_gb2)d"%(x),
         "-R",'"rusage[mem=%(memory_limit_gb)d]"'%(x),
         "-R",'"%s"'%select,
         "-P","%(project)s"%(x),
         "-cwd",PythonDir(x),
         "-g","/imaging/batch/%(batch_id)d"%(x),
         "-J","/imaging/batch/%(batch_id)d/%(start)s_to_%(end)s"%(x),
         "-o",'"%(data_dir)s/txt_output/%(start)s_to_%(end)s.txt"'%(x),
         "./python-2.6.sh",
         "CellProfiler.py",
         "-p",'"%(data_dir)s/Batch_data.mat"'%(x),
         "-c",
         "-r","-b",
         "-f","%(start)d"%(x),
         "-l","%(end)d"%(x),
         "-d",'"%(done_file)s"'%(x)]
    if x["group_or_null"] is not None:
        cmd += ["-g",'"%s"'%(x["group_or_null"])]
    if x["write_data"]:
        cmd += ['"%(data_dir)s/%(start)s_to_%(end)s.mat"'%(x)]
    cmd = ' '.join(cmd)
    p=os.popen(". /broad/lsf/conf/profile.lsf;umask 2;"+cmd,'r')
    output=p.read()
    exit_code=p.close()
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
    p=os.popen(". /broad/lsf/conf/profile.lsf;bkill -s 6 %(job_id)d"%(run),"r")
    

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
    '''Get the status of a single job or a sequence of jobs on imageweb'''
    if isinstance(job_id, basestring) or getattr(job_id, '__iter__', False):
        result = {}
        for i in job_id:
            result[i] = {}
        p=os.popen(". /broad/lsf/conf/profile.lsf;bjobs -w -u imageweb")
        fields = p.readline().strip()
        if fields.startswith('No unfinished'):
            return result
        fields = fields.split()
        for line in p.readlines():
            line = line.strip()
            values = line.split()
            id = int(values[0])
            for field,value in zip(fields,values):
                if result.has_key(id):
                    result[id][field] = value
        return result
    else:
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
        match = re.compile(".*\s+CPU time\s+:\s+([0-9.]+)\s+sec",
                           re.DOTALL).search(text)
        return float(match.group(1))
    except:
        return

