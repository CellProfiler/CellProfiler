# Utility functions for Batch Profiler
import os
import sys
from cStringIO import StringIO
import subprocess
import tempfile
import stat

PREFIX = "/imaging/analysis/CPCluster/CellProfiler-2.0/builds/redhat_6"
LC_ALL = "en_US.UTF-8"

def get_batch_data_version_and_githash(batch_filename):
    """Get the commit's GIT hash stored in the batch file's pipeline
    
    batch_filename - Batch_data.h5 file to look at
    
    returns the GIT hash as stored in the file.
    """
    import cellprofiler.measurements as cpmeas
    import cellprofiler.pipeline as cpp
    
    m = cpmeas.load_measurements(batch_filename, mode="r")
    pipeline_txt = m[cpmeas.EXPERIMENT, cpp.M_PIPELINE]
    pipeline = cpp.Pipeline()
    version, git_hash = pipeline.loadtxt(StringIO(pipeline_txt))
    return version, git_hash

def get_cellprofiler_location(
    batch_filename = None, version = None, git_hash=None):
    '''Get the location of the CellProfiler source to use

    There are two choices - get by batch name or by version and git hash
    '''
    
    if version is None or git_hash is None:
        version, git_hash = get_batch_data_version_and_githash(batch_filename)
    path = os.path.join(os.environ["CPCLUSTER"], "%s_%s" % (version, git_hash))
    return path

def get_queues():
    '''Return a list of queues'''
    try:
        host_fd, host_scriptfile = tempfile.mkstemp(suffix=".sh") 
        os.fchmod(host_fd, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        os.write(host_fd, "#!/bin/sh\n")
        os.write(host_fd, ". /broad/software/scripts/useuse\n")
        os.write(host_fd, "use GridEngine8\n")
        os.write(host_fd, "qconf -sql\n")
        os.close(host_fd)
        process = subprocess.Popen(
            [host_scriptfile],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return [x.strip() for x in stdout.split("\n")]
    finally:
        os.unlink(host_scriptfile)
    
    
def run_on_tgt_os(script, group_name, job_name, queue_name, memory_limit, cwd, deps=None):
    '''Run the given script on the target operating system'''
    if deps is not None:
        dep_cond = " && ".join(["done(%s)" % d for d in deps])
        dep_cond = "-w \"" + dep_cond + "\" "
    else:
        dep_cond = ""
    host_script = (
        ". /broad/software/scripts/useuse && "
        "reuse GridEngine8 && "
        "qsub -N %(job_name)s -q %(queue_name)s "
        "-M %(memory_limit)s "
        "-cwd %(cwd)s "
        )  % locals()
    host_script += dep_cond
    p = subprocess.check_call(["sh", "-c", host_script], 
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         cwd=cwd)
    stdout, stderr = p.communicate(script)
    return stdout, stderr
   
def python_on_tgt_os(args, group_name, job_name, queue_name, memory_limit, cwd, deps = None):
    '''Run Python with the given arguments on a target machine'''
    script = (
        "PATH=%(PREFIX)s/bin:$PATH "
        "LD_LIBRARY_PATH=%(PREFIX)s/lib:$LD_LIBRARY_PATH:%(PREFIX)/lib/mysql:"
        "$JAVA_HOME/jre/lib/amd64/server " 
        "LC_ALL=$(LC_ALL)s "
        "python ") % globals()
    script += " ".join(args)
    return run_on_tgt_os(script, group_name, job_name, queue_name, memory_limit, deps)
    
def build_cellprofiler(version = None, git_hash=None, group_name="imaging"):
    path = get_cellprofiler_location(version = version, git_hash = git_hash)
    os.makedirs(path)
    if os.path.isdir(os.path.join(path, ".git")):
        subprocess.check_call(["git", "clean", "-d", "-f"], cwd=path)
    else:
        subprocess.check_call([
            "git", "clone", "https://github.com/CellProfiler/CellProfiler", path])
        subprocess.check_call(["git", "checkout", git_hash], cwd=path)
    mvn_job = "CellProfiler-mvn-%s" % git_hash
    python_on_tgt_os(
        ["external_dependencies.py" "-o"],
        group_name, 
        mvn_job,
        "hour", "4200000", path)
    python_on_tgt_os(
        ["CellProfiler.py", "--build-and-exit", "--do-not-fetch"],
        group_name, 
        "CellProfiler-build-%s" % git_hash,
        "interactive", "4200000", path, [mvn_job])
    

