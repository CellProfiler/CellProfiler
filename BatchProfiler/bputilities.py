# Utility functions for Batch Profiler
import datetime
import dateutil.parser
import numpy as np
import os
import re
import sys
from cStringIO import StringIO
import subprocess
import tempfile
import stat
from bpformdata import PREFIX, LC_ALL, BATCHPROFILER_CPCHECKOUT, \
     BATCHPROFILER_CELLPROFILER_REPO

QSUB_WORKS = True

SENDMAIL="/usr/sbin/sendmail"
BUILD_TOUCHFILE = ".cellprofiler.built"

ROOT_DIR = BATCHPROFILER_CELLPROFILER_REPO

DATEVERSION_PATTERN="(?P<year>20\\d{2})(?P<month>\\d{2})(?P<day>\\d{2})(?P<hour>\\d{2})(?P<minute>\\d{2})(?P<second>\\d{2})"
def get_batch_data_version_and_githash(batch_filename):
    """Get the commit's GIT hash stored in the batch file's pipeline
    
    batch_filename - Batch_data.h5 file to look at
    
    returns the GIT hash as stored in the file.
    """
    from cellprofiler.utilities.hdf5_dict import HDF5Dict
    import cellprofiler.measurements as cpmeas
    import cellprofiler.pipeline as cpp
    
    m = HDF5Dict(batch_filename, mode="r")
    pipeline_txt = cpmeas.Measurements.unwrap_string(
        m[cpmeas.EXPERIMENT, cpp.M_PIPELINE, 0][0])
    if isinstance(pipeline_txt, unicode):
        pipeline_txt = pipeline_txt.encode("utf-8")
    #
    # TODO:
    # It's not such a good idea to duplicate parsing of the pipeline
    # text. Create a method to read the header info.
    #
    try:
        git_hash = None
        version = None
        pipeline_version = None
        lines = [_.strip() for _ in pipeline_txt.split("\n")]
        if lines[0] != cpp.COOKIE:
            raise ValueError("Failed to recognize cookie")
        for line in lines[1:]:
            if len(line) == 0:
                break
            k, v = line.split(":")
            if k == cpp.H_DATE_REVISION:
                version = v
            elif k == cpp.H_GIT_HASH:
                git_hash = v
            elif k == cpp.H_VERSION:
                pipeline_version = int(v)
    except:
        pipeline = cpp.Pipeline()
        version, git_hash = pipeline.loadtxt(StringIO(pipeline_txt))
    if git_hash is None:
        for log_version, log_git_hash in get_versions_and_githashes():
            if int(log_version) == version:
                git_hash = log_git_hash
                break
    else:
        version, git_hash = get_version_and_githash(git_hash)
    return version, git_hash

def get_batch_image_numbers(batch_filename):
    from cellprofiler.utilities.hdf5_dict import HDF5Dict
    import cellprofiler.measurements as cpmeas
    m = HDF5Dict(batch_filename, mode="r")
    image_numbers = np.array(
        m.get_indices(cpmeas.IMAGE, cpmeas.IMAGE_NUMBER).keys(), int)
    image_numbers.sort()
    return image_numbers

def get_batch_groups(batch_filename):
    from cellprofiler.utilities.hdf5_dict import HDF5Dict
    import cellprofiler.measurements as cpmeas
    image_numbers = get_batch_image_numbers(batch_filename)
    m = HDF5Dict(batch_filename, mode="r")
    if m.has_feature(cpmeas.IMAGE, cpmeas.GROUP_NUMBER):
        group_numbers = np.array(
            [_[0] for _ in m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_numbers]])
        if len(np.unique(group_numbers)) <= 1:
            return
        group_indices = np.array(
            [_[0] for _ in m[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_numbers]])
        return group_numbers, group_indices

def get_version_and_githash(treeish):
    subprocess.check_call(
        ["git", "fetch", "origin", "master"], cwd=ROOT_DIR)
    if re.match(DATEVERSION_PATTERN, treeish):
        for dateversion, githash in get_versions_and_githashes():
            if dateversion == treeish:
                return dateversion, githash
        else:
            raise ValueError("Could not find commit: %s", treeish)    
    #
    # Give me another reason to hate GIT:
    #     2.1.1 preferentially matches 2.1.1-docker
    #
    tag_pattern = "[0-9]+\\.[0-9]+\\.[0-9]+"
    if re.match(tag_pattern, treeish) is not None:
        line = subprocess.check_output(
            ["git", "log", "-n", "1", "--tags=%s" % 
             tag_pattern.replace("\\", "\\\\"), 
             "--pretty=%ai_%H", treeish],
            cwd=ROOT_DIR)
    else:
        line = subprocess.check_output(
            ["git", "log", "-n", "1", "--pretty=%ai_%H", treeish],
            cwd=ROOT_DIR)
    time_str, git_hash = [x.strip() for x in line.split("_")]
    return get_version_from_timestr(time_str), git_hash
    
def get_version_from_timestr(time_str):
    '''convert ISO date to dateversion format
    
    Returns the UTC time of the time string in the format
    YYYYMMDDHHMMSS
    '''
    t = dateutil.parser.parse(time_str).utctimetuple()
    return "%04d%02d%02d%02d%02d%02d" % \
           ( t.tm_year, t.tm_mon, t.tm_mday, 
             t.tm_hour, t.tm_min, t.tm_sec)
                           
def get_versions_and_githashes():
    '''Get the versions and githashes via git-log'''
    splooge = subprocess.check_output(
        ["git", "log", "--pretty=%ai_%H"], cwd = ROOT_DIR)
    result = []
    for line in splooge.split("\n"):
        try:
            time_str, git_hash = [x.strip() for x in line.split("_")]
            result.append((get_version_from_timestr(time_str), git_hash))
        except:
            pass
    return result
    
def get_cellprofiler_location(
    batch_filename = None, version = None, git_hash=None):
    '''Get the location of the CellProfiler source to use

    There are two choices - get by batch name or by version and git hash
    '''
    
    if version is None or git_hash is None:
        version, git_hash = get_batch_data_version_and_githash(batch_filename)
    path = os.path.join(BATCHPROFILER_CPCHECKOUT, 
                        "%s_%s" % (version, git_hash))
    return path

def get_queues():
    '''Return a list of queues'''
    try:
        host_fd, host_scriptfile = tempfile.mkstemp(suffix=".sh") 
        os.fchmod(host_fd, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        os.write(host_fd, "#!/bin/sh\n")
        os.write(host_fd, "set -v\n")
        os.write(host_fd, ". /broad/software/scripts/useuse\n")
        os.write(host_fd, "use GridEngine8\n")
        os.write(host_fd, "set +v\n")
        os.write(host_fd, "qconf -sql\n")
        os.close(host_fd)
        process = subprocess.Popen(
            [host_scriptfile],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return filter((lambda x: len(x) > 0), [x.strip() for x in stdout.split("\n")])
    finally:
        os.unlink(host_scriptfile)
        
def get_jobs():
    '''Return a list of all jobs for the webserver user'''
    script = """#!/bin/sh
set -v
. /broad/software/scripts/useuse
use GridEngine8
set +v
qstat
"""
    scriptfile = make_temp_script(script)
    try:
        output = subprocess.check_output([scriptfile])
        result = []
        for line in output.split("\n"):
            fields = [x.strip() for x in line.strip().split(" ")]
            try:
                if len(fields) > 0:
                    result.append(int(fields[0]))
            except:
                pass
    finally:
        os.rmdir(scriptfile)
    return result
    
def make_temp_script(script):
    '''Write a script to a tempfile
    '''
    host_fd, host_scriptfile = tempfile.mkstemp(suffix=".sh") 
    os.fchmod(host_fd, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    os.write(host_fd, script)
    os.close(host_fd)
    return host_scriptfile
    
def run_on_tgt_os(script, 
                  group_name, 
                  job_name, 
                  queue_name, 
                  output,
                  err_output = None,
                  priority = None,
                  cwd=None, 
                  deps=None,
                  mail_before = False,
                  mail_error = True,
                  mail_after = True,
                  email_address = None):
    '''Run the given script on the target operating system
    
    script - the script to be run with shebang header line
             (e.g. #!/bin/sh)
    group_name - charge to this group
    job_name - name of the job
    queue_name - run on this queue
    output - send stdout to this file
    err_output - send stderr to this file
    priority - the priority # for the job
    cwd - change to this directory on remote machine to run script
    deps - a list of job IDs to wait for before starting this one
    mail_before - true to send email before job starts
    mail_error - true to send email on error
    mail_after - true to send email after job finishes
    email_address - address of email recipient
    '''
    if deps is not None:
        dep_cond = "-hold_jid %s" % (",".join(deps))
    else:
        dep_cond = ""
    if cwd is not None:
        cwd_switch = "-wd %s" % cwd
    else:
        cwd_switch = ""
    if email_address is None or not any([mail_before, mail_error, mail_after]):
        email_switches = ""
    else:
        email_events = "".join([x for x, y in (("b", mail_before),
                                               ("e", mail_error),
                                               ("a", mail_after))
                                if y])
        email_switches = "-m %(email_events)s -M %(email_address)s" % locals()
        
    if err_output is None:
        err_output = output+".err"
    if queue_name is None:
        queue_switch = ""
    else:
        queue_switch = "-q %s" % queue_name
    #
    # TODO: memory and priority, possibly more
    #
    tgt_script = make_temp_script(script)
    host_script = make_temp_script("""#!/bin/sh
qsub -N %(job_name)s \\
    -e %(err_output)s \\
    -o %(output)s \\
    -terse %(dep_cond)s %(email_switches)s %(cwd_switch)s %(queue_switch)s\\
    %(tgt_script)s
""" %locals())
    try:
        p = subprocess.Popen(
            host_script, stdout=subprocess.PIPE, stderr = subprocess.PIPE)
        stdout, stderr = p.communicate(script)
        if p.returncode != 0:
            raise RuntimeError("Failed to submit job: %s" % stderr)
        return int(stdout.strip())
    finally:
        os.unlink(host_script)
        os.unlink(tgt_script)
   
def kill_job(job_id):
    host_script = make_temp_script("""#!/bin/sh
    . /broad/software/scripts/useuse
    reuse -q GridEngine8
    qdel %d"
    """ % job_id)
    try:
        p = subprocess.Popen(
            host_script, stdout=subprocess.PIPE, stderr = subprocess.PIPE)
        stdout, stderr = p.communicate(host_script)
        return stdout, stderr
    finally:
        os.unlink(host_script)
        
def kill_jobs(job_ids):
    host_script = make_temp_script("""#!/bin/sh
    . /broad/software/scripts/useuse
    reuse -q GridEngine8
    qdel %s"
    """ % " ".join(map(str, job_ids)))
    try:
        p = subprocess.Popen(
            host_script, stdout=subprocess.PIPE, stderr = subprocess.PIPE)
        stdout, stderr = p.communicate(host_script)
        return stdout, stderr
    finally:
        os.unlink(host_script)
    
def python_on_tgt_os(args, group_name, job_name, queue_name, output,
                     err_output=None,
                     cwd=None, deps = None,
                     priority=None,
                     mail_before = False,
                     mail_error = True,
                     mail_after = True,
                     email_address = None,
                     with_xvfb = False):
    '''Run Python with the given arguments on a target machine'''
    if cwd is None:
        cd_command = ""
    else:
        cd_command = "cd %s" % cwd
        
    if with_xvfb:
        xvfb_run = "xvfb-run"
    else:
        xvfb_run = ""
    argstr = '"%s"' % '" "'.join(args)
    script = ("""#!/bin/sh
%%(cd_command)s
. /broad/software/scripts/useuse
reuse -q Java-1.6
export PATH=%(PREFIX)s/bin:$PATH
export LD_LIBRARY_PATH=%(PREFIX)s/lib:$LD_LIBRARY_PATH:%(PREFIX)s/lib/mysql:$JAVA_HOME/jre/lib/amd64/server
export LC_ALL=%(LC_ALL)s
export PYTHONNOUSERSITE=1
%%(xvfb_run)s python %%(argstr)s
""" % globals()) % locals()
    return run_on_tgt_os(script, group_name, job_name, queue_name, output, 
                         cwd = cwd, 
                         deps=deps,
                         err_output=err_output,
                         mail_before=mail_before,
                         mail_error=mail_error,
                         mail_after=mail_after,
                         email_address=email_address)
    
def build_cellprofiler(
    version = None, 
    git_hash=None, 
    queue_name=None,
    group_name="imaging", 
    email_address = None):
    '''Build/rebuild a version of CellProfiler

    version - numeric version # based on commit date
    git_hash - git hash of version
    group_name - fairshare group for remote jobs
    email_address - send email notifications here
    
    returns a sequence of job numbers.
    '''
    path = get_cellprofiler_location(version = version, git_hash = git_hash)
    if os.path.isdir(os.path.join(path, ".git")):
        subprocess.check_call(["git", "clean", "-d", "-f", "-x", "-q"], cwd=path)
    else:
        if not os.path.isdir(path):
            os.makedirs(path)
        subprocess.check_call([
            "git", "clone", "https://github.com/CellProfiler/CellProfiler", path])
        subprocess.check_call(["git", "checkout", "-q", git_hash], cwd=path)
    mvn_job = "CellProfiler-mvn-%s" % git_hash
    build_job = "CellProfiler-build-%s" % git_hash
    touch_job = "CellProfiler-touch-%s" % git_hash
    if version > "20120607000000":
        python_on_tgt_os(
            ["external_dependencies.py", "-o"],
            group_name, 
            mvn_job,
            queue_name,
            os.path.join(path, mvn_job+".log"),
            cwd = path,
            mail_after = False,
            email_address=email_address)
        python_on_tgt_os(
            ["CellProfiler.py", "--build-and-exit", "--do-not-fetch"],
            group_name, 
            build_job,
            queue_name, 
            os.path.join(path, build_job+".log"),
            cwd = path,
            deps=[mvn_job],
            mail_after = False,
            email_address=email_address,
            with_xvfb=True)
    else:
        python_on_tgt_os(
            ["CellProfiler.py", "--build-and-exit"],
            group_name, 
            build_job,
            queue_name, 
            os.path.join(path, build_job+".log"),
            cwd = path,
            mail_after = False,
            email_address=email_address,
            with_xvfb=True)
    
    touchfile = os.path.join(path, BUILD_TOUCHFILE)
    python_on_tgt_os(
        args = ["-c", 
                ("import cellprofiler.pipeline;"
                 "open('%s', 'w').close();"
                 "from BatchProfiler.bputilities import shutdown_cellprofiler;"
                 "shutdown_cellprofiler()") % touchfile],
        group_name = group_name,
        job_name = touch_job,
        queue_name = queue_name,
        output = "/dev/null",
        err_output = touch_job+".err",
        deps = [build_job],
        cwd = path,
        email_address = email_address,
        mail_after = True,
        with_xvfb=True)

def is_built(version, git_hash):
    path = get_cellprofiler_location(version=version, git_hash=git_hash)
    return os.path.isfile(os.path.join(path, BUILD_TOUCHFILE))
        
def send_mail(recipient, subject, content_type, body):
    '''Send mail to a single recipient
    
    recipient - email address of recipient
    subject - subject field of email
    content_type - mime type of the message body
    body - the payload of the mail message
    '''
    pipe= subprocess.Popen([SENDMAIL, "-t"], stdin=subprocess.PIPE)
    doc = """To: %(recipient)s
Subject: %(subject)s
Content-Type: %(content_type)s; charset=UTF-8

%(body)s

""" % locals()
    pipe.communicate(doc)
    
def send_html_mail(recipient, subject, html):
    '''Send mail that has HTML in the body

    recipient - email address of recipient
    subject - subject field of email
    content_type - mime type of the message body
    body - the payload of the mail message
    '''
    send_mail(recipient=recipient,
              subject = subject,
              content_type="text/html",
              body=html)

class CellProfilerContext(object):
    '''Wrap CellProfiler operations in a handy context
    
    Redirect stderr and stdout to prevent log printing
    On exit, shut things down
    '''
    def __enter__(self):
        devnull = open("/dev/null", "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        return self
        
    def __exit__(self, exc_1, exc_2, exc_3):
        shutdown_cellprofiler()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        
    
def shutdown_cellprofiler():
    '''Oh sadness and so many threads that won't die...
    
    '''
    try:
        import javabridge
        javabridge.kill_vm()
    except:
        pass
    try:
        from ilastik.core.jobMachine import GLOBAL_WM
        GLOBAL_WM.stopWorkers()
    except:
        pass

    
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    output = []
    with CellProfilerContext():
        for arg in sys.argv[1:]:
            output.append(get_batch_data_version_and_githash(arg))
    for line in output:
        print line
        
