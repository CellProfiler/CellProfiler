"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

# This file allows developers working in the git repository to fetch
# binary files from SVN (or other site) so that the git repository
# doesn't have to track large files.

import logging
logger = logging.getLogger(__package__)
import re
import os.path
import hashlib
import urllib2
import shutil
import subprocess
import sys
import traceback
import zipfile

#From https://gist.github.com/edufelipe/1027906
def check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.
 
    Backported from Python 2.7 as it's implemented as pure python on stdlib.
 
    >>> check_output(['/usr/bin/python', '--version'])
    Python 2.6.2

    """
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        error = subprocess.CalledProcessError(retcode, cmd)
        error.output = output
        raise error
    return output

ACTION_MAVEN = "Maven"

CELLPROFILER_DEPENDENCIES_URL = \
    'http://www.cellprofiler.org/linked_files/CellProfilerDependencies'
OMERO_CLIENTS_URL = CELLPROFILER_DEPENDENCIES_URL + '/OMERO.clients-5.0.0-ice35-b19'
# The list of files (relative path) to fetch, their SHA1, and their source URL.
files = [
    [['imagej', 'apache-maven-3.0.4-bin.zip'], 
     '29cfd351206016b67dd0d556098513d2b259c69b',
     CELLPROFILER_DEPENDENCIES_URL + '/apache-maven-3.0.4-bin.zip',
     ACTION_MAVEN],
    [['imagej', 'jars', 'blitz.jar'], 
     '537bb9c05adc23cb07be21991bc4511aefe92dfd',
     OMERO_CLIENTS_URL + '/blitz.jar', None],
    [['imagej', 'jars', 'common.jar'], 
     '8c6926ef5c77d1606dfb2483232ddff4716553f9',
     OMERO_CLIENTS_URL + '/common.jar', None],
    [['imagej', 'jars', 'model-psql.jar'], 
     'aeaf122dbb2ffa2fe716194daf818c12bc764183',
     OMERO_CLIENTS_URL + '/model-psql.jar', None],
    [['imagej', 'jars', 'ice.jar'], 
     'f11f38c0f643cafe933089827395c8e5d29162e7',
     OMERO_CLIENTS_URL + '/ice.jar', None],
    [['imagej', 'jars', 'ice-glacier2.jar'],
     '90b6cbc3d05c3610f00e23efe7067a11a74b84b2',
     OMERO_CLIENTS_URL + '/ice-glacier2.jar', None],
    [['imagej', 'jars', 'ice-storm.jar'],
     'b3ecbee2e7f25daf2adf5c890b65965ed518dcb9',
     OMERO_CLIENTS_URL + '/ice-storm.jar', None],
    [['imagej', 'jars', 'ice-grid.jar'],
     '108cb942775d3d4763d7d2a9d1c646f6c5f21354',
     OMERO_CLIENTS_URL + '/ice-grid.jar', None]
]

pom_folders = ["imagej", "java"]
classpath_filenames = ("cellprofiler-dependencies-classpath.txt",
                       "cellprofiler-java-dependencies-classpath.txt")

def filehash(filename):
    sha1 = hashlib.sha1()
    try:
        f = open(filename, 'rb')
        for chunk in iter(lambda: f.read(8192), ''):
            sha1.update(chunk)
        return sha1.hexdigest()
    except:
        return ''

def fetchfile(filename, url):
    print "fetching %s to %s"%(url, filename)
    # no try/except, it's wrapped below, and we just fail and whine to the user.
    path = os.path.split(filename)[0]
    if not os.path.isdir(path):
        os.makedirs(path)
    src = urllib2.urlopen(url)
    dest = open(filename, 'wb')
    shutil.copyfileobj(src, dest)

def get_cellprofiler_root_dir():
    if __name__ == "__main__":
        root = os.path.abspath(os.curdir)
    else:
        root = os.path.abspath(os.path.split(__file__)[0])
    return root
    
def get_maven_install_path():
    '''Return the location of the Maven install'''
    root = get_cellprofiler_root_dir()
    return os.path.join(root, 'imagej', 'maven')
    
def fetch_external_dependencies(overwrite=False):
    # look for each file, check its hash, download if missing, or out
    # of date if overwrite==True, complain if it fails.  If overwrite
    # is 'fail', die on mismatches hashes.
    root = get_cellprofiler_root_dir()
    for path, hash, url, action in files:
        path = os.path.join(root, *path)
        try:
            assert os.path.isfile(path)
            if overwrite == True:
                assert filehash(path) == hash
            else:
                if filehash(path) != hash:
                    sys.stderr.write("Warning: hash of depenency %s does not match expected value.\n"%(path))
                    if overwrite == 'fail':
                        raise RuntimeError('Mismatched hash for %s'%(path))
            continue
        except AssertionError, e:
            # fetch the file
            try:
                fetchfile(path, url)
                assert os.path.isfile(path)
                assert filehash(path) == hash, 'Hashes do not match!'
                if action == ACTION_MAVEN:
                    install_maven(path)
            except:
                sys.stderr.write(traceback.format_exc())
                sys.stderr.write("Could not fetch external binary dependency %s from %s.  Some functionality may be missing.  You might try installing it by hand.\n"%(path, url))
    if overwrite == 'fail':
        return
                
    logging.info("Updating Java dependencies using Maven.")
    for pom_folder in pom_folders:
        pom_dir = os.path.join(root, pom_folder)
        try:
            try:
                if check_maven_repositories(pom_dir):
                    aggressive_update = overwrite
                else:
                    aggressive_update = None
            except:
                # check_maven_repositories runs with the -o switch to prevent it
                # from going to the Internet. If the local repository doesn't
                # have all the necessary pieces, mvn returns an error code
                # and check_output throws to here.
                #
                # Tell run_maven to update aggressively
                aggressive_update = True
    
            if overwrite:
                run_maven(pom_dir,
                          goal="clean",
                          aggressive_update = aggressive_update)
            run_maven(pom_dir, 
                      quiet = not overwrite,
                      run_tests = overwrite,
                      aggressive_update = aggressive_update)
        except:
            sys.stderr.write(traceback.format_exc())
            sys.stderr.write("Maven failed to update Java dependencies.\n")
            if not overwrite:
                sys.stderr.write("Run external_dependencies with the -o switch to get full output.\n")
        
    
def install_maven(zipfile_path):
    '''Install the Maven jars from a zipfile
    
    zipfile_path - path to the zipfile
    zip_jar_path - path to the jar files within the zip file
    jar_path - destination for the jar files
    '''
    install_path = get_maven_install_path()
    zf = zipfile.ZipFile(zipfile_path)
    zf.extractall(install_path)
    if sys.platform != 'win32':
        import stat
        executeable_path = get_mvn_executable_path(install_path)
        os.chmod(executeable_path,
                 stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR|
                 stat.S_IRGRP | stat.S_IXGRP|
                 stat.S_IROTH | stat.S_IXOTH)
    
def get_mvn_executable_path(maven_install_path):
    subdir = reduce(max, [x for x in os.listdir(maven_install_path)
                          if x.startswith('apache-maven')])
    
    if sys.platform == 'win32':
        executeable = 'mvn.bat'
    else:
        executeable = 'mvn'
    executeable_path = os.path.join(maven_install_path, subdir, 'bin', 
                                    executeable)
    return executeable_path

def run_maven(pom_path, goal="package",
              quiet=False, run_tests=True, aggressive_update = True,
              return_stdout = False, additional_args = []):
    '''Run a Maven pom to install all of the needed jars
    
    pom_path - the directory hosting the Maven POM
    
    goal - the maven goal. "package" is the default which is pretty much
           "do whatever the POM was built to do"
    
    quiet - feed Maven the -q switch if true to make it run in quiet mode
    
    run_tests - if False, set the "skip tests" maven flag. This is appropriate
                if you're silently building a known-good downloaded source.
    
    aggressive_update - if True, use the -U switch to make Maven go to the
                internet and check for updates. If False, default behavior.
                If None, use the -o switch to prevent Maven from any online
                updates.
                
    return_stdout - redirect stdout to capture a string and return it if True,
                    dump Maven output to console if False
                    
    additional_args - additional arguments for the command-line
    
    Runs mvn package on the POM
    '''
    from cellprofiler.utilities.setup import find_jdk
    
    maven_install_path = get_maven_install_path()
    jdk_home = find_jdk()
    env = os.environ.copy()
    if jdk_home is not None:
        env["JAVA_HOME"] = jdk_home.encode("utf-8")
            
    executeable_path = get_mvn_executable_path(maven_install_path)
    args = [executeable_path]
    if aggressive_update:
        args.append("-U")
    elif aggressive_update is None:
        args.append("-o")
    if quiet:
        args.append("-q")
    if not run_tests:
        args.append("-Dmaven.test.skip=true")
    args += additional_args
    args.append(goal)
    logging.debug("Running %s" % (" ".join(args)))
    if return_stdout:
        return check_output(args, cwd = pom_path, env=env)
    else:
        subprocess.check_call(args, cwd = pom_path, env=env)
            
def check_maven_repositories(pom_path):
    '''Check the repositories used by the POM for internet connectivity
    
    pom_path - location of the pom.xml file
    
    returns True if we can reach all repositories in a reasonable amount of time,
            False if the ping failed.
    Throws an exception if Maven failed, possibly because, given the -o switch
    it didn't have what it needed to run the POM.
    
    the goal, "dependency-list-repositories", lists the repositories needed
    by a POM.
    '''
    output = run_maven(pom_path, 
                       goal="dependency:list-repositories",
                       aggressive_update = None,
                       return_stdout=True)
    pattern = r"\s*url:\s+((?:http|ftp|https):.+)"
    for line in output.split("\n"):
        line = line.strip()
        match = re.match(pattern, line)
        if match is not None:
            url = match.groups()[0]
            try:
                urllib2.urlopen(url, timeout=1)
            except urllib2.URLError, e:
                return False
    return True

def get_cellprofiler_jars():
    '''Return the class path for the Java dependencies
    
    NOTE: should not be called for the frozen version of CP
    '''
    root = get_cellprofiler_root_dir()
    jars = set(filter(lambda x:x.endswith(".jar"), [x[0][-1] for x in files]))
    aggressive_update = None
    #
    # Our jars come first because of patches
    #
    jar_dir = os.path.join(root, "imagej", "jars")
    our_jars = ["cellprofiler-java.jar"]
    for filename in classpath_filenames:
        path = os.path.join(jar_dir, filename)
        if not os.path.isfile(path):
            raise RuntimeError(
                "Can't determine CellProfiler java dependencies because %s is missing. Please re-run external_dependencies with the -o switch" % path)
        jar_line = open(path, "r").readline().strip()
        jar_list = jar_line.split(os.pathsep)
        jar_filenames = [os.path.split(jar_path)[1] for jar_path in jar_list]
        if len(our_jars) > 0:
            jar_set = set(our_jars)
            jar_filenames = filter((lambda x: x not in jar_set), jar_filenames)
        our_jars += jar_filenames
    return our_jars + sorted(jars)

if __name__=="__main__":
    import optparse
    usage = """Fetch external dependencies from internet
usage: %prog [options]"""
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-m", "--missing-only",
                      action="store_const",
                      const=False,
                      dest="overwrite",
                      default=False,
                      help="Download external dependency only if missing")
    parser.add_option("-o", "--overwrite",
                      action="store_const",
                      const=True,
                      dest="overwrite",
                      help="Download external dependency if hash doesn't match")
    parser.add_option("-f", "--fail",
                      action="store_const",
                      const="fail",
                      dest="overwrite",
                      help="Fail if a dependency exists and its hash is wrong")
    options, args = parser.parse_args()
    print "Fetching external dependencies..."
    fetch_external_dependencies(options.overwrite)
    print "Fetch complete"
