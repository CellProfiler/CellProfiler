"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

# This file allows developers working in the git repository to fetch
# binary files from SVN (or other site) so that the git repository
# doesn't have to track large files.

import os.path
import hashlib
import urllib2
import shutil
import subprocess
import sys
import zipfile

ACTION_MAVEN = "Maven"

CELLPROFILER_JAVA_JAR = "cellprofiler-java-0.0.1-SNAPSHOT.jar"
CELLPROFILER_DEPENDENCIES_URL = \
    'http://www.cellprofiler.org/linked_files/CellProfilerDependencies'
OMERO_CLIENTS_URL = CELLPROFILER_DEPENDENCIES_URL + '/OMERO.clients-4.4.5'
# The list of files (relative path) to fetch, their SHA1, and their source URL.
files = [
    [['imagej', 'apache-maven-3.0.4-bin.zip'], 
     '29cfd351206016b67dd0d556098513d2b259c69b',
     CELLPROFILER_DEPENDENCIES_URL + '/apache-maven-3.0.4-bin.zip',
     ACTION_MAVEN],
    [['imagej', 'jars', 'blitz.jar'], 
     '106111c58509a05035e8b26b49d214d0ee1e6442',
     OMERO_CLIENTS_URL + '/blitz.jar', None],
    [['imagej', 'jars', 'common.jar'], 
     '83733cd16a498bb6d30c829de5daead061fb7769',
     OMERO_CLIENTS_URL + '/common.jar', None],
    [['imagej', 'jars', 'model-psql.jar'], 
     '7774ffcd7fb0b76a075b39601208a254915a7c49',
     OMERO_CLIENTS_URL + '/model-psql.jar', None],
    [['imagej', 'jars', 'ice.jar'], 
     '017c5f3960be550673ff491bbcc7184c6d6388f1',
     OMERO_CLIENTS_URL + '/ice.jar', None]]


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
    src = urllib2.urlopen(url)
    dest = open(filename, 'wb')
    shutil.copyfileobj(src, dest)

def fetch_external_dependencies(overwrite=False):
    # look for each file, check its hash, download if missing, or out
    # of date if overwrite==True, complain if it fails.  If overwrite
    # is 'fail', die on mismatches hashes.
    if __name__ == "__main__":
        root = os.path.abspath(os.curdir)
    else:
        root = os.path.split(__file__)[0]

    maven_install_path = os.path.join(root, 'imagej', 'maven')
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
                    install_maven(path,
                                  maven_install_path)
            except:
                import traceback
                sys.stderr.write(traceback.format_exc())
                sys.stderr.write("Could not fetch external binary dependency %s from %s.  Some functionality may be missing.  You might try installing it by hand.\n"%(path, url))
                
    imagej_dir = os.path.join(root, 'imagej')
    if overwrite or not os.path.isdir(os.path.join(imagej_dir, "jars")):
        run_maven(imagej_dir, maven_install_path)
    if (overwrite or not 
        os.path.isfile(os.path.join(imagej_dir, CELLPROFILER_JAVA_JAR))):
        run_maven(os.path.join(root, "java"), maven_install_path)
    
def install_maven(zipfile_path, install_path):
    '''Install the Maven jars from a zipfile
    
    zipfile_path - path to the zipfile
    zip_jar_path - path to the jar files within the zip file
    jar_path - destination for the jar files
    '''
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

def run_maven(pom_path, maven_install_path):
    '''Run a Maven pom to install all of the needed jars
    
    pom_path - the directory hosting the Maven POM
    maven_install_path - the path to the maven install
    
    Runs mvn package on the POM
    '''
    from cellprofiler.utilities.setup import find_jdk
    
    jdk_home = find_jdk()
    old_java_home = None
    if jdk_home is not None:
        old_java_home = os.environ.get("JAVA_HOME", None)
        os.environ["JAVA_HOME"] = jdk_home
            
    executeable_path = get_mvn_executable_path(maven_install_path)
    current_directory = os.path.abspath(os.getcwd())
    os.chdir(pom_path)
    try:
        subprocess.check_call([executeable_path, '-U', 'package'])
    finally:
        os.chdir(current_directory)
        if old_java_home is not None:
            os.environ["JAVA_HOME"] = old_java_home


                
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
