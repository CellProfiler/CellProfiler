# This file allows developers working in the git repository to fetch
# binary files from SVN (or other site) so that the git repository
# doesn't have to track large files.

import os.path
import hashlib
import urllib2
import shutil
import sys

# The list of files (relative path) to fetch, their SHA1, and their source URL.
files = [
    [['bioformats', 'loci_tools.jar'], 'a1d08a9dfb648eb86290a036fe0b5a4e47f2c44a', 'https://svn.broadinstitute.org/CellProfiler/!svn/bc/11312/trunk/CellProfiler/bioformats/loci_tools.jar'],
    [['imagej', 'ij.jar'], 'f675dc28e38a9a2612e55db049f4d4bb47d774b1', 'https://svn.broadinstitute.org/CellProfiler/!svn/bc/11073/trunk/CellProfiler/imagej/ij.jar'],
    [['imagej', 'imglib.jar'], '7e0e68aa371706012e224df0b31925317a3dc284', 'https://svn.broadinstitute.org/CellProfiler/!svn/bc/11073/trunk/CellProfiler/imagej/imglib.jar'],
    [['imagej', 'javacl-1.0-beta-4-shaded.jar'], '62b3b41c4759652595070534d79d1294eeba8139', 'https://svn.broadinstitute.org/CellProfiler/!svn/bc/11073/trunk/CellProfiler/imagej/javacl-1.0-beta-4-shaded.jar'],
    [['imagej', 'junit-4.5.jar'], '41eb8ac5586d163d61518be18a13626983f9ece6', 'https://svn.broadinstitute.org/CellProfiler/!svn/bc/11073/trunk/CellProfiler/imagej/junit-4.5.jar'],
    [['imagej', 'precompiled_headless.jar'], '39ef36e6cfbd7f48e8c0e3033b30b0e3f5b5d24e', 'https://svn.broadinstitute.org/CellProfiler/!svn/bc/11073/trunk/CellProfiler/imagej/precompiled_headless.jar']]

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
    root = os.path.split(__file__)[0]
    for path, hash, url in files:
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
            except:
                import traceback
                sys.stderr.write(traceback.format_exc())
                sys.stderr.write("Could not fetch external binary dependency %s from %s.  Some functionality may be missing.  You might try installing it by hand.\n"%(path, url))
                
