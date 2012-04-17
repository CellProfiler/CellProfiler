# This file allows developers working in the git repository to fetch
# binary files from SVN (or other site) so that the git repository
# doesn't have to track large files.

import os.path
import hashlib
import urllib2
import shutil
import sys

# The list of files (relative path) to fetch, their SHA1, and their source URL.
subimager_version = "db06f96da25b28b87577ce55a31be7d69963542e"
files = [
    [['subimager', 'subimager.jar'], '7c5b417a8b7a4b88b16c7752760b5450ca283a7d', 
     'http://www.cellprofiler.org/linked_files/subimager/builds/subimager-%s.jar' % subimager_version]
]


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
                
