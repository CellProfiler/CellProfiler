'''read_directory_url.py - get a directory listing from a URL

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''


import os
import urllib
import urllib2
import xml.dom.minidom as dom

IS_DIRECTORY = 1
IS_FILE = 2
IS_UNKNOWN = 0
def read_directory_url(url):
    '''Given a URL representing a directory, return a list of subdirectories and files
    
    Returns a list of two-tuples:
    * First entry is the file name
    * Second entry is either IS_DIRECTORY, IS_FILE or IS_UNKNOWN depending
      on whether the filename is the name of a directory, file or
      is not known.
    '''
    fd = urllib2.urlopen(url)
    data = ""
    discard = True
    for line in fd:
        if discard:
            if line.lower().find("<ul>") != -1:
                discard = False
            else:
                continue
        elif line.lower().find("/ul") != -1:
            data += line
            break
        data += line
    doc = dom.parseString(data)
    result = []
    for a in doc.getElementsByTagName("a"):
        href = a.getAttribute("href")
        if len(href) > 0:
            if href.startswith("/"):
                continue # Global like "/foo/"
            elif href in ("./", "../"):
                continue
            elif href.endswith("/"):
                result.append((urllib.unquote(href[:-1]), IS_DIRECTORY))
            else:
                result.append((urllib.unquote(href), IS_FILE))
    return result

def walk_url(url, topdown = False):
    files = [f for f,d in read_directory_url(url) if d == IS_FILE]
    directories = [f for f,d in read_directory_url(url) if d == IS_DIRECTORY]
    if topdown:
        yield (url, directories, files)
    for directory in directories:
        full_path = url + "/" + directory
        for result in walk_url(full_path, topdown):
            yield result
    if not topdown:
        yield (url, directories, files)
        
if __name__ == "__main__":
    result = read_directory_url("http://www.broadinstitute.org/~leek/tracking")
    print "\n".join(["%s\t%s" % ( f, "f" if d == IS_FILE else "d")
                     for f, d in result])
    for path, files, directories in walk_url("https://svn.broadinstitute.org/CellProfiler/trunk/ExampleImages", True):
        print path
        for filename in files:
            print "\t"+filename
            
    
    
