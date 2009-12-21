'''Check for new versions on a web page, in a separate thread, and
call a callback with the new version information if there is one.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import threading
import urllib2

class VersionChecker(threading.Thread):
    def __init__(self, url, current_version, callback):
        super(VersionChecker, self).__init__()
        self.url = url
        self.current_version = current_version
        self.callback = callback
        self.daemon = True # if we hang it's no big deal
    
    def run(self):
        try:
            response = urllib2.urlopen(self.url)
            html = response.read()
            # format should be version number in first line followed by html
            new_version, info = html.split('\n', 1)
            new_version = int(new_version)
            if new_version > self.current_version:
                self.callback(new_version, info)
        except Exception, e:
            print "Exception fetching new version information:", e
            pass # no worries

def check_for_updates(url, current_version, callback):
    vc = VersionChecker(url, current_version, callback)
    vc.start()
