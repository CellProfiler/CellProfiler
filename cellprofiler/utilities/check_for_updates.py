'''Check for new versions on a web page, in a separate thread, and
call a callback with the new version information if there is one.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''


import logging
import threading
import urllib2

logger = logging.getLogger(__name__)

class VersionChecker(threading.Thread):
    def __init__(self, url, current_version, callback, user_agent):
        super(VersionChecker, self).__init__()
        self.url = url
        self.user_agent = user_agent
        self.current_version = current_version
        self.callback = callback
        self.daemon = True # if we hang it's no big deal
        self.setName("VersionChecker")
    
    def run(self):
        try:
            req = urllib2.Request(self.url, None, {'User-Agent' : self.user_agent})
            response = urllib2.urlopen(req)
            html = response.read()
            response.close()
            # format should be version number in first line followed by html
            new_version, info = html.split('\n', 1)
            new_version = int(new_version)
            if new_version > self.current_version:
                self.callback(new_version, info)
        except Exception, e:
            logger.warning("Exception fetching new version information from %s: %s"%(self.url, e))
            pass # no worries

def check_for_updates(url, current_version, callback, user_agent='CellProfiler_cfu'):
    vc = VersionChecker(url, current_version, callback, user_agent)
    vc.start()

