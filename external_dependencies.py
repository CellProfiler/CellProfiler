"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import os
import urllib2


def install_prokaryote(version="1.0.0"):
    path = "build/prokaryote-{}.jar".format(version)

    if not os.path.isfile(path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "wb") as JAR:
            endpoint = "https://github.com/CellProfiler/prokaryote/releases/download"

            resource = "{}/{}".format(endpoint, "{}/prokaryote-{}.jar".format(version, version))

            prokaryote = urllib2.urlopen(resource).read()

            JAR.write(prokaryote)

            JAR.close()

if __name__ == "__main__":
    install_prokaryote()
