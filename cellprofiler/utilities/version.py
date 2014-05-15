'''version.py - Version fetching and comparison.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import datetime
import sys
import subprocess
import re
import os.path
import logging

def datetime_from_isoformat(dt_str):
    return datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")

def get_version():
    '''Get a version as "timestamp version", where timestamp is when the last
    commit was made, and version is a git hash, or if that fails, SVN version
    (relative to whatever SVN repository is in the source), and otherwise, just
    the current time and "unkown".'''

    if not hasattr(sys, 'frozen'):
        import cellprofiler
        cellprofiler_basedir = os.path.abspath(os.path.join(os.path.dirname(cellprofiler.__file__), '..'))
        # Evil GIT without GIT. Look for what we want in the log files.
        try:
            while True:
                git_dir = os.path.join(cellprofiler_basedir, ".git")
                if not os.path.isdir(git_dir):
                    break
                with open(os.path.join(git_dir, "HEAD"), "r") as fd:
                    line = fd.readline().strip()
                # Line is like this:
                #
                # ref: refs/heads/master
                #
                match = re.match("ref:\\s+(.+)", line)
                if match is None:
                    break
                treeish = match.groups()[0]
                #
                # The log is in .git/logs/<treeish>
                #
                log_file = os.path.join(git_dir, "logs", treeish)
                if not os.path.isfile(log_file):
                    break
                pattern = (
                    "(?P<oldhash>[0-9a-f]+) (?P<newhash>[0-9a-f]+) "
                    ".+? (?P<timestamp>[0-9]+) (?P<timezone>[-+]?[0-9]{4})[\t\n]")
                last_hash = None
                with open(log_file, "r") as fd:
                    for line in fd:
                        match = re.search(pattern, line)
                        if match is not None:
                            last_hash = match.groupdict()["newhash"]
                            last_timestamp = match.groupdict()["timestamp"]
                if last_hash is not None:
                    t = datetime.datetime.utcfromtimestamp(float(last_timestamp))
                    return "%s %s" %(t.isoformat("T"), last_hash[:7])
        except:
            pass

        # GIT
        try:
            timestamp, hash = subprocess.Popen(['git', 'log', '--format=%ct %h', '-n', '1'],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT,
                                               cwd=cellprofiler_basedir).communicate()[0].strip().split(' ')
            return '%s %s' % (datetime.datetime.utcfromtimestamp(float(timestamp)).isoformat('T'), hash)
        except (OSError, subprocess.CalledProcessError, ValueError), e:
            pass

        # SVN
        try:
            if os.path.isdir(os.path.join(cellprofiler_basedir, ".svn")):
                # use svn info because it doesn't require the network.
                output = subprocess.Popen(['svn', 'info', '--xml'],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT,
                                          cwd=cellprofiler_basedir).communicate()[0]
                date = re.search('<date>([^.]*)(\\.[0-9]*Z)</date>', output).group(1)
                version = re.search('revision="(.*)">', output).group(1)
                return '%s SVN:%s' % (datetime_from_isoformat(date).isoformat('T'), version)
        except (OSError, subprocess.CalledProcessError), e:
            pass
        except (AttributeError,), e:
            import logging
            logging.root.error("Could not parse SVN XML output while finding version.\n" + output)

        # Give up
        return '%s Unknown rev.' % (datetime.datetime.utcnow().isoformat('T').split('.')[0])
    else:
        import cellprofiler.frozen_version
        return cellprofiler.frozen_version.version_string

'''Code version'''
version_string = get_version()
version_number = int(datetime_from_isoformat(version_string.split(' ')[0]).strftime('%Y%m%d%H%M%S'))
dotted_version = '2.1.1'
git_hash = version_string.split(' ', 1)[1]
title_string = '%s (rev %s)' % (dotted_version, git_hash)

if __name__ == '__main__':
    print version_string
