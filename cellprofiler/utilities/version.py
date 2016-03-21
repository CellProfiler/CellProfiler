'''version.py - Version fetching and comparison.
'''

import datetime
import logging
import os.path
import re
import subprocess
import sys


def get_git_dir():
    import cellprofiler
    cellprofiler_basedir = os.path.abspath(
            os.path.join(os.path.dirname(cellprofiler.__file__), '..'))
    git_dir = os.path.join(cellprofiler_basedir, ".git")
    if not os.path.isdir(git_dir):
        return None
    return cellprofiler_basedir


def datetime_from_isoformat(dt_str):
    return datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")


def get_version():
    '''Get a version as "timestamp version", where timestamp is when the last
    commit was made, and version is a git hash, or if that fails, SVN version
    (relative to whatever SVN repository is in the source), and otherwise, just
    the current time and "unkown".'''

    unknown_rev = '%s Unknown rev.' % (
        datetime.datetime.utcnow().isoformat('T').split('.')[0])
    if not hasattr(sys, 'frozen'):
        # Evil GIT without GIT. Look for what we want in the log files.
        try:
            cellprofiler_basedir = get_git_dir()
            if cellprofiler_basedir is None:
                try:
                    import cellprofiler.frozen_version
                    return cellprofiler.frozen_version.version_string
                except ImportError:
                    return unknown_rev
            logging.debug("Traversing file system")
            while True:
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
                    return "%s %s" % (t.isoformat("T"), last_hash[:7])
        except:
            pass

        # GIT
        try:
            with open(os.devnull, "r") as devnull:
                timestamp, hash = subprocess.check_output(
                        ['git', 'log', '--format=%ct %h', '-n', '1'],
                        stdin=devnull,
                        cwd=cellprofiler_basedir).strip().split(' ')
            return '%s %s' % (datetime.datetime.utcfromtimestamp(float(timestamp)).isoformat('T'), hash)
        except (OSError, subprocess.CalledProcessError, ValueError), e:
            pass

        try:
            import cellprofiler.frozen_version
            return cellprofiler.frozen_version.version_string
        except ImportError:
            pass
        # Give up
        return unknown_rev
    else:
        import cellprofiler.frozen_version
        return cellprofiler.frozen_version.version_string


def get_dotted_version():
    if not hasattr(sys, 'frozen'):
        try:
            cellprofiler_dir = get_git_dir()
            if cellprofiler_dir is None:
                try:
                    import cellprofiler.frozen_version
                    return cellprofiler.frozen_version.dotted_version
                except ImportError:
                    return "0.0.0"
            with open(os.devnull, "r") as devnull:
                output = subprocess.check_output(
                        ["git", "describe", "--tags"],
                        stdin=devnull,
                        cwd=cellprofiler_dir)
            return output.strip().partition("-")[0]
        except:
            logging.root.warning("Unable to find version - no GIT")
            return "0.0.0"
    else:
        import cellprofiler.frozen_version
        return cellprofiler.frozen_version.dotted_version


'''Code version'''
version_string = get_version()
version_number = int(datetime_from_isoformat(version_string.split(' ')[0]).strftime('%Y%m%d%H%M%S'))
dotted_version = get_dotted_version()
git_hash = version_string.split(' ', 1)[1]
title_string = '%s (rev %s)' % (dotted_version, git_hash)

if __name__ == '__main__':
    print version_string
