"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

from __future__ import with_statement
import threading
import Queue
import uuid
import os
import time
import stat
import traceback
import errno

class TimeOutException(Exception):
    pass

class TimeOutFunction(object):
    def __init__(self, func):
        self.func = func
        self.t = None

    def __call__(self, timeout, *args):
        def do_call(temp):
            try:
                temp['value'] = self.func(*args)
            except Exception, e:
                temp['exception'] = e
        retval = {}
        t = threading.Thread(target=do_call, args=(retval,))
        t.daemon = True
        t.start()
        t.join(timeout)
        if 'value' in retval:
            return retval['value']
        if 'exception' in retval:
            raise retval['exception']
        raise TimeOutException('timeout in %s, %s secs' % (self.func.__name__, timeout))

timeout_listdir = TimeOutFunction(os.listdir)
timeout_stat = TimeOutFunction(os.stat)

# Helper class for Locator.
# Note that these all live within a lock-protected dictionary within the Locator class.
class Info(object):
    def __init__(self, key, path, isdir, status, depth, parent=None, num_failures=0, num_timeouts=0):
        self.key = key
        self.path = path
        self.isdir = isdir
        self.depth = depth
        self.status = status
        self.error = False
        self.parent = parent
        self.num_failures = num_failures
        self.num_timeouts = num_timeouts
        self.children = []
        self.in_progress = False
        self.blocked = False
        self.data = None
        self.high_priority = False

# status values
FOUND, STAT, LISTING, METADATA, REMOVED, ERROR, FINISHED = [0, 1, 2, 3, 4, 5, 6]

# priority levels
PRI_IMMEDIATE, PRI_DIRECTORY, PRI_STAT, PRI_METADATA = [0, 1, 2, 3]

class Locator(object):
    '''A queue-and-callback driven file finder, using multiple threads to
    search for files and extract metadata from them.

    creation:
    loc = Locator(callback, metadata_function, descend=True, num_threads=8,
                  max_fallbacks=5, listdir_time=4, stat_time=1)

    callback should be a function with signature:
      callback(pathname, isdir, key, parent, status, data)

      pathname - directory or filename of the discovered object
      isdir - True for directories
      key - arbitrary key for use in the remove() and prioritize() methods
      parent - key of the parent of this object, or None
      status - one of (FOUND, STAT, LISTING, METADATA, FINISHED, ERROR)
      data - when status is FINISHED, metadata for files or list of files for directories.
             when status is EXCEPTION, exception information

      Note that callback() will be called at least three times, once for each
      status change (unless removed with Locator.remove()).  LISTING and METADATA
      are for directories and files, respectively.  ERROR is used to report
      exceptions (timeouts and failures).

      NB: callback() will be called from multiple different threads, and should
      handle (and take advantage of) this in an appropriate manner.

    metadata_function should take a path and return an arbitrary object.

    listdir_time, stat_time, and max_fallbacks are use to control retries after
    filesystem errors and timeouts.  Each timeout will double the timeout used
    next time to the maximum of fallbacks.  Timeouts are advisory, and may
    resolve themselves in the indefinite future.  Errors will continue to retry
    without the fallbacks.

    usage:
    key = loc.queue(path)  # directory or individual file
    loc.pause()
    loc.unpause()
    loc.remove(key)  # remove a directory and all of its children from the search
    path, parent, status = loc.info(key)
    loc.prioritize(key)

    TODO:
    loc.block(key)  # prevents (and interrupts) searching key's directory and all of its children
    loc.unblock(key)  # reallows searching a key's directory and children

    '''

    def __init__(self, callback, metadata_function, descend=True, num_threads=20, max_fallbacks=4, listdir_time=30, stat_time=1):
        self.callback = callback
        self.metadata_function = metadata_function
        self.descend = descend
        self.max_fallbacks = max_fallbacks
        self.listdir_time = listdir_time
        self.stat_time = stat_time

        self.paused = False
        self.num_active_threads = 0
        self.hipri_count = 0
        self.lopri_count = 0

        self._info = {}  # {key : Info}
        self._info_lock = threading.RLock()
        self._active_cv = threading.Condition()
        # priorities are (priority, depth) to do a breadth-first search
        self._queue = Queue.PriorityQueue()
        for tidx in range(num_threads):
            t = threading.Thread(target=self.search, name='Searcher %d' % tidx)
            t.daemon = True
            t.start()

    def _make_callback(self, key):
        with self._info_lock:
            f_info = self._info[key]
            self.callback(f_info.path,
                          f_info.isdir,
                          key,
                          f_info.parent,
                          ERROR if f_info.error else f_info.status,
                          f_info.data)

    def queue(self, path):
        return self._enqueue(path, PRI_STAT, 0, None)

    def _enqueue(self, path, priority, depth, parent, key=None):
        key = key or uuid.uuid4()
        f_info = Info(key, path, None, FOUND, depth, parent=parent)
        # self._info must be updated before this path is queued
        with self._info_lock:
            if parent:
                # make sure parent hasn't been removed
                if (parent not in self._info) or (self._info[parent].status == REMOVED):
                    return
                self._info[parent].children.append(key)
            self._info[key] = f_info
            # make a callback to ensure whatever code is using the Locator sees
            # a parent before any of its children.
            self._make_callback(key)
            # move to STAT state
            f_info.status = STAT
        self._queue.put((priority, depth, key))
        with self._active_cv:
            self._active_cv.notify()
        return key

    def _requeue(self, priority, depth, key):
        with self._info_lock:
            f_info = self._info[key]
            if f_info.parent:
                if (f_info.parent not in self._info) or (self._info[f_info.parent].status == REMOVED):
                    return
        self._queue.put((priority, depth, key))
        with self._active_cv:
            self._active_cv.notify()

    def pause(self):
        with self._active_cv:
            self.paused = True

    def unpause(self):
        with self._active_cv:
            self.paused = False
            self._active_cv.notify_all()

    def remove(self, key):
        with self._info_lock:
            f_info = self._info[key]
            for child in f_info.children:
                self.remove(child)
            if f_info.in_progress:
                # processing in progress... subthread will remove when it notices
                f_info.status = REMOVED
            else:
                del self._info[key]

    def put_back(self, key, path, parent):
        with self._info_lock:
            if parent:
                if parent not in self._info:
                    return
                self._enqueue(path, PRI_STAT, self._info[parent].depth + 1, parent, key)
            else:
                self._enqueue(path, PRI_STAT, 0, None, key)

    def prioritize(self, key):
        with self._info_lock:
            f_info = self._info[key]
            f_info.high_priority = True
            self._requeue(PRI_IMMEDIATE, f_info.depth, key)

    def loc_info(self, key):
        with self._info_lock:
            f_info = self._info[key]
            return f_info.path, f_info.parent, f_info.status, f_info.data

    def search(self):
        while True:
            # get a job (hippie!)
            priority, depth, key = self._queue.get()
            with self._info_lock:
                f_info = self._info.get(key, None)
                # it's possible this key was queued, then removed.
                if not f_info:
                    continue
                # Check if another thread is handling this key (keys can be
                # double-queued when raising priority), or it's already done.
                if f_info.in_progress or f_info.status == FINISHED:
                    continue
                # clear any error state
                f_info.error = False
                f_info.in_progress = True
                self._make_callback(key)
                path = f_info.path
                status = f_info.status
                num_timeouts = f_info.num_timeouts
                if priority == PRI_IMMEDIATE:
                    self.hipri_count += 1
                else:
                    self.lopri_count += 1
                unreadable = False

            # do the actual work
            try:
                st = time.time()
                if status == STAT:
                    if num_timeouts < self.max_fallbacks:
                        timeout = self.stat_time * (2 ** num_timeouts)
                    else:
                        # XXX - to prevent starvation of the thread pool,
                        # perhaps we should track the number of threads
                        # potentially hung without a timeout, and spawn a new
                        # one to replace it (up to 2x the base number).
                        timeout = None
                    mode = timeout_stat(timeout, path).st_mode
                    if f_info.status == REMOVED:
                        # has this key been removed? - handle below.
                        raise TimeOutException()
                    with self._info_lock:
                        f_info.isdir = stat.S_ISDIR(mode)
                        f_info.num_failures = f_info.num_timeouts = 0
                        f_info.status = LISTING if f_info.isdir else METADATA
                        # adjust priority for requeue below
                        priority = PRI_DIRECTORY if f_info.isdir else PRI_METADATA
                elif status == LISTING:
                    timeout = None
                    if num_timeouts < self.max_fallbacks:
                        timeout = self.listdir_time * (2 ** num_timeouts)
                    # some network filesystems (OSX+smb) drop some directory
                    # entries if the filesystem gets overloaded.  Serializing
                    # calls to listdir seems to prevent this.  We use the
                    # _info_lock to throttle all the other threads, as well.
                    #
                    # Oddly enough, this seems to actually speed up file discovery.
                    with self._info_lock:
                        children = timeout_listdir(timeout, path)
                    if f_info.status == REMOVED:
                        # have we been removed? - handle below.
                        raise TimeOutException()
                    self._check_for_pause()  # next step might be lengthy
                    with self._info_lock:
                        for c in children:
                            # what should child priority be?
                            self._enqueue(os.path.join(path, c), PRI_STAT, depth + 1, key)
                        f_info.data = children
                        f_info.status = FINISHED
                elif status == METADATA:
                    # XXX - should be timed out as well
                    d = self.metadata_function(path)
                    with self._info_lock:
                        f_info.data = d
                        f_info.status = FINISHED
                else:
                    raise ValueError('unknown status %d' % status)
            except TimeOutException, e:
                with self._info_lock:
                    f_info.error = True
                    f_info.data = e
                    f_info.num_timeouts += 1
            except (IOError, OSError), e:
                if e.errno in (errno.EACCES, errno.ENOENT):
                    # XXX - should we requeue these errors with a low priority?
                    unreadable = True
            except Exception, e:
                with self._info_lock:
                    f_info.error = True
                    f_info.data = e
                    f_info.num_failures += 1
                # XXX - if the exception was in the metadata callback, we should probably get the traceback.
            self._check_for_pause()
            with self._info_lock:
                if f_info.status == REMOVED:
                    # entry was removed during processing
                    del self._info[key]
                else:
                    if f_info.error or (f_info.status == FINISHED):
                        self._make_callback(key)
                    if (f_info.error or (f_info.status != FINISHED)) and not unreadable:
                        self._requeue(PRI_IMMEDIATE if f_info.high_priority else priority, depth, key)
                    if unreadable:
                        # give up on these files
                        # XXX - should signal with an UNREADABLE state?
                        f_info.status = FINISHED
                        self._make_callback(key)
                    f_info.in_progress = False
            self._check_for_pause()

    def _check_for_pause(self):
        with self._active_cv:
            while self.paused:
                self._active_cv.wait()

if __name__ == '__main__':
    import sys
    counts = {}
    def cb(path, isdir, key, parent, status, data):
        if status == ERROR:
            if isinstance(data, TimeOutException):
                status = FINISHED + 1
        counts[status] = counts.get(status, 0) + 1
    def metadata_cb(path):
        return None
    loc = Locator(cb, metadata_cb)
    loc.queue(sys.argv[1])
    print '%010s' % 'THREADS',
    for idx, n in enumerate('FOUND, STAT, LISTING, METADATA, REMOVED, ERROR, FINISHED, TIMEOUT'.split(', ')):
        print '%010s' % n,
    print
    st = time.time()
    while True:
        time.sleep(3)
        print '%010s' % threading.active_count(),
        for idx, n in enumerate('FOUND, STAT, LISTING, METADATA, REMOVED, ERROR, FINISHED, TIMEOUT'.split(', ')):
            print '%010s' % counts.get(idx, 0),
        print time.time() - st, "secs", loc.hipri_count
        if counts.get(FOUND, 0) == counts.get(FINISHED, -1):
            break
