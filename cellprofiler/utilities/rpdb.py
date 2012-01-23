'''Remote debugger class, useful for multiprocess and distributed debugging.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

# modified version of http://snippets.dzone.com/posts/show/7248

import pdb
import socket
import sys


class Rdb(pdb.Pdb):
    # XXX - add documentation (basically, telnet to the port)
    # XXX - we should add some sort of verification for security.
    def __init__(self, port=4444):
        self.old_stds = (sys.stdin, sys.stdout)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', port))
        self.socket.listen(1)
        (clientsocket, address) = self.socket.accept()
        handle = clientsocket.makefile('rw')
        pdb.Pdb.__init__(self, completekey='tab', stdin=handle, stdout=handle)
        sys.stdout = sys.stdin = handle

    def do_continue(self, arg):
        sys.stdin, sys.stdout = self.old_stds
        self.socket.close()
        self.set_continue()
        return 1

    do_c = do_cont = do_continue


def post_mortem(t=None):
    # handling the default
    if t is None:
        # sys.exc_info() returns (type, value, traceback) if an exception is
        # being handled, otherwise it returns None
        t = sys.exc_info()[2]
        if t is None:
            raise ValueError("A valid traceback must be passed if no "
                             "exception is being handled")
        p = Rdb()
        p.reset()
        p.interaction(None, t)
