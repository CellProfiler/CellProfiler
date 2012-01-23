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
import hashlib


class Rdb(pdb.Pdb):
    '''A remote python debugger.

    Create with Rdb(port, verification_hash, port_callback).  

    port - the port number to bind to, or 0 to choose a random port.  The bound
           port is available as the .port attribute afterward.  Default = 4444.

    verification_hash - the SHA1 hash hexdigest of a string to be sent as the
           first message to the port, or None for no verification.  Default = None.

    port_callback - this function (if not None) will be called with the port
           number once the socket is bound.  Default = None.
    '''
    # XXXX - The verification_hash and port_callback mechanisms are pretty
    # ugly.  It would be better to have the remote debugger connect to a known
    # port on the machine where debugging is happening.  We already require
    # that it be able to communicate with that machine to receive work, anyway.
    # (Just make it use ZMQ on a known port).
    def __init__(self, port=4444, verification_hash=None, port_callback=None):
        self.old_stds = (sys.stdin, sys.stdout)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', port))
        if port_callback is not None:
            port_callback(self.socket.getsockname()[1])
        self.socket.listen(1)
        (clientsocket, address) = self.socket.accept()
        handle = clientsocket.makefile('rw')
        self.verified = True
        if verification_hash is not None:
            verification_string = handle.readline()
            if hashlib.sha1(verification_string.rstrip()).hexdigest() != verification_hash:
                sys.stderr.write("RDB verification failed!  Closing.\n")
                handle.close()
                clientsocket.close()
                self.verified = False
        pdb.Pdb.__init__(self, completekey='tab', stdin=handle, stdout=handle)
        if self.verified:
            sys.stdout = sys.stdin = handle

    def do_continue(self, arg):
        sys.stdin, sys.stdout = self.old_stds
        self.socket.close()
        self.set_continue()
        return 1

    do_c = do_cont = do_continue


def post_mortem(t=None, port=0, verification_hash=None, port_callback=None):
    # handling the default
    if t is None:
        # sys.exc_info() returns (type, value, traceback) if an exception is
        # being handled, otherwise it returns None
        t = sys.exc_info()[2]
    if t is None:
        raise ValueError("A valid traceback must be passed if no "
                         "exception is being handled.")
    p = Rdb(port=port, verification_hash=verification_hash, port_callback=port_callback)
    if not p.verified:
        return
    p.reset()
    p.interaction(None, t)
