'''Remote debugger class, useful for multiprocess and distributed debugging.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

# modified version of http://snippets.dzone.com/posts/show/7248

import pdb
import socket
import sys
import hashlib
import readline  # otherwise, pdb.Pdb.__init__ hangs


class Rpdb(pdb.Pdb):
    '''A remote python debugger.

    Create with Rpdb(port, verification_hash), then call verify() to complete
    creation, and post_mortem(traceback=None) to debug a particular traceback.

    port - the port number to bind to, or 0 (the default) to choose a random
           port.  The bound port is available after initialization in the .port
           attribute.

    verification_hash - the SHA1 hash hexdigest of a string to be sent as the
           first message to the port, or None for no verification.  Default =
           None.
    '''
    def __init__(self, port=0, verification_hash=None, port_callback=None):
        self.old_stds = (sys.stdin, sys.stdout)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', port))
        self.socket.listen(1)
        self.port = self.socket.getsockname()[1]
        self.verification_hash = verification_hash
        if port_callback is not None:
            port_callback(self.port)
        self.verified = False
        (self.clientsocket, self.address) = self.socket.accept()
        self.handle = self.clientsocket.makefile('rw')
        pdb.Pdb.__init__(self, completekey='tab', stdin=self.handle, stdout=self.handle)
        sys.stdout = sys.stdin = self.handle

    def verify(self):
        if self.verification_hash is not None:
            self.handle.write('Verification: ')
            self.handle.flush()
            verification = self.handle.readline().rstrip()
            if hashlib.sha1(verification).hexdigest() != self.verification_hash:
                sys.stderr.write('Verification hash from %s does not match in Rpdb.verify()!  Closing.\n' % str(self.address))
                sys.stderr.write('Got: %s, expected: %s\n' % (hashlib.sha1(verification).hexdigest(), self.verification_hash))
                sys.stdin, sys.stdout = self.old_stds
                self.clientsocket.close()
                self.socket.close()
                self.handle.close()
                return
        self.verified = True

    def do_continue(self, arg):
        sys.stdin, sys.stdout = self.old_stds
        self.socket.close()
        self.set_continue()
        return 1

    do_c = do_cont = do_continue

    def post_mortem(self, t=None):
        if not self.verified:
            return
        # handling the default
        if t is None:
            # sys.exc_info() returns (type, value, traceback) if an exception is
            # being handled, otherwise it returns None
            t = sys.exc_info()[2]
        if t is None:
            raise ValueError("A valid traceback must be passed if no "
                             "exception is being handled.")
        self.reset()
        self.interaction(None, t)
        sys.stdin, sys.stdout = self.old_stds
        self.socket.close()


if __name__ == '__main__':
    try:
        a = []
        a[0] = 1
    except:
        def pc(x):
            print x
        rpdb = Rpdb(verification_hash=hashlib.sha1("testing").hexdigest(), port_callback=pc)
        print "debugger listing on port", rpdb.port
        rpdb.verify()
        rpdb.post_mortem()
