import sys
import threading
import zmq
import Queue


class Communicable(object):
    '''Base class for Requests and Replies.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    '''
    # XXX - include type information
    def send(self, socket, routing=[]):
        if hasattr(self, '_remote'):
            assert not self._remote, "send() called on a non-local Communicable object."
        sendable_keys = [k for k in self.__dict__
                         if (not k.startswith('_'))
                         and (not callable(self.__dict__[k]))]
        interleaved = [x for key in sendable_keys for x in (key, self.__dict__[key])]
        socket.send_multipart(routing +
                              [self.__class__.__module__, self.__class__.__name__] +
                              interleaved)

    class MultipleReply(RuntimeError):
        pass

    @classmethod
    def recv(cls, socket, routed=False):
        message = socket.recv_multipart()
        if routed:
            split = message.index('') + 1
            routing = message[:split]
            message = message[split:]
        else:
            routing = []
        module, classname = message[:2]
        attribute_dict = dict(zip(message[2::2], message[3::2]))
        instance = sys.modules[module].__dict__[classname](**attribute_dict)
        instance._remote = True
        instance._routing = routing
        instance._socket = socket
        instance._replied = False
        return instance

    def reply(self, reply_obj):
        assert self._remote, "Replying to a local Communicable!"
        if self._replied:
            raise MultipleReply("Can't reply to a Communicable more than once!")
        Communicable.send(reply_obj, self._socket, self._routing)
        self._replied = True


class Request(Communicable):
    '''A class for making requests and receiving replies across a zmq REQ
    socket.  When communicated through a socket, the class will self-serialize
    any non-callable attributes that do not start with an underscore.

    When received, reply() can be used to send a Reply object to the original
    Request.

    Note that both sides of the connection must have imported the same module
    under the same name defining any objects that will be communicated.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    '''
    def __init__(self, **kwargs):
        # all keywords become attributes
        self.__dict__.update(kwargs)

    def send(self, socket):
        Communicable.send(self, socket)
        return Communicable.recv(socket)

    def reply(self, reply_obj):
        assert isinstance(reply_obj, Reply), "send_reply() called with something other than a Reply object!"
        Communicable.reply(self, reply_obj)


class Reply(Communicable):
    '''The counterpart to a Request.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    '''
    def __init__(self, **kwargs):
        # all keywords become attributes
        self.__dict__.update(kwargs)


class BoundaryExited(Reply):
    pass


class Boundary(object):
    '''This object serves as the interface between a ZMQ socket passing
    Requests and Replies, and a thread or threads serving those requests.
    Received requests are received on a ZMQ socket and placed on upward_queue,
    and notify_all() is called on updward_cv.  Replies (via the Request.reply()
    method) are dispatched to their requesters via a downward queue.

    The address of the request
    '''
    def __init__(self, zmq_address, upward_queue, upward_cv, random_port=True):
        self.zmq_context = zmq.Context()
        self.upq = upward_queue
        self.upcv = upward_cv
        self.downward_queue = Queue.Queue()
        self.reqs_pending = set()
        self._stop = False

        # socket for handling downward notifications
        self.selfnotify_socket = self.zmq_context.socket(zmq.SUB)
        self.selfnotify_socket.bind('inproc://BoundaryNotifications')
        self.selfnotify_socket.setsockopt(zmq.SUBSCRIBE, '')
        self.notifier = threading.local()  # for connecting to notification socket

        # socket where we receive Requests
        self.request_socket = self.zmq_context.socket(zmq.ROUTER)
        if random_port:
            self.request_port = self.request_socket.bind_to_random_port(zmq_address)
            self.request_address = zmq_address + (':%d' % self.request_port)
        else:
            self.request_port = self.request_socket.bind(zmq_address)
            self.request_address = zmq_address

        self.thread = threading.Thread(target=self.spin,
                                       args=(self.selfnotify_socket, self.request_socket),
                                       name="Boundary spin()")
        self.thread.daemon = True
        self.thread.start()

    def spin(self, selfnotify_socket, request_socket):
        poller = zmq.Poller()
        poller.register(selfnotify_socket, zmq.POLLIN)
        poller.register(request_socket, zmq.POLLIN)
        while (not self._stop) or (not self.downward_queue.empty()):
            socks = dict(poller.poll(1000))  # milliseconds
            if socks.get(selfnotify_socket, None) == zmq.POLLIN:
                selfnotify_socket.recv()  # drop notification
                req, orig_reply, rep = self.downward_queue.get()
                orig_reply(rep)
                self.reqs_pending.remove(req)
            if socks.get(request_socket, None) == zmq.POLLIN:
                req = Communicable.recv(request_socket, routed=True)

                # ZMQ requires that replies sent out on the socket be from the
                # thread that received them.  Further, REQ sockets must receive
                # a reply before sending another request, so we ensure that any
                # not-yet-replied-to reqs get a reply when we exit to keep from
                # freezing the other process.
                def wrap_reply(req, orig_reply):
                    def reply_in_boundary_thread(rep):
                        self.downward_queue.put((req, orig_reply, rep))
                        # signal boundary thread to process reply
                        if not hasattr(self.notifier, 'notify_socket'):
                            self.notifier.notify_socket = self.zmq_context.socket(zmq.PUB)
                            self.notifier.notify_socket.connect('inproc://BoundaryNotifications')
                        self.notifier.notify_socket.send("reply ready")
                    return reply_in_boundary_thread

                req.reply = wrap_reply(req, req.reply)
                self.reqs_pending.add(req)

                # XXX - put req in set of reqs pending replies, remove when
                # replied, clean up last ones on spin() exit
                self.upq.put(req)
                with self.upcv:
                    self.upcv.notify_all()

        # ensure every pending req gets a reply
        for req in self.reqs_pending:
            req.reply(BoundaryExited())
        while not self.downward_queue.empty():
            req, orig_reply, rep = self.downward_queue.get()
            try:
                orig_reply(rep)
            except Communicable.MultipleReply:
                pass

        self.request_socket.close()  # will linger until messages are delivered

    def stop(self):
        self._stop = True


if __name__ == '__main__':
    context = zmq.Context()

    def subproc():
        address = sys.argv[sys.argv.index('subproc') + 1]
        mysock = context.socket(zmq.REQ)
        mysock.connect(address)
        req = Request(this='is', a='test')
        rep = req.send(mysock)
        print "subproc received", rep, rep.__dict__

    if 'subproc' in sys.argv[1:]:
        subproc()
    else:
        import subprocess
        upq = Queue.Queue()
        cv = threading.Condition()
        boundary = Boundary('tcp://127.0.0.1', upq, cv)
        s = subprocess.Popen(['python', sys.argv[0], 'subproc', boundary.request_address])
        boundary = Boundary('tcp://127.0.0.1', upq, cv)

        with cv:
            while upq.empty():
                cv.wait()
            req = upq.get()
            print "mainproc received", req, req.__dict__
            rep = Reply(this='is', your='reply')
            req.reply(rep)
        s.wait()
