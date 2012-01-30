import sys
import threading
import zmq
import Queue


class Communicable(object):
    '''Base class for Requests and Replies.

    All subclasses must accept zero arguments to __init__().
    '''
    def send(self, socket, routing=[]):
        if hasattr(self, '_remote'):
            assert not self._remote, "send() called on a non-local Communicable object."
        sendable_keys = [k for k in self.__dict__
                         if (not k.startswith('_'))
                         and (not callable(self.__dict__[k]))]
        interleaved = [x for key in sendable_keys for x in (key, self.__dict__[key])]
        socket.send_multipart(routing + [self.__class__.__module__, self.__class__.__name__] + interleaved)

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
        instance = sys.modules[module].__dict__[classname]()
        instance.__dict__.update(dict(zip(message[2::2], message[3::2])))
        instance._remote = True
        instance._routing = routing
        instance._socket = socket
        instance._replied = False
        return instance

    def reply(self, reply_obj):
        assert self._remote, "Replying to a local Communicable!"
        assert not self._replied, "Can't reply to a Communicable more than once!"
        if self._thread_cb is not None:  # ZMQ requires same-thread communication
            cb = self._thread_cb
            self._thread_cb = None
            cb(self.reply, reply_obj)
        else:
            Communicable.send(reply_obj, self._socket, self._routing)
            self._replied = True


class Request(Communicable):
    '''A class for making requests and receiving synchronous replies across a
    zmq socket.  When communicated through a socket, the class will
    self-serialize any non-callable attributes that do not start with an
    underscore.

    When received, reply() can be used to send a Reply object to the original
    Request.

    Note that both sides of the connection must have imported the same module
    under the same name defining any objects that will be communicated.

    All subclasses must accept zero arguments to __init__().
    '''
    def __init__(self, **kwargs):
        # all keywords become attributes
        self.__dict__.update(kwargs)

    def send(self, socket):
        Communicable.send(self, socket)
        return Communicable.recv(socket)

    def reply(self, reply_obj):
        assert isinstance(reply_obj, Reply), "send_reply() called with something other than a Reply object!"
        assert not self._replied, "Can't reply to a Request more than once!"
        Communicable.reply(self, reply_obj)


class Reply(Communicable):
    '''The counterpart to a Request.

    All subclasses must accept zero arguments to __init__().
    '''
    def __init__(self, **kwargs):
        # all keywords become attributes
        self.__dict__.update(kwargs)


class Boundary(object):
    '''This object serves as the interface between a ZMQ socket passing
    Requests and Replies, and a thread or threads serving those requests.
    Received requests are received on a ZMQ socket and placed on upward_queue,
    and notify_all() is called on updward_cv.  Replies (via the Request.reply()
    method) are dispatched to their requesters via a downward queue.

    The address of the request
    '''
    def __init__(self, zmq_address, upward_queue, upward_cv):
        self.zmq_context = zmq.Context()
        self.upq = upward_queue
        self.upcv = upward_cv
        self.downward_queue = Queue.Queue()

        # socket for handling downward notifications
        self.selfnotify_socket = self.zmq_context.socket(zmq.SUB)
        self.selfnotify_socket.bind('inproc://BoundaryNotifications')
        self.selfnotify_socket.setsockopt(zmq.SUBSCRIBE, '')
        self.notifier = threading.local()  # for connecting to notification socket

        # socket where we receive Requests
        self.request_socket = self.zmq_context.socket(zmq.ROUTER)
        self.request_port = self.request_socket.bind_to_random_port(zmq_address)
        self.request_address = zmq_address + (':%d' % self.request_port)

        self.thread = threading.Thread(target=self.spin,
                                       args=(self.selfnotify_socket, self.request_socket),
                                       name="Boundary spin()")
        self.thread.daemon = True
        self.thread.start()

    def spin(self, selfnotify_socket, request_socket):
        poller = zmq.Poller()
        poller.register(selfnotify_socket, zmq.POLLIN)
        poller.register(request_socket, zmq.POLLIN)
        while True:
            socks = dict(poller.poll())
            if socks.get(selfnotify_socket, None) == zmq.POLLIN:
                selfnotify_socket.recv()  # drop notification
                reply_args = self.downward_queue.get()
                reply_args[0](*reply_args[1:])
            if socks.get(request_socket, None) == zmq.POLLIN:
                req = Communicable.recv(request_socket, routed=True)
                # ZMQ requires that replies sent out on the socket be from the
                # thread that received them.
                req._thread_cb = self.reply_wrapper
                self.upq.put(req)
                with self.upcv:
                    self.upcv.notify_all()

    def reply_wrapper(self, *args):
        self.downward_queue.put(args)
        if not hasattr(self.notifier, 'notify_socket'):
            self.notifier.notify_socket = self.zmq_context.socket(zmq.PUB)
            self.notifier.notify_socket.connect('inproc://BoundaryNotifications')
        self.notifier.notify_socket.send("reply ready")

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
