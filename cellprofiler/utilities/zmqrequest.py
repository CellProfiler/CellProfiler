import sys
import threading
import zmq
import Queue
import numpy as np

class Communicable(object):
    '''Base class for Requests and Replies.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    '''
    def send(self, socket, routing=[]):
        if hasattr(self, '_remote'):
            assert not self._remote, "send() called on a non-local Communicable object."
        sendable_dict = dict((k, v) for k,v in self.__dict__.items()
                             if (not k.startswith('_'))
                             and (not callable(self.__dict__[k])))

        # replace each numpy array with its metadata, and send it separately
        numpy_arrays = []
        def numpy_encoder(data):
            if isinstance(data, np.ndarray):
                idx = len(numpy_arrays)
                numpy_arrays.append(np.ascontiguousarray(data))
                return {'__ndarray__' : True,
                        'dtype' : str(data.dtype),
                        'shape' : data.shape,
                        'idx' : idx}
            if isinstance(data, np.generic):
                # http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
                return data.astype(object)
            raise TypeError("%r of type %r is not JSON serializable" % (data, type(data)))

        json_str = zmq.utils.jsonapi.dumps(sendable_dict, default=numpy_encoder)
        socket.send_multipart(routing +
                              [self.__class__.__module__, self.__class__.__name__] +
                              [json_str] +
                              numpy_arrays, copy=False)

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
        numpy_arrays = message[3:]
        def numpy_decoder(dct):
            if '__ndarray__' in dct:
                buf = buffer(numpy_arrays[dct['idx']])
                return np.frombuffer(buf, dtype=dct['dtype']).reshape(dct['shape'])
            return dct
        attribute_dict = zmq.utils.jsonapi.loads(message[2], object_hook=numpy_decoder)
        try:
            instance = sys.modules[module].__dict__[classname](**attribute_dict)
        except:
            print "Communicable could not instantiate %s from module %s with kwargs %s" % (module, classname, attribute_dict)
            raise
        instance._remote = True
        instance._routing = routing
        instance._socket = socket
        instance._replied = False
        return instance

    def routing(self):
        return "/".join(self._routing)

    def reply(self, reply_obj, please_reply=False):
        assert self._remote, "Replying to a local Communicable!"
        if self._replied:
            raise self.MultipleReply("Can't reply to a Communicable more than once!")
        Communicable.send(reply_obj, self._socket, self._routing)
        self._replied = True
        if please_reply:
            return reply_obj.recv(self._socket)


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

    def reply(self, reply_obj, please_reply=False):
        '''send a reply to a request.  If please_reply is True, wait for and
        return a reply to the reply.  Note that that reply should be treated
        like a Request object, i.e., it should be replied to.'''
        assert isinstance(reply_obj, Reply), "send_reply() called with something other than a Reply object!"
        return Communicable.reply(self, reply_obj, please_reply)


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
        self.threadlocal = threading.local()  # for connecting to notification socket, and receiving replies

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

        # Dict of routing info for replies using please_reply=True, with values
        # of the Queue to put the result on.
        waiting_for_reply = {}

        while (not self._stop) or (not self.downward_queue.empty()):
            socks = dict(poller.poll(1000))  # milliseconds
            if socks.get(selfnotify_socket, None) == zmq.POLLIN:
                selfnotify_socket.recv()  # drop notification
                req, orig_reply, rep, please_reply, please_reply_queue = self.downward_queue.get()
                if please_reply:
                    waiting_for_reply[req.routing()] = please_reply_queue
                orig_reply(rep)
                self.reqs_pending.remove(req)
            if socks.get(request_socket, None) == zmq.POLLIN:
                req = Communicable.recv(request_socket, routed=True)

                # ZMQ requires that replies sent out on the socket be from the
                # thread that received them.
                def wrap_reply(req, orig_reply):
                    def reply_in_boundary_thread(rep, please_reply=False):
                        # connect to the notify socket using thread-local data.
                        if not hasattr(self.threadlocal, 'notify_socket'):
                            self.threadlocal.notify_socket = self.zmq_context.socket(zmq.PUB)
                            self.threadlocal.notify_socket.connect('inproc://BoundaryNotifications')
                            self.threadlocal.please_reply_queue = Queue.Queue()
                        self.downward_queue.put((req, orig_reply, rep,
                                                 please_reply, self.threadlocal.please_reply_queue))
                        # signal boundary thread to process reply
                        self.threadlocal.notify_socket.send("reply ready")
                        if please_reply:
                            return self.threadlocal.please_reply_queue.get()
                    return reply_in_boundary_thread

                req.reply = wrap_reply(req, req.reply)

                # Further, REQ sockets must receive a reply before sending
                # another request, so we ensure that any not-yet-replied-to
                # reqs get a reply when we exit to keep from freezing the other
                # process
                self.reqs_pending.add(req)

                # If this req is actually a Reply in response to a Reply with
                # please_reply=True, put it in the relevant queue.
                if req.routing() in waiting_for_reply:
                    waiting_for_reply[req.routing()].put(req)
                    del waiting_for_reply[req.routing()]
                else:
                    # otherwise, put it in the default queue
                    self.upq.put(req)
                    with self.upcv:
                        self.upcv.notify_all()

        # ensure every pending req gets a reply
        for req in self.reqs_pending:
            req.reply(BoundaryExited())
        while not self.downward_queue.empty():
            req, orig_reply, rep, please_reply, please_reply_queue = self.downward_queue.get()
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
        req = Request(this='is', a='test', b=5, c=1.3, d=np.arange(10), e=[{'q' : np.arange(5)}])
        rep = req.send(mysock)
        print "subproc received", rep, rep.__dict__
        rep = rep.reply(Reply(msg='FOO'), please_reply=True)
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
            rep2 = req.reply(rep, please_reply=True)
            print "mainproc received", rep2, rep2.__dict__
            rep2.reply(Reply(message='done'))

        s.wait()
