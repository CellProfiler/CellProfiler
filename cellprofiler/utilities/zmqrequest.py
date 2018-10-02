from __future__ import print_function
import errno
import logging

logger = logging.getLogger(__name__)
import json
import os
import socket
import sys
import threading
import uuid
import zmq
import queue
import numpy as np
import six
import cellprofiler.grid as cpg

try:
    buffer         # Python 2
except NameError:  # Python 3
    buffer = memoryview

NOTIFY_SOCKET_ADDR = 'inproc://BoundaryNotifications'
SD_KEY_DICT = "__keydict__"


def make_CP_encoder(buffers):
    '''create an encoder for CellProfiler data and numpy arrays (which will be
    stored in the input argument)'''

    def encoder(data, buffers=buffers):
        if isinstance(data, np.ndarray):
            #
            # Maybe it's nice to save memory by converting 64-bit to 32-bit
            # but the purpose here is to fix a bug on the Mac where a
            # 32-bit worker gets a 64-bit array or unsigned 32-bit array,
            # tries to use it for indexing and fails because the integer
            # is wider than a 32-bit pointer
            #
            info32 = np.iinfo(np.int32)
            if data.dtype.kind == "i" and data.dtype.itemsize > 4 or \
                                    data.dtype.kind == "u" and data.dtype.itemsize >= 4:
                if np.prod(data.shape) == 0 or \
                        (np.min(data) >= info32.min and np.max(data) <= info32.max):
                    data = data.astype(np.int32)
            idx = len(buffers)
            buffers.append(np.ascontiguousarray(data))
            dtype = str(data.dtype)
            return {'__ndarray__': True,
                    'dtype': str(data.dtype),
                    'shape': data.shape,
                    'idx': idx}
        if isinstance(data, np.generic):
            # http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
            return data.astype(object)
        if isinstance(data, cpg.Grid):
            d = data.serialize()
            d['__CPGridInfo__'] = True
            return d
        if isinstance(data, buffer):
            # arbitrary data
            idx = len(buffers)
            buffers.append(data)
            return {'__buffer__': True,
                    'idx': idx}
        raise TypeError("%r of type %r is not JSON serializable" % (data, type(data)))

    return encoder


def make_CP_decoder(buffers):
    def decoder(dct, buffers=buffers):
        if '__ndarray__' in dct:
            buf = buffer(buffers[dct['idx']])
            shape = dct['shape']
            dtype = dct['dtype']
            if np.prod(shape) == 0:
                return np.zeros(shape, dtype)
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        if '__buffer__' in dct:
            return buffer(buffers[dct['idx']])
        if '__CPGridInfo__' in dct:
            grid = cpg.Grid()
            grid.deserialize(dct)
            return grid
        return dct

    return decoder


def make_sendable_dictionary(d):
    '''Make a dictionary that passes muster with JSON'''
    result = {}
    fake_key_idx = 1
    for k, v in d.items():
        if (isinstance(k, six.string_types) and k.startswith('_')) or callable(d[k]):
            continue
        if isinstance(v, dict):
            v = make_sendable_dictionary(v)
        elif isinstance(v, (list, tuple)):
            v = make_sendable_sequence(v)
        if not isinstance(k, str):
            if SD_KEY_DICT not in result:
                result[SD_KEY_DICT] = {}
            fake_key = "__%d__" % fake_key_idx
            fake_key_idx += 1
            result[SD_KEY_DICT][fake_key] = k
            result[fake_key] = v
        else:
            result[k] = v
    return result


def make_sendable_sequence(l):
    '''Make a list that passes muster with JSON'''
    result = []
    for v in l:
        if isinstance(v, (list, tuple)):
            result.append(make_sendable_sequence(v))
        elif isinstance(v, dict):
            result.append(make_sendable_dictionary(v))
        else:
            result.append(v)
    return tuple(result)


def decode_sendable_dictionary(d):
    '''Decode the dictionary encoded by make_sendable_dictionary'''
    result = {}
    for k, v in d.items():
        if k == SD_KEY_DICT:
            continue
        if isinstance(v, dict):
            v = decode_sendable_dictionary(v)
        elif isinstance(v, list):
            v = decode_sendable_sequence(v, list)
        if k.startswith("__") and k.endswith("__"):
            k = d[SD_KEY_DICT][k]
            if isinstance(k, list):
                k = decode_sendable_sequence(k, tuple)
        result[k] = v
    return result


def decode_sendable_sequence(l, desired_type):
    '''Decode a tuple encoded by make_sendable_sequence'''
    result = []
    for v in l:
        if isinstance(v, dict):
            result.append(decode_sendable_dictionary(v))
        elif isinstance(v, (list, tuple)):
            result.append(decode_sendable_sequence(v, desired_type))
        else:
            result.append(v)
    return result if isinstance(result, desired_type) else desired_type(result)


def json_encode(o):
    '''Encode an object as a JSON string

    o - object to encode

    returns a 2-tuple of json-encoded object + buffers of binary stuff
    '''
    sendable_dict = make_sendable_dictionary(o)

    # replace each buffer with its metadata, and send it separately
    buffers = []
    encoder = make_CP_encoder(buffers)
    json_str = json.dumps(sendable_dict, default=encoder)
    return json_str, buffers


def json_decode(json_str, buffers):
    '''Decode a JSON-encoded string

    json_str - the JSON string

    buffers - buffers of binary data to feed into the decoder of special cases

    return the decoded dictionary
    '''
    decoder = make_CP_decoder(buffers)
    attribute_dict = json.loads(json_str, object_hook=decoder)
    return decode_sendable_dictionary(attribute_dict)


class Communicable(object):
    '''Base class for Requests and Replies.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    '''

    def send(self, socket, routing=[]):
        if hasattr(self, '_remote'):
            assert not self._remote, "send() called on a non-local Communicable object."
        json_str, buffers = json_encode(self.__dict__)

        socket.send_multipart(routing +
                              [self.__class__.__module__, self.__class__.__name__] +
                              [json_str] +
                              buffers, copy=False)

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
        buffers = message[3:]
        attribute_dict = json_decode(message[2], buffers)
        try:
            instance = sys.modules[module].__dict__[classname](**attribute_dict)
        except:
            print("Communicable could not instantiate %s from module %s with kwargs %s" % (
                module, classname, attribute_dict))
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
            raise NotImplementedError(
                    "Req / rep / rep / rep pattern is no longer supported")


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
        self._boundary = None

    def send(self, socket):
        Communicable.send(self, socket)
        return Communicable.recv(socket)

    def send_only(self, socket):
        '''Send the request but don't perform the .recv

        socket - send on this socket

        First part of a two-part client-side request: send the request
        with an expected .recv, possibly after polling to make the .recv
        non-blocking.
        '''
        Communicable.send(self, socket)

    def set_boundary(self, boundary):
        '''Set the boundary object to use when sending the reply

        boundary - the reply will be enqueued on this boundary's transmit thread
        '''
        self._boundary = boundary

    def reply(self, reply_obj, please_reply=False):
        '''send a reply to a request.  If please_reply is True, wait for and
        return a reply to the reply.  Note that that reply should be treated
        like a Request object, i.e., it should be replied to.'''
        assert isinstance(reply_obj, Reply), "send_reply() called with something other than a Reply object!"
        if self._boundary is None:
            return Communicable.reply(self, reply_obj, please_reply)
        else:
            self._boundary.enqueue_reply(self, reply_obj)


class AnalysisRequest(Request):
    '''A request associated with an analysis

    Every analysis request is made with an analysis ID. The Boundary
    will reply with BoundaryExited if the analysis associated with the
    analysis ID has been cancelled.
    '''

    def __init__(self, analysis_id, **kwargs):
        Request.__init__(self, **kwargs)
        self.analysis_id = analysis_id


class LockStatusRequest(Request):
    '''A request for the status on some locked file

    uid - the unique ID stored inside the file's lock
    '''

    def __init__(self, uid, **kwargs):
        self.uid = uid
        Request.__init__(self, **kwargs)


class Reply(Communicable):
    '''The counterpart to a Request.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    '''

    def __init__(self, **kwargs):
        # all keywords become attributes
        self.__dict__.update(kwargs)


# Two level hierarchy so other classes can inherit from UpstreamExit
class UpstreamExit(Reply):
    pass


class BoundaryExited(UpstreamExit):
    pass


class LockStatusReply(Reply):
    '''A reply to the LockStatusRequest

    self.uid - the unique ID of the locked file
    self.locked - true if locked, false if not
    '''

    def __init__(self, uid, locked, **kwargs):
        Reply.__init__(self, **kwargs)
        self.uid = uid
        self.locked = locked


the_boundary = None


def start_boundary():
    global the_boundary
    if the_boundary is None:
        the_boundary = Boundary("tcp://127.0.0.1")
    return the_boundary


def get_announcer_address():
    return start_boundary().announce_address


def register_analysis(analysis_id, upward_queue):
    '''Register for all analysis request messages with the given ID

    analysis_id - the analysis ID present in every AnalysisRequest

    upward_queue - requests are placed on this queue

    upward_cv - the condition variable used to signal the queue's thread

    returns the boundary singleton.
    '''
    global the_boundary
    start_boundary()
    the_boundary.register_analysis(analysis_id, upward_queue)
    return the_boundary


def cancel_analysis(analysis_id):
    '''Cancel an analysis.

    analysis_id - analysis ID of the analysis to be cancelled

    Calling cancel_analysis guarantees that all AnalysisRequests with the
    given analysis_id without matching replies will receive replies of
    BoundaryExited and that no request will be added to the upward_queue
    after the call returns.
    '''
    global the_boundary
    the_boundary.cancel_analysis(analysis_id)


def join_to_the_boundary():
    '''Send a stop signal to the boundary thread and join to it'''
    global the_boundary
    if the_boundary is not None:
        the_boundary.join()
        the_boundary = None


class AnalysisContext(object):
    '''The analysis context holds the pieces needed to route analysis requests'''

    def __init__(self, analysis_id, upq, lock):
        self.lock = lock
        self.analysis_id = analysis_id
        self.upq = upq
        self.cancelled = False
        # A map of requests pending to the closure that can be used to
        # reply to the request
        self.reqs_pending = set()

    def reply(self, req, rep):
        '''Reply to a AnalysisRequest with this analysis ID

        rep - the intended reply

        Returns True if the intended reply was sent, returns False
        if BoundaryExited was sent instead.

        Always executed on the boundary thread.
        '''
        with self.lock:
            if self.cancelled:
                return False
            if req in self.reqs_pending:
                Communicable.reply(req, rep)
                self.reqs_pending.remove(req)
            return True

    def enqueue(self, req):
        '''Enqueue a request on the upward queue

        req - request to be enqueued. The enqueue should be done before
              req.reply is replaced.

        returns True if the request was enqueued, False if the analysis
        has been cancelled. It is up to the caller to send a BoundaryExited
        reply to the request.

        Always executes on the boundary thread.
        '''
        with self.lock:
            if not self.cancelled:
                assert req not in self.reqs_pending
                self.reqs_pending.add(req)
                self.upq.put(req)
                return True
            else:
                Communicable.reply(req, BoundaryExited())
                return False

    def cancel(self):
        '''Cancel this analysis

        All analysis requests will receive BoundaryExited() after this
        method returns.
        '''
        with self.lock:
            if self.cancelled:
                return
            self.cancelled = True
            self.upq = None

    def handle_cancel(self):
        '''Handle a cancel in the boundary thread.

        Take care of workers expecting replies.
        '''
        with self.lock:
            for req in list(self.reqs_pending):
                Communicable.reply(req, BoundaryExited())
            self.reqs_pending = set()


class Boundary(object):
    '''This object serves as the interface between a ZMQ socket passing
    Requests and Replies, and a thread or threads serving those requests.
    Received requests are received on a ZMQ socket and placed on upward_queue,
    and notify_all() is called on updward_cv.  Replies (via the Request.reply()
    method) are dispatched to their requesters via a downward queue.

    The Boundary wakes up the socket thread via the notify socket. This lets
    the socket thread poll for changes on the notify and request sockets, but
    allows it to receive Python objects via the downward queue.
    '''

    def __init__(self, zmq_address, port=None):
        '''Construction

        zmq_address - the address for announcements and requests
        port - the port for announcements, defaults to random
        '''
        self.analysis_dictionary = {}
        self.analysis_dictionary_lock = threading.RLock()
        #
        # Dictionary of request dictionary to queue for handler
        # (not including AnalysisRequest)
        #
        self.request_dictionary = {}
        self.zmq_context = zmq.Context()
        # The downward queue is used to feed replies to the socket thread
        self.downward_queue = queue.Queue()

        # socket for handling downward notifications
        self.selfnotify_socket = self.zmq_context.socket(zmq.SUB)
        self.selfnotify_socket.bind(NOTIFY_SOCKET_ADDR)
        self.selfnotify_socket.setsockopt(zmq.SUBSCRIBE, '')
        self.threadlocal = threading.local()  # for connecting to notification socket, and receiving replies

        # announce socket
        # zmq.PUB - publish half of publish / subscribe
        # LINGER = 0 to not wait for transmission during shutdown

        self.announce_socket = self.zmq_context.socket(zmq.PUB)
        self.announce_socket.setsockopt(zmq.LINGER, 0)
        if port is None:
            self.announce_port = self.announce_socket.bind_to_random_port(zmq_address)
            self.announce_address = "%s:%d" % (zmq_address, self.announce_port)
        else:
            self.announce_address = "%s:%d" % (zmq_address, port)
            self.announce_port = self.announce_socket.bind(self.announce_address)

        # socket where we receive Requests
        self.request_socket = self.zmq_context.socket(zmq.ROUTER)
        self.request_socket.setsockopt(zmq.LINGER, 0)
        self.request_port = self.request_socket.bind_to_random_port(zmq_address)
        self.request_address = zmq_address + (':%d' % self.request_port)
        #
        # socket for requests outside of the loopback port
        #
        self.external_request_socket = self.zmq_context.socket(zmq.ROUTER)
        self.external_request_socket.setsockopt(zmq.LINGER, 0)
        try:
            fqdn = socket.getfqdn()
            # make sure that this isn't just an entry in /etc/somethingorother
            socket.gethostbyname(fqdn)
        except:
            try:
                fqdn = socket.gethostbyname(socket.gethostname())
            except:
                fqdn = "127.0.0.1"
        self.external_request_port = \
            self.external_request_socket.bind_to_random_port("tcp://*")
        self.external_request_address = "tcp://%s:%d" % (
            fqdn, self.external_request_port)

        self.thread = threading.Thread(
                target=self.spin,
                args=(self.selfnotify_socket, self.request_socket,
                      self.external_request_socket),
                name="Boundary spin()")
        self.thread.start()

    '''Notify the socket thread that an analysis was added'''
    NOTIFY_REGISTER_ANALYSIS = "register analysis"
    '''Notify a request class handler of a request'''
    NOTIFY_REQUEST = "request"
    '''Notify the socket thread that a reply is ready to be sent'''
    NOTIFY_REPLY_READY = "reply ready"
    '''Cancel an analysis. The analysis ID is the second part of the message'''
    NOTIFY_CANCEL_ANALYSIS = "cancel analysis"
    '''Stop the socket thread'''
    NOTIFY_STOP = "stop"

    def register_analysis(self, analysis_id, upward_queue):
        '''Register a queue to receive analysis requests

        analysis_id - the analysis ID embedded in each analysis request

        upward_queue - place the requests on this queue
        '''
        with self.analysis_dictionary_lock:
            self.analysis_dictionary[analysis_id] = AnalysisContext(
                    analysis_id, upward_queue,
                    self.analysis_dictionary_lock)
        response_queue = queue.Queue()
        self.send_to_boundary_thread(self.NOTIFY_REGISTER_ANALYSIS,
                                     (analysis_id, response_queue))
        response_queue.get()

    def register_request_class(self, cls_request, upward_queue):
        '''Register a queue to receive requests of the given class

        cls_request - requests that match isinstance(request, cls_request) will
                      be routed to the upward_queue

        upward_queue - queue that will receive the requests
        '''
        self.request_dictionary[cls_request] = upward_queue

    def enqueue_reply(self, req, rep):
        '''Enqueue a reply to be sent from the boundary thread

        req - original request
        rep - the reply to the request
        '''
        self.send_to_boundary_thread(self.NOTIFY_REPLY_READY, (req, rep))

    def cancel(self, analysis_id):
        '''Cancel an analysis

        All requests with the given analysis ID will get a BoundaryExited
        reply after this call returns.
        '''
        with self.analysis_dictionary_lock:
            if self.analysis_dictionary[analysis_id].cancelled:
                return
            self.analysis_dictionary[analysis_id].cancel()
        response_queue = queue.Queue()
        self.send_to_boundary_thread(self.NOTIFY_CANCEL_ANALYSIS,
                                     (analysis_id, response_queue))
        response_queue.get()

    def handle_cancel(self, analysis_id, response_queue):
        '''Handle cancellation in the boundary thread'''
        with self.analysis_dictionary_lock:
            self.analysis_dictionary[analysis_id].handle_cancel()
        self.announce_analyses()
        response_queue.put("OK")

    def join(self):
        '''Join to the boundary thread.

        Note that this should only be done at a point where no worker truly
        expects a reply to its requests.
        '''
        self.send_to_boundary_thread(self.NOTIFY_STOP, None)
        self.thread.join()

    def spin(self, selfnotify_socket, request_socket, external_request_socket):
        try:
            poller = zmq.Poller()
            poller.register(selfnotify_socket, zmq.POLLIN)
            poller.register(request_socket, zmq.POLLIN)
            poller.register(external_request_socket, zmq.POLLIN)

            received_stop = False

            while not received_stop:
                self.announce_analyses()
                poll_result = poller.poll(1000)
                #
                # Under all circumstances, read everything from the queue
                #
                try:
                    while True:
                        notification, arg = self.downward_queue.get_nowait()
                        if notification == self.NOTIFY_REPLY_READY:
                            req, rep = arg
                            self.handle_reply(req, rep)
                        elif notification == self.NOTIFY_CANCEL_ANALYSIS:
                            analysis_id, response_queue = arg
                            self.handle_cancel(analysis_id, response_queue)
                        elif notification == self.NOTIFY_REGISTER_ANALYSIS:
                            analysis_id, response_queue = arg
                            self.handle_register_analysis(
                                    analysis_id, response_queue)
                        elif notification == self.NOTIFY_STOP:
                            received_stop = True
                except queue.Empty:
                    pass
                #
                # Then process the poll result
                #
                for s, state in poll_result:
                    if s == selfnotify_socket and state == zmq.POLLIN:
                        # Discard the actual contents
                        _ = selfnotify_socket.recv()
                    if (s not in (request_socket, external_request_socket) or
                                state != zmq.POLLIN):
                        continue
                    req = Communicable.recv(s, routed=True)
                    req.set_boundary(self)
                    if not isinstance(req, AnalysisRequest):
                        for request_class in self.request_dictionary:
                            if isinstance(req, request_class):
                                q = self.request_dictionary[request_class]
                                q.put([self, self.NOTIFY_REQUEST, req])
                                break
                        else:
                            logger.warn(
                                    "Received a request that wasn't an AnalysisRequest: %s" %
                                    str(type(req)))
                            req.reply(BoundaryExited())
                        continue
                    if s != request_socket:
                        # Request is on the external socket
                        logger.warn("Received a request on the external socket")
                        req.reply(BoundaryExited())
                        continue
                    #
                    # Filter out requests for cancelled analyses.
                    #
                    with self.analysis_dictionary_lock:
                        analysis_context = self.analysis_dictionary[req.analysis_id]
                        if not analysis_context.enqueue(req):
                            continue

            #
            # We assume here that workers trying to communicate with us will
            # be shut down abruptly without needing replies to pending requests.
            # There's not much we can do in terms of handling that in a more
            # orderly fashion since workers might be formulating requests as or
            # after we have shut down. But calling cancel on all the analysis
            # contexts will raise exceptions in any thread waiting for a rep/rep.
            #
            # You could call analysis_context.handle_cancel() here, what if it
            # blocks?
            self.announce_socket.close()
            with self.analysis_dictionary_lock:
                for analysis_context in self.analysis_dictionary.values():
                    analysis_context.cancel()
                for request_class_queue in self.request_dictionary.values():
                    #
                    # Tell each response class to stop. Wait for a reply
                    # which may be a thread instance. If so, join to the
                    # thread so there will be an orderly shutdown.
                    #
                    response_queue = queue.Queue()
                    request_class_queue.put(
                            [self, self.NOTIFY_STOP, response_queue])
                    thread = response_queue.get()
                    if isinstance(thread, threading.Thread):
                        thread.join()

            self.request_socket.close()
            logger.info("Exiting the boundary thread")
        except:
            #
            # Pretty bad - a logic error or something extremely unexpected
            #              We're close to hosed here, best to die an ugly death.
            #
            logger.critical("Unhandled exception in boundary thread.",
                            exc_info=10)
            import os
            os._exit(-1)

    def send_to_boundary_thread(self, msg, arg):
        '''Send a message to the boundary thread via the notify socket

        Send a wakeup call to the boundary thread by sending arbitrary
        data to the notify socket, placing the real objects of interest
        on the downward queue.

        msg - message placed in the downward queue indicating the purpose
              of the wakeup call

        args - supplementary arguments passed to the boundary thread via
               the downward queue.
        '''
        if not hasattr(self.threadlocal, 'notify_socket'):
            self.threadlocal.notify_socket = self.zmq_context.socket(zmq.PUB)
            self.threadlocal.notify_socket.setsockopt(zmq.LINGER, 0)
            self.threadlocal.notify_socket.connect(NOTIFY_SOCKET_ADDR)
        self.downward_queue.put((msg, arg))
        self.threadlocal.notify_socket.send('WAKE UP!')

    def announce_analyses(self):
        with self.analysis_dictionary_lock:
            valid_analysis_ids = [
                analysis_id for analysis_id in self.analysis_dictionary.keys()
                if not self.analysis_dictionary[analysis_id].cancelled]
        self.announce_socket.send_json([
                                           (analysis_id, self.request_address)
                                           for analysis_id in valid_analysis_ids])

    def handle_reply(self, req, rep):
        if not isinstance(req, AnalysisRequest):
            assert isinstance(req, Request)
            Communicable.reply(req, rep)
            return

        with self.analysis_dictionary_lock:
            analysis_context = self.analysis_dictionary.get(req.analysis_id)
            analysis_context.reply(req, rep)

    def handle_register_analysis(self, analysis_id, response_queue):
        '''Handle a request to register an analysis

        analysis_id - analysis_id of new analysis
        response_queue - response queue. Any announce subscriber that registers
                         after the response is placed in this queue
                         will receive an announcement of the analysis.
        '''
        self.announce_analyses()
        response_queue.put("OK")


__lock_queue = queue.Queue()
__lock_thread = None

LOCK_REQUEST = "Lock request"
UNLOCK_REQUEST = "Unlock request"
UNLOCK_OK = "OK"


def start_lock_thread():
    '''Start the thread that handles file locking'''
    global __lock_thread
    if __lock_thread is not None:
        return
    the_boundary.register_request_class(LockStatusRequest, __lock_queue)

    def lock_thread_fn():
        global __lock_thread
        locked_uids = {}
        locked_files = {}
        while True:
            msg = __lock_queue.get()
            boundary = msg[0]
            if msg[1] == Boundary.NOTIFY_STOP:
                msg[2].put(__lock_thread)
                break
            elif msg[1] == Boundary.NOTIFY_REQUEST:
                request = msg[2]
                assert isinstance(request, LockStatusRequest)
                assert isinstance(boundary, Boundary)
                logger.info("Received lock status request for %s" % request.uid)
                reply = LockStatusReply(request.uid,
                                        request.uid in locked_uids)
                if reply.locked:
                    logger.info("Denied lock request for %s" % locked_uids[request.uid])
                boundary.enqueue_reply(request, reply)
            elif msg[1] == LOCK_REQUEST:
                uid, path = msg[2]
                locked_uids[uid] = path
                locked_files[path] = uid
                msg[3].put("OK")
            elif msg[1] == UNLOCK_REQUEST:
                try:
                    uid = locked_files[msg[2]]
                    del locked_uids[uid]
                    del locked_files[msg[2]]
                    msg[3].put("OK")
                except Exception as e:
                    msg[3].put(e)
        __lock_thread = None
        logger.info("Exiting the lock thread")

    __lock_thread = threading.Thread(target=lock_thread_fn)
    __lock_thread.setName("FileLockThread")
    __lock_thread.start()


def get_lock_path(path):
    '''Return the path to the lockfile'''
    pathpart, filepart = os.path.split(path)
    return os.path.join(pathpart, u"." + filepart + u".lock")


def lock_file(path, timeout=3):
    '''Lock a file

    path - path to the file

    timeout - timeout in seconds when waiting for announcement

    returns True if we obtained the lock, False if the file is already owned.
    '''
    lock_path = get_lock_path(path)
    start_boundary()
    uid = uuid.uuid4().hex
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        with os.fdopen(fd, "a") as f:
            f.write(the_boundary.external_request_address + "\n" + uid)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        logger.info("Lockfile for %s already exists - contacting owner" % path)
        with open(lock_path, "r") as f:
            remote_address = f.readline().strip()
            remote_uid = f.readline().strip()
        if len(remote_address) > 0 and len(remote_uid) > 0:
            logger.info("Owner is %s" % remote_address)
            request_socket = the_boundary.zmq_context.socket(zmq.REQ)
            request_socket.setsockopt(zmq.LINGER, 0)
            assert isinstance(request_socket, zmq.Socket)
            request_socket.connect(remote_address)

            lock_request = LockStatusRequest(remote_uid)
            lock_request.send_only(request_socket)
            poller = zmq.Poller()
            poller.register(request_socket, zmq.POLLIN)
            keep_polling = True
            while keep_polling:
                keep_polling = False
                for socket, status in poller.poll(timeout * 1000):
                    keep_polling = True
                    if socket == request_socket and status == zmq.POLLIN:
                        lock_response = lock_request.recv(socket)
                        if isinstance(lock_response, LockStatusReply):
                            if lock_response.locked:
                                logger.info("%s is locked" % path)
                                return False
                            keep_polling = False
        #
        # Fall through if we believe that the other end is dead
        #
        with open(lock_path, "w") as f:
            f.write(the_boundary.request_address + "\n" + uid)
    #
    # The coast is clear to lock
    #
    q = queue.Queue()
    start_lock_thread()
    __lock_queue.put((None, LOCK_REQUEST, (uid, path), q))
    q.get()
    return True


def unlock_file(path):
    '''Unlock the file at the given path'''
    if the_boundary is None:
        return
    q = queue.Queue()
    start_lock_thread()
    __lock_queue.put((None, UNLOCK_REQUEST, path, q))
    result = q.get()
    if result != UNLOCK_OK:
        raise result
    lock_path = get_lock_path(path)
    os.remove(lock_path)


if __name__ == '__main__':
    context = zmq.Context()


    def subproc():
        address = sys.argv[sys.argv.index('subproc') + 1]
        mysock = context.socket(zmq.REQ)
        mysock.connect(address)
        req = Request(this='is', a='test', b=5, c=1.3, d=np.arange(10), e=[{'q': np.arange(5)}])
        rep = req.send(mysock)
        print("subproc received", rep, rep.__dict__)
        rep = rep.reply(Reply(msg='FOO'), please_reply=True)
        print("subproc received", rep, rep.__dict__)


    if 'subproc' in sys.argv[1:]:
        subproc()
    else:
        import subprocess

        upq = queue.Queue()
        cv = threading.Condition()
        boundary = Boundary('tcp://127.0.0.1', upq, cv)
        s = subprocess.Popen(['python', sys.argv[0], 'subproc', boundary.request_address])
        boundary = Boundary('tcp://127.0.0.1', upq, cv)

        with cv:
            while upq.empty():
                cv.wait()
            req = upq.get()
            print("mainproc received", req, req.__dict__)
            rep = Reply(this='is', your='reply')
            rep2 = req.reply(rep, please_reply=True)
            print("mainproc received", rep2, rep2.__dict__)
            rep2.reply(Reply(message='done'))

        s.wait()
