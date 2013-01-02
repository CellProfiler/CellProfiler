"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2013 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import logging
logger = logging.getLogger(__name__)
import json
import sys
import threading
import zmq
import Queue
import numpy as np
import cellprofiler.cpgridinfo as cpg

NOTIFY_SOCKET_ADDR = 'inproc://BoundaryNotifications'
def make_CP_encoder(buffers):
    '''create an encoder for CellProfiler data and numpy arrays (which will be
    stored in the input argument)'''
    def encoder(data, buffers=buffers):
        if isinstance(data, np.ndarray):
            idx = len(buffers)
            buffers.append(np.ascontiguousarray(data))
            return {'__ndarray__': True,
                    'dtype': str(data.dtype),
                    'shape': data.shape,
                    'idx': idx}
        if isinstance(data, np.generic):
            # http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
            return data.astype(object)
        if isinstance(data, cpg.CPGridInfo):
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
            return np.frombuffer(buf, dtype=dct['dtype']).reshape(dct['shape']).copy()
        if '__buffer__' in dct:
            return buffer(buffers[dct['idx']])
        if '__CPGridInfo__' in dct:
            grid = cpg.CPGridInfo()
            grid.deserialize(dct)
            return grid
        return dct
    return decoder

class Communicable(object):
    '''Base class for Requests and Replies.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    '''
    def send(self, socket, routing=[]):
        if hasattr(self, '_remote'):
            assert not self._remote, "send() called on a non-local Communicable object."
        sendable_dict = dict((k, v) for k, v in self.__dict__.items()
                             if (not k.startswith('_'))
                             and (not callable(self.__dict__[k])))

        # replace each buffer with its metadata, and send it separately
        buffers = []
        encoder = make_CP_encoder(buffers)
        json_str = json.dumps(sendable_dict, default=encoder)
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
        decoder = make_CP_decoder(buffers)
        attribute_dict = json.loads(message[2], object_hook=decoder)
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
        self.zmq_context = zmq.Context()
        # The downward queue is used to feed replies to the socket thread
        self.downward_queue = Queue.Queue()

        # socket for handling downward notifications
        self.selfnotify_socket = self.zmq_context.socket(zmq.SUB)
        self.selfnotify_socket.bind(NOTIFY_SOCKET_ADDR)
        self.selfnotify_socket.setsockopt(zmq.SUBSCRIBE, '')
        self.threadlocal = threading.local()  # for connecting to notification socket, and receiving replies

        # announce socket
        
        self.announce_socket = self.zmq_context.socket(zmq.PUB)
        if port is None:
            self.announce_port = self.announce_socket.bind_to_random_port(zmq_address)
            self.announce_address = "%s:%d" % (zmq_address, self.announce_port)
        else:
            self.announce_address = "%s:%d" % (zmq_address, port)
            self.announce_port = self.announce_socket.bind(self.announce_address)
            
        # socket where we receive Requests
        self.request_socket = self.zmq_context.socket(zmq.ROUTER)
        self.request_port = self.request_socket.bind_to_random_port(zmq_address)
        self.request_address = zmq_address + (':%d' % self.request_port)
            
        self.thread = threading.Thread(
            target=self.spin,
            args=(self.selfnotify_socket, self.request_socket),
            name="Boundary spin()")
        self.thread.daemon = True
        self.thread.start()
        
    '''Notify the socket thread that an analysis was added'''
    NOTIFY_REGISTER_ANALYSIS = "register analysis"
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
        response_queue = Queue.Queue()
        self.send_to_boundary_thread(self.NOTIFY_REGISTER_ANALYSIS,
                                     (analysis_id, response_queue))
        response_queue.get()
        
    def enqueue_reply(self, req, rep):
        '''Enqueue a reply to be sent from the boundary thread
        
        req - original request
        rep - the reply to the request
        '''
        self.send_to_boundary_thread(self.NOTIFY_REPLY_READY,(req, rep))
            
    def cancel(self, analysis_id):
        '''Cancel an analysis
        
        All requests with the given analysis ID will get a BoundaryExited
        reply after this call returns.
        '''
        with self.analysis_dictionary_lock:
            if self.analysis_dictionary[analysis_id].cancelled:
                return
            self.analysis_dictionary[analysis_id].cancel()
        response_queue = Queue.Queue()
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

    def spin(self, selfnotify_socket, request_socket):
        try:
            poller = zmq.Poller()
            poller.register(selfnotify_socket, zmq.POLLIN)
            poller.register(request_socket, zmq.POLLIN)
            
            received_stop = False
    
            while not received_stop:
                self.announce_analyses()
                socks = dict(poller.poll(1000))  # milliseconds
                if socks.get(selfnotify_socket, None) == zmq.POLLIN:
                    # Discard the actual contents
                    _ = selfnotify_socket.recv()
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
                            self.handle_register_analysis(analysis_id, response_queue)
                        elif notification == self.NOTIFY_STOP:
                            received_stop = True
                except Queue.Empty:
                    pass
                if socks.get(request_socket, None) == zmq.POLLIN:
                    req = Communicable.recv(request_socket, routed=True)
                    req.set_boundary(self)
                    if not isinstance(req, AnalysisRequest):
                        logger.warn(
                            "Received a request that wasn't an AnalysisRequest: %s"% 
                            str(type(req)))
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
    
            self.request_socket.close()  # will linger until messages are delivered
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
