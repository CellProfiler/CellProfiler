import logging
import queue
import threading

import zmq

import cellprofiler_core.utilities.zmq
from cellprofiler_core.utilities.zmq._analysis_context import AnalysisContext
from .communicable import Communicable
from .communicable.reply.upstream_exit import BoundaryExited
from .communicable.request import AnalysisRequest, Request
from ...constants.worker import NOTIFY_STOP, NOTIFY_RUN


LOGGER = logging.getLogger(__name__)

class Boundary:
    """This object serves as the interface between a ZMQ socket passing
    Requests and Replies, and a thread or threads serving those requests.
    Received requests are received on a ZMQ socket and placed on upward_queue,
    and notify_all() is called on updward_cv.  Replies (via the Request.reply()
    method) are dispatched to their requesters via a downward queue.

    The Boundary wakes up the socket thread via the notify socket. This lets
    the socket thread poll for changes on the notify and request sockets, but
    allows it to receive Python objects via the downward queue.
    """

    def __init__(self, zmq_address, port=None):
        """Construction

        zmq_address - the address for announcements and requests
        port - the port for announcements, defaults to random
        """
        self.analysis_context = None
        self.analysis_context_lock = threading.RLock()
        #
        # Dictionary of request dictionary to queue for handler
        # (not including AnalysisRequest)
        #
        self.request_dictionary = {}
        self.zmq_context = zmq.Context()
        # Set linger to 0 so that all sockets close without
        # waiting for transmission during shutdown.
        self.zmq_context.setsockopt(zmq.LINGER, 0)
        self.zmq_address = zmq_address
        # The downward queue is used to feed replies to the socket thread
        self.downward_queue = queue.Queue()

        # Not entirely convinced that threadlocal is still needed, but meh.
        self.threadlocal = (
            threading.local()
        )  # for connecting to notification socket, and receiving replies

        #
        # Socket for telling workers whether to keep running
        # We should ONLY send over this socket from the spin thread
        # We need to create it here so that we can transmit the address to
        # workers on creation.
        #
        self.keepalive_socket = self.zmq_context.socket(zmq.PUB)

        self.keepalive_socket_port = self.keepalive_socket.bind_to_random_port(
            "tcp://127.0.0.1"
        )
        self.keepalive_address = f"tcp://127.0.0.1:{self.keepalive_socket_port}"

        self.thread = threading.Thread(
            target=self.spin,
            name="Boundary spin() thread",
        )
        self.thread.start()

    """Notify a request class handler of a request"""
    NOTIFY_REQUEST = "request"
    """Notify the socket thread that a reply is ready to be sent"""
    NOTIFY_REPLY_READY = "reply ready"
    """Cancel an analysis. The analysis ID is the second part of the message"""
    NOTIFY_CANCEL_ANALYSIS = "cancel analysis"
    """Stop the socket thread"""
    NOTIFY_STOP = "stop"

    def register_analysis(self, analysis_id, upward_queue):
        """Register a queue to receive analysis requests

        analysis_id - the analysis ID embedded in each analysis request

        upward_queue - place the requests on this queue
        """
        with self.analysis_context_lock:
            self.analysis_context = AnalysisContext(
                analysis_id, upward_queue, self.analysis_context_lock
            )
        LOGGER.debug(f"Registered analysis as id {analysis_id}")

    def register_request_class(self, cls_request, upward_queue):
        """Register a queue to receive requests of the given class

        cls_request - requests that match isinstance(request, cls_request) will
                      be routed to the upward_queue

        upward_queue - queue that will receive the requests
        """
        self.request_dictionary[cls_request] = upward_queue

    def enqueue_reply(self, req, rep):
        """Enqueue a reply to be sent from the boundary thread

        req - original request
        rep - the reply to the request
        """
        self.send_to_boundary_thread(self.NOTIFY_REPLY_READY, (req, rep))

    def cancel(self, analysis_id):
        """Cancel an analysis

        All requests with the given analysis ID will get a BoundaryExited
        reply after this call returns.
        """
        with self.analysis_context_lock:
            if self.analysis_context.cancelled:
                return
            self.analysis_context.cancel()
        response_queue = queue.Queue()
        self.send_to_boundary_thread(
            self.NOTIFY_CANCEL_ANALYSIS, (analysis_id, response_queue)
        )
        response_queue.get()

    def handle_cancel(self, analysis_id, response_queue):
        """Handle cancellation in the boundary thread"""
        with self.analysis_context_lock:
            self.analysis_context.handle_cancel()
        response_queue.put("OK")

    def join(self):
        """Join to the boundary thread.

        Note that this should only be done at a point where no worker truly
        expects a reply to its requests.
        """
        self.send_to_boundary_thread(self.NOTIFY_STOP, None)
        self.thread.join()

    def spin(self):
        try:
            # socket for handling downward notifications
            selfnotify_socket = self.zmq_context.socket(zmq.SUB)
            selfnotify_socket.connect(
                cellprofiler_core.utilities.zmq.NOTIFY_SOCKET_ADDR)
            selfnotify_socket.setsockopt(zmq.SUBSCRIBE, b"")

            # socket where we receive Requests
            request_socket = self.zmq_context.socket(zmq.ROUTER)
            request_socket.setsockopt(zmq.LINGER, 0)
            request_socket.set_hwm(2000)
            request_port = request_socket.bind_to_random_port(
                self.zmq_address)
            self.request_address = self.zmq_address + (":%d" % request_port)

            poller = zmq.Poller()
            poller.register(selfnotify_socket, zmq.POLLIN)
            poller.register(request_socket, zmq.POLLIN)

            received_stop = False
            while not received_stop:
                # Tell the workers to stay alive
                self.keepalive_socket.send(NOTIFY_RUN)
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
                        elif notification == self.NOTIFY_STOP:
                            received_stop = True
                except queue.Empty:
                    pass
                #
                # Then process the poll result
                #
                for s, state in poll_result:
                    if s == selfnotify_socket:
                        # This isn't really used for message transmission.
                        # Instead, it pokes this spin thread to get it to check
                        # the non-ZMQ message queue.
                        msg = selfnotify_socket.recv()
                        # Let's watch out for stop signals anyway
                        if msg == NOTIFY_STOP:
                            LOGGER.warning("Captured a stop message over zmq")
                            received_stop = True
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
                            LOGGER.warning(
                                "Received a request that wasn't an AnalysisRequest: %s"
                                % str(type(req))
                            )
                            req.reply(BoundaryExited())
                        continue
                    if s != request_socket:
                        # Request is on the external socket
                        LOGGER.warning("Received a request on the external socket")
                        req.reply(BoundaryExited())
                        continue
                    #
                    # Filter out requests for cancelled analyses.
                    #
                    with self.analysis_context_lock:
                        if not self.analysis_context.enqueue(req):
                            continue
            LOGGER.debug("boundary received stop")
            # Give the workers an explicit stop command
            self.keepalive_socket.send(NOTIFY_STOP)
            self.keepalive_socket.close()
            #
            # We assume here that workers trying to communicate with us will
            # be shut down without needing replies to pending requests.
            # There's not much we can do in terms of handling that in a more
            # orderly fashion since workers might be formulating requests as or
            # after we have shut down. But calling cancel on all the analysis
            # contexts will raise exceptions in threads waiting for a rep/rep.
            #
            # You could call analysis_context.handle_cancel() here, what if it
            # blocks?
            with self.analysis_context_lock:
                self.analysis_context.cancel()
                request_dict_values = self.request_dictionary.values()
                LOGGER.debug(f"joining {len(request_dict_values)} threads")
                for request_class_queue in request_dict_values:
                    #
                    # Tell each response class to stop. Wait for a reply
                    # which may be a thread instance. If so, join to the
                    # thread so there will be an orderly shutdown.
                    #
                    response_queue = queue.Queue()
                    request_class_queue.put(
                        [self, self.NOTIFY_STOP, response_queue]
                    )
                    thread = response_queue.get()
                    LOGGER.debug("joining on thread from response_queue")
                    if isinstance(thread, threading.Thread):
                        thread.join()

            request_socket.close()
            selfnotify_socket.close()
            LOGGER.debug("Shutting down the boundary thread")
        except:
            #
            # Pretty bad - a logic error or something extremely unexpected
            #              We're close to hosed here, better die an ugly death.
            #
            LOGGER.critical("Unhandled exception in boundary thread.",
                             exc_info=True)
            import os
            os._exit(-1)

    def send_to_boundary_thread(self, msg, arg):
        """Send a message to the boundary thread via the notify socket

        Send a wakeup call to the boundary thread by sending arbitrary
        data to the notify socket, placing the real objects of interest
        on the downward queue.

        msg - message placed in the downward queue indicating the purpose
              of the wakeup call

        args - supplementary arguments passed to the boundary thread via
               the downward queue.
        """
        if not hasattr(self.threadlocal, "notify_socket"):
            self.threadlocal.notify_socket = self.zmq_context.socket(zmq.PUB)
            self.threadlocal.notify_socket.setsockopt(zmq.LINGER, 0)
            self.threadlocal.notify_socket.connect(
                cellprofiler_core.utilities.zmq.NOTIFY_SOCKET_ADDR
            )
        self.downward_queue.put((msg, arg))
        self.threadlocal.notify_socket.send(b"WAKE UP!")

    def handle_reply(self, req, rep):
        if not isinstance(req, AnalysisRequest):
            assert isinstance(req, Request)
            Communicable.reply(req, rep)
            return

        with self.analysis_context_lock:
            self.analysis_context.reply(req, rep)
