import logging
import queue
import socket
import threading

import zmq

import cellprofiler_core.utilities.zmq
from cellprofiler_core.utilities.zmq._analysis_context import AnalysisContext
from .communicable import Communicable
from .communicable.reply.upstream_exit import BoundaryExited
from .communicable.request import AnalysisRequest, Request


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
        self.selfnotify_socket.bind(cellprofiler_core.utilities.zmq.NOTIFY_SOCKET_ADDR)
        self.selfnotify_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.threadlocal = (
            threading.local()
        )  # for connecting to notification socket, and receiving replies

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
        self.request_address = zmq_address + (":%d" % self.request_port)
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
        self.external_request_port = self.external_request_socket.bind_to_random_port(
            "tcp://*"
        )
        self.external_request_address = "tcp://%s:%d" % (
            fqdn,
            self.external_request_port,
        )

        self.thread = threading.Thread(
            target=self.spin,
            args=(
                self.selfnotify_socket,
                self.request_socket,
                self.external_request_socket,
            ),
            name="Boundary spin()",
        )
        self.thread.start()

    """Notify the socket thread that an analysis was added"""
    NOTIFY_REGISTER_ANALYSIS = "register analysis"
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
        with self.analysis_dictionary_lock:
            self.analysis_dictionary[analysis_id] = AnalysisContext(
                analysis_id, upward_queue, self.analysis_dictionary_lock
            )
        response_queue = queue.Queue()
        self.send_to_boundary_thread(
            self.NOTIFY_REGISTER_ANALYSIS, (analysis_id, response_queue)
        )
        response_queue.get()

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
        with self.analysis_dictionary_lock:
            if self.analysis_dictionary[analysis_id].cancelled:
                return
            self.analysis_dictionary[analysis_id].cancel()
        response_queue = queue.Queue()
        self.send_to_boundary_thread(
            self.NOTIFY_CANCEL_ANALYSIS, (analysis_id, response_queue)
        )
        response_queue.get()

    def handle_cancel(self, analysis_id, response_queue):
        """Handle cancellation in the boundary thread"""
        with self.analysis_dictionary_lock:
            self.analysis_dictionary[analysis_id].handle_cancel()
        self.announce_analyses()
        response_queue.put("OK")

    def join(self):
        """Join to the boundary thread.

        Note that this should only be done at a point where no worker truly
        expects a reply to its requests.
        """
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
                            self.handle_register_analysis(analysis_id, response_queue)
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
                    if (
                        s not in (request_socket, external_request_socket)
                        or state != zmq.POLLIN
                    ):
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
                            logging.warning(
                                "Received a request that wasn't an AnalysisRequest: %s"
                                % str(type(req))
                            )
                            req.reply(BoundaryExited())
                        continue
                    if s != request_socket:
                        # Request is on the external socket
                        logging.warning("Received a request on the external socket")
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
                for analysis_context in list(self.analysis_dictionary.values()):
                    analysis_context.cancel()
                for request_class_queue in list(self.request_dictionary.values()):
                    #
                    # Tell each response class to stop. Wait for a reply
                    # which may be a thread instance. If so, join to the
                    # thread so there will be an orderly shutdown.
                    #
                    response_queue = queue.Queue()
                    request_class_queue.put([self, self.NOTIFY_STOP, response_queue])
                    thread = response_queue.get()
                    if isinstance(thread, threading.Thread):
                        thread.join()

            self.request_socket.close()
            logging.info("Exiting the boundary thread")
        except:
            #
            # Pretty bad - a logic error or something extremely unexpected
            #              We're close to hosed here, best to die an ugly death.
            #
            logging.critical("Unhandled exception in boundary thread.", exc_info=10)
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

    def announce_analyses(self):
        with self.analysis_dictionary_lock:
            valid_analysis_ids = [
                analysis_id
                for analysis_id in list(self.analysis_dictionary.keys())
                if not self.analysis_dictionary[analysis_id].cancelled
            ]
        self.announce_socket.send_json(
            [(analysis_id, self.request_address) for analysis_id in valid_analysis_ids]
        )

    def handle_reply(self, req, rep):
        if not isinstance(req, AnalysisRequest):
            assert isinstance(req, Request)
            Communicable.reply(req, rep)
            return

        with self.analysis_dictionary_lock:
            analysis_context = self.analysis_dictionary.get(req.analysis_id)
            analysis_context.reply(req, rep)

    def handle_register_analysis(self, analysis_id, response_queue):
        """Handle a request to register an analysis

        analysis_id - analysis_id of new analysis
        response_queue - response queue. Any announce subscriber that registers
                         after the response is placed in this queue
                         will receive an announcement of the analysis.
        """
        self.announce_analyses()
        response_queue.put("OK")
