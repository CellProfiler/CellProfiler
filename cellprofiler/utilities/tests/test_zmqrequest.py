"""test_zmqrequest.py - test the zmqrequest framework
"""
import logging
import logging.handlers

logger = logging.getLogger(__name__)
import Queue
import os
import threading
import tempfile
import zmq
import unittest
import uuid
import numpy as np

import cellprofiler.utilities.zmqrequest as Z

CLIENT_MESSAGE = "Hello, server"
SERVER_MESSAGE = "Hello, client"


class TestZMQRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_hostname = os.environ.get("HOSTNAME")
        if not cls.old_hostname is None:
            del os.environ['HOSTNAME']
        cls.zmq_context = zmq.Context()

    @classmethod
    def tearDownClass(cls):
        Z.join_to_the_boundary()
        cls.zmq_context.term()
        if cls.old_hostname is not None:
            os.environ['HOSTNAME'] = cls.old_hostname

    class ZMQClient(threading.Thread):
        '''A mockup of a ZMQ client to the boundary

        must be instantiated after an analysis has been started
        '''
        MSG_STOP = "STOP"
        MSG_SEND = "SEND"

        def __init__(self, analysis_id, name="Client thread"):
            threading.Thread.__init__(self, name=name)
            self.notify_addr = "inproc://" + uuid.uuid4().hex
            self.setDaemon(True)
            self.queue = Queue.Queue()
            self.response_queue = Queue.Queue()
            self.start_signal = threading.Semaphore(0)
            self.keep_going = True
            self.analysis_id = analysis_id
            logger.info("Starting client thread")
            self.start()
            self.start_signal.acquire()
            logger.info("Client thread started")
            self.send_notify_socket = TestZMQRequest.zmq_context.socket(zmq.PAIR)
            self.send_notify_socket.connect(self.notify_addr)

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            logger.info("Stopping client thread")
            self.stop()
            self.join()
            logger.info("Client thread stopped")
            self.send_notify_socket.close()

        def run(self):
            self.work_socket = TestZMQRequest.zmq_context.socket(zmq.REQ)
            self.work_socket.connect(Z.the_boundary.request_address)
            self.notify_socket = TestZMQRequest.zmq_context.socket(zmq.PAIR)
            self.notify_socket.bind(self.notify_addr)
            poller = zmq.Poller()
            poller.register(self.work_socket, zmq.POLLIN)
            poller.register(self.notify_socket, zmq.POLLIN)
            self.start_signal.release()
            try:
                while self.keep_going:
                    for sock, state in poller.poll():
                        logger.info("Client interrupt")
                        if sock == self.work_socket:
                            logger.info("Receiving message on work socket")
                            rep = Z.Communicable.recv(self.work_socket)
                            logger.info("Message received")
                            self.response_queue.put((None, rep))
                        elif sock == self.notify_socket:
                            logger.info("Client receiving message on notify socket")
                            msg = self.notify_socket.recv()
                            logger.info("Client message received")
                            if msg == self.MSG_STOP:
                                return
                            elif msg == self.MSG_SEND:
                                logger.info("Client sending message")
                                req = self.queue.get_nowait()
                                req.send_only(self.work_socket)
                                logger.info("Client message sent")
            except Exception, e:
                self.response_queue.put((e, None))
            finally:
                logger.info("Client thread exiting")
                self.work_socket.close()
                self.notify_socket.close()

        def stop(self):
            self.keep_going = False
            self.send_notify_socket.send(self.MSG_STOP)

        def send(self, req):
            self.queue.put(req)
            self.send_notify_socket.send(self.MSG_SEND)

        def recv(self):
            exception, result = self.response_queue.get()
            if exception is not None:
                raise exception
            else:
                return result

    class ZMQServer(object):
        def __enter__(self):
            self.analysis_id = uuid.uuid4().hex
            self.upq = Queue.Queue()
            logger.info("Server registering")
            self.boundary = Z.register_analysis(self.analysis_id,
                                                self.upq)
            logger.info("Server has registered")
            return self

        def recv(self, timeout):
            '''Receive a message'''
            try:
                req = self.upq.get(timeout)
                return req
            except Queue.Empty:
                raise AssertionError("Failed to receive message within timeout of %f sec" % timeout)

        def __exit__(self, type, value, traceback):
            self.cancel()

        def cancel(self):
            if self.boundary is not None:
                self.boundary.cancel(self.analysis_id)
                self.boundary = None

    def test_01_01_start(self):
        with self.ZMQServer() as server:
            pass

    def test_01_02_send_and_receive(self):
        logger.info("Executing test_01_02_send_and_receive")
        with self.ZMQServer() as server:
            with self.ZMQClient(server.analysis_id) as client:
                logger.info("Sending client an analysis request message")
                client.send(Z.AnalysisRequest(server.analysis_id,
                                              msg=CLIENT_MESSAGE))
                logger.info("Message given to client")
                req = server.recv(10.)
                logger.info("Message received from server")
                self.assertIsInstance(req, Z.AnalysisRequest)
                self.assertEqual(req.msg, CLIENT_MESSAGE)
                req.reply(Z.Reply(msg=SERVER_MESSAGE))
                response = client.recv()
                self.assertEqual(response.msg, SERVER_MESSAGE)

    def test_02_01_boundary_exit_after_send(self):
        with self.ZMQServer() as server:
            with self.ZMQClient(server.analysis_id) as client:
                client.send(Z.AnalysisRequest(server.analysis_id,
                                              msg=CLIENT_MESSAGE))
                req = server.recv(10.)
                self.assertIsInstance(req, Z.AnalysisRequest)
                self.assertEqual(req.msg, CLIENT_MESSAGE)
                server.cancel()
                req.reply(Z.Reply(msg=SERVER_MESSAGE))
                response = client.recv()
                self.assertIsInstance(response, Z.BoundaryExited)

    def test_02_02_boundary_exit_before_send(self):
        with self.ZMQServer() as server:
            with self.ZMQClient(server.analysis_id) as client:
                server.cancel()
                client.send(Z.AnalysisRequest(server.analysis_id,
                                              msg=CLIENT_MESSAGE))
                response = client.recv()
                self.assertIsInstance(response, Z.BoundaryExited)

    def test_03_01_announce_nothing(self):
        boundary = Z.start_boundary()
        socket = self.zmq_context.socket(zmq.SUB)
        socket.connect(boundary.announce_address)
        socket.setsockopt(zmq.SUBSCRIBE, '')
        obj = socket.recv_json()
        self.assertEqual(len(obj), 0)

    def test_03_02_announce_something(self):
        boundary = Z.start_boundary()
        with self.ZMQServer() as server:
            socket = self.zmq_context.socket(zmq.SUB)
            socket.connect(boundary.announce_address)
            socket.setsockopt(zmq.SUBSCRIBE, '')
            obj = socket.recv_json()
            self.assertEqual(len(obj), 1)
            self.assertEqual(len(obj[0]), 2)
            analysis_id, address = obj[0]
            self.assertEqual(address, Z.the_boundary.request_address)
            self.assertEqual(analysis_id, server.analysis_id)
        #
        #
        req_socket = self.zmq_context.socket(zmq.REQ)
        req_socket.connect(address)

        #
        # The analysis should be gone immediately after the
        # server has shut down
        #
        obj = socket.recv_json()
        self.assertEqual(len(obj), 0)

    def test_03_03_test_lock_file(self):
        t = tempfile.NamedTemporaryFile()
        self.assertTrue(Z.lock_file(t.name))
        self.assertFalse(Z.lock_file(t.name))
        Z.unlock_file(t.name)
        self.assertTrue(Z.lock_file(t.name))
        Z.unlock_file(t.name)

    def test_03_04_json_encode(self):
        r = np.random.RandomState()
        r.seed(15)
        test_cases = [
            {"k": "v"},
            {"k": (1, 2, 3)},
            {(1, 2, 3): "k"},
            {1: {u"k": "v"}},
            {"k": [{1: 2}, {3: 4}]},
            {"k": ((1, 2, {"k1": "v1"}),)},
            {"k": r.uniform(size=(5, 8))},
            {"k": r.uniform(size=(7, 3)) > .5}
        ]
        for test_case in test_cases:
            json_string, buf = Z.json_encode(test_case)
            result = Z.json_decode(json_string, buf)
            self.same(test_case, result)

    def test_03_05_json_encode_uint64(self):
        for dtype in np.uint64, np.int64, np.uint32:
            json_string, buf = Z.json_encode(
                    dict(foo=np.arange(10).astype(dtype)))
            result = Z.json_decode(json_string, buf)
            self.assertEqual(result["foo"].dtype, np.int32)

        json_string, buf = Z.json_encode(
                dict(foo=np.arange(10).astype(np.int16)))
        result = Z.json_decode(json_string, buf)
        self.assertEqual(result["foo"].dtype, np.int16)

    def test_03_06_json_encode_zero_length_uint(self):
        for dtype in np.uint64, np.int64, np.uint32:
            json_string, buf = Z.json_encode(
                    dict(foo=np.zeros(0, dtype)))
            result = Z.json_decode(json_string, buf)
            self.assertEqual(len(result["foo"]), 0)

    def same(self, a, b):
        if isinstance(a, (float, int)):
            self.assertAlmostEquals(a, b)
        elif isinstance(a, basestring):
            self.assertEquals(a, b)
        elif isinstance(a, dict):
            self.assertTrue(isinstance(b, dict))
            for k in a:
                self.assertTrue(k in b)
                self.same(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for aa, bb in zip(a, b):
                self.same(aa, bb)
        elif not np.isscalar(a):
            np.testing.assert_almost_equal(a, b)
        else:
            self.assertEqual(a, b)
