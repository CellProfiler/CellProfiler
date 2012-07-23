"""test_zmqrequest.py - test the zmqrequest framework

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import Queue
import threading
import zmq
import unittest
import uuid

import cellprofiler.utilities.zmqrequest as Z

CLIENT_MESSAGE = "Hello, server"
SERVER_MESSAGE = "Hello, client"

class TestZMQRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zmq_context = zmq.Context()
        
    @classmethod
    def tearDownClass(cls):
        Z.join_to_the_boundary()
        cls.zmq_context.term()
        
    class ZMQClient(threading.Thread):
        '''A mockup of a ZMQ client to the boundary
        
        must be instantiated after an analysis has been started
        '''
        def __init__(self, analysis_id, name="Client thread"):
            threading.Thread.__init__(self, name = name)
            self.setDaemon(True)
            self.queue = Queue.Queue()
            self.response_queue = Queue.Queue()
            self.cv = threading.Condition()
            self.keep_going = True
            self.analysis_id = analysis_id
            self.start()
            with self.cv:
                self.cv.wait()
                
        def __enter__(self):
            return self
            
        def __exit__(self, type, value, traceback):
            self.stop()
            self.join()
            
        def run(self):
            self.work_socket = TestZMQRequest.zmq_context.socket(zmq.REQ)
            self.work_socket.connect(Z.the_boundary.request_address)
            with self.cv:
                self.cv.notify_all()
                
            while True:
                with self.cv:
                    self.cv.wait(10.)
                    try:
                        if not self.keep_going:
                            break
                        req = self.queue.get_nowait()
                    except Queue.Empty:
                        continue
                try:
                    self.response_queue.put((None, req.send(self.work_socket)))
                except Exception,e:
                    self.response_queue.put((e, None))
            self.work_socket.close()
                
        def stop(self):
            with self.cv:
                self.keep_going = False
                self.cv.notify_all()
            
        def send(self, req):
            with self.cv:
                self.queue.put(req)
                self.cv.notify_all()
                
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
            self.upcv = threading.Condition()
            self.boundary = Z.register_analysis(self.analysis_id,
                                                self.upq,
                                                self.upcv)
            return self
            
        def recv(self, timeout):
            '''Receive a message'''
            with self.upcv:
                self.upcv.wait(timeout)
                try:
                    req = self.upq.get_nowait()
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
        with self.ZMQServer() as server:
            with self.ZMQClient(server.analysis_id) as client:
                client.send(Z.AnalysisRequest(server.analysis_id,
                                              msg=CLIENT_MESSAGE))
                req = server.recv(10.)
                self.assertIsInstance(req, Z.AnalysisRequest)
                self.assertEqual(req.msg, CLIENT_MESSAGE)
                req.reply(Z.Reply(msg = SERVER_MESSAGE))
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
                req.reply(Z.Reply(msg = SERVER_MESSAGE))
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
        
            
            
                