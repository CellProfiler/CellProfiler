"""test_analysis.py - test the analysis server framework
"""

import logging

logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.DEBUG)
from cStringIO import StringIO
import inspect
import numpy as np
import os
import Queue
import tempfile
import threading
import traceback
import unittest
import uuid
import zmq

import cellprofiler.analysis as cpanalysis
import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.preferences as cpprefs
import cellprofiler.measurement as cpmeas
import cellprofiler.utilities.zmqrequest as cpzmq
from tests.modules import example_images_directory, testimages_directory

IMAGE_NAME = "imagename"
OBJECTS_NAME = "objectsname"
IMAGE_FEATURE = "imagefeature"
OBJECTS_FEATURE = "objectsfeature"


class TestAnalysis(unittest.TestCase):
    class FakeWorker(threading.Thread):
        '''A mockup of a ZMQ client to the boundary

        '''

        def __init__(self, name="Client thread"):
            threading.Thread.__init__(self, name=name)
            self.setDaemon(True)
            self.zmq_context = zmq.Context()
            self.queue = Queue.Queue()
            self.response_queue = Queue.Queue()
            self.start_signal = threading.Semaphore(0)
            self.keep_going = True
            self.notify_addr = "inproc://%s" % uuid.uuid4().hex
            self.notify_socket = self.zmq_context.socket(zmq.PUB)
            self.notify_socket.bind(self.notify_addr)
            self.start()
            self.start_signal.acquire()

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            self.stop()
            self.join()
            self.notify_socket.close()

        def run(self):
            logger.debug("Client thread starting")
            try:
                self.work_socket = self.zmq_context.socket(zmq.REQ)
                self.recv_notify_socket = self.zmq_context.socket(zmq.SUB)
                self.recv_notify_socket.setsockopt(zmq.SUBSCRIBE, '')
                self.recv_notify_socket.connect(self.notify_addr)
                self.announce_socket = None
                self.poller = zmq.Poller()
                self.poller.register(self.recv_notify_socket, zmq.POLLIN)
                self.start_signal.release()

                while self.keep_going:
                    socks = dict(self.poller.poll(1000))
                    if socks.get(self.recv_notify_socket, None) == zmq.POLLIN:
                        # Discard whatever comes down the notify socket.
                        # It's only used to wake us up.
                        msg = self.recv_notify_socket.recv()
                    while True:
                        try:
                            if not self.keep_going:
                                break
                            fn_and_args = self.queue.get_nowait()

                        except Queue.Empty:
                            break
                        try:
                            response = fn_and_args[0](*fn_and_args[1:])
                            self.response_queue.put((None, response))
                        except Exception, e:
                            traceback.print_exc()
                            self.response_queue.put((e, None))
            except:
                logger.warn("Client thread caught exception", exc_info=10)
                self.start_signal.release()
            finally:
                logger.debug("Client thread exiting")

        def stop(self):
            self.keep_going = False
            self.notify_socket.send("Stop")

        def send(self, req):
            logger.debug("    Enqueueing send of %s" % str(type(req)))
            self.queue.put((self.do_send, req))
            self.notify_socket.send("Send")
            return self.recv

        def request_work(self):
            '''Send a work request until we get a WorkReply'''
            while True:
                reply = self.send(cpanalysis.WorkRequest(self.analysis_id))()
                if isinstance(reply, cpanalysis.WorkReply):
                    return reply
                elif not isinstance(reply, cpanalysis.NoWorkReply):
                    raise NotImplementedError(
                            "Received a reply of %s for a work request" %
                            str(type(reply)))

        def do_send(self, req):
            logger.debug("    Sending %s" % str(type(req)))
            cpzmq.Communicable.send(req, self.work_socket)
            self.poller.register(self.work_socket, zmq.POLLIN)
            try:
                while True:
                    socks = dict(self.poller.poll())
                    if socks.get(self.recv_notify_socket, None) == zmq.POLLIN:
                        msg = self.recv_notify_socket.recv()
                        if not self.keep_going:
                            raise Exception("Cancelled")
                    if socks.get(self.work_socket, None) == zmq.POLLIN:
                        logger.debug("    Received response for %s" % str(type(req)))
                        return cpzmq.Communicable.recv(self.work_socket)
            finally:
                self.poller.unregister(self.work_socket)

        def recv(self):
            logger.debug("     Waiting for client thread")
            exception, result = self.response_queue.get()
            if exception is not None:
                logger.debug("    Client thread communicated exception")
                raise exception
            else:
                logger.debug("    Client thread communicated result")
                return result

        def listen_for_announcements(self, work_announce_address):
            self.queue.put((self.do_listen_for_announcements,
                            work_announce_address))
            self.notify_socket.send("Listen for announcements")
            return self.recv

        def do_listen_for_announcements(self, work_announce_address):
            self.announce_socket = self.zmq_context.socket(zmq.SUB)
            self.announce_socket.setsockopt(zmq.SUBSCRIBE, '')
            self.announce_socket.connect(work_announce_address)
            self.poller.register(self.announce_socket, zmq.POLLIN)
            try:
                while True:
                    socks = dict(self.poller.poll())
                    if socks.get(self.recv_notify_socket, None) == zmq.POLLIN:
                        msg = self.recv_notify_socket.recv()
                        if not self.keep_going:
                            raise Exception("Cancelled")
                    if socks.get(self.announce_socket, None) == zmq.POLLIN:
                        announcements = \
                            self.announce_socket.recv_json()
                        return announcements
            finally:
                self.poller.unregister(self.announce_socket)
                self.announce_socket.close()
                self.announce_socket = None

        def connect(self, work_announce_address):
            self.analysis_id, work_queue_address = \
                self.listen_for_announcements(work_announce_address)()[0]

            self.queue.put((self.do_connect, work_queue_address))
            self.notify_socket.send("Do connect")
            return self.recv()

        def do_connect(self, work_queue_address):
            self.work_socket.connect(work_queue_address)

    @classmethod
    def setUpClass(cls):
        cls.zmq_context = zmq.Context()
        from cellprofiler.modules import fill_modules
        fill_modules()

    @classmethod
    def tearDownClass(cls):
        cpzmq.join_to_the_boundary()

        cls.zmq_context.term()

    def setUp(self):
        fd, self.filename = tempfile.mkstemp(".h5")
        os.close(fd)
        self.event_queue = Queue.Queue()
        self.analysis = None
        self.wants_analysis_finished = False
        self.wants_pipeline_events = False
        self.measurements_to_close = None

    def tearDown(self):
        self.cancel_analysis()
        if self.measurements_to_close is not None:
            self.measurements_to_close.close()
        if os.path.exists(self.filename):
            os.unlink(self.filename)

    def cancel_analysis(self):
        if self.analysis is not None:
            self.analysis.cancel()
            self.analysis = None

    def analysis_event_handler(self, event):
        if isinstance(event, cpanalysis.AnalysisProgress):
            return
        if isinstance(event, cpanalysis.AnalysisFinished):
            self.measurements_to_close = event.measurements
            if not self.wants_analysis_finished:
                return
        if (isinstance(event, cpp.AbstractPipelineEvent) and
                not self.wants_pipeline_events):
            return
        self.event_queue.put(event)

    def make_pipeline_and_measurements(self,
                                       nimage_sets=1,
                                       group_numbers=None,
                                       group_indexes=None,
                                       **kwargs):
        m = cpmeas.Measurements(mode="memory")
        for i in range(1, nimage_sets + 1):
            if group_numbers is not None:
                group_number = group_numbers[i - 1]
                group_index = group_indexes[i - 1]
            else:
                group_number = 1
                group_index = i
            m[cpmeas.IMAGE, cpmeas.C_URL + "_" + IMAGE_NAME, i] = "file:/%d.tif" % i
            m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, i] = group_number
            m[cpmeas.IMAGE, cpmeas.GROUP_INDEX, i] = group_index
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(StringIO(SBS_PIPELINE), raise_on_error=True)
        return pipeline, m

    def make_pipeline_and_measurements_and_start(self, **kwargs):
        pipeline, m = self.make_pipeline_and_measurements(**kwargs)
        if "status" in kwargs:
            overwrite = False
            for i, status in enumerate(kwargs["status"]):
                m.add_measurement(cpmeas.IMAGE,
                                  cpanalysis.AnalysisRunner.STATUS,
                                  status, image_set_number=i + 1)
        else:
            overwrite = True
        self.analysis = cpanalysis.Analysis(pipeline, self.filename, m)

        self.analysis.start(self.analysis_event_handler,
                            num_workers=0, overwrite=overwrite)
        analysis_started = self.event_queue.get()
        self.assertIsInstance(analysis_started, cpanalysis.AnalysisStarted)
        return pipeline, m

    def check_display_post_run_requests(self, pipeline):
        '''Read the DisplayPostRunRequest messages during the post_run phase'''

        for module in pipeline.modules():
            if module.show_window and \
                            module.__class__.display_post_run != cpm.Module.display_post_run:
                result = self.event_queue.get()
                self.assertIsInstance(
                        result, cpanalysis.DisplayPostRunRequest)
                self.assertEqual(result.module_num, module.module_num)

    def test_01_01_start_and_stop(self):

        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.make_pipeline_and_measurements_and_start()
        self.wants_analysis_finished = True
        self.cancel_analysis()
        # The last should be AnalysisFinished. There may be AnalysisProgress
        # prior to that.
        analysis_finished = self.event_queue.get()
        self.assertIsInstance(analysis_finished, cpanalysis.AnalysisFinished)
        self.assertTrue(analysis_finished.cancelled)
        self.assertIsInstance(analysis_finished.measurements,
                              cpmeas.Measurements)
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_02_01_announcement(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()

        with self.FakeWorker() as worker:
            work_announce_address = self.analysis.runner.work_announce_address
            response = worker.listen_for_announcements(work_announce_address)()
            self.assertEqual(len(response), 1)
            analysis_id, request_address = response[0]
            self.assertEqual(analysis_id, self.analysis.analysis_in_progress)
            self.cancel_analysis()
            response = worker.listen_for_announcements(work_announce_address)()
            self.assertEqual(len(response), 0)
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_03_01_get_work(self):
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.send(cpanalysis.WorkRequest(worker.analysis_id))()
            self.assertIsInstance(response, cpanalysis.WorkReply)
            self.assertSequenceEqual(response.image_set_numbers, (1,))
            self.assertFalse(response.worker_runs_post_group)
            self.assertTrue(response.wants_dictionary)
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_03_02_get_work_twice(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()

        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.send(cpanalysis.WorkRequest(worker.analysis_id))()
            self.assertIsInstance(response, cpanalysis.WorkReply)
            response = worker.send(cpanalysis.WorkRequest(worker.analysis_id))()
            self.assertIsInstance(response, cpanalysis.NoWorkReply)
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_03_03_cancel_before_work(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()

        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            self.cancel_analysis()
            response = worker.send(cpanalysis.WorkRequest(worker.analysis_id))()
            self.assertIsInstance(response, cpzmq.BoundaryExited)
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_04_01_pipeline_preferences(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        cpprefs.set_headless()
        title_font_name = "Rosewood Std Regular"
        cpprefs.set_title_font_name(title_font_name)
        cpprefs.set_default_image_directory(example_images_directory())
        cpprefs.set_default_output_directory(testimages_directory())
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.send(cpanalysis.PipelinePreferencesRequest(
                    worker.analysis_id))()
            #
            # Compare pipelines
            #
            client_pipeline = cpp.Pipeline()
            pipeline_txt = response.pipeline_blob.tostring()
            client_pipeline.loadtxt(StringIO(pipeline_txt),
                                    raise_on_error=True)
            self.assertEqual(len(pipeline.modules()),
                             len(client_pipeline.modules()))
            for smodule, cmodule in zip(pipeline.modules(),
                                        client_pipeline.modules()):
                self.assertEqual(smodule.module_name, cmodule.module_name)
                self.assertEqual(len(smodule.settings()),
                                 len(cmodule.settings()))
                for ssetting, csetting in zip(smodule.settings(),
                                              cmodule.settings()):
                    self.assertEqual(ssetting.get_value_text(),
                                     csetting.get_value_text())
            preferences = response.preferences
            self.assertIn(cpprefs.TITLE_FONT_NAME, preferences)
            self.assertEqual(preferences[cpprefs.TITLE_FONT_NAME],
                             title_font_name)
            self.assertIn(cpprefs.DEFAULT_IMAGE_DIRECTORY, preferences)
            self.assertEqual(preferences[cpprefs.DEFAULT_IMAGE_DIRECTORY],
                             cpprefs.get_default_image_directory())
            self.assertIn(cpprefs.DEFAULT_OUTPUT_DIRECTORY, preferences)
            self.assertEqual(preferences[cpprefs.DEFAULT_OUTPUT_DIRECTORY],
                             cpprefs.get_default_output_directory())

        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_04_02_initial_measurements_request(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            client_measurements = cpmeas.load_measurements_from_buffer(
                    response.buf)
            try:
                assert isinstance(client_measurements, cpmeas.Measurements)
                assert isinstance(m, cpmeas.Measurements)
                self.assertSequenceEqual(
                        m.get_image_numbers(),
                        client_measurements.get_image_numbers())
                image_numbers = m.get_image_numbers()
                self.assertItemsEqual(m.get_object_names(),
                                      client_measurements.get_object_names())
                for object_name in m.get_object_names():
                    self.assertItemsEqual(
                            m.get_feature_names(object_name),
                            client_measurements.get_feature_names(object_name))
                    for feature_name in m.get_feature_names(object_name):
                        for image_number in image_numbers:
                            sv = m.get_measurement(
                                    object_name, feature_name,
                                    image_set_number=image_number)
                            cv = client_measurements.get_measurement(
                                    object_name, feature_name,
                                    image_set_number=image_number)
                            self.assertEqual(np.isscalar(sv),
                                             np.isscalar(cv))
                            if np.isscalar(sv):
                                self.assertEqual(sv, cv)
                            else:
                                np.testing.assert_almost_equal(sv, cv)
            finally:
                client_measurements.close()
                logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_04_03_interaction(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            fn_interaction_reply = worker.send(
                    cpanalysis.InteractionRequest(
                            worker.analysis_id,
                            foo="bar"))
            request = self.event_queue.get()
            self.assertIsInstance(request, cpanalysis.InteractionRequest)
            self.assertEqual(request.foo, "bar")
            request.reply(cpanalysis.InteractionReply(hello="world"))
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, cpanalysis.InteractionReply)
            self.assertEqual(reply.hello, "world")
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_04_04_01_display(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            fn_interaction_reply = worker.send(
                    cpanalysis.DisplayRequest(
                            worker.analysis_id,
                            foo="bar"))
            #
            # The event queue should be hooked up to the interaction callback
            #
            request = self.event_queue.get()
            self.assertIsInstance(request, cpanalysis.DisplayRequest)
            self.assertEqual(request.foo, "bar")
            request.reply(cpanalysis.Ack(message="Gimme Pony"))
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, cpanalysis.Ack)
            self.assertEqual(reply.message, "Gimme Pony")
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_04_04_02_display_post_group(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            fn_interaction_reply = worker.send(
                    cpanalysis.DisplayPostGroupRequest(
                            worker.analysis_id, 1,
                            dict(foo="bar"), 3))
            #
            # The event queue should be hooked up to the interaction callback
            #
            request = self.event_queue.get()
            self.assertIsInstance(request, cpanalysis.DisplayPostGroupRequest)
            display_data = request.display_data
            self.assertEqual(display_data["foo"], "bar")
            request.reply(cpanalysis.Ack(message="Gimme Pony"))
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, cpanalysis.Ack)
            self.assertEqual(reply.message, "Gimme Pony")
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_04_05_exception(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            fn_interaction_reply = worker.send(
                    cpanalysis.ExceptionReport(
                            worker.analysis_id,
                            image_set_number=1,
                            module_name="Images",
                            exc_type="Exception",
                            exc_message="Not really an exception",
                            exc_traceback=traceback.extract_stack(),
                            filename="test_analysis.py",
                            line_number=374))
            #
            # The event queue should be hooked up to the interaction callback
            #
            request = self.event_queue.get()
            self.assertIsInstance(request, cpanalysis.ExceptionReport)
            function = request.exc_traceback[-1][2]
            self.assertEqual(function, inspect.getframeinfo(inspect.currentframe()).function)
            self.assertEqual(request.filename, "test_analysis.py")
            request.reply(cpanalysis.ExceptionPleaseDebugReply(
                    disposition=1, verification_hash="corned beef"))
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, cpanalysis.ExceptionPleaseDebugReply)
            self.assertEqual(reply.verification_hash, "corned beef")
            self.assertEqual(reply.disposition, 1)
            #
            # Try DebugWaiting and DebugComplete as well
            #
            for req in (cpanalysis.DebugWaiting(worker.analysis_id, 8080),
                        cpanalysis.DebugComplete(worker.analysis_id)):
                fn_interaction_reply = worker.send(req)
                request = self.event_queue.get()
                self.assertEqual(type(request), type(req))
                request.reply(cpanalysis.Ack())
                reply = fn_interaction_reply()
                self.assertIsInstance(reply, cpanalysis.Ack)
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_05_01_imageset_with_dictionary(self):
        #
        # Go through the steps for the first imageset and see if the
        # dictionary that we sent is the one we get.
        #
        # WorkRequest - to get the rights to report the dictionary
        # ImageSetSuccessWithDictionary - to report the dictionary
        # WorkRequest (with spin until WorkReply received)
        # SharedDictionaryRequest
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start(
                nimage_sets=2)
        r = np.random.RandomState()
        r.seed(51)
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            dictionaries = [dict([(uuid.uuid4().hex, r.uniform(size=(10, 15)))
                                  for _ in range(10)])
                            for module in pipeline.modules()]
            response = worker.send(cpanalysis.ImageSetSuccessWithDictionary(
                    worker.analysis_id, response.image_set_numbers[0],
                    dictionaries))()
            self.assertIsInstance(response, cpanalysis.Ack)
            response = worker.request_work()
            self.assertSequenceEqual(response.image_set_numbers, [2])
            response = worker.send(cpanalysis.SharedDictionaryRequest(
                    worker.analysis_id))()
            self.assertIsInstance(response, cpanalysis.SharedDictionaryReply)
            result = response.dictionaries
            self.assertEqual(len(dictionaries), len(result))
            for ed, d in zip(dictionaries, result):
                self.assertItemsEqual(ed.keys(), d.keys())
                for k in ed.keys():
                    np.testing.assert_almost_equal(ed[k], d[k])
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_05_02_groups(self):
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        pipeline, m = self.make_pipeline_and_measurements_and_start(
                nimage_sets=4,
                group_numbers=[1, 1, 2, 2],
                group_indexes=[1, 2, 1, 2])
        r = np.random.RandomState()
        r.seed(52)
        with self.FakeWorker() as worker:
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            self.assertTrue(response.worker_runs_post_group)
            self.assertFalse(response.wants_dictionary)
            self.assertSequenceEqual(response.image_set_numbers, [1, 2])
            response = worker.send(cpanalysis.ImageSetSuccess(
                    worker.analysis_id, response.image_set_numbers[0]))()
            response = worker.request_work()
            self.assertSequenceEqual(response.image_set_numbers, [3, 4])
            self.assertTrue(response.worker_runs_post_group)
            self.assertFalse(response.wants_dictionary)
        logger.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_06_01_single_imageset(self):
        #
        # Test a full cycle of analysis with an image set list
        # with a single image set
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        r = np.random.RandomState()
        r.seed(61)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            client_measurements = cpmeas.load_measurements_from_buffer(
                    response.buf)
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            dictionaries = [dict([(uuid.uuid4().hex, r.uniform(size=(10, 15)))
                                  for _ in range(10)])
                            for module in pipeline.modules()]
            response = worker.send(cpanalysis.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries))()
            objects_measurements = r.uniform(size=10)
            client_measurements[cpmeas.IMAGE, IMAGE_FEATURE, 1] = "Hello"
            client_measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1] = \
                objects_measurements
            req = cpanalysis.MeasurementsReport(
                    worker.analysis_id,
                    client_measurements.file_contents(),
                    image_set_numbers=[1])
            client_measurements.close()
            response_fn = worker.send(req)

            self.check_display_post_run_requests(pipeline)
            #####################################################
            #
            # The server should receive the measurements report.
            # It should merge the measurements and post an
            # AnalysisFinished event.
            #
            #####################################################

            result = self.event_queue.get()
            self.assertIsInstance(result, cpanalysis.AnalysisFinished)
            self.assertFalse(result.cancelled)
            measurements = result.measurements
            self.assertSequenceEqual(measurements.get_image_numbers(), [1])
            self.assertEqual(measurements[cpmeas.IMAGE, IMAGE_FEATURE, 1],
                             "Hello")
            np.testing.assert_almost_equal(
                    measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1],
                    objects_measurements)

    def test_06_02_test_three_imagesets(self):
        # Test an analysis of three imagesets
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(
                nimage_sets=3)
        r = np.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            client_measurements = cpmeas.load_measurements_from_buffer(
                    response.buf)
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            dictionaries = [dict([(uuid.uuid4().hex, r.uniform(size=(10, 15)))
                                  for _ in range(10)])
                            for module in pipeline.modules()]
            response = worker.send(cpanalysis.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries))()
            #####################################################
            #
            # The analysis server should be ready to send us two
            # more jobs to do.
            #
            #####################################################
            expected_jobs = [2, 3]
            for _ in range(2):
                response = worker.request_work()
                image_numbers = response.image_set_numbers
                self.assertEqual(len(image_numbers), 1)
                self.assertIn(image_numbers[0], expected_jobs)
                expected_jobs.remove(image_numbers[0])
            #####################################################
            #
            # Send the measurement groups
            #
            #####################################################
            objects_measurements = [r.uniform(size=10) for _ in range(3)]
            for i, om in enumerate(objects_measurements):
                image_number = i + 1
                if image_number > 0:
                    worker.send(cpanalysis.ImageSetSuccess(
                            worker.analysis_id,
                            image_set_number=image_number))
                m = cpmeas.Measurements(copy=client_measurements)
                m[cpmeas.IMAGE, IMAGE_FEATURE, image_number] = \
                    "Hello %d" % image_number
                m[OBJECTS_NAME, OBJECTS_FEATURE, image_number] = om
                req = cpanalysis.MeasurementsReport(
                        worker.analysis_id,
                        m.file_contents(),
                        image_set_numbers=[image_number])
                m.close()
                response = worker.send(req)()
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # AnalysisFinished event.
            #
            #####################################################

            self.check_display_post_run_requests(pipeline)
            result = self.event_queue.get()
            self.assertIsInstance(result, cpanalysis.AnalysisFinished)
            self.assertFalse(result.cancelled)
            measurements = result.measurements
            self.assertSequenceEqual(list(measurements.get_image_numbers()),
                                     [1, 2, 3])
            for i in range(1, 4):
                self.assertEqual(measurements[cpmeas.IMAGE, IMAGE_FEATURE, i],
                                 "Hello %d" % i)
                np.testing.assert_almost_equal(
                        measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                        objects_measurements[i - 1])

    def test_06_03_test_grouped_imagesets(self):
        # Test an analysis of four imagesets in two groups
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(
                nimage_sets=4,
                group_numbers=[1, 1, 2, 2],
                group_indexes=[1, 2, 1, 2])
        r = np.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            client_measurements = cpmeas.load_measurements_from_buffer(
                    response.buf)
            response = worker.send(cpanalysis.ImageSetSuccess(
                    worker.analysis_id, 1))()
            #####################################################
            #
            # Get the second group.
            #
            #####################################################
            response = worker.request_work()
            image_numbers = response.image_set_numbers
            self.assertSequenceEqual(list(image_numbers), [3, 4])
            #####################################################
            #
            # Send the measurement groups
            #
            #####################################################
            objects_measurements = [r.uniform(size=10) for _ in range(4)]
            for image_number in range(2, 5):
                worker.send(cpanalysis.ImageSetSuccess(
                        worker.analysis_id,
                        image_set_number=image_number))
            for image_numbers in ((1, 2), (3, 4)):
                m = cpmeas.Measurements(copy=client_measurements)
                for image_number in image_numbers:
                    m[cpmeas.IMAGE, IMAGE_FEATURE, image_number] = \
                        "Hello %d" % image_number
                    m[OBJECTS_NAME, OBJECTS_FEATURE, image_number] = \
                        objects_measurements[image_number - 1]
                req = cpanalysis.MeasurementsReport(
                        worker.analysis_id,
                        m.file_contents(),
                        image_set_numbers=image_numbers)
                m.close()
                response = worker.send(req)()
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # AnalysisFinished event.
            #
            #####################################################

            self.check_display_post_run_requests(pipeline)
            result = self.event_queue.get()
            self.assertIsInstance(result, cpanalysis.AnalysisFinished)
            self.assertFalse(result.cancelled)
            measurements = result.measurements
            self.assertSequenceEqual(list(measurements.get_image_numbers()),
                                     [1, 2, 3, 4])
            for i in range(1, 5):
                self.assertEqual(measurements[cpmeas.IMAGE, IMAGE_FEATURE, i],
                                 "Hello %d" % i)
                np.testing.assert_almost_equal(
                        measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                        objects_measurements[i - 1])

    def test_06_04_test_restart(self):
        # Test a restart of an analysis
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(
                nimage_sets=3,
                status=[cpanalysis.AnalysisRunner.STATUS_UNPROCESSED,
                        cpanalysis.AnalysisRunner.STATUS_DONE,
                        cpanalysis.AnalysisRunner.STATUS_IN_PROCESS])
        r = np.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            client_measurements = cpmeas.load_measurements_from_buffer(
                    response.buf)
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            dictionaries = [dict([(uuid.uuid4().hex, r.uniform(size=(10, 15)))
                                  for _ in range(10)])
                            for module in pipeline.modules()]
            response = worker.send(cpanalysis.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries))()
            #####################################################
            #
            # The analysis server should be ready to send us just
            # the third job.
            #
            #####################################################
            response = worker.request_work()
            image_numbers = response.image_set_numbers
            self.assertEqual(len(image_numbers), 1)
            self.assertEqual(image_numbers[0], 3)
            #####################################################
            #
            # Send the measurement groups
            #
            #####################################################
            objects_measurements = [r.uniform(size=10) for _ in range(3)]
            for image_number, om in ((1, objects_measurements[0]),
                                     (3, objects_measurements[2])):
                worker.send(cpanalysis.ImageSetSuccess(
                        worker.analysis_id,
                        image_set_number=image_number))
                m = cpmeas.Measurements(copy=client_measurements)
                m[cpmeas.IMAGE, IMAGE_FEATURE, image_number] = \
                    "Hello %d" % image_number
                m[OBJECTS_NAME, OBJECTS_FEATURE, image_number] = om
                req = cpanalysis.MeasurementsReport(
                        worker.analysis_id,
                        m.file_contents(),
                        image_set_numbers=[image_number])
                m.close()
                response = worker.send(req)()
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # AnalysisFinished event.
            #
            #####################################################

            self.check_display_post_run_requests(pipeline)
            result = self.event_queue.get()
            self.assertIsInstance(result, cpanalysis.AnalysisFinished)
            self.assertFalse(result.cancelled)
            measurements = result.measurements
            assert isinstance(measurements, cpmeas.Measurements)
            self.assertSequenceEqual(list(measurements.get_image_numbers()),
                                     [1, 2, 3])
            for i in range(1, 4):
                if i == 2:
                    for feature in (IMAGE_FEATURE, OBJECTS_FEATURE):
                        self.assertFalse(measurements.has_measurements(
                                cpmeas.IMAGE, feature, 2))
                else:
                    self.assertEqual(measurements[cpmeas.IMAGE, IMAGE_FEATURE, i],
                                     "Hello %d" % i)
                    np.testing.assert_almost_equal(
                            measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                            objects_measurements[i - 1])

    def test_06_05_test_grouped_restart(self):
        # Test an analysis of four imagesets in two groups with all but one
        # complete.
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(
                nimage_sets=4,
                group_numbers=[1, 1, 2, 2],
                group_indexes=[1, 2, 1, 2],
                status=[cpanalysis.AnalysisRunner.STATUS_DONE,
                        cpanalysis.AnalysisRunner.STATUS_UNPROCESSED,
                        cpanalysis.AnalysisRunner.STATUS_DONE,
                        cpanalysis.AnalysisRunner.STATUS_DONE]
        )
        r = np.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            self.assertSequenceEqual(response.image_set_numbers, [1, 2])
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            client_measurements = cpmeas.load_measurements_from_buffer(
                    response.buf)
            for image_number in (1, 2):
                response = worker.send(cpanalysis.ImageSetSuccess(
                        worker.analysis_id, image_number))()
            m = cpmeas.Measurements(copy=client_measurements)
            objects_measurements = [r.uniform(size=10) for _ in range(2)]
            for image_number in (1, 2):
                m[cpmeas.IMAGE, IMAGE_FEATURE, image_number] = \
                    "Hello %d" % image_number
                m[OBJECTS_NAME, OBJECTS_FEATURE, image_number] = \
                    objects_measurements[image_number - 1]
            req = cpanalysis.MeasurementsReport(
                    worker.analysis_id,
                    m.file_contents(),
                    image_set_numbers=(1, 2))
            m.close()
            response = worker.send(req)()
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # AnalysisFinished event.
            #
            #####################################################

            self.check_display_post_run_requests(pipeline)
            result = self.event_queue.get()
            self.assertIsInstance(result, cpanalysis.AnalysisFinished)
            self.assertFalse(result.cancelled)
            measurements = result.measurements
            for i in range(1, 3):
                self.assertEqual(measurements[cpmeas.IMAGE, IMAGE_FEATURE, i],
                                 "Hello %d" % i)
                np.testing.assert_almost_equal(
                        measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                        objects_measurements[i - 1])

    def test_06_06_relationships(self):
        #
        # Test a transfer of the relationships table.
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        r = np.random.RandomState()
        r.seed(61)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            client_measurements = cpmeas.load_measurements_from_buffer(
                    response.buf)
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            dictionaries = [dict([(uuid.uuid4().hex, r.uniform(size=(10, 15)))
                                  for _ in range(10)])
                            for module in pipeline.modules()]
            response = worker.send(cpanalysis.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries))()
            n_objects = 10
            objects_measurements = r.uniform(size=n_objects)
            objects_relationship = r.permutation(n_objects) + 1
            client_measurements[cpmeas.IMAGE, IMAGE_FEATURE, 1] = "Hello"
            client_measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1] = \
                objects_measurements
            client_measurements.add_relate_measurement(
                    1, "Foo", OBJECTS_NAME, OBJECTS_NAME,
                    np.ones(n_objects, int), np.arange(1, n_objects + 1),
                    np.ones(n_objects, int), objects_relationship)
            req = cpanalysis.MeasurementsReport(
                    worker.analysis_id,
                    client_measurements.file_contents(),
                    image_set_numbers=[1])
            client_measurements.close()
            response_fn = worker.send(req)

            self.check_display_post_run_requests(pipeline)
            #####################################################
            #
            # The server should receive the measurements report.
            # It should merge the measurements and post an
            # AnalysisFinished event.
            #
            #####################################################

            result = self.event_queue.get()
            self.assertIsInstance(result, cpanalysis.AnalysisFinished)
            self.assertFalse(result.cancelled)
            measurements = result.measurements
            assert isinstance(measurements, cpmeas.Measurements)
            self.assertSequenceEqual(measurements.get_image_numbers(), [1])
            self.assertEqual(measurements[cpmeas.IMAGE, IMAGE_FEATURE, 1],
                             "Hello")
            np.testing.assert_almost_equal(
                    measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1],
                    objects_measurements)
            rg = measurements.get_relationship_groups()
            self.assertEqual(len(rg), 1)
            rk = rg[0]
            assert isinstance(rk, cpmeas.RelationshipKey)
            self.assertEqual(rk.module_number, 1)
            self.assertEqual(rk.object_name1, OBJECTS_NAME)
            self.assertEqual(rk.object_name2, OBJECTS_NAME)
            self.assertEqual(rk.relationship, "Foo")
            r = measurements.get_relationships(
                    1, "Foo", OBJECTS_NAME, OBJECTS_NAME)
            self.assertEqual(len(r), n_objects)
            np.testing.assert_array_equal(r[cpmeas.R_FIRST_IMAGE_NUMBER], 1)
            np.testing.assert_array_equal(r[cpmeas.R_SECOND_IMAGE_NUMBER], 1)
            np.testing.assert_array_equal(r[cpmeas.R_FIRST_OBJECT_NUMBER],
                                          np.arange(1, n_objects + 1))
            np.testing.assert_array_equal(r[cpmeas.R_SECOND_OBJECT_NUMBER],
                                          objects_relationship)

    def test_06_07_worker_cancel(self):
        #
        # Test worker sending AnalysisCancelRequest
        #
        logger.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        r = np.random.RandomState()
        r.seed(61)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            worker.connect(self.analysis.runner.work_announce_address)
            response = worker.request_work()
            response = worker.send(cpanalysis.InitialMeasurementsRequest(
                    worker.analysis_id))()
            #####################################################
            #
            # The worker sends an AnalysisCancelRequest. The
            # server should send AnalysisFinished.
            #
            #####################################################

            response = worker.send(cpanalysis.AnalysisCancelRequest(
                    worker.analysis_id))()
            result = self.event_queue.get()
            self.assertIsInstance(result, cpanalysis.AnalysisFinished)
            self.assertTrue(result.cancelled)


SBS_PIPELINE = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120424205644
ModuleCount:8
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    :
    Filter based on rules:No
    Filter:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:Yes
    Extraction method count:1
    Extraction method:Manual
    Source:From file name
    Regular expression:Channel(?P<C>\x5B12\x5D)-\x5B0-9\x5D{2}-(?P<WellRow>\x5BA-H\x5D)-(?P<WellColumn>\x5B0-9\x5D{2})
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Assignment method:Assign images matching rules
    Load as:Grayscale image
    Image name:DNA
    :\x5B{u\'Cytoplasm\'\x3A u\'WellRow\', u\'DNACorr\'\x3A None, \'DNA\'\x3A u\'WellRow\', u\'CytoplasmCorr\'\x3A None}, {u\'Cytoplasm\'\x3A u\'WellColumn\', u\'DNACorr\'\x3A None, \'DNA\'\x3A u\'WellColumn\', u\'CytoplasmCorr\'\x3A None}\x5D
    Match channels by:Metadata
    Assignments count:4
    Match this rule:and (extension does istif) (metadata does C "2")
    Image name:DNA
    Objects name:Cells
    Load as:Grayscale image
    Match this rule:and (extension does istif) (metadata does C "1")
    Image name:Cytoplasm
    Objects name:Cells
    Load as:Grayscale image
    Match this rule:or (file does startwith "Channel1ILLUM")
    Image name:DNACorr
    Objects name:Cells
    Load as:Grayscale image
    Match this rule:or (file does contain "Channel2ILLUM")
    Image name:CytoplasmCorr
    Objects name:Cells
    Load as:Grayscale image

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Do you want to group your images?:No
    grouping metadata count:1
    Image name:DNA
    Metadata category:None

CorrectIlluminationApply:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input image:Cytoplasm
    Name the output image:CorrCytoplasm
    Select the illumination function:CytoplasmCorr
    Select how the illumination function is applied:Divide

CorrectIlluminationApply:[module_num:6|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input image:DNA
    Name the output image:CorrDNA
    Select the illumination function:DNACorr
    Select how the illumination function is applied:Divide

IdentifyPrimaryObjects:[module_num:7|svn_version:\'Unknown\'|variable_revision_number:9|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input image:CorrDNA
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):10,40
    Discard objects outside the diameter range?:Yes
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:Yes
    Select the thresholding method:Otsu Global
    Threshold correction factor:1
    Lower and upper bounds on threshold:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:7
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Manual threshold:0.0
    Select binary image:None
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:0.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Select the measurement to threshold with:None
    Method to calculate adaptive window size:Image size
    Size of adaptive window:10

IdentifySecondaryObjects:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:8|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input objects:Nuclei
    Name the objects to be identified:Cells
    Select the method to identify the secondary objects:Propagation
    Select the input image:CorrCytoplasm
    Select the thresholding method:Otsu Global
    Threshold correction factor:1
    Lower and upper bounds on threshold:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Number of pixels by which to expand the primary objects:10
    Regularization factor:0.05
    Name the outline image:SecondaryOutlines
    Manual threshold:0.0
    Select binary image:None
    Retain outlines of the identified secondary objects?:No
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Discard secondary objects touching the border of the image?:No
    Discard the associated primary objects?:No
    Name the new primary objects:FilteredNuclei
    Retain outlines of the new primary objects?:No
    Name the new primary object outlines:FilteredNucleiOutlines
    Select the measurement to threshold with:None
    Fill holes in identified objects?:Yes
    Method to calculate adaptive window size:Image size
    Size of adaptive window:10
"""
