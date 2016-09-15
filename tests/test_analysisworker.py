"""test_analysisworker.py - test the analysis client framework"""

import Queue
import cStringIO
import os
import tempfile
import threading
import traceback
import unittest
import uuid

import cellprofiler.analysis
import cellprofiler.gui.errordialog
import cellprofiler.measurement
import cellprofiler.modules.identify
import cellprofiler.modules.loadimages
import cellprofiler.modules.namesandtypes
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.utilities.zmqrequest
import cellprofiler.worker
import javabridge
import numpy
import tests.modules
import zmq


class TestAnalysisWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zmq_context = cellprofiler.worker.the_zmq_context
        cls.notify_pub_socket = cellprofiler.worker.get_the_notify_pub_socket()
        #
        # Install a bogus display_post_group method in FlipAndRotate
        # to elicit a post-group interaction request
        #
        from cellprofiler.modules.flipandrotate import FlipAndRotate
        def bogus_display_post_group(self, workspace, figure):
            pass

        FlipAndRotate.display_post_group = bogus_display_post_group
        tests.modules.maybe_download_sbs()

    @classmethod
    def tearDownClass(cls):
        cls.notify_pub_socket.close()

    def cancel(self):
        self.notify_pub_socket.send(cellprofiler.worker.NOTIFY_STOP)

    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        cellprofiler.preferences.set_default_output_directory(self.out_dir)
        self.announce_addr = "inproc://" + uuid.uuid4().hex
        self.work_addr = "inproc://" + uuid.uuid4().hex
        self.announce_socket = self.zmq_context.socket(zmq.PUB)
        self.announce_socket.bind(self.announce_addr)
        self.work_socket = self.zmq_context.socket(zmq.REP)
        self.work_socket.bind(self.work_addr)
        self.awthread = None

    def tearDown(self):
        if self.awthread:
            self.cancel()
            self.awthread.down_queue.put(None)
            self.awthread.join(10000)
            self.assertFalse(self.awthread.isAlive())
        self.work_socket.close()
        self.announce_socket.close()
        #
        # No .h5 mouseturds
        #
        h5_files = [f for f in os.listdir(self.out_dir)
                    if f.endswith(".h5")]
        self.assertEqual(len(h5_files), 0,
                         msg="Left the following files: " + str(h5_files))

    class AWThread(threading.Thread):

        def __init__(self, announce_addr, *args, **kwargs):
            threading.Thread.__init__(self, *args, **kwargs)
            self.announce_addr = announce_addr
            self.cancelled = False

        def start(self):
            self.setDaemon(True)
            self.setName("Analysis worker thread")
            self.up_queue = Queue.Queue()
            self.notify_addr = "inproc://" + uuid.uuid4().hex
            self.up_queue_recv_socket = cellprofiler.worker.the_zmq_context.socket(zmq.SUB)
            self.up_queue_recv_socket.setsockopt(zmq.SUBSCRIBE, "")
            self.up_queue_recv_socket.bind(self.notify_addr)
            self.down_queue = Queue.Queue()
            threading.Thread.start(self)
            self.up_queue.get()

        def run(self):
            up_queue_send_socket = cellprofiler.worker.the_zmq_context.socket(zmq.PUB)
            up_queue_send_socket.connect(self.notify_addr)
            with cellprofiler.worker.AnalysisWorker(self.announce_addr,
                                                    with_stop_run_loop=False) as aw:
                aw.enter_thread()
                self.aw = aw
                self.up_queue.put("OK")
                while True:
                    fn = self.down_queue.get()
                    if fn is None:
                        break
                    try:
                        result = fn()
                        self.up_queue.put((result, None))
                        up_queue_send_socket.send("OK")
                    except Exception, e:
                        traceback.print_exc()
                        self.up_queue.put((None, e))
                        up_queue_send_socket.send("EXCEPTION")
                aw.exit_thread()

        def recv(self, work_socket, timeout=None):
            '''Receive a request from the worker

            work_socket - receive a request on this socket

            timeout - if request isn't received by the timeout, raise Queue.Empty
                      default = blocks forever

            This polls on both the worker and up_queue sockets and
            will throw an exception if there is anything available on
            the up-queue as this indicates that nothing is running.
            '''
            poller = zmq.Poller()
            poller.register(self.up_queue_recv_socket, zmq.POLLIN)
            poller.register(work_socket, zmq.POLLIN)
            for socket, state in poller.poll(timeout):
                if socket == self.up_queue_recv_socket and state == zmq.POLLIN:
                    result, e = self.up_queue.get()
                    if e is not None:
                        raise e
                    else:
                        raise cellprofiler.pipeline.CancelledException("Unexpected exit during recv")
                if socket == work_socket and state == zmq.POLLIN:
                    return cellprofiler.utilities.zmqrequest.Communicable.recv(work_socket)
            raise Queue.Empty

        def join(self, timeout=None):
            if self.isAlive():
                def cancel_me():
                    self.aw.cancelled = True

                self.down_queue.put(cancel_me)
                threading.Thread.join(self, timeout)

        def execute(self, fn, *args, **kwargs):
            '''Execute a closure on the AnalysisWorker thread

            fn - closure to execute

            Returns the function's result or throws whatever exception
            was thrown by the function.
            '''
            self.ex(fn, *args, **kwargs)
            return self.ecute()

        def ex(self, fn, *args, **kwargs):
            '''Do the first part of a functional execution'''
            if len(args) == 0 and len(kwargs) == 0:
                self.down_queue.put(fn)
            else:
                def closure():
                    return fn(*args, **kwargs)

                self.down_queue.put(closure)

        def ecute(self):
            '''Retrieve the results of self.ex()'''
            msg = self.up_queue_recv_socket.recv()
            result, e = self.up_queue.get()
            if e is not None:
                raise e
            return result

    def set_work_socket(self):
        '''Artificially set up the worker's work socket

        This sets self.aw.work_socket so that methods other than "run"
        can be tested in the worker.
        '''
        self.analysis_id = uuid.uuid4().hex

        def do_set_work_socket(aw):
            aw.work_socket = cellprofiler.worker.the_zmq_context.socket(zmq.REQ)
            aw.work_socket.connect(self.work_addr)
            aw.work_request_address = self.work_addr
            aw.current_analysis_id = self.analysis_id

        self.awthread.execute(do_set_work_socket, self.awthread.aw)

    def send_announcement_get_work_request(self):
        '''Announce the work address until we get some sort of a request'''
        self.analysis_id = uuid.uuid4().hex
        while True:
            self.announce_socket.send_json(
                    ((self.analysis_id, self.work_addr),))
            try:
                return self.awthread.recv(self.work_socket, 250)
            except Queue.Empty:
                continue

    def test_01_01_get_announcement(self):
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.awthread.ex(self.awthread.aw.get_announcement)
        while True:
            self.announce_socket.send_json((("foo", "bar"),))
            try:
                result, exception = self.awthread.up_queue.get_nowait()
                break
            except Queue.Empty:
                continue

        self.assertIsNone(exception)
        self.assertSequenceEqual(result, ("foo", "bar"))

    def test_01_02_announcement_cancellation(self):
        #
        # Call AnalysisWorker.get_announcement, then notify the worker
        # that it should exit.
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.awthread.ex(self.awthread.aw.get_announcement)
        self.cancel()
        self.assertRaises(cellprofiler.pipeline.CancelledException, self.awthread.ecute)

    def test_02_01_send(self):
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        def send_something():
            reply = self.awthread.aw.send(cellprofiler.analysis.WorkRequest("foo"))
            return reply
        self.awthread.ex(send_something)
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.WorkRequest)
        self.assertEqual(req.analysis_id, "foo")
        req.reply(cellprofiler.analysis.WorkReply(foo="bar"))
        reply = self.awthread.ecute()
        self.assertIsInstance(reply, cellprofiler.analysis.WorkReply)
        self.assertEqual(reply.foo, "bar")

    def test_02_02_send_cancellation(self):
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        def send_something():
            reply = self.awthread.aw.send(cellprofiler.analysis.WorkRequest("foo"))
            return reply
        self.awthread.ex(send_something)
        self.cancel()
        self.assertRaises(cellprofiler.pipeline.CancelledException, self.awthread.ecute)

    def test_02_03_send_upstream_exit(self):
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        def send_something():
            reply = self.awthread.aw.send(cellprofiler.analysis.WorkRequest("foo"))
            return reply
        self.awthread.ex(send_something)
        req = self.awthread.recv(self.work_socket)
        req.reply(cellprofiler.analysis.ServerExited())
        self.assertRaises(cellprofiler.pipeline.CancelledException, self.awthread.ecute)

    def test_03_01_work_request(self):
        #
        # Walk the worker through the connect sequence through
        # WorkRequest, then kill it.
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.awthread.ex(self.awthread.aw.run)
        #
        # Get the work request
        #
        req = self.send_announcement_get_work_request()
        self.assertEqual(self.analysis_id, req.analysis_id)

    def test_03_02_pipeline_preferences(self):
        #
        # Walk the worker up through pipelines and preferences.
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        self.awthread.ex(self.awthread.aw.do_job,
                         cellprofiler.analysis.WorkReply(
                             image_set_numbers = [1],
                             worker_runs_post_group = False,
                             wants_dictionary = True))
        #
        # The worker should ask for the pipeline and preferences next.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.PipelinePreferencesRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)

        tests.modules.maybe_download_example_image(["ExampleSBSImages"],
                                     "Channel1-01-A-01.tif")
        tests.modules.maybe_download_example_image(["ExampleHT29"],
                                     "AS_09125_050116030001_D03f00d0.tif")
        input_dir = os.path.normcase(
            os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages"))
        output_dir = os.path.normcase(
            os.path.join(tests.modules.example_images_directory(), "ExampleHT29"))
        cellprofiler.preferences.set_default_image_directory(input_dir)
        input_dir = cellprofiler.preferences.get_default_image_directory()
        cellprofiler.preferences.set_default_output_directory(output_dir)
        output_dir = cellprofiler.preferences.get_default_output_directory()
        preferences = {cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY:
                       cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY),
                       cellprofiler.preferences.DEFAULT_OUTPUT_DIRECTORY:
                       cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_OUTPUT_DIRECTORY)}
        cellprofiler.preferences.set_default_image_directory(tests.modules.example_images_directory())
        cellprofiler.preferences.set_default_output_directory(tests.modules.example_images_directory())
        rep = cellprofiler.analysis.Reply(
            pipeline_blob = numpy.array(GOOD_PIPELINE),
            preferences = preferences)
        req.reply(rep)
        #
        # Get the next request so that we know the worker has
        # processed the preferences.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertEqual(cellprofiler.preferences.get_default_image_directory(),
                         input_dir)
        self.assertEqual(cellprofiler.preferences.get_default_output_directory(),
                         output_dir)
        self.assertIn(self.analysis_id,
                      self.awthread.aw.pipelines_and_preferences)
        pipe, prefs = self.awthread.aw.pipelines_and_preferences[
            self.analysis_id]
        self.assertEqual(len(pipe.modules()), 7)
        #
        # Cancel and check for exit
        #
        req.reply(cellprofiler.analysis.ServerExited())
        self.assertRaises(cellprofiler.pipeline.CancelledException, self.awthread.ecute)

    def test_03_03_initial_measurements(self):
        #
        # Walk to the initial measurements
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        self.awthread.ex(self.awthread.aw.do_job,
                         cellprofiler.analysis.WorkReply(
                             image_set_numbers = [1],
                             worker_runs_post_group = False,
                             wants_dictionary = True))
        #
        # The worker should ask for the pipeline and preferences next.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.PipelinePreferencesRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)

        input_dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
        cellprofiler.preferences.set_default_image_directory(input_dir)
        preferences = {cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY:
                       cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY)}

        rep = cellprofiler.analysis.Reply(
            pipeline_blob = numpy.array(GOOD_PIPELINE),
            preferences = preferences)
        req.reply(rep)
        #
        # The worker asks for the initial measurements.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.InitialMeasurementsRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)
        m = get_measurements_for_good_pipeline()
        try:
            req.reply(cellprofiler.analysis.Reply(buf = m.file_contents()))
            req = self.awthread.recv(self.work_socket)
            #
            # Check that they were installed
            #
            self.assertIn(self.analysis_id,
                          self.awthread.aw.initial_measurements)
            cm = self.awthread.aw.initial_measurements[self.analysis_id]
            for object_name in m.get_object_names():
                for feature_name in m.get_feature_names(object_name):
                    self.assertTrue(cm.has_feature(object_name, feature_name))
                    if feature_name == cellprofiler.modules.namesandtypes.M_IMAGE_SET:
                        numpy.testing.assert_array_equal(
                            cm[object_name, feature_name, 1],
                                     m[object_name, feature_name, 1])
                    else:
                        self.assertEqual(cm[object_name, feature_name, 1],
                                         m[object_name, feature_name, 1])
            #
            # Cancel and check for exit
            #
            req.reply(cellprofiler.analysis.ServerExited())
            self.assertRaises(cellprofiler.pipeline.CancelledException, self.awthread.ecute)
        finally:
            m.close()

    def test_03_04_shared_dictionary_request(self):
        #
        # The SharedDictionaryRequest
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        self.awthread.ex(self.awthread.aw.do_job,
                         cellprofiler.analysis.WorkReply(
                             image_set_numbers = [1],
                             worker_runs_post_group = False,
                             wants_dictionary = True))
        #
        # The worker should ask for the pipeline and preferences next.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.PipelinePreferencesRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)

        input_dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
        cellprofiler.preferences.set_default_image_directory(input_dir)
        preferences = {cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY:
                       cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY)}

        rep = cellprofiler.analysis.Reply(
            pipeline_blob = numpy.array(DISPLAY_PIPELINE),
            preferences = preferences)
        req.reply(rep)
        #
        # The worker asks for the initial measurements.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.InitialMeasurementsRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)
        m = get_measurements_for_good_pipeline()
        try:
            req.reply(cellprofiler.analysis.Reply(buf = m.file_contents()))
        finally:
            m.close()
        #
        # Next, the worker asks for the shared dictionary
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.SharedDictionaryRequest)
        rep = cellprofiler.analysis.SharedDictionaryReply(
            dictionaries = [{ ("foo%d" % i):"bar%d" % i} for i in range(1,8)])
        req.reply(rep)
        #
        # Sneaky way to get pipeline. First, synchronize with the next message
        #
        req = self.awthread.recv(self.work_socket)
        pipe, prefs = self.awthread.aw.pipelines_and_preferences[
            self.analysis_id]
        for d, module in zip(rep.dictionaries, pipe.modules()):
            self.assertDictEqual(module.get_dictionary(), d)
        #
        # Might as well torpedo the app. It should be stalled waiting
        # for the FlipAndRotate display.
        #
        self.cancel()
        self.awthread.ecute()

    def test_03_05_the_happy_path_chapter_1(self):
        #
        # Run the worker clear through to the end
        # for the first imageset
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        self.awthread.ex(self.awthread.aw.do_job,
                         cellprofiler.analysis.WorkReply(
                             image_set_numbers = [1],
                             worker_runs_post_group = False,
                             wants_dictionary = True))
        #
        # The worker should ask for the pipeline and preferences next.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.PipelinePreferencesRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)

        input_dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
        cellprofiler.preferences.set_default_image_directory(input_dir)
        preferences = {cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY:
                       cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY)}

        rep = cellprofiler.analysis.Reply(
            pipeline_blob = numpy.array(DISPLAY_PIPELINE),
            preferences = preferences)
        req.reply(rep)
        #
        # The worker asks for the initial measurements.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.InitialMeasurementsRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)
        m = get_measurements_for_good_pipeline()
        try:
            req.reply(cellprofiler.analysis.Reply(buf = m.file_contents()))
        finally:
            m.close()
        #
        # Next, the worker asks for the shared dictionary
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.SharedDictionaryRequest)
        shared_dictionaries = [{ ("foo%d" % i):"bar%d" % i} for i in range(1,8)]
        rep = cellprofiler.analysis.SharedDictionaryReply(
            dictionaries = shared_dictionaries)
        req.reply(rep)
        #
        # The worker sends a display request for FlipAndRotate
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.DisplayRequest)
        self.assertEqual(req.image_set_number, 1)
        d = req.display_data_dict
        # Possibly, this will break if someone edits FlipAndRotate. Sorry.
        self.assertItemsEqual(d.keys(),
                              ['vmax', 'output_image_pixel_data',
                               'image_pixel_data', 'vmin'])
        self.assertIsInstance(d['output_image_pixel_data'], numpy.ndarray)
        req.reply(cellprofiler.analysis.Ack())
        #
        # The worker sends ImageSetSuccessWithDictionary.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.ImageSetSuccessWithDictionary)
        self.assertEqual(req.image_set_number, 1)
        for expected, actual in zip(shared_dictionaries, req.shared_dicts):
            self.assertDictEqual(expected, actual)
        req.reply(cellprofiler.analysis.Ack())
        #
        # The worker sends the measurement report
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.MeasurementsReport)
        self.assertSequenceEqual(req.image_set_numbers, [1])
        m = cellprofiler.measurement.load_measurements_from_buffer(req.buf)
        #
        # Spot check for some expected stuff
        #
        self.assertTrue(m.has_feature(cellprofiler.measurement.IMAGE, cellprofiler.modules.identify.C_COUNT + "_Nuclei"))
        self.assertTrue(m.has_feature("Nuclei", cellprofiler.modules.identify.M_LOCATION_CENTER_X))
        self.assertTrue(m.has_feature("Nuclei", "AreaShape_Area"))
        req.reply(cellprofiler.analysis.Ack())
        self.awthread.ecute()

    def test_03_06_the_happy_path_chapter_2(self):
        #
        # Give the worker image sets # 2 and 3 and tell it to run post_group
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        self.awthread.ex(self.awthread.aw.do_job,
                         cellprofiler.analysis.WorkReply(
                             image_set_numbers = [2, 3],
                             worker_runs_post_group = True,
                             wants_dictionary = False))
        #
        # The worker should ask for the pipeline and preferences next.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.PipelinePreferencesRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)

        input_dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
        cellprofiler.preferences.set_default_image_directory(input_dir)
        preferences = {cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY:
                       cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY)}

        rep = cellprofiler.analysis.Reply(
            pipeline_blob = numpy.array(DISPLAY_PIPELINE),
            preferences = preferences)
        req.reply(rep)
        #
        # The worker asks for the initial measurements.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.InitialMeasurementsRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)
        m = get_measurements_for_good_pipeline(nimages=3)
        try:
            req.reply(cellprofiler.analysis.Reply(buf = m.file_contents()))
        finally:
            m.close()
        #
        # In group mode, the worker issues a display request and constructs
        # its own dictonaries
        #
        for image_number in (2, 3):
            #
            # The worker sends a display request for FlipAndRotate
            #
            req = self.awthread.recv(self.work_socket)
            self.assertIsInstance(req, cellprofiler.analysis.DisplayRequest)
            req.reply(cellprofiler.analysis.Ack())
            #
            # The worker sends ImageSetSuccess.
            #
            req = self.awthread.recv(self.work_socket)
            self.assertIsInstance(req, cellprofiler.analysis.ImageSetSuccess)
            self.assertEqual(req.image_set_number, image_number)
            req.reply(cellprofiler.analysis.Ack())
        #
        # The worker sends a DisplayPostGroup request for FlipAndRotate
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.DisplayPostGroupRequest)
        self.assertEqual(req.image_set_number, 3)
        req.reply(cellprofiler.analysis.Ack())
        #
        # The worker sends a measurement report for image sets 2 and 3
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.MeasurementsReport)
        self.assertSequenceEqual(req.image_set_numbers, [2, 3])
        m = cellprofiler.measurement.load_measurements_from_buffer(req.buf)
        #
        # Spot check for some expected stuff
        #
        self.assertTrue(m.has_feature(cellprofiler.measurement.IMAGE, cellprofiler.modules.identify.C_COUNT + "_Nuclei"))
        self.assertTrue(m.has_feature("Nuclei", cellprofiler.modules.identify.M_LOCATION_CENTER_X))
        self.assertTrue(m.has_feature("Nuclei", "AreaShape_Area"))
        req.reply(cellprofiler.analysis.Ack())
        self.awthread.ecute()

    def test_03_08_a_sad_moment(self):
        #
        # Run using the good pipeline, but change one of the URLs so
        # an exception is thrown.
        #
        self.awthread = self.AWThread(self.announce_addr)
        self.awthread.start()
        self.set_work_socket()
        self.awthread.ex(self.awthread.aw.do_job,
                         cellprofiler.analysis.WorkReply(
                             image_set_numbers = [2, 3],
                             worker_runs_post_group = False,
                             wants_dictionary = False))
        #
        # The worker should ask for the pipeline and preferences next.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.PipelinePreferencesRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)

        input_dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
        cellprofiler.preferences.set_default_image_directory(input_dir)
        preferences = {cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY:
                       cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY)}

        rep = cellprofiler.analysis.Reply(
            pipeline_blob = numpy.array(GOOD_PIPELINE),
            preferences = preferences)
        req.reply(rep)
        #
        # The worker asks for the initial measurements.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.InitialMeasurementsRequest)
        self.assertEqual(req.analysis_id, self.analysis_id)
        m = get_measurements_for_good_pipeline(nimages=3)
        m[cellprofiler.measurement.IMAGE, cellprofiler.modules.namesandtypes.M_IMAGE_SET, 2] = numpy.zeros(100, numpy.uint8)
        try:
            req.reply(cellprofiler.analysis.Reply(buf = m.file_contents()))
        finally:
            m.close()
        #
        # Next, the worker asks for the shared dictionary
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.SharedDictionaryRequest)
        shared_dictionaries = [{ ("foo%d" % i):"bar%d" % i} for i in range(1,8)]
        rep = cellprofiler.analysis.SharedDictionaryReply(
            dictionaries = shared_dictionaries)
        req.reply(rep)
        #
        # The worker should choke somewhere in NamesAndTypes, but we
        # tell the worker to skip the rest of the imageset.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.ExceptionReport)
        req.reply(cellprofiler.analysis.ExceptionPleaseDebugReply(disposition = cellprofiler.gui.errordialog.ED_SKIP))
        #
        # The worker should send ImageSetSuccess for image set 2 anyway.
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.ImageSetSuccess)
        self.assertEqual(req.image_set_number, 2)
        req.reply(cellprofiler.analysis.Ack())
        #
        # And then it tells us about image set 3
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.ImageSetSuccess)
        self.assertEqual(req.image_set_number, 3)
        req.reply(cellprofiler.analysis.Ack())
        #
        # The worker should then report the measurements for both 2 and 3
        #
        req = self.awthread.recv(self.work_socket)
        self.assertIsInstance(req, cellprofiler.analysis.MeasurementsReport)
        self.assertSequenceEqual(req.image_set_numbers, [2, 3])
        m = cellprofiler.measurement.load_measurements_from_buffer(req.buf)
        #
        # Spot check for some expected stuff
        #
        self.assertTrue(m.has_feature(cellprofiler.measurement.IMAGE, cellprofiler.modules.identify.C_COUNT + "_Nuclei"))
        self.assertTrue(m.has_feature("Nuclei", cellprofiler.modules.identify.M_LOCATION_CENTER_X))
        self.assertTrue(m.has_feature("Nuclei", "AreaShape_Area"))
        #
        # The count for the skipped image should be None
        #
        count = m[cellprofiler.measurement.IMAGE, cellprofiler.modules.identify.C_COUNT + "_Nuclei", 2]
        self.assertIsNone(count)
        count = m[cellprofiler.measurement.IMAGE, cellprofiler.modules.identify.C_COUNT + "_Nuclei", 3]
        center_x = m["Nuclei", cellprofiler.modules.identify.M_LOCATION_CENTER_X, 3]
        self.assertEqual(count, len(center_x))
        req.reply(cellprofiler.analysis.Ack())
        self.awthread.ecute()


    def test_03_09_flag_image_abort(self):
                #
                # Regression test of issue #1210
                # Make a pipeline that aborts during FlagImage
                #
                data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
        Version:3
        DateRevision:20140918122611
        GitHash:ded6939
        ModuleCount:6
        HasImagePlaneDetails:False

        Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'To begin creating your project, use the Images module to compile a list of files and/or folders that you want to analyze. You can also specify a set of rules to include only the desired files in your selected folders.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
            :
            Filter images?:Images only
            Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "\x5B\\\\\\\\\\\\\\\\/\x5D\\\\\\\\.")

        Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\'The Metadata module optionally allows you to extract information describing your images (i.e, metadata) which will be stored along with your measurements. This information can be contained in the file name and/or location, or in an external file.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
            Extract metadata?:No
            Metadata data type:Text
            Metadata types:{}
            Extraction method count:1
            Metadata extraction method:Extract from file/folder names
            Metadata source:File name
            Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
            Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
            Extract metadata from:All images
            Select the filtering criteria:and (file does contain "")
            Metadata file location:
            Match file and image metadata:\x5B\x5D
            Use case insensitive matching?:No

        NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:5|show_window:False|notes:\x5B\'The NamesAndTypes module allows you to assign a meaningful name to each image by which other modules will refer to it.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
            Assign a name to:All images
            Select the image type:Grayscale image
            Name to assign these images:DNA
            Match metadata:\x5B\x5D
            Image set matching method:Order
            Set intensity range from:Image metadata
            Assignments count:1
            Single images count:0
            Select the rule criteria:and (file does contain "")
            Name to assign these images:DNA
            Name to assign these objects:Cell
            Select the image type:Grayscale image
            Set intensity range from:Image metadata
            Retain outlines of loaded objects?:No
            Name the outline image:LoadedOutlines

        Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'The Groups module optionally allows you to split your list of images into image subsets (groups) which will be processed independently of each other. Examples of groupings include screening batches, microtiter plates, time-lapse movies, etc.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
            Do you want to group your images?:No
            grouping metadata count:1
            Metadata category:None

        FlagImage:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
            Hidden:1
            Hidden:1
            Name the flag\'s category:Metadata
            Name the flag:QCFlag
            Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
            Skip image set if flagged?:Yes
            Flag is based on:Whole-image measurement
            Select the object to be used for flagging:None
            Which measurement?:Height_DNA
            Flag images based on low values?:No
            Minimum value:0.0
            Flag images based on high values?:Yes
            Maximum value:1.0
            Rules file location:Elsewhere...\x7C
            Rules file name:rules.txt
            Class number:

        MeasureImageIntensity:[module_num:6|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
            Select the image to measure:DNA
            Measure the intensity only from areas enclosed by objects?:No
            Select the input objects:None
        """
                self.awthread = self.AWThread(self.announce_addr)
                self.awthread.start()
                self.set_work_socket()
                self.awthread.ex(self.awthread.aw.do_job,
                                 cellprofiler.analysis.WorkReply(
                                     image_set_numbers = [1],
                                     worker_runs_post_group = False,
                                     wants_dictionary = True))
                #
                # The worker should ask for the pipeline and preferences next.
                #
                req = self.awthread.recv(self.work_socket)
                self.assertIsInstance(req, cellprofiler.analysis.PipelinePreferencesRequest)
                self.assertEqual(req.analysis_id, self.analysis_id)

                input_dir = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
                cellprofiler.preferences.set_default_image_directory(input_dir)
                preferences = {cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY:
                               cellprofiler.preferences.config_read(cellprofiler.preferences.DEFAULT_IMAGE_DIRECTORY)}

                rep = cellprofiler.analysis.Reply(
                    pipeline_blob = numpy.array(data),
                    preferences = preferences)
                req.reply(rep)
                #
                # The worker asks for the initial measurements.
                #
                req = self.awthread.recv(self.work_socket)
                self.assertIsInstance(req, cellprofiler.analysis.InitialMeasurementsRequest)
                self.assertEqual(req.analysis_id, self.analysis_id)
                m = get_measurements_for_good_pipeline()
                pipeline = cellprofiler.pipeline.Pipeline()
                pipeline.loadtxt(cStringIO.StringIO(data))
                pipeline.write_pipeline_measurement(m)

                try:
                    req.reply(cellprofiler.analysis.Reply(buf = m.file_contents()))
                finally:
                    m.close()
                #
                # Next, the worker asks for the shared dictionary
                #
                req = self.awthread.recv(self.work_socket)
                self.assertIsInstance(req, cellprofiler.analysis.SharedDictionaryRequest)
                shared_dictionaries = [{ ("foo%d" % i):"bar%d" % i} for i in range(1,7)]
                rep = cellprofiler.analysis.SharedDictionaryReply(
                    dictionaries = shared_dictionaries)
                req.reply(rep)
                #
                # MeasureImageIntensity follows FlagImage and it is poised to ask
                # for a display. So if we get that, we know the module has been run
                # and we fail the test.
                #
                req = self.awthread.recv(self.work_socket)
                self.assertFalse(isinstance(req, cellprofiler.analysis.DisplayRequest))
                self.assertFalse(isinstance(req, cellprofiler.analysis.ExceptionReport))


GOOD_PIPELINE = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120712182756
ModuleCount:7
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    :
    Filter based on rules:No
    Filter:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:No
    Extraction method count:1
    Extraction method:Automatic
    Source:From file name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Assignment method:Assign all images
    Load as:Grayscale image
    Image name:DNA
    :\x5B\x5D
    Assign channels by:Order
    Assignments count:1
    Match this rule:or (file does contain "")
    Image name:DNA
    Objects name:Cell
    Load as:Grayscale image

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Do you want to group your images?:No
    grouping metadata count:1
    Image name:DNA
    Metadata category:None

FlipAndRotate:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input image:DNA
    Name the output image:DNACopy
    Select method to flip image:Do not flip
    Select method to rotate image:Do not rotate
    Crop away the rotated edges?:Yes
    Calculate rotation:Individually
    Enter coordinates of the top or left pixel:0,0
    Enter the coordinates of the bottom or right pixel:0,100
    Select how the specified points should be aligned:horizontally
    Enter angle of rotation:0

IdentifyPrimaryObjects:[module_num:6|svn_version:\'Unknown\'|variable_revision_number:9|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select the input image:DNA
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

MeasureObjectSizeShape:[module_num:7|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Select objects to measure:Nuclei
    Calculate the Zernike features?:Yes
"""

'''This pipeline should raise an exception when NamesAndTypes is run'''
BAD_PIPELINE = GOOD_PIPELINE.replace("Image name:DNA", "Image name:RNA")

'''This pipeline should issue a DisplayRequest when FlipAndRotate is run'''
DISPLAY_PIPELINE = GOOD_PIPELINE.replace(
        r"[module_num:5|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False",
        r"[module_num:5|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True")


def get_measurements_for_good_pipeline(nimages=1,
                                       group_numbers=None):
    '''Get an appropriately initialized measurements structure for the good pipeline'''
    path = os.path.join(tests.modules.example_images_directory(), "ExampleSBSImages")
    m = cellprofiler.measurement.Measurements()
    if group_numbers is None:
        group_numbers = [1] * nimages
    group_indexes = [1]
    last_group_number = group_numbers[0]
    group_index = 1
    for group_number in group_numbers:
        if group_number == last_group_number:
            group_index += 1
        else:
            group_index = 1
        group_indexes.append(group_index)
    for i in range(1, nimages + 1):
        filename = ("Channel2-%02d-%s-%02d.tif" %
                    (i, "ABCDEFGH"[int((i - 1) / 12)], ((i - 1) % 12) + 1))
        url = cellprofiler.modules.loadimages.pathname2url(os.path.join(path, filename))
        m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_FILE_NAME + "_DNA", i] = filename
        m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_PATH_NAME + "_DNA", i] = path
        m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_URL + "_DNA", i] = url
        m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_NUMBER, i] = group_numbers[i - 1]
        m[cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_INDEX, i] = group_indexes[i - 1]
        jblob = javabridge.run_script("""
        importPackage(Packages.org.cellprofiler.imageset);
        importPackage(Packages.org.cellprofiler.imageset.filter);
        var imageFile=new ImageFile(new java.net.URI(url));
        var imageFileDetails = new ImageFileDetails(imageFile);
        var imageSeries=new ImageSeries(imageFile, 0);
        var imageSeriesDetails = new ImageSeriesDetails(imageSeries, imageFileDetails);
        var imagePlane=new ImagePlane(imageSeries, 0, ImagePlane.ALWAYS_MONOCHROME);
        var ipd = new ImagePlaneDetails(imagePlane, imageSeriesDetails);
        var stack = ImagePlaneDetailsStack.makeMonochromeStack(ipd);
        var stacks = java.util.Collections.singletonList(stack);
        var keys = java.util.Collections.singletonList(imageNumber);
        var imageSet = new ImageSet(stacks, keys);
        imageSet.compress(java.util.Collections.singletonList("DNA"), null);
        """, dict(url=url, imageNumber=str(i)))
        blob = javabridge.get_env().get_byte_array_elements(jblob)
        m[cellprofiler.measurement.IMAGE, cellprofiler.modules.namesandtypes.M_IMAGE_SET, i, blob.dtype] = blob
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.loadtxt(cStringIO.StringIO(GOOD_PIPELINE))
    pipeline.write_pipeline_measurement(m)
    return m
