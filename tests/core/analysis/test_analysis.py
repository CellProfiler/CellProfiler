"""test_analysis.py - test the analysis server framework
"""

import logging
import queue
import inspect
import numpy
import os
import tempfile
import threading
import traceback
import unittest
import uuid
import zmq
import queue
from io import StringIO
from importlib.util import find_spec

import cellprofiler_core.constants.measurement
import cellprofiler_core.utilities.measurement
from cellprofiler_core.analysis._analysis import Analysis
from cellprofiler_core.analysis._runner import Runner
from cellprofiler_core.analysis.reply import ImageSetSuccess
import cellprofiler_core.analysis
import cellprofiler_core.analysis.request as anarequest
import cellprofiler_core.analysis.reply as anareply
import cellprofiler_core.analysis.event
import cellprofiler_core.pipeline
import cellprofiler_core.module
import cellprofiler_core.preferences
import cellprofiler_core.measurement
import cellprofiler_core.utilities.zmq
import cellprofiler_core.utilities.zmq.communicable.reply.upstream_exit


LOGGER = logging.getLogger(__name__)

IMAGE_NAME = "imagename"
OBJECTS_NAME = "objectsname"
IMAGE_FEATURE = "imagefeature"
OBJECTS_FEATURE = "objectsfeature"


class TestAnalysis(unittest.TestCase):
    class FakeWorker(threading.Thread):
        """A mockup of a ZMQ client to the boundary

        """

        def __init__(self, name="Client thread"):
            threading.Thread.__init__(self, name=name)
            self.setDaemon(True)
            self.zmq_context = zmq.Context()
            self.zmq_context.setsockopt(zmq.LINGER, 0)
            self.request_queue = queue.Queue()
            self.response_queue = queue.Queue()
            self.start_signal = threading.Semaphore(0)
            self.keep_going = True
            self.analysis_id = None
            self.notify_addr = "inproc://%s" % uuid.uuid4().hex
            self.notify_socket = self.zmq_context.socket(zmq.PUB)
            self.notify_socket.bind(self.notify_addr)
            self.work_socket = None
            self.recv_notify_socket = None
            self.keepalive_socket = None
            self.sockets = (self.notify_socket, self.work_socket, self.recv_notify_socket, self.keepalive_socket)
            self.start()
            self.start_signal.acquire()

            logging.root.setLevel(10)
            if len(logging.root.handlers) == 0:
                stream_handler = logging.StreamHandler()
                # fmt = logging.Formatter("%(process)d|%(levelno)s|%(name)s::%(funcName)s: %(message)s")
                # stream_handler.setFormatter(fmt)
                logging.root.addHandler(stream_handler)

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            LOGGER.debug("FakeWorker::__exit__ - Stopping the worker")
            self.stop()
            LOGGER.debug("FakeWorker::__exit__ - Closing sockets (notify, work, recv_notify, and keepalive)")
            for i, socket in enumerate(self.sockets):
                if socket is not None:
                    LOGGER.debug(f"  FakeWorker::__exit__ - Closing socket {i}")
                    socket.close(linger=0)
                else:
                    LOGGER.debug(f"  FakeWorker::__exit__ - sockets {i} is None")
            self.zmq_context.setsockopt(zmq.LINGER, 0)
            LOGGER.debug("FakeWorker::__exit__ - Destroying zmq context")
            self.zmq_context.destroy()

            LOGGER.debug("FakeWorker::__exit__ - waiting on join")
            self.join()
            LOGGER.debug("FakeWorker::__exit__ - did join")

        def run(self):
            LOGGER.info("FakeWorker::run - starting")
            try:
                self.work_socket = self.zmq_context.socket(zmq.REQ)
                self.recv_notify_socket = self.zmq_context.socket(zmq.SUB)
                self.recv_notify_socket.setsockopt(zmq.SUBSCRIBE, b"")
                self.recv_notify_socket.connect(self.notify_addr)
                self.keepalive_socket = None
                self.poller = zmq.Poller()
                self.poller.register(self.recv_notify_socket, zmq.POLLIN)
                self.start_signal.release()

                while self.keep_going:
                    LOGGER.debug("FakeWorker::run - keep going, poll for a second")
                    socks = dict(self.poller.poll(1000))
                    if socks.get(self.recv_notify_socket, None) == zmq.POLLIN:
                        # Discard whatever comes down the notify socket.
                        # It's only used to wake us up.
                        LOGGER.debug("FakeWorker::run - Blocking on recv_notify_socket")
                        msg = self.recv_notify_socket.recv()
                        LOGGER.debug("FakeWorker::run - Received message on recv_notify_socket")
                    while True:
                        try:
                            if not self.keep_going:
                                LOGGER.debug("FakeWorker::run - keep going is false, breaking")
                                break
                            fn_and_args = self.request_queue.get_nowait()

                        except queue.Empty:
                            LOGGER.debug("FakeWorker::run - queue empty, breaking")
                            break
                        try:
                            response = fn_and_args[0](*fn_and_args[1:])
                            self.response_queue.put((None, response))
                        except Exception as e:
                            traceback.print_exc()
                            self.response_queue.put((e, None))
            except:
                LOGGER.warning("FakeWorker::run - thread caught exception", exc_info=True)
                self.start_signal.release()
            finally:
                LOGGER.debug("FakeWorker::run - thread exiting")

        def stop(self):
            self.keep_going = False
            self.notify_socket.send(b"Stop")

        def send(self, req):
            LOGGER.debug("    FakeWorker::send - Enqueueing send of %s" % str(type(req)))
            self.request_queue.put((self.do_send, req))
            self.notify_socket.send(b"Send")
            return self.recv

        def request_work(self):
            """Send a work request until we get a WorkReply"""
            while True:
                reply = self.send(anarequest.Work(self.analysis_id))()
                if isinstance(reply, anareply.Work):
                    return reply
                elif not isinstance(reply, anareply.NoWork):
                    raise NotImplementedError(
                        "Received a reply of %s for a work request" % str(type(reply))
                    )

        def do_send(self, req):
            LOGGER.info("    FakeWorker::do_send - Sending %s" % str(type(req)))
            cellprofiler_core.utilities.zmq.communicable.Communicable.send(
                req, self.work_socket
            )
            self.poller.register(self.work_socket, zmq.POLLIN)
            try:
                while True:
                    socks = dict(self.poller.poll())
                    if socks.get(self.recv_notify_socket, None) == zmq.POLLIN:
                        msg = self.recv_notify_socket.recv()
                        if not self.keep_going:
                            raise Exception("Cancelled")
                    if socks.get(self.work_socket, None) == zmq.POLLIN:
                        LOGGER.debug("    Received response for %s" % str(type(req)))
                        return cellprofiler_core.utilities.zmq.communicable.Communicable.recv(
                            self.work_socket
                        )
            finally:
                self.poller.unregister(self.work_socket)

        def recv(self):
            LOGGER.debug("     FakeWorker::recv - Waiting for client thread")
            exception, result = self.response_queue.get()
            if exception is not None:
                LOGGER.debug("    FakeWorker::recv - Client thread communicated exception")
                raise exception
            else:
                LOGGER.debug("    FakeWorker::recv - Client thread communicated result")
                return result

        def listen_for_heartbeat(self, address):
            self.request_queue.put((self.do_listen_for_heartbeat, address))
            self.notify_socket.send(b"Listen for announcements")
            return self.recv

        def do_listen_for_heartbeat(self, address):
            self.keepalive_socket = self.zmq_context.socket(zmq.SUB)
            self.keepalive_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.keepalive_socket.connect(address)
            self.poller.register(self.keepalive_socket, zmq.POLLIN)
            try:
                while True:
                    socks = dict(self.poller.poll(5000))
                    if socks.get(self.recv_notify_socket, None) == zmq.POLLIN:
                        msg = self.recv_notify_socket.recv()
                        if not self.keep_going:
                            raise Exception("Cancelled")
                    if socks.get(self.keepalive_socket, None) == zmq.POLLIN:
                        beat = self.keepalive_socket.recv()
                        return beat
            except:
                LOGGER.info("Failed to listen")
            finally:
                self.poller.unregister(self.keepalive_socket)
                self.keepalive_socket.close()
                self.keepalive_socket = None

        def connect(self, request_address, analysis_id):
            self.analysis_id = analysis_id
            self.request_queue.put((self.do_connect, request_address))
            self.notify_socket.send(b"Do connect")
            return self.recv()

        def do_connect(self, work_queue_address):
            self.work_socket.connect(work_queue_address)

    @classmethod
    def setUpClass(cls):
        cls.zmq_context = zmq.Context()
        from cellprofiler_core.utilities.core.modules import fill_modules
        fill_modules()

    def setUp(self):
        fd, self.filename = tempfile.mkstemp(".h5")
        os.close(fd)
        self.event_queue = queue.Queue()
        self.analysis = None
        self.wants_analysis_finished = False
        self.wants_pipeline_events = False
        self.measurements_to_close = None
        self.cpinstalled = find_spec("cellprofiler") != None

    def tearDown(self):
        LOGGER.debug("Tearing down")
        self.cancel_analysis()
        if self.measurements_to_close is not None:
            self.measurements_to_close.close()
        if os.path.exists(self.filename):
            os.unlink(self.filename)
        LOGGER.debug("Tear down complete")

    def cancel_analysis(self):
        LOGGER.debug("Canceling analysis")
        if self.analysis is not None:
            self.analysis.cancel()
            self.analysis = None

    def analysis_event_handler(self, event):
        if isinstance(event, cellprofiler_core.analysis.event.Progress):
            return
        if isinstance(event, cellprofiler_core.analysis.event.Finished):
            self.measurements_to_close = event.measurements
            if not self.wants_analysis_finished:
                return
        if (
            isinstance(event, cellprofiler_core.pipeline.Event)
            and not self.wants_pipeline_events
        ):
            return
        self.event_queue.put(event)

    def make_pipeline_and_measurements(
        self, nimage_sets=1, group_numbers=None, group_indexes=None, **kwargs
    ):
        m = cellprofiler_core.measurement.Measurements(mode="memory")
        for i in range(1, nimage_sets + 1):
            if group_numbers is not None:
                group_number = group_numbers[i - 1]
                group_index = group_indexes[i - 1]
            else:
                group_number = 1
                group_index = i
            m[
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.C_URL + "_" + IMAGE_NAME,
                i,
            ] = ("file:/%d.tif" % i)
            m[
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.GROUP_NUMBER,
                i,
            ] = group_number
            m[
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.GROUP_INDEX,
                i,
            ] = group_index
        pipeline = cellprofiler_core.pipeline.Pipeline()
        if self.cpinstalled:
            pipeline.loadtxt(StringIO(SBS_PIPELINE), raise_on_error=True)
        else:
            pipeline.loadtxt(StringIO(SBS_PIPELINE_CORE_ONLY), raise_on_error=True)
        return pipeline, m

    def make_pipeline_and_measurements_and_start(self, **kwargs):
        LOGGER.debug("Making pipeline and measurements, and starting analysis")
        pipeline, m = self.make_pipeline_and_measurements(**kwargs)
        if "status" in kwargs:
            overwrite = False
            for i, status in enumerate(kwargs["status"]):
                m.add_measurement(
                    cellprofiler_core.constants.measurement.IMAGE,
                    Runner.STATUS,
                    status,
                    image_set_number=i + 1,
                )
        else:
            overwrite = True
        self.analysis = Analysis(pipeline, m)

        LOGGER.debug("Starting Analysis")
        self.analysis.start(
            self.analysis_event_handler, num_workers=0, overwrite=overwrite
        )
        analysis_started = self.event_queue.get()
        self.assertIsInstance(
            analysis_started, cellprofiler_core.analysis.event.Started
        )
        return pipeline, m

    def check_display_post_run_requests(self, pipeline):
        """Read the request.DisplayPostRun messages during the post_run phase"""

        LOGGER.debug("check_display_post_run_requests: for all pipeline modules, get from event queue and assert DisplayPostRun request")
        for module in pipeline.modules():
            if (
                module.show_window
                and module.__class__.display_post_run
                != cellprofiler_core.module.Module.display_post_run
            ):
                result = self.event_queue.get()
                self.assertIsInstance(result, anarequest.DisplayPostRun)
                self.assertEqual(result.module_num, module.module_num)
        LOGGER.debug("check_display_post_run_requests: all passed")

    def test_01_01_start_and_stop(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.make_pipeline_and_measurements_and_start()
        self.wants_analysis_finished = True
        self.cancel_analysis()
        # The last should be event.Finished. There may be AnalysisProgress
        # prior to that.
        analysis_finished = self.event_queue.get()
        self.assertIsInstance(
            analysis_finished, cellprofiler_core.analysis.event.Finished
        )
        self.assertTrue(analysis_finished.cancelled)
        self.assertIsInstance(
            analysis_finished.measurements, cellprofiler_core.measurement.Measurements
        )
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_02_01_heartbeat(self):
        """The heartbeat should send b'KeepAlive' each second. When the
        analysis is cancelled or finishes we should see b'Stop' as the final
        transmission on the socket"""
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()

        with self.FakeWorker() as worker:
            heartbeat_address = self.analysis.runner.boundary.keepalive_address
            LOGGER.debug("test_02_01: listening for heartbeat")
            response = worker.listen_for_heartbeat(heartbeat_address)()
            from cellprofiler_core.constants.worker import NOTIFY_RUN, NOTIFY_STOP
            assert response == NOTIFY_RUN

            def collect_messages(container):
                LOGGER.debug("test_02_01: spy thred - subscribing to heartbeat")
                print("test_02_01: spy thread - container contents: ", container)
                sock = worker.zmq_context.socket(zmq.SUB)
                sock.setsockopt(zmq.SUBSCRIBE, b"")
                sock.connect(heartbeat_address)
                poller = zmq.Poller()
                poller.register(sock, zmq.POLLIN)
                while len(container) < 10:
                    msg = poller.poll(2000)
                    LOGGER.debug("test_02_01: spy thread - polled 2s and got message, adding to")
                    print("test_02_01 - spy thread - messages is", msg)
                    if msg:
                        container.append(sock.recv())
                    else:
                        break

            messages = []

            spy_thread = threading.Thread(target=collect_messages, name="test heartbeat thread", args=(messages, ))
            LOGGER.debug("test_02_01: starting spy thread, collecting messages")
            spy_thread.start()
            LOGGER.debug("test_02_01: canceling analysis")
            self.cancel_analysis()
            LOGGER.debug("test_02_01: joingin on spy thread 2s")
            spy_thread.join(2000)
            assert not spy_thread.is_alive()
            LOGGER.debug("test_02_01: spy thread is now dead (joined)")
            assert len(messages)
            LOGGER.debug("test_02_01: messages has length")
            assert messages[-1] == NOTIFY_STOP
            LOGGER.debug("test_02_01: the last message has NOTIFY_STOP")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_03_01_get_work(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            LOGGER.debug("test_03_01: connecting worker to boundary request addr, with analysis_id")
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            print("test_03_01 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            LOGGER.debug("test_03_01: sending work to worker")
            response = worker.send(anarequest.Work(worker.analysis_id))()
            LOGGER.debug("test_03_01: got work response from worker")
            self.assertIsInstance(response, anareply.Work)
            LOGGER.debug("test_03_01: wwork response valid")
            self.assertSequenceEqual(response.image_set_numbers, (1,))
            LOGGER.debug("test_03_01: image set nuber valid")
            self.assertFalse(response.worker_runs_post_group)
            LOGGER.debug("test_03_01: worker does not run post group")
            self.assertTrue(response.wants_dictionary)
            LOGGER.debug("test_03_01: worker wants dictionary")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_03_02_get_work_twice(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()

        with self.FakeWorker() as worker:
            LOGGER.debug("test_03_02: connecting worker to boundary request addr with analysis id")
            print("test_03_02 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_03_02: sending work request to worker")
            response = worker.send(anarequest.Work(worker.analysis_id))()
            LOGGER.debug("test_03_02: got repsonse from worker")
            self.assertIsInstance(response, anareply.Work)
            LOGGER.debug("test_03_02: repsonse is Work repsonse, sending another work request")
            response = worker.send(anarequest.Work(worker.analysis_id))()
            LOGGER.debug("test_03_02: got response second time")
            self.assertIsInstance(response, anareply.NoWork)
            LOGGER.debug("test_03_02: second response is NoWork")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_03_03_cancel_before_work(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()

        with self.FakeWorker() as worker:
            LOGGER.debug("test_03_03: connecting worker to boundary request addr with analysis id")
            print("test_03_03 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_03_03: canceling analysis before work")
            self.cancel_analysis()
            LOGGER.debug("test_03_03: finished canceling analysis")
            assert self.analysis is None
            LOGGER.debug("test_03_03: analysis is none")
            # The boundary thread used to spin eternally and reply with
            # BoundaryExited if analysis was cancelled. From CP5 it'll shut
            # down with the workers and so any open sockets will be closed.

            # response = worker.send(anarequest.Work(worker.analysis_id))()
            # self.assertIsInstance(
            #     response,
            #     cellprofiler_core.utilities.zmq.communicable.reply.upstream_exit.BoundaryExited,
            # )
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    # FIXME: wxPython 4 PR
    # def test_04_01_pipeline_preferences(self):
    #     LOGGER.debug("Entering %s" % inspect.getframeinfo(inspect.currentframe()).function)
    #     pipeline, m = self.make_pipeline_and_measurements_and_start()
    #     cellprofiler_core.preferences.set_headless()
    #     title_font_name = "Rosewood Std Regular"
    #     cellprofiler_core.preferences.set_title_font_name(title_font_name)
    #     cellprofiler_core.preferences.set_default_image_directory(example_images_directory())
    #     cellprofiler_core.preferences.set_default_output_directory(testimages_directory())
    #     with self.FakeWorker() as worker:
    #         worker.connect(self.analysis.runner.work_announce_address)
    #         response = worker.send(anarequest.PipelinePreferences(
    #                 worker.analysis_id))()
    #         #
    #         # Compare pipelines
    #         #
    #         client_pipeline = cellprofiler_core.pipeline.Pipeline()
    #         pipeline_txt = response.pipeline_blob.tostring()
    #         client_pipeline.loadtxt(StringIO(pipeline_txt),
    #                                 raise_on_error=True)
    #         self.assertEqual(len(pipeline.modules()),
    #                          len(client_pipeline.modules()))
    #         for smodule, cmodule in zip(pipeline.modules(),
    #                                     client_pipeline.modules()):
    #             self.assertEqual(smodule.module_name, cmodule.module_name)
    #             self.assertEqual(len(smodule.settings()),
    #                              len(cmodule.settings()))
    #             for ssetting, csetting in zip(smodule.settings(),
    #                                           cmodule.settings()):
    #                 self.assertEqual(ssetting.get_value_text(),
    #                                  csetting.get_value_text())
    #         preferences = response.preferences
    #         self.assertIn(cellprofiler_core.preferences.TITLE_FONT_NAME, preferences)
    #         self.assertEqual(preferences[cellprofiler_core.preferences.TITLE_FONT_NAME],
    #                          title_font_name)
    #         self.assertIn(cellprofiler_core.preferences.DEFAULT_IMAGE_DIRECTORY, preferences)
    #         self.assertEqual(preferences[cellprofiler_core.preferences.DEFAULT_IMAGE_DIRECTORY],
    #                          cellprofiler_core.preferences.get_default_image_directory())
    #         self.assertIn(cellprofiler_core.preferences.DEFAULT_OUTPUT_DIRECTORY, preferences)
    #         self.assertEqual(preferences[cellprofiler_core.preferences.DEFAULT_OUTPUT_DIRECTORY],
    #                          cellprofiler_core.preferences.get_default_output_directory())
    #
    #     LOGGER.debug("Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function)

    def test_04_02_initial_measurements_request(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            LOGGER.debug("test_04_02: connecting worker to boundary request addr with analysis id")
            print("test_04_02 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_04_02: connected, sending worker InitialMeasurments request")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            LOGGER.debug("test_04_02: got response, loading the measurments form buffer")
            client_measurements = cellprofiler_core.utilities.measurement.load_measurements_from_buffer(
                response.buf
            )
            try:
                assert isinstance(
                    client_measurements, cellprofiler_core.measurement.Measurements
                )
                LOGGER.debug("test_04_02: measurements received are indeed measurements")
                assert isinstance(m, cellprofiler_core.measurement.Measurements)
                LOGGER.debug("test_04_02: the measurments we sent were measurements too")
                self.assertSequenceEqual(
                    m.get_image_numbers(), client_measurements.get_image_numbers()
                )
                LOGGER.debug("test_04_02: image set numbers are correct")
                image_numbers = m.get_image_numbers()
                self.assertCountEqual(
                    m.get_object_names(), client_measurements.get_object_names()
                )
                LOGGER.debug("test_04_02: image numbers are correct")
                for object_name in m.get_object_names():
                    self.assertCountEqual(
                        m.get_feature_names(object_name),
                        client_measurements.get_feature_names(object_name),
                    )
                    for feature_name in m.get_feature_names(object_name):
                        for image_number in image_numbers:
                            sv = m.get_measurement(
                                object_name, feature_name, image_set_number=image_number
                            )
                            cv = client_measurements.get_measurement(
                                object_name, feature_name, image_set_number=image_number
                            )
                            self.assertEqual(numpy.isscalar(sv), numpy.isscalar(cv))
                            if numpy.isscalar(sv):
                                self.assertEqual(sv, cv)
                            else:
                                numpy.testing.assert_almost_equal(sv, cv)
                LOGGER.debug("test_04_02: for all objets, and all features per object, and all image numbers per object, they are correct")
            finally:
                client_measurements.close()
                LOGGER.debug("test_04_02: closed measurements")
                LOGGER.debug(
                    "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
                )
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_04_03_interaction(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            LOGGER.debug("test_04_03: connecting worker to boundary request addr with analysis id")
            print("test_04_03 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_04_03: sending worker Interaction request with foo=bar, expecting fn reply back")
            fn_interaction_reply = worker.send(
                anarequest.Interaction(worker.analysis_id, foo="bar")
            )
            LOGGER.debug("test_04_03: received a fn reply, getting from event queue")
            request = self.event_queue.get()
            LOGGER.debug("test_04_03: got request from event queue")
            self.assertIsInstance(request, anarequest.Interaction)
            LOGGER.debug("test_04_03: event queue request is an Interaction request")
            self.assertEqual(request.foo, "bar")
            LOGGER.debug("test_04_03: requst's foo is bar, replying with hello=world Interaction reply")
            request.reply(anareply.Interaction(hello="world"))
            LOGGER.debug("test_04_03: invoking the fn reply from before and getting a reply")
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, anareply.Interaction)
            LOGGER.debug("test_04_03: the reply from fn is Interaction")
            self.assertEqual(reply.hello, "world")
            LOGGER.debug("test_04_03: reply's hello is world")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_04_04_01_display(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            LOGGER.debug("test_04_04_01: connecting worker to boundary request addr with analysis id")
            print("test_04_04_01 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_04_04_01: sending worker Display request, with foo=bar, expecting a fn reply back")
            fn_interaction_reply = worker.send(
                anarequest.Display(worker.analysis_id, foo="bar")
            )
            #
            # The event queue should be hooked up to the interaction callback
            #
            LOGGER.debug("test_04_04_01: getting request from event queue")
            request = self.event_queue.get()
            self.assertIsInstance(request, anarequest.Display)
            LOGGER.debug("test_04_04_01: event queue request is Display request")
            self.assertEqual(request.foo, "bar")
            LOGGER.debug("test_04_04_01: event queue Display request's foo is bar, replying with Ack, with message as Gimme Pony")
            request.reply(anareply.Ack(message="Gimme Pony"))
            LOGGER.debug("test_04_04_01: invoking fn reply from before")
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, anareply.Ack)
            LOGGER.debug("test_04_04_01: fn reply is indeed Ack")
            self.assertEqual(reply.message, "Gimme Pony")
            LOGGER.debug("test_04_04_01: fn reply message is indeed Gimme Pony")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_04_04_02_display_post_group(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            LOGGER.debug("test_04_04_02: connecting worker to boundary request addr with analysis id")
            print("test_04_04_02 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_04_04_02: sending worker DisplayPostGroup request, with 1 dict(foo=bar) and 3, expecting fn reply back")
            fn_interaction_reply = worker.send(
                anarequest.DisplayPostGroup(worker.analysis_id, 1, dict(foo="bar"), 3)
            )
            LOGGER.debug("test_04_04_02: got fn reply back, getting request from event queue")
            #
            # The event queue should be hooked up to the interaction callback
            #
            request = self.event_queue.get()
            LOGGER.debug("test_04_04_02: got event queue request")
            self.assertIsInstance(request, anarequest.DisplayPostGroup)
            LOGGER.debug("test_04_04_02: event queue request is indeed DisplayPostGroup")
            display_data = request.display_data
            self.assertEqual(display_data["foo"], "bar")
            LOGGER.debug("test_04_04_02: DisplayPostGroup's display data's foo is indeed bar, replying with Ack, with message of Gimme Pony")
            request.reply(anareply.Ack(message="Gimme Pony"))
            LOGGER.debug("test_04_04_02: invoking fn from before")
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, anareply.Ack)
            LOGGER.debug("test_04_04_02: reply fn is indeed Ack")
            self.assertEqual(reply.message, "Gimme Pony")
            LOGGER.debug("test_04_04_02: reply fn message is indeed Gimmme Pony")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_04_05_exception(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        with self.FakeWorker() as worker:
            LOGGER.debug("test_04_5: connecting worker to boundary request addr with analysis id")
            print("test_04_5 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_04_05: constructing fake traceback")
            fake_traceback = ''.join(traceback.format_list(traceback.extract_stack()))
            LOGGER.debug("test_04_05: sending ExceptionReport request with fake traceback, receiving fn reply")
            fn_interaction_reply = worker.send(
                anarequest.ExceptionReport(
                    worker.analysis_id,
                    image_set_number=1,
                    module_name="Images",
                    exc_type="Exception",
                    exc_message="Not really an exception",
                    exc_traceback=fake_traceback,
                    filename="test_analysis.py",
                    line_number=374,
                )
            )
            #
            # The event queue should be hooked up to the interaction callback
            #
            LOGGER.debug("test_04_05: getting request from event queue")
            request = self.event_queue.get()
            self.assertIsInstance(request, anarequest.ExceptionReport)
            LOGGER.debug("test_04_05: event queue request is indeed ExceptionReport")
            self.assertEqual(
                fake_traceback, request.exc_traceback
            )
            LOGGER.debug("test_04_05: event queue ExceptionReport is as expected")
            self.assertEqual(request.filename, "test_analysis.py")
            LOGGER.debug("test_04_05: event queue ExceptionReport filename is test_analysis.py; replying with ExceptionPleaseDebug with verification hash corned beef and dispostion 1")
            request.reply(
                anareply.ExceptionPleaseDebug(
                    disposition=1, verification_hash="corned beef"
                )
            )
            LOGGER.debug("test_04_05: invoking reply fn from before")
            reply = fn_interaction_reply()
            self.assertIsInstance(reply, anareply.ExceptionPleaseDebug)
            LOGGER.debug("test_04_05: fn reply is indeed ExceptionPleaseDebug")
            self.assertEqual(reply.verification_hash, "corned beef")
            LOGGER.debug("test_04_05: fn reply's verification hash is indeed corned beef")
            self.assertEqual(reply.disposition, 1)
            LOGGER.debug("test_04_05: fn reply's dispostion is indeed 1")
            #
            # Try DebugWaiting and DebugComplete as well
            #
            LOGGER.debug("test_04_05: for DebugWaiting on 8080 and DebugComplete:")
            for req in (
                anarequest.DebugWaiting(worker.analysis_id, 8080),
                anarequest.DebugComplete(worker.analysis_id),
            ):
                LOGGER.debug("  test_04_05: sending the req, expecting fn reply")
                fn_interaction_reply = worker.send(req)
                LOGGER.debug("  test_04_05: getting event queue request")
                request = self.event_queue.get()
                self.assertEqual(type(request), type(req))
                LOGGER.debug("  test_04_05: request is DebugWaiting or DebugComplete, sending ACK")
                request.reply(anareply.Ack())
                LOGGER.debug("test_04_05: invoking fn, getting reply")
                reply = fn_interaction_reply()
                self.assertIsInstance(reply, anareply.Ack)
                LOGGER.debug("test_04_05: reply is indeed Ack")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_05_01_imageset_with_dictionary(self):
        #
        # Go through the steps for the first imageset and see if the
        # dictionary that we sent is the one we get.
        #
        # request.Work - to get the rights to report the dictionary
        # ImageSetSuccessWithDictionary - to report the dictionary
        # request.Work (with spin until WorkReply received)
        # request.SharedDictionary
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start(nimage_sets=2)
        r = numpy.random.RandomState()
        r.seed(51)
        with self.FakeWorker() as worker:
            LOGGER.debug("test_05_01: connecting worker to boundary request addr with analysis id")
            print("test_05_01 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_05_01: requesting work from worker, getting response")
            response = worker.request_work()
            dictionaries = [
                dict([(uuid.uuid4().hex, r.uniform(size=(10, 15))) for _ in range(10)])
                for module in pipeline.modules()
            ]
            LOGGER.debug("test_05_01: sending worker ImageSetSuccessWithDictionary, from constructed dictionary, IIF reply fn")
            response = worker.send(
                anareply.ImageSetSuccessWithDictionary(
                    worker.analysis_id, response.image_set_numbers[0], dictionaries
                )
            )()
            self.assertIsInstance(response, anareply.Ack)
            LOGGER.debug("test_05_01: reply is indeed Ack, requesting additional work")
            response = worker.request_work()
            self.assertSequenceEqual(response.image_set_numbers, [2])
            LOGGER.debug("test_05_01: reply image set numbers is [2]; sending SharedDictionary, IIF reply fn")
            response = worker.send(anarequest.SharedDictionary(worker.analysis_id))()
            self.assertIsInstance(response, anareply.SharedDictionary)
            LOGGER.debug("test_05_01: response is indeed SharedDictionary")
            result = response.dictionaries
            self.assertEqual(len(dictionaries), len(result))
            for ed, d in zip(dictionaries, result):
                self.assertCountEqual(list(ed.keys()), list(d.keys()))
                for k in list(ed.keys()):
                    numpy.testing.assert_almost_equal(ed[k], d[k])
            LOGGER.debug("test_05_01: dictionary is correctly constructed")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_05_02_groups(self):
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        pipeline, m = self.make_pipeline_and_measurements_and_start(
            nimage_sets=4, group_numbers=[1, 1, 2, 2], group_indexes=[1, 2, 1, 2]
        )
        r = numpy.random.RandomState()
        r.seed(52)
        with self.FakeWorker() as worker:
            LOGGER.debug("test_05_02: connecting worker to boundary request addr with analysis id")
            print("test_05_02 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_05_02: requesting work from worker")
            response = worker.request_work()
            self.assertTrue(response.worker_runs_post_group)
            LOGGER.debug("test_05_02: worker does runs post group")
            self.assertFalse(response.wants_dictionary)
            LOGGER.debug("test_05_02: worker does want dictionary")
            self.assertSequenceEqual(response.image_set_numbers, [1, 2])
            LOGGER.debug("test_05_02: worker image set numbers as expected, [1,2]; sending ImageSetSuccess, IIF response fn")
            response = worker.send(
                ImageSetSuccess(worker.analysis_id, response.image_set_numbers[0])
            )()
            LOGGER.debug("test_05_02: not testing previous response; requesting additional work")
            response = worker.request_work()
            self.assertSequenceEqual(response.image_set_numbers, [3, 4])
            LOGGER.debug("test_05_02: response image set numbers indeed [3,4]")
            self.assertTrue(response.worker_runs_post_group)
            LOGGER.debug("test_05_02: response indeed runs post group")
            self.assertFalse(response.wants_dictionary)
            LOGGER.debug("test_05_02: response indeed wants dictionary")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_06_01_single_imageset(self):
        #
        # Test a full cycle of analysis with an image set list
        # with a single image set
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        r = numpy.random.RandomState()
        r.seed(61)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            LOGGER.debug("test_06_01: connecting worker to boundary request addr with analysis id")
            print("test_06_01 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_06_01: requesting work from worker")
            response = worker.request_work()
            LOGGER.debug("test_06_01: sending worker InitialMeasurements; IIF response fn")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            LOGGER.debug("test_06_01: loading response buf")
            client_measurements = cellprofiler_core.utilities.measurement.load_measurements_from_buffer(
                response.buf
            )
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            dictionaries = [
                dict([(uuid.uuid4().hex, r.uniform(size=(10, 15))) for _ in range(10)])
                for module in pipeline.modules()
            ]
            LOGGER.debug("test_06_01: sending ImageSetSuccessWithDictionary with constructed dictionary; IIF response fn")
            response = worker.send(
                anareply.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries
                )
            )()
            objects_measurements = r.uniform(size=10)
            client_measurements[
                cellprofiler_core.constants.measurement.IMAGE, IMAGE_FEATURE, 1
            ] = "Hello"
            client_measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1] = objects_measurements
            req = anarequest.MeasurementsReport(
                worker.analysis_id,
                client_measurements.file_contents(),
                image_set_numbers=[1],
            )
            LOGGER.debug("test_06_01: closing ClientMeasurements")
            client_measurements.close()
            LOGGER.debug("test_06_01: sending MeasurementsReport")
            response_fn = worker.send(req)

            self.check_display_post_run_requests(pipeline)
            #####################################################
            #
            # The server should receive the measurements report.
            # It should merge the measurements and post an
            # event.Finished event.
            #
            #####################################################

            LOGGER.debug("test_06_01: getting from event queue")
            result = self.event_queue.get()
            self.assertIsInstance(result, cellprofiler_core.analysis.event.Finished)
            LOGGER.debug("test_06_01: result is indeed event Finished")
            self.assertFalse(result.cancelled)
            LOGGER.debug("test_06_01: result is not cancelled")
            measurements = result.measurements
            self.assertSequenceEqual(measurements.get_image_numbers(), [1])
            LOGGER.debug("test_06_01: result measurments image numbers as expected")
            self.assertEqual(
                measurements[
                    cellprofiler_core.constants.measurement.IMAGE, IMAGE_FEATURE, 1
                ],
                "Hello",
            )
            numpy.testing.assert_almost_equal(
                measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1], objects_measurements
            )
            LOGGER.debug("test_06_01: result mesurements internals matches expectation")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_06_02_test_three_imagesets(self):
        # Test an analysis of three imagesets
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(nimage_sets=3)
        r = numpy.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            LOGGER.debug("test_06_02: connecting worker to boundary request addr with analysis id")
            print("test_06_02 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_06_02: requesting work from worker")
            response = worker.request_work()
            LOGGER.debug("test_06_02: got response, not doing anything with it, sending worker InitialMeasurements, IIF fn reply")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            LOGGER.debug("test_06_02: loading measurements from reply buffer")
            client_measurements = cellprofiler_core.utilities.measurement.load_measurements_from_buffer(
                response.buf
            )
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            LOGGER.debug("test_06_02: constructing dictionary")
            dictionaries = [
                dict([(uuid.uuid4().hex, r.uniform(size=(10, 15))) for _ in range(10)])
                for module in pipeline.modules()
            ]
            LOGGER.debug("test_06_02: sending ImageSetSuccessWithDictionary, with constructed dictionary; IIF fn response")
            response = worker.send(
                anareply.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries
                )
            )()
            #####################################################
            #
            # The analysis server should be ready to send us two
            # more jobs to do.
            #
            #####################################################
            LOGGER.debug("test_06_02: for expected jobs 2 and 3:")
            expected_jobs = [2, 3]
            for _ in range(2):
                LOGGER.debug("  test_06_02: requesting work from worker")
                response = worker.request_work()
                image_numbers = response.image_set_numbers
                self.assertEqual(len(image_numbers), 1)
                LOGGER.debug("  test_06_02: image set numbers are expected length")
                self.assertIn(image_numbers[0], expected_jobs)
                LOGGER.debug("  test_06_02: image set num 0 in expected jobs 2 and 3")
                expected_jobs.remove(image_numbers[0])
            #####################################################
            #
            # Send the measurement groups
            #
            #####################################################
            objects_measurements = [r.uniform(size=10) for _ in range(3)]
            LOGGER.debug("test_06_02: for three random object measurements")
            for i, om in enumerate(objects_measurements):
                image_number = i + 1
                if image_number > 0:
                    LOGGER.debug("  test_06_02: first image number - send ImageSetSuccess")
                    worker.send(
                        ImageSetSuccess(
                            worker.analysis_id, image_set_number=image_number
                        )
                    )
                LOGGER.debug("  test_06_02: construct new Measurements from one returned before")
                m = cellprofiler_core.measurement.Measurements(copy=client_measurements)
                m[
                    cellprofiler_core.constants.measurement.IMAGE,
                    IMAGE_FEATURE,
                    image_number,
                ] = ("Hello %d" % image_number)
                m[OBJECTS_NAME, OBJECTS_FEATURE, image_number] = om
                LOGGER.debug("  test_06_02: construct MeasurementsReport request")
                req = anarequest.MeasurementsReport(
                    worker.analysis_id,
                    m.file_contents(),
                    image_set_numbers=[image_number],
                )
                LOGGER.debug("  test_06_02: close measurements")
                m.close()
                LOGGER.debug("  test_06_02: send the MeasurementsReport request; IIF reply fn")
                response = worker.send(req)()
            LOGGER.debug("test_06_02: close client measurements returned previously")
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # event.Finished event.
            #
            #####################################################

            LOGGER.debug("test_06_02: check_display_post_run_requests")
            self.check_display_post_run_requests(pipeline)
            LOGGER.debug("test_06_02: get from event queue")
            result = self.event_queue.get()
            self.assertIsInstance(result, cellprofiler_core.analysis.event.Finished)
            LOGGER.debug("test_06_02: event queue result is Finished event")
            self.assertFalse(result.cancelled)
            LOGGER.debug("test_06_02: result is not cancelled")
            measurements = result.measurements
            self.assertSequenceEqual(list(measurements.get_image_numbers()), [1, 2, 3])
            for i in range(1, 4):
                self.assertEqual(
                    measurements[
                        cellprofiler_core.constants.measurement.IMAGE, IMAGE_FEATURE, i
                    ],
                    "Hello %d" % i,
                )
                numpy.testing.assert_almost_equal(
                    measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                    objects_measurements[i - 1],
                )
            LOGGER.debug("test_06_02: result measurements are internally as expected")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_06_03_test_grouped_imagesets(self):
        # Test an analysis of four imagesets in two groups
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(
            nimage_sets=4, group_numbers=[1, 1, 2, 2], group_indexes=[1, 2, 1, 2]
        )
        r = numpy.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            LOGGER.debug("test_06_03: connecting worker to boundary request addr with analysis id")
            print("test_06_03 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_06_03: requesting work from worker")
            response = worker.request_work()
            LOGGER.debug("test_06_03: ignroing response and sending InitialMeasurements; IIF reply fn")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            LOGGER.debug("test_06_03: loading measurements from response buffer")
            client_measurements = cellprofiler_core.utilities.measurement.load_measurements_from_buffer(
                response.buf
            )
            LOGGER.debug("test_06_03: sending ImageSetSuccess; IIF reply fn")
            response = worker.send(ImageSetSuccess(worker.analysis_id, 1))()
            #####################################################
            #
            # Get the second group.
            #
            #####################################################
            LOGGER.debug("test_06_03: requesting second work group")
            response = worker.request_work()
            image_numbers = response.image_set_numbers
            self.assertSequenceEqual(list(image_numbers), [3, 4])
            LOGGER.debug("test_06_03: got back expected images et numbers [3,4]")
            #####################################################
            #
            # Send the measurement groups
            #
            #####################################################
            objects_measurements = [r.uniform(size=10) for _ in range(4)]
            LOGGER.debug("test_06_03: sending ImageSetSuccess with image set number 2, 3, and 4")
            for image_number in range(2, 5):
                worker.send(
                    ImageSetSuccess(worker.analysis_id, image_set_number=image_number)
                )
            for image_numbers in ((1, 2), (3, 4)):
                m = cellprofiler_core.measurement.Measurements(copy=client_measurements)
                for image_number in image_numbers:
                    m[
                        cellprofiler_core.constants.measurement.IMAGE,
                        IMAGE_FEATURE,
                        image_number,
                    ] = ("Hello %d" % image_number)
                    m[
                        OBJECTS_NAME, OBJECTS_FEATURE, image_number
                    ] = objects_measurements[image_number - 1]
                req = anarequest.MeasurementsReport(
                    worker.analysis_id,
                    m.file_contents(),
                    image_set_numbers=image_numbers,
                )
                m.close()
                response = worker.send(req)()
            LOGGER.debug("test_06_03: for image numbers in ((1,2),(3,4)): did copy Measurements from client measurements, request MeasurementsReport, close constructed measurements, and sent the MeasurementsReport; IIF reply fn")
            LOGGER.debug("test_06_03: close client measurements")
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # event.Finished event.
            #
            #####################################################

            LOGGER.debug("test_06_03: check_display_post_run_requests")
            self.check_display_post_run_requests(pipeline)
            LOGGER.debug("test_06_03: get from event queue")
            result = self.event_queue.get()
            self.assertIsInstance(result, cellprofiler_core.analysis.event.Finished)
            LOGGER.debug("test_06_03: event queue result is Finished event")
            self.assertFalse(result.cancelled)
            LOGGER.debug("test_06_03: not cancelled")
            measurements = result.measurements
            self.assertSequenceEqual(
                list(measurements.get_image_numbers()), [1, 2, 3, 4]
            )
            for i in range(1, 5):
                self.assertEqual(
                    measurements[
                        cellprofiler_core.constants.measurement.IMAGE, IMAGE_FEATURE, i
                    ],
                    "Hello %d" % i,
                )
                numpy.testing.assert_almost_equal(
                    measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                    objects_measurements[i - 1],
                )
            LOGGER.debug("test_06_03: result measurements internally as expected")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_06_04_test_restart(self):
        # Test a restart of an analysis
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(
            nimage_sets=3,
            status=[
                Runner.STATUS_UNPROCESSED,
                Runner.STATUS_DONE,
                Runner.STATUS_IN_PROCESS,
            ],
        )
        r = numpy.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            LOGGER.debug("test_06_04: connecting worker to boundary request addr with analysis id")
            print("test_06_04 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            response = worker.request_work()
            LOGGER.debug("test_06_04: sinding InitialMeasurements to worker; IIF reply fn")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            LOGGER.debug("test_06_04: contructing client measurements from response buffer")
            client_measurements = cellprofiler_core.utilities.measurement.load_measurements_from_buffer(
                response.buf
            )
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            LOGGER.debug("test_06_04: constructing dictionaries")
            dictionaries = [
                dict([(uuid.uuid4().hex, r.uniform(size=(10, 15))) for _ in range(10)])
                for module in pipeline.modules()
            ]
            LOGGER.debug("test_06_04: sending ImageSetSuccessWithDictionary with constructed dictionary; IIF reply fn")
            response = worker.send(
                anareply.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries
                )
            )()
            #####################################################
            #
            # The analysis server should be ready to send us just
            # the third job.
            #
            #####################################################
            LOGGER.debug("test_06_04: ignoring previous response; requesting work")
            response = worker.request_work()
            image_numbers = response.image_set_numbers
            self.assertEqual(len(image_numbers), 1)
            self.assertEqual(image_numbers[0], 3)
            LOGGER.debug("test_06_04: response image set numbers properly structured")
            #####################################################
            #
            # Send the measurement groups
            #
            #####################################################
            objects_measurements = [r.uniform(size=10) for _ in range(3)]
            for image_number, om in (
                (1, objects_measurements[0]),
                (3, objects_measurements[2]),
            ):
                worker.send(
                    ImageSetSuccess(worker.analysis_id, image_set_number=image_number)
                )
                m = cellprofiler_core.measurement.Measurements(copy=client_measurements)
                m[
                    cellprofiler_core.constants.measurement.IMAGE,
                    IMAGE_FEATURE,
                    image_number,
                ] = ("Hello %d" % image_number)
                m[OBJECTS_NAME, OBJECTS_FEATURE, image_number] = om
                req = anarequest.MeasurementsReport(
                    worker.analysis_id,
                    m.file_contents(),
                    image_set_numbers=[image_number],
                )
                m.close()
                response = worker.send(req)()
            LOGGER.debug("test_06_04: for each image number, object mesurement pair, send ImageSetSuccess, copy client measurements, construct MeasurementsReport, close copied measurements, send MesurementsReport; IIF rply fn")
            LOGGER.debug("test_06_04: close client mesurements")
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # event.Finished event.
            #
            #####################################################

            LOGGER.debug("test_06_04: check_dipslay_post_run_requests")
            self.check_display_post_run_requests(pipeline)
            LOGGER.debug("test_06_04: get form event queue")
            result = self.event_queue.get()
            self.assertIsInstance(result, cellprofiler_core.analysis.event.Finished)
            LOGGER.debug("test_06_04: event queue event is Finished event")
            self.assertFalse(result.cancelled)
            LOGGER.debug("test_06_04: event queue event is not cancelled")
            measurements = result.measurements
            assert isinstance(measurements, cellprofiler_core.measurement.Measurements)
            self.assertSequenceEqual(list(measurements.get_image_numbers()), [1, 2, 3])
            for i in range(1, 4):
                if i == 2:
                    for feature in (IMAGE_FEATURE, OBJECTS_FEATURE):
                        self.assertFalse(
                            measurements.has_measurements(
                                cellprofiler_core.constants.measurement.IMAGE,
                                feature,
                                2,
                            )
                        )
                else:
                    self.assertEqual(
                        measurements[
                            cellprofiler_core.constants.measurement.IMAGE,
                            IMAGE_FEATURE,
                            i,
                        ],
                        "Hello %d" % i,
                    )
                    numpy.testing.assert_almost_equal(
                        measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                        objects_measurements[i - 1],
                    )
            LOGGER.debug("test_06_04: event queue result measurements is as expected")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_06_05_test_grouped_restart(self):
        # Test an analysis of four imagesets in two groups with all but one
        # complete.
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start(
            nimage_sets=4,
            group_numbers=[1, 1, 2, 2],
            group_indexes=[1, 2, 1, 2],
            status=[
                Runner.STATUS_DONE,
                Runner.STATUS_UNPROCESSED,
                Runner.STATUS_DONE,
                Runner.STATUS_DONE,
            ],
        )
        r = numpy.random.RandomState()
        r.seed(62)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            LOGGER.debug("test_06_05: connecting worker to boundary request addr with analysis id")
            print("test_06_05 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_06_05: requesting work from worker")
            response = worker.request_work()
            self.assertSequenceEqual(response.image_set_numbers, [1, 2])
            LOGGER.debug("test_06_05: response image set numbers as expected, sending InitialMeasurements; IIF reply fn")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            LOGGER.debug("test_06_05: loading client measurements from response buffer")
            client_measurements = cellprofiler_core.utilities.measurement.load_measurements_from_buffer(
                response.buf
            )
            for image_number in (1, 2):
                response = worker.send(
                    ImageSetSuccess(worker.analysis_id, image_number)
                )()
            LOGGER.debug("test_06_05: for each image number, send ImageSetSuccess; IFF reply fn")
            LOGGER.debug("test_06_05: construct Measurements from client measurements")
            m = cellprofiler_core.measurement.Measurements(copy=client_measurements)
            objects_measurements = [r.uniform(size=10) for _ in range(2)]
            for image_number in (1, 2):
                m[
                    cellprofiler_core.constants.measurement.IMAGE,
                    IMAGE_FEATURE,
                    image_number,
                ] = ("Hello %d" % image_number)
                m[OBJECTS_NAME, OBJECTS_FEATURE, image_number] = objects_measurements[
                    image_number - 1
                ]
            LOGGER.debug("test_06_05: construct MeasurementsReport")
            req = anarequest.MeasurementsReport(
                worker.analysis_id, m.file_contents(), image_set_numbers=(1, 2)
            )
            LOGGER.debug("test_06_05: close client measurements")
            m.close()
            LOGGER.debug("test_06_05: send MeasurementsReport; IFF rply fn")
            response = worker.send(req)()
            LOGGER.debug("test_06_05: close client measurements")
            client_measurements.close()
            #####################################################
            #
            # The server should receive the measurements reports,
            # It should merge the measurements and post an
            # event.Finished event.
            #
            #####################################################

            LOGGER.debug("test_06_05: check_display_post_run_requessts")
            self.check_display_post_run_requests(pipeline)
            LOGGER.debug("test_06_05: get from event queue")
            result = self.event_queue.get()
            self.assertIsInstance(result, cellprofiler_core.analysis.event.Finished)
            LOGGER.debug("test_06_05: event queue event is Finished")
            self.assertFalse(result.cancelled)
            LOGGER.debug("test_06_05: event queue event is not cancelled")
            measurements = result.measurements
            for i in range(1, 3):
                self.assertEqual(
                    measurements[
                        cellprofiler_core.constants.measurement.IMAGE, IMAGE_FEATURE, i
                    ],
                    "Hello %d" % i,
                )
                numpy.testing.assert_almost_equal(
                    measurements[OBJECTS_NAME, OBJECTS_FEATURE, i],
                    objects_measurements[i - 1],
                )
            LOGGER.debug("test_06_05: event queue result measurements is structured as expected")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_06_06_relationships(self):
        #
        # Test a transfer of the relationships table.
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        r = numpy.random.RandomState()
        r.seed(61)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            LOGGER.debug("test_06_06: connecting worker to boundary request addr with analysis id")
            print("test_06_06 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address,
                           self.analysis.runner.analysis_id)
            LOGGER.debug("test_06_06: requesting work")
            response = worker.request_work()
            LOGGER.debug("test_06_06: ignore response, sending InitialMeasuremnts; IFF reply fn")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            LOGGER.debug("test_06_06: load client measurements from response buffer")
            client_measurements = cellprofiler_core.utilities.measurement.load_measurements_from_buffer(
                response.buf
            )
            #####################################################
            #
            # Report the dictionary, add some measurements and
            # report the results of the first job
            #
            #####################################################
            LOGGER.debug("test_06_06: create dictionaries")
            dictionaries = [
                dict([(uuid.uuid4().hex, r.uniform(size=(10, 15))) for _ in range(10)])
                for module in pipeline.modules()
            ]
            LOGGER.debug("test_06_06: send ImageSetSuccessWithDictionary; IFF reply fn")
            response = worker.send(
                anareply.ImageSetSuccessWithDictionary(
                    worker.analysis_id, 1, dictionaries
                )
            )()
            LOGGER.debug("test_06_06: mutate client measurements")
            n_objects = 10
            objects_measurements = r.uniform(size=n_objects)
            objects_relationship = r.permutation(n_objects) + 1
            client_measurements[
                cellprofiler_core.constants.measurement.IMAGE, IMAGE_FEATURE, 1
            ] = "Hello"
            client_measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1] = objects_measurements
            client_measurements.add_relate_measurement(
                1,
                "Foo",
                OBJECTS_NAME,
                OBJECTS_NAME,
                numpy.ones(n_objects, int),
                numpy.arange(1, n_objects + 1),
                numpy.ones(n_objects, int),
                objects_relationship,
            )
            LOGGER.debug("test_06_06: create MeasurementsReport from client measurements file contents")
            req = anarequest.MeasurementsReport(
                worker.analysis_id,
                client_measurements.file_contents(),
                image_set_numbers=[1],
            )
            LOGGER.debug("test_06_06: close client measurements")
            client_measurements.close()
            LOGGER.debug("test_06_06: send MeasurementsReport")
            response_fn = worker.send(req)

            LOGGER.debug("test_06_06: check_display_post_run")
            self.check_display_post_run_requests(pipeline)
            #####################################################
            #
            # The server should receive the measurements report.
            # It should merge the measurements and post an
            # event.Finished event.
            #
            #####################################################

            LOGGER.debug("test_06_06: get from event queue")
            result = self.event_queue.get()
            self.assertIsInstance(result, cellprofiler_core.analysis.event.Finished)
            LOGGER.debug("test_06_06: event queue event is Finished")
            self.assertFalse(result.cancelled)
            LOGGER.debug("test_06_06: event queue event is not cancelled")
            measurements = result.measurements
            assert isinstance(measurements, cellprofiler_core.measurement.Measurements)
            self.assertSequenceEqual(measurements.get_image_numbers(), [1])
            self.assertEqual(
                measurements[
                    cellprofiler_core.constants.measurement.IMAGE, IMAGE_FEATURE, 1
                ],
                "Hello",
            )
            numpy.testing.assert_almost_equal(
                measurements[OBJECTS_NAME, OBJECTS_FEATURE, 1], objects_measurements
            )
            LOGGER.debug("test_06_06: event queue events result measurements are structured as expected, checking relationship groups")
            rg = measurements.get_relationship_groups()
            self.assertEqual(len(rg), 1)
            rk = rg[0]
            assert isinstance(rk, cellprofiler_core.measurement.RelationshipKey)
            self.assertEqual(rk.module_number, 1)
            self.assertEqual(rk.object_name1, OBJECTS_NAME)
            self.assertEqual(rk.object_name2, OBJECTS_NAME)
            self.assertEqual(rk.relationship, "Foo")
            r = measurements.get_relationships(1, "Foo", OBJECTS_NAME, OBJECTS_NAME)
            self.assertEqual(len(r), n_objects)
            numpy.testing.assert_array_equal(
                r[cellprofiler_core.constants.measurement.R_FIRST_IMAGE_NUMBER], 1
            )
            numpy.testing.assert_array_equal(
                r[cellprofiler_core.constants.measurement.R_SECOND_IMAGE_NUMBER], 1
            )
            numpy.testing.assert_array_equal(
                r[cellprofiler_core.constants.measurement.R_FIRST_OBJECT_NUMBER],
                numpy.arange(1, n_objects + 1),
            )
            numpy.testing.assert_array_equal(
                r[cellprofiler_core.constants.measurement.R_SECOND_OBJECT_NUMBER],
                objects_relationship,
            )
            LOGGER.debug("test_06_06: all relationship groups as expected")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

    def test_06_07_worker_cancel(self):
        #
        # Test worker sending AnalysisCancelRequest
        #
        LOGGER.debug(
            "Entering %s" % inspect.getframeinfo(inspect.currentframe()).function
        )
        self.wants_analysis_finished = True
        pipeline, m = self.make_pipeline_and_measurements_and_start()
        r = numpy.random.RandomState()
        r.seed(61)
        with self.FakeWorker() as worker:
            #####################################################
            #
            # Connect the worker to the analysis server and get
            # the initial measurements.
            #
            #####################################################
            LOGGER.debug("test_06_07: connecting worker to boundary request addr with analysis id")
            print("test_06_07 - boundary request addr", self.analysis.runner.boundary.request_address, ", analysis_id", self.analysis.runner.analysis_id)
            worker.connect(self.analysis.runner.boundary.request_address, self.analysis.runner.analysis_id)
            LOGGER.debug("test_06_07: requesting work from worker")
            response = worker.request_work()
            LOGGER.debug("test_06_07: sending InitialMeasurements; IFF reply fn")
            response = worker.send(anarequest.InitialMeasurements(worker.analysis_id))()
            #####################################################
            #
            # The worker sends an AnalysisCancelRequest. The
            # server should send event.Finished.
            #
            #####################################################

            LOGGER.debug("test_06_07: sending AnalysisCancel; IFF reply fn")
            response = worker.send(anarequest.AnalysisCancel(worker.analysis_id))()
            LOGGER.debug("test_06_07: getting from event queue")
            result = self.event_queue.get()
            self.assertIsInstance(result, cellprofiler_core.analysis.event.Finished)
            LOGGER.debug("test_06_07: event queue event is Finished")
            self.assertTrue(result.cancelled)
            LOGGER.debug("test_06_07: event queue event is cancelled")
        LOGGER.debug(
            "Exiting %s" % inspect.getframeinfo(inspect.currentframe()).function
        )

# Sample pipeline - should only be used if cellprofiler is installed
SBS_PIPELINE = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:300
GitHash:
ModuleCount:8
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:No filtering
    Select the rule criteria:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\'\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:
    Regular expression:(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$
    Extract metadata from:All images
    Select the filtering criteria:or (file does contain "")
    Metadata file location:
    Match file and image metadata:[]
    Use case insensitive matching?:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:Images matching rules
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:[{u'Cytoplasm': u'WellRow', u'DNACorr': None, 'DNA': u'WellRow', u'CytoplasmCorr': None}, {u'Cytoplasm': u'WellColumn', u'DNACorr': None, 'DNA': u'WellColumn', u'CytoplasmCorr': None}]
    Image set matching method:Metadata
    Set intensity range from:Yes
    Assignments count:4
    Single images count:0
    Maximum intensity:255.0
    Volumetric:No
    x:1.0
    y:1.0
    z:1.0
    Select the rule criteria:and (extension does istif) (metadata does C "2")
    Name to assign these images:DNA
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0
    Select the rule criteria:and (extension does istif) (metadata does C "1")
    Name to assign these images:Cytoplasm
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0
    Select the rule criteria:or (file does startwith "Channel1ILLUM")
    Name to assign these images:DNACorr
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0
    Select the rule criteria:or (file does contain "Channel2ILLUM")
    Name to assign these images:CytoplasmCorr
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

CorrectIlluminationApply:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:Cytoplasm
    Name the output image:CorrCytoplasm
    Select the illumination function:CytoplasmCorr
    Select how the illumination function is applied:Divide

CorrectIlluminationApply:[module_num:6|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:DNA
    Name the output image:CorrDNA
    Select the illumination function:DNACorr
    Select how the illumination function is applied:Divide

IdentifyPrimaryObjects:[module_num:7|svn_version:\'Unknown\'|variable_revision_number:13|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input image:CorrDNA
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):10,40
    Discard objects outside the diameter range?:Yes
    Discard objects touching the border of the image?:Yes
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:7
    Speed up by using lower-resolution image to find local maxima?:Yes
    Fill holes in identified objects?:After both thresholding and declumping
    Automatically calculate size of smoothing filter for declumping?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Use advanced settings?:Yes
    Threshold setting version:3
    Threshold strategy:Global
    Thresholding method:Otsu
    Threshold smoothing scale:1.3488
    Threshold correction factor:1
    Lower and upper bounds on threshold:0.000000,1.000000
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Two-class or three-class thresholding?:Two classes
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Size of adaptive window:10
    Lower outlier fraction:0.05
    Upper outlier fraction:0.05
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2

IdentifySecondaryObjects:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:9|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Select the input objects:Nuclei
    Name the objects to be identified:Cells
    Select the method to identify the secondary objects:Propagation
    Select the input image:CorrCytoplasm
    Number of pixels by which to expand the primary objects:10
    Regularization factor:0.05
    Name the outline image:SecondaryOutlines
    Retain outlines of the identified secondary objects?:No
    Discard secondary objects touching the border of the image?:No
    Discard the associated primary objects?:No
    Name the new primary objects:FilteredNuclei
    Retain outlines of the new primary objects?:No
    Name the new primary object outlines:FilteredNucleiOutlines
    Fill holes in identified objects?:Yes
    Threshold setting version:3
    Threshold strategy:Global
    Thresholding method:Otsu
    Threshold smoothing scale:0
    Threshold correction factor:1
    Lower and upper bounds on threshold:0.000000,1.000000
    Manual threshold:0.0
    Select the measurement to threshold with:None
    Two-class or three-class thresholding?:Two classes
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Size of adaptive window:10
    Lower outlier fraction:0.05
    Upper outlier fraction:0.05
    Averaging method:Mean
    Variance method:Standard deviation
    # of deviations:2
"""

# simpler sample pipeline - when cellprofiler not installed
SBS_PIPELINE_CORE_ONLY = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:300
GitHash:
ModuleCount:4
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:No filtering
    Select the rule criteria:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\'\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:
    Regular expression:(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$
    Extract metadata from:All images
    Select the filtering criteria:or (file does contain "")
    Metadata file location:
    Match file and image metadata:[]
    Use case insensitive matching?:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Assign a name to:Images matching rules
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:[{u'Cytoplasm': u'WellRow', u'DNACorr': None, 'DNA': u'WellRow', u'CytoplasmCorr': None}, {u'Cytoplasm': u'WellColumn', u'DNACorr': None, 'DNA': u'WellColumn', u'CytoplasmCorr': None}]
    Image set matching method:Metadata
    Set intensity range from:Yes
    Assignments count:4
    Single images count:0
    Maximum intensity:255.0
    Volumetric:No
    x:1.0
    y:1.0
    z:1.0
    Select the rule criteria:and (extension does istif) (metadata does C "2")
    Name to assign these images:DNA
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0
    Select the rule criteria:and (extension does istif) (metadata does C "1")
    Name to assign these images:Cytoplasm
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0
    Select the rule criteria:or (file does startwith "Channel1ILLUM")
    Name to assign these images:DNACorr
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0
    Select the rule criteria:or (file does contain "Channel2ILLUM")
    Name to assign these images:CytoplasmCorr
    Name to assign these objects:Cells
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedObjects
    Maximum intensity:255.0

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None
"""
