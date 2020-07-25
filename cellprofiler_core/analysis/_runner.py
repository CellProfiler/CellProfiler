import collections
import io
import queue
import subprocess
import tempfile

import numpy
import psutil
import zmq

import cellprofiler_core.analysis.event
import cellprofiler_core.analysis.reply
import cellprofiler_core.analysis.request
import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.workspace
from cellprofiler_core.analysis import *
from ..utilities.zmq import get_announcer_address, register_analysis
from ..utilities.zmq.communicable.reply._reply import Reply

logger = logging.getLogger(__name__)


class Runner:
    """The Runner manages two threads (per instance) and all of the
    workers (per class, i.e., singletons).

    The two threads run interface() and jobserver(), below.

    interface() is responsible grouping jobs for dispatch, tracking job
    progress, integrating measurements returned from workers.

    jobserver() is a lightweight thread that serves jobs and receives their
    requests, acting as a switchboard between workers, interface(), and
    whatever event_listener is present (via post_event()).

    workers are stored in AnalysisRunner.workers[], and are separate processes.
    They are expected to exit if their stdin() closes, e.g., if the parent
    process goes away.

    interface() and jobserver() communicate via Queues and using condition
    variables to get each other's attention.  zmqrequest is used to communicate
    between jobserver() and workers[].
    """

    # worker pool - shared by all instances
    workers = []
    deadman_switches = []

    # measurement status
    STATUS = "ProcessingStatus"
    STATUS_UNPROCESSED = "Unprocessed"
    STATUS_IN_PROCESS = "InProcess"
    STATUS_FINISHED_WAITING = "FinishedWaitingMeasurements"
    STATUS_DONE = "Done"
    STATUSES = [
        STATUS_UNPROCESSED,
        STATUS_IN_PROCESS,
        STATUS_FINISHED_WAITING,
        STATUS_DONE,
    ]

    def __init__(
        self, analysis_id, pipeline, initial_measurements_buf, event_listener,
    ):
        self.initial_measurements_buf = initial_measurements_buf

        self.analysis_id = analysis_id
        self.pipeline = pipeline.copy()
        self.event_listener = event_listener

        self.interface_work_cv = threading.Condition()
        self.jobserver_work_cv = threading.Condition()
        self.paused = False
        self.cancelled = False

        self.work_queue = queue.Queue()
        self.in_process_queue = queue.Queue()
        self.finished_queue = queue.Queue()

        # We use a queue size of 10 because we keep measurements in memory (as
        # their HDF5 file contents) until they get merged into the full
        # measurements set.  If at some point, this size is too limiting, we
        # should have jobserver() call load_measurements_from_buffer() rather
        # than interface() doing so.  Currently, passing measurements in this
        # way seems like it might be buggy:
        # http://code.google.com/p/h5py/issues/detail?id=244
        self.received_measurements_queue = queue.Queue(maxsize=10)

        self.shared_dicts = None

        self.interface_thread = None
        self.jobserver_thread = None

    # External control interfaces
    def start(self, num_workers=None, overwrite=True):
        """start the analysis run

        num_workers - # of workers to run, default = # of cores
        overwrite - if True, overwrite existing image set measurements, False
                    try to reuse them.
        """

        # Although it would be nice to reuse the worker pool, I'm not entirely
        # sure they recover correctly from the user cancelling an analysis
        # (e.g., during an interaction request).  This should be handled by
        # zmqRequest.cancel_analysis, but just in case, we stop the workers and
        # restart them.  Note that this creates a new announce port, so we
        # don't have to worry about old workers taking a job before noticing
        # that their stdin has closed.
        self.stop_workers()

        start_signal = threading.Semaphore(0)
        self.interface_thread = cellprofiler_core.analysis.start_daemon_thread(
            target=self.interface,
            args=(start_signal,),
            kwargs=dict(overwrite=overwrite),
            name="AnalysisRunner.interface",
        )
        #
        # Wait for signal on interface started.
        #
        start_signal.acquire()
        self.jobserver_thread = cellprofiler_core.analysis.start_daemon_thread(
            target=self.jobserver,
            args=(self.analysis_id, start_signal),
            name="AnalysisRunner.jobserver",
        )
        #
        # Wait for signal on jobserver started.
        #
        start_signal.acquire()
        # start worker pool via class method (below)
        self.start_workers(num_workers)

    def check(self):
        return (
            (self.interface_thread is not None)
            and (self.jobserver_thread is not None)
            and self.interface_thread.is_alive()
            and self.jobserver_thread.is_alive()
        )

    def notify_threads(self):
        with self.interface_work_cv:
            self.interface_work_cv.notify()
        with self.jobserver_work_cv:
            self.jobserver_work_cv.notify()

    def cancel(self):
        """cancel the analysis run"""
        logger.debug("Stopping workers")
        self.stop_workers()
        logger.debug("Canceling run")
        self.cancelled = True
        self.paused = False
        self.notify_threads()
        logger.debug("Waiting on interface thread")
        self.interface_thread.join()
        logger.debug("Waiting on jobserver thread")
        self.jobserver_thread.join()
        logger.debug("Cancel complete")

    def pause(self):
        """pause the analysis run"""
        self.paused = True
        self.notify_threads()

    def resume(self):
        """resume a paused analysis run"""
        self.paused = False
        self.notify_threads()

    # event posting
    def post_event(self, evt):
        self.event_listener(evt)

    def post_run_display_handler(self, workspace, module):
        event = cellprofiler_core.analysis.request.DisplayPostRun(
            module.module_num, workspace.display_data
        )
        self.event_listener(event)

    # XXX - catch and deal with exceptions in interface() and jobserver() threads
    def interface(
        self, start_signal, image_set_start=1, image_set_end=None, overwrite=True
    ):
        """Top-half thread for running an analysis.  Sets up grouping for jobs,
        deals with returned measurements, reports status periodically.

        start_signal- signal this semaphore when jobs are ready.
        image_set_start - beginning image set number to process
        image_set_end - last image set number to process
        overwrite - whether to recompute imagesets that already have data in initial_measurements.
        """
        from javabridge import attach, detach

        posted_analysis_started = False
        acknowledged_thread_start = False
        measurements = None
        workspace = None
        attach()
        try:
            # listen for pipeline events, and pass them upstream
            self.pipeline.add_listener(lambda pipe, evt: self.post_event(evt))

            initial_measurements = None
            # Make a temporary measurements file.
            fd, filename = tempfile.mkstemp(
                ".h5", dir=cellprofiler_core.preferences.get_temporary_directory()
            )
            try:
                fd = os.fdopen(fd, "wb")
                fd.write(self.initial_measurements_buf)
                fd.close()
                initial_measurements = cellprofiler_core.measurement.Measurements(
                    filename=filename, mode="r"
                )
                measurements = cellprofiler_core.measurement.Measurements(
                    image_set_start=None, copy=initial_measurements, mode="a"
                )
            finally:
                if initial_measurements is not None:
                    initial_measurements.close()
                os.unlink(filename)

            # The shared dicts are needed in jobserver()
            self.shared_dicts = [m.get_dictionary() for m in self.pipeline.modules()]
            workspace = cellprofiler_core.workspace.Workspace(
                self.pipeline,
                None,
                None,
                None,
                measurements,
                cellprofiler_core.image.ImageSetList(),
            )

            if image_set_end is None:
                image_set_end = measurements.get_image_numbers()[-1]
            image_sets_to_process = list(
                filter(
                    lambda x: image_set_start <= x <= image_set_end,
                    measurements.get_image_numbers(),
                )
            )

            self.post_event(cellprofiler_core.analysis.event.Started())
            posted_analysis_started = True

            # reset the status of every image set that needs to be processed
            has_groups = measurements.has_groups()
            if self.pipeline.requires_aggregation():
                overwrite = True
            if has_groups and not overwrite:
                if not measurements.has_feature(
                    cellprofiler_core.measurement.IMAGE, self.STATUS
                ):
                    overwrite = True
                else:
                    group_status = {}
                    for image_number in measurements.get_image_numbers():
                        group_number = measurements[
                            cellprofiler_core.measurement.IMAGE,
                            cellprofiler_core.measurement.GROUP_NUMBER,
                            image_number,
                        ]
                        status = measurements[
                            cellprofiler_core.measurement.IMAGE,
                            self.STATUS,
                            image_number,
                        ]
                        if status != self.STATUS_DONE:
                            group_status[group_number] = self.STATUS_UNPROCESSED
                        elif group_number not in group_status:
                            group_status[group_number] = self.STATUS_DONE

            new_image_sets_to_process = []
            for image_set_number in image_sets_to_process:
                needs_reset = False
                if (
                    overwrite
                    or (
                        not measurements.has_measurements(
                            cellprofiler_core.measurement.IMAGE,
                            self.STATUS,
                            image_set_number,
                        )
                    )
                    or (
                        measurements[
                            cellprofiler_core.measurement.IMAGE,
                            self.STATUS,
                            image_set_number,
                        ]
                        != self.STATUS_DONE
                    )
                ):
                    needs_reset = True
                elif has_groups:
                    group_number = measurements[
                        cellprofiler_core.measurement.IMAGE,
                        cellprofiler_core.measurement.GROUP_NUMBER,
                        image_set_number,
                    ]
                    if group_status[group_number] != self.STATUS_DONE:
                        needs_reset = True
                if needs_reset:
                    measurements[
                        cellprofiler_core.measurement.IMAGE,
                        self.STATUS,
                        image_set_number,
                    ] = self.STATUS_UNPROCESSED
                    new_image_sets_to_process.append(image_set_number)
            image_sets_to_process = new_image_sets_to_process

            # Find image groups.  These are written into measurements prior to
            # analysis.  Groups are processed as a single job.
            if has_groups or self.pipeline.requires_aggregation():
                worker_runs_post_group = True
                job_groups = {}
                for image_set_number in image_sets_to_process:
                    group_number = measurements[
                        cellprofiler_core.measurement.IMAGE,
                        cellprofiler_core.measurement.GROUP_NUMBER,
                        image_set_number,
                    ]
                    group_index = measurements[
                        cellprofiler_core.measurement.IMAGE,
                        cellprofiler_core.measurement.GROUP_INDEX,
                        image_set_number,
                    ]
                    job_groups[group_number] = job_groups.get(group_number, []) + [
                        (group_index, image_set_number)
                    ]
                job_groups = [
                    [isn for _, isn in sorted(job_groups[group_number])]
                    for group_number in sorted(job_groups)
                ]
            else:
                worker_runs_post_group = False  # prepare_group will be run in worker, but post_group is below.
                job_groups = [
                    [image_set_number] for image_set_number in image_sets_to_process
                ]

            # XXX - check that any constructed groups are complete, i.e.,
            # image_set_start and image_set_end shouldn't carve them up.

            if not worker_runs_post_group:
                # put the first job in the queue, then wait for the first image to
                # finish (see the check of self.finish_queue below) to post the rest.
                # This ensures that any shared data from the first imageset is
                # available to later imagesets.
                self.work_queue.put((job_groups[0], worker_runs_post_group, True))
                waiting_for_first_imageset = True
                del job_groups[0]
            else:
                waiting_for_first_imageset = False
                for job in job_groups:
                    self.work_queue.put((job, worker_runs_post_group, False))
                job_groups = []
            start_signal.release()
            acknowledged_thread_start = True

            # We loop until every image is completed, or an outside event breaks the loop.
            while not self.cancelled:

                # gather measurements
                while not self.received_measurements_queue.empty():
                    image_numbers, buf = self.received_measurements_queue.get()
                    image_numbers = [int(i) for i in image_numbers]
                    recd_measurements = cellprofiler_core.measurement.load_measurements_from_buffer(
                        buf
                    )
                    self.copy_recieved_measurements(
                        recd_measurements, measurements, image_numbers
                    )
                    recd_measurements.close()
                    del recd_measurements

                # check for jobs in progress
                while not self.in_process_queue.empty():
                    image_set_numbers = self.in_process_queue.get()
                    for image_set_number in image_set_numbers:
                        measurements[
                            cellprofiler_core.measurement.IMAGE,
                            self.STATUS,
                            int(image_set_number),
                        ] = self.STATUS_IN_PROCESS

                # check for finished jobs that haven't returned measurements, yet
                while not self.finished_queue.empty():
                    finished_req = self.finished_queue.get()
                    measurements[
                        cellprofiler_core.measurement.IMAGE,
                        self.STATUS,
                        int(finished_req.image_set_number),
                    ] = self.STATUS_FINISHED_WAITING
                    if waiting_for_first_imageset:
                        assert isinstance(
                            finished_req,
                            cellprofiler_core.analysis.reply.ImageSetSuccessWithDictionary,
                        )
                        self.shared_dicts = finished_req.shared_dicts
                        waiting_for_first_imageset = False
                        assert len(self.shared_dicts) == len(self.pipeline.modules())
                        # if we had jobs waiting for the first image set to finish,
                        # queue them now that the shared state is available.
                        for job in job_groups:
                            self.work_queue.put((job, worker_runs_post_group, False))
                    finished_req.reply(cellprofiler_core.analysis.reply.Ack())

                # check progress and report
                counts = collections.Counter(
                    measurements[
                        cellprofiler_core.measurement.IMAGE,
                        self.STATUS,
                        image_set_number,
                    ]
                    for image_set_number in image_sets_to_process
                )
                self.post_event(cellprofiler_core.analysis.event.Progress(counts))

                # Are we finished?
                if counts[self.STATUS_DONE] == len(image_sets_to_process):
                    last_image_number = measurements.get_image_numbers()[-1]
                    measurements.image_set_number = last_image_number
                    if not worker_runs_post_group:
                        self.pipeline.post_group(workspace, {})

                    workspace = cellprofiler_core.workspace.Workspace(
                        self.pipeline, None, None, None, measurements, None, None
                    )
                    workspace.post_run_display_handler = self.post_run_display_handler
                    self.pipeline.post_run(workspace)
                    break

                measurements.flush()
                # not done, wait for more work
                with self.interface_work_cv:
                    while self.paused or (
                        (not self.cancelled)
                        and self.in_process_queue.empty()
                        and self.finished_queue.empty()
                        and self.received_measurements_queue.empty()
                    ):
                        self.interface_work_cv.wait()  # wait for a change of status or work to arrive
        finally:
            detach()
            # Note - the measurements file is owned by the queue consumer
            #        after this post_event.
            #
            if not acknowledged_thread_start:
                start_signal.release()
            if posted_analysis_started:
                was_cancelled = self.cancelled
                self.post_event(
                    cellprofiler_core.analysis.event.Finished(
                        measurements, was_cancelled
                    )
                )
            self.stop_workers()
        self.analysis_id = False  # this will cause the jobserver thread to exit

    def copy_recieved_measurements(
        self, recd_measurements, measurements, image_numbers
    ):
        """Copy the received measurements to the local process' measurements

        recd_measurements - measurements received from worker

        measurements - local measurements = destination for copy

        image_numbers - image numbers processed by worker
        """
        measurements.copy_relationships(recd_measurements)
        for o in recd_measurements.get_object_names():
            if o == cellprofiler_core.measurement.EXPERIMENT:
                continue  # Written during prepare_run / post_run
            elif o == cellprofiler_core.measurement.IMAGE:
                # Some have been previously written. It's worth the time
                # to check values and only write changes
                for feature in recd_measurements.get_feature_names(o):
                    if not measurements.has_feature(
                        cellprofiler_core.measurement.IMAGE, feature
                    ):
                        f_image_numbers = image_numbers
                    else:
                        local_values = measurements[
                            cellprofiler_core.measurement.IMAGE, feature, image_numbers
                        ]
                        remote_values = recd_measurements[
                            cellprofiler_core.measurement.IMAGE, feature, image_numbers
                        ]
                        f_image_numbers = [
                            i
                            for i, lv, rv in zip(
                                image_numbers, local_values, remote_values
                            )
                            if (
                                numpy.any(rv != lv)
                                if isinstance(lv, numpy.ndarray)
                                else lv != rv
                            )
                        ]
                    if len(f_image_numbers) > 0:
                        measurements[o, feature, f_image_numbers] = recd_measurements[
                            o, feature, f_image_numbers
                        ]
            else:
                for feature in recd_measurements.get_feature_names(o):
                    measurements[o, feature, image_numbers] = recd_measurements[
                        o, feature, image_numbers
                    ]
        for image_set_number in image_numbers:
            measurements[
                cellprofiler_core.measurement.IMAGE, self.STATUS, image_set_number
            ] = self.STATUS_DONE

    def jobserver(self, analysis_id, start_signal):
        # this server subthread should be very lightweight, as it has to handle
        # all the requests from workers, of which there might be several.

        # start the zmqrequest Boundary
        request_queue = queue.Queue()
        boundary = register_analysis(analysis_id, request_queue)
        #
        # The boundary is announcing our analysis at this point. Workers
        # will get announcements if they connect.
        #
        start_signal.release()

        # XXX - is this just to keep from posting another AnalysisPaused event?
        # If so, probably better to simplify the code and keep sending them
        # (should be only one per second).
        i_was_paused_before = False

        # start serving work until the analysis is done (or changed)
        while not self.cancelled:

            with self.jobserver_work_cv:
                if self.paused and not i_was_paused_before:
                    self.post_event(cellprofiler_core.analysis.event.Paused())
                    i_was_paused_before = True
                if self.paused or request_queue.empty():
                    self.jobserver_work_cv.wait(
                        1
                    )  # we timeout in order to keep announcing ourselves.
                    continue  # back to while... check that we're still running

            if i_was_paused_before:
                self.post_event(cellprofiler_core.analysis.event.Resumed())
                i_was_paused_before = False

            try:
                req = request_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if isinstance(req, cellprofiler_core.analysis.request.PipelinePreferences):
                logger.debug("Received pipeline preferences request")
                req.reply(
                    Reply(
                        pipeline_blob=numpy.array(self.pipeline_as_string()),
                        preferences=cellprofiler_core.preferences.preferences_as_dict(),
                    )
                )
                logger.debug("Replied to pipeline preferences request")
            elif isinstance(
                req, cellprofiler_core.analysis.request.InitialMeasurements
            ):
                logger.debug("Received initial measurements request")
                req.reply(Reply(buf=self.initial_measurements_buf))
                logger.debug("Replied to initial measurements request")
            elif isinstance(req, cellprofiler_core.analysis.request.Work):
                if not self.work_queue.empty():
                    logger.debug("Received work request")
                    (
                        job,
                        worker_runs_post_group,
                        wants_dictionary,
                    ) = self.work_queue.get()
                    req.reply(
                        cellprofiler_core.analysis.reply.Work(
                            image_set_numbers=job,
                            worker_runs_post_group=worker_runs_post_group,
                            wants_dictionary=wants_dictionary,
                        )
                    )
                    self.queue_dispatched_job(job)
                    logger.debug(
                        "Dispatched job: image sets=%s"
                        % ",".join([str(i) for i in job])
                    )
                else:
                    # there may be no work available, currently, but there
                    # may be some later.
                    req.reply(cellprofiler_core.analysis.reply.NoWork())
            elif isinstance(req, cellprofiler_core.analysis.reply.ImageSetSuccess):
                # interface() is responsible for replying, to allow it to
                # request the shared_state dictionary if needed.
                logger.debug("Received ImageSetSuccess")
                self.queue_imageset_finished(req)
                logger.debug("Enqueued ImageSetSuccess")
            elif isinstance(req, cellprofiler_core.analysis.request.SharedDictionary):
                logger.debug("Received shared dictionary request")
                req.reply(
                    cellprofiler_core.analysis.reply.SharedDictionary(
                        dictionaries=self.shared_dicts
                    )
                )
                logger.debug("Sent shared dictionary reply")
            elif isinstance(req, cellprofiler_core.analysis.request.MeasurementsReport):
                logger.debug("Received measurements report")
                self.queue_received_measurements(req.image_set_numbers, req.buf)
                req.reply(cellprofiler_core.analysis.reply.Ack())
                logger.debug("Acknowledged measurements report")
            elif isinstance(req, cellprofiler_core.analysis.request.AnalysisCancel):
                # Signal the interface that we are cancelling
                logger.debug("Received analysis worker cancel request")
                with self.interface_work_cv:
                    self.cancelled = True
                    self.interface_work_cv.notify()
                req.reply(cellprofiler_core.analysis.reply.Ack())
            elif isinstance(
                req,
                (
                    cellprofiler_core.analysis.request.Interaction,
                    cellprofiler_core.analysis.request.Display,
                    cellprofiler_core.analysis.request.DisplayPostGroup,
                    cellprofiler_core.analysis.request.ExceptionReport,
                    cellprofiler_core.analysis.request.DebugWaiting,
                    cellprofiler_core.analysis.request.DebugComplete,
                    cellprofiler_core.analysis.request.OmeroLogin,
                ),
            ):
                logger.debug("Enqueueing interactive request")
                # bump upward
                self.post_event(req)
                logger.debug("Interactive request enqueued")
            else:
                msg = "Unknown request from worker: %s of type %s" % (req, type(req))
                logger.error(msg)
                raise ValueError(msg)

        # stop the ZMQ-boundary thread - will also deal with any requests waiting on replies
        boundary.cancel(analysis_id)

    def queue_job(self, image_set_number):
        self.work_queue.put(image_set_number)

    def queue_dispatched_job(self, job):
        self.in_process_queue.put(job)
        # notify interface thread
        with self.interface_work_cv:
            self.interface_work_cv.notify()

    def queue_imageset_finished(self, finished_req):
        self.finished_queue.put(finished_req)
        # notify interface thread
        with self.interface_work_cv:
            self.interface_work_cv.notify()

    def queue_received_measurements(self, image_set_numbers, measurements):
        self.received_measurements_queue.put((image_set_numbers, measurements))
        # notify interface thread
        with self.interface_work_cv:
            self.interface_work_cv.notify()

    def pipeline_as_string(self):
        s = io.StringIO()
        cellprofiler_core.pipeline.dump(self.pipeline, s, version=5)
        return s.getvalue()

    # Class methods for managing the worker pool
    @classmethod
    def start_workers(cls, num=None):
        if cls.workers:
            return

        if num is None:
            num = psutil.cpu_count(logical=False)

        cls.work_announce_address = get_announcer_address()
        logger.info("Starting workers on address %s" % cls.work_announce_address)
        if "CP_DEBUG_WORKER" in os.environ:
            if os.environ["CP_DEBUG_WORKER"] == "NOT_INPROC":
                return
            from cellprofiler_core.worker import Worker, NOTIFY_ADDR, NOTIFY_STOP
            from cellprofiler_core.pipeline.event import CancelledException

            class WorkerRunner(threading.Thread):
                def __init__(self, work_announce_address):
                    threading.Thread.__init__(self)
                    self.work_announce_address = work_announce_address
                    self.notify_socket = zmq.Context.instance().socket(zmq.PUB)
                    self.notify_socket.bind(NOTIFY_ADDR)

                def run(self):
                    with Worker(self.work_announce_address) as aw:
                        try:
                            aw.run()
                        except CancelledException:
                            logger.info("Exiting debug worker thread")

                def wait(self):
                    self.notify_socket.send(NOTIFY_STOP)
                    self.join()

            thread = WorkerRunner(cls.work_announce_address)
            thread.setDaemon(True)
            thread.start()
            cls.workers.append(thread)
            return

        close_fds = False
        # start workers
        for idx in range(num):
            if sys.platform == "darwin":
                cellprofiler_core.analysis.close_all_on_exec()

            aw_args = [
                "--work-announce",
                cls.work_announce_address,
                "--plugins-directory",
                cellprofiler_core.preferences.get_plugin_directory(),
            ]
            # stdin for the subprocesses serves as a deadman's switch.  When
            # closed, the subprocess exits.
            if hasattr(sys, "frozen"):
                if sys.platform == "darwin":
                    executable = os.path.join(os.path.dirname(sys.executable), "cp")
                    args = [executable] + aw_args
                elif sys.platform.startswith("linux"):
                    aw_path = os.path.join(os.path.dirname(__file__), "__init__.py")
                    args = [sys.executable, aw_path] + aw_args
                else:
                    args = [sys.executable] + aw_args

                worker = subprocess.Popen(
                    args,
                    env=cellprofiler_core.analysis.find_worker_env(idx),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    close_fds=close_fds,
                )
            else:
                worker = subprocess.Popen(
                    [
                        cellprofiler_core.analysis.find_python(),
                        "-u",
                        cellprofiler_core.analysis.find_analysis_worker_source(),
                    ]  # unbuffered
                    + aw_args,
                    env=cellprofiler_core.analysis.find_worker_env(idx),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    close_fds=close_fds,
                )

            def run_logger(workR, widx):
                while True:
                    try:
                        line = workR.stdout.readline()
                        line = line.decode("utf-8")
                        if not line:
                            break
                        logger.info("Worker %d: %s", widx, line.rstrip())
                    except:
                        break

            cellprofiler_core.analysis.start_daemon_thread(
                target=run_logger, args=(worker, idx), name="worker stdout logger"
            )

            cls.workers += [worker]
            cls.deadman_switches += [worker.stdin]  # closing stdin will kill subprocess

    @classmethod
    def stop_workers(cls):
        for deadman_switch in cls.deadman_switches:
            deadman_switch.close()
        for worker in cls.workers:
            worker.wait()
        cls.workers = []
        cls.deadman_switches = []
