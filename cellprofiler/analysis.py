"""analysis.py - Run pipelines on imagesets to produce measurements.
"""
from __future__ import with_statement

import Queue
import cStringIO as StringIO
import collections
import logging
import multiprocessing
import os
import os.path
import subprocess
import sys
import tempfile
import threading
import uuid

import h5py
import numpy as np
import zmq

import cellprofiler
import cellprofiler.image as cpimage
import cellprofiler.measurement as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.workspace as cpw
from cellprofiler.utilities.zmqrequest import AnalysisRequest, Request, Reply, UpstreamExit
from cellprofiler.utilities.zmqrequest import get_announcer_address
from cellprofiler.utilities.zmqrequest import register_analysis, cancel_analysis

logger = logging.getLogger(__name__)

use_analysis = True

DEBUG = 'DEBUG'
ANNOUNCE_DONE = "DONE"


class Analysis(object):
    '''An Analysis is the application of a particular pipeline of modules to a
    set of images to produce measurements.

    Multiprocessing for analyses is handled by multiple layers of threads and
    processes, to keep the GUI responsive and simplify the code.  Threads and
    processes are organized as below.  Display/Interaction requests and
    Exceptions are sent directly to the pipeline listener.

    +------------------------------------------------+
    |           CellProfiler GUI/WX thread           |
    |                                                |
    +- Analysis() methods down,  Events/Requests up -+
    |                                                |
    |       AnalysisRunner.interface() thread        |
    |                                                |
    +----------------  Queues  ----------------------+
    |                                                |
    |  AnalysisRunner.jobserver()/announce() threads |
    |                                                |
    +----------------------------------------------- +
    |              zmqrequest.Boundary()             |
    +---------------+----------------+---------------+
    |     Worker    |     Worker     |   Worker      |
    +---------------+----------------+---------------+

    Workers are managed by class variables in the AnalysisRunner.
    '''

    def __init__(self, pipeline, measurements_filename,
                 initial_measurements=None):
        '''create an Analysis applying pipeline to a set of images, writing out
        to measurements_filename, optionally starting with previous
        measurements.'''
        self.pipeline = pipeline
        initial_measurements = cpmeas.Measurements(copy=initial_measurements)
        self.initial_measurements_buf = initial_measurements.file_contents()
        initial_measurements.close()
        self.output_path = measurements_filename
        self.debug_mode = False
        self.analysis_in_progress = False
        self.runner = None

        self.runner_lock = threading.Lock()  # defensive coding purposes

    def start(self, analysis_event_callback, num_workers=None, overwrite=True):
        '''Start the analysis runner

        analysis_event_callback - callback from runner to UI thread for
                                  event progress and UI handlers

        num_workers - # of worker processes to instantiate, default is # of cores

        overwrite - True (default) to process all image sets, False to only
                    process incomplete ones (or incomplete groups if grouping)
        '''
        with self.runner_lock:
            assert not self.analysis_in_progress
            self.analysis_in_progress = uuid.uuid1().hex

            self.runner = AnalysisRunner(self.analysis_in_progress,
                                         self.pipeline,
                                         self.initial_measurements_buf,
                                         self.output_path,
                                         analysis_event_callback)
            self.runner.start(num_workers=num_workers, overwrite=overwrite)
            return self.analysis_in_progress

    def pause(self):
        with self.runner_lock:
            assert self.analysis_in_progress
            self.runner.pause()

    def resume(self):
        with self.runner_lock:
            assert self.analysis_in_progress
            self.runner.resume()

    def cancel(self):
        with self.runner_lock:
            if not self.analysis_in_progress:
                return
            self.analysis_in_progress = False
            self.runner.cancel()
            self.runner = None

    def check_running(self):
        '''Verify that an analysis is running, allowing the GUI to recover even
        if the AnalysisRunner fails in some way.

        Returns True if analysis is still running (threads are still alive).
        '''
        with self.runner_lock:
            if self.analysis_in_progress:
                return self.runner.check()
            return False


class AnalysisRunner(object):
    '''The AnalysisRunner manages two threads (per instance) and all of the
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
    '''

    # worker pool - shared by all instances
    workers = []
    deadman_switches = []

    # measurement status
    STATUS = "ProcessingStatus"
    STATUS_UNPROCESSED = "Unprocessed"
    STATUS_IN_PROCESS = "InProcess"
    STATUS_FINISHED_WAITING = "FinishedWaitingMeasurements"
    STATUS_DONE = "Done"
    STATUSES = [STATUS_UNPROCESSED, STATUS_IN_PROCESS, STATUS_FINISHED_WAITING, STATUS_DONE]

    def __init__(self, analysis_id, pipeline,
                 initial_measurements_buf,
                 output_path, event_listener):
        self.initial_measurements_buf = initial_measurements_buf

        # for writing results
        self.output_path = output_path

        self.analysis_id = analysis_id
        self.pipeline = pipeline.copy()
        self.event_listener = event_listener

        self.interface_work_cv = threading.Condition()
        self.jobserver_work_cv = threading.Condition()
        self.paused = False
        self.cancelled = False

        self.work_queue = Queue.Queue()
        self.in_process_queue = Queue.Queue()
        self.finished_queue = Queue.Queue()

        # We use a queue size of 10 because we keep measurements in memory (as
        # their HDF5 file contents) until they get merged into the full
        # measurements set.  If at some point, this size is too limiting, we
        # should have jobserver() call load_measurements_from_buffer() rather
        # than interface() doing so.  Currently, passing measurements in this
        # way seems like it might be buggy:
        # http://code.google.com/p/h5py/issues/detail?id=244
        self.received_measurements_queue = Queue.Queue(maxsize=10)

        self.shared_dicts = None

        self.interface_thread = None
        self.jobserver_thread = None

    # External control interfaces
    def start(self, num_workers=None, overwrite=True):
        '''start the analysis run

        num_workers - # of workers to run, default = # of cores
        overwrite - if True, overwrite existing image set measurements, False
                    try to reuse them.
        '''

        # Although it would be nice to reuse the worker pool, I'm not entirely
        # sure they recover correctly from the user cancelling an analysis
        # (e.g., during an interaction request).  This should be handled by
        # zmqRequest.cancel_analysis, but just in case, we stop the workers and
        # restart them.  Note that this creates a new announce port, so we
        # don't have to worry about old workers taking a job before noticing
        # that their stdin has closed.
        self.stop_workers()

        start_signal = threading.Semaphore(0)
        self.interface_thread = start_daemon_thread(
                target=self.interface,
                args=(start_signal,),
                kwargs=dict(overwrite=overwrite),
                name='AnalysisRunner.interface')
        #
        # Wait for signal on interface started.
        #
        start_signal.acquire()
        self.jobserver_thread = start_daemon_thread(
                target=self.jobserver,
                args=(self.analysis_id, start_signal),
                name='AnalysisRunner.jobserver')
        #
        # Wait for signal on jobserver started.
        #
        start_signal.acquire()
        # start worker pool via class method (below)
        self.start_workers(num_workers)

    def check(self):
        return ((self.interface_thread is not None) and
                (self.jobserver_thread is not None) and
                self.interface_thread.is_alive() and
                self.jobserver_thread.is_alive())

    def notify_threads(self):
        with self.interface_work_cv:
            self.interface_work_cv.notify()
        with self.jobserver_work_cv:
            self.jobserver_work_cv.notify()

    def cancel(self):
        '''cancel the analysis run'''
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
        '''pause the analysis run'''
        self.paused = True
        self.notify_threads()

    def resume(self):
        '''resume a paused analysis run'''
        self.paused = False
        self.notify_threads()

    # event posting
    def post_event(self, evt):
        self.event_listener(evt)

    def post_run_display_handler(self, workspace, module):
        event = DisplayPostRunRequest(module.module_num,
                                      workspace.display_data)
        self.event_listener(event)

    # XXX - catch and deal with exceptions in interface() and jobserver() threads
    def interface(self,
                  start_signal,
                  image_set_start=1,
                  image_set_end=None,
                  overwrite=True):
        '''Top-half thread for running an analysis.  Sets up grouping for jobs,
        deals with returned measurements, reports status periodically.

        start_signal- signal this semaphore when jobs are ready.
        image_set_start - beginning image set number to process
        image_set_end - last image set number to process
        overwrite - whether to recompute imagesets that already have data in initial_measurements.
        '''
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
            if self.output_path is None:
                # Caller wants a temporary measurements file.
                fd, filename = tempfile.mkstemp(
                        ".h5", dir=cpprefs.get_temporary_directory())
                try:
                    fd = os.fdopen(fd, "wb")
                    fd.write(self.initial_measurements_buf)
                    fd.close()
                    initial_measurements = cpmeas.Measurements(
                            filename=filename, mode="r")
                    measurements = cpmeas.Measurements(
                            image_set_start=None,
                            copy=initial_measurements,
                            mode="a")
                finally:
                    if initial_measurements is not None:
                        initial_measurements.close()
                    os.unlink(filename)
            else:
                with open(self.output_path, "wb") as fd:
                    fd.write(self.initial_measurements_buf)
                measurements = cpmeas.Measurements(image_set_start=None,
                                                   filename=self.output_path,
                                                   mode="a")
            # The shared dicts are needed in jobserver()
            self.shared_dicts = [m.get_dictionary() for m in self.pipeline.modules()]
            workspace = cpw.Workspace(self.pipeline, None, None, None,
                                      measurements, cpimage.ImageSetList())

            if image_set_end is None:
                image_set_end = measurements.get_image_numbers()[-1]
            image_sets_to_process = filter(
                    lambda x: x >= image_set_start and x <= image_set_end,
                    measurements.get_image_numbers())

            self.post_event(AnalysisStarted())
            posted_analysis_started = True

            # reset the status of every image set that needs to be processed
            has_groups = measurements.has_groups()
            if self.pipeline.requires_aggregation():
                overwrite = True
            if has_groups and not overwrite:
                if not measurements.has_feature(cpmeas.IMAGE, self.STATUS):
                    overwrite = True
                else:
                    group_status = {}
                    for image_number in measurements.get_image_numbers():
                        group_number = measurements[
                            cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_number]
                        status = measurements[cpmeas.IMAGE, self.STATUS,
                                              image_number]
                        if status != self.STATUS_DONE:
                            group_status[group_number] = self.STATUS_UNPROCESSED
                        elif group_number not in group_status:
                            group_status[group_number] = self.STATUS_DONE

            new_image_sets_to_process = []
            for image_set_number in image_sets_to_process:
                needs_reset = False
                if (overwrite or
                        (not measurements.has_measurements(
                                cpmeas.IMAGE, self.STATUS, image_set_number)) or
                        (measurements[cpmeas.IMAGE, self.STATUS, image_set_number]
                             != self.STATUS_DONE)):
                    needs_reset = True
                elif has_groups:
                    group_number = measurements[
                        cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_set_number]
                    if group_status[group_number] != self.STATUS_DONE:
                        needs_reset = True
                if needs_reset:
                    measurements[cpmeas.IMAGE, self.STATUS, image_set_number] = \
                        self.STATUS_UNPROCESSED
                    new_image_sets_to_process.append(image_set_number)
            image_sets_to_process = new_image_sets_to_process

            # Find image groups.  These are written into measurements prior to
            # analysis.  Groups are processed as a single job.
            if has_groups or self.pipeline.requires_aggregation():
                worker_runs_post_group = True
                job_groups = {}
                for image_set_number in image_sets_to_process:
                    group_number = measurements[cpmeas.IMAGE,
                                                cpmeas.GROUP_NUMBER,
                                                image_set_number]
                    group_index = measurements[cpmeas.IMAGE,
                                               cpmeas.GROUP_INDEX,
                                               image_set_number]
                    job_groups[group_number] = job_groups.get(group_number, []) + [(group_index, image_set_number)]
                job_groups = [[isn for _, isn in sorted(job_groups[group_number])]
                              for group_number in sorted(job_groups)]
            else:
                worker_runs_post_group = False  # prepare_group will be run in worker, but post_group is below.
                job_groups = [[image_set_number] for image_set_number in image_sets_to_process]

            # XXX - check that any constructed groups are complete, i.e.,
            # image_set_start and image_set_end shouldn't carve them up.

            if not worker_runs_post_group:
                # put the first job in the queue, then wait for the first image to
                # finish (see the check of self.finish_queue below) to post the rest.
                # This ensures that any shared data from the first imageset is
                # available to later imagesets.
                self.work_queue.put((job_groups[0],
                                     worker_runs_post_group,
                                     True))
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
                    recd_measurements = cpmeas.load_measurements_from_buffer(buf)
                    self.copy_recieved_measurements(recd_measurements, measurements, image_numbers)
                    recd_measurements.close()
                    del recd_measurements

                # check for jobs in progress
                while not self.in_process_queue.empty():
                    image_set_numbers = self.in_process_queue.get()
                    for image_set_number in image_set_numbers:
                        measurements[cpmeas.IMAGE, self.STATUS, int(image_set_number)] = self.STATUS_IN_PROCESS

                # check for finished jobs that haven't returned measurements, yet
                while not self.finished_queue.empty():
                    finished_req = self.finished_queue.get()
                    measurements[
                        cpmeas.IMAGE, self.STATUS, int(finished_req.image_set_number)] = self.STATUS_FINISHED_WAITING
                    if waiting_for_first_imageset:
                        assert isinstance(finished_req,
                                          ImageSetSuccessWithDictionary)
                        self.shared_dicts = finished_req.shared_dicts
                        waiting_for_first_imageset = False
                        assert len(self.shared_dicts) == len(self.pipeline.modules())
                        # if we had jobs waiting for the first image set to finish,
                        # queue them now that the shared state is available.
                        for job in job_groups:
                            self.work_queue.put((job, worker_runs_post_group, False))
                    finished_req.reply(Ack())

                # check progress and report
                counts = collections.Counter(measurements[cpmeas.IMAGE, self.STATUS, image_set_number]
                                             for image_set_number in image_sets_to_process)
                self.post_event(AnalysisProgress(counts))

                # Are we finished?
                if counts[self.STATUS_DONE] == len(image_sets_to_process):
                    last_image_number = measurements.get_image_numbers()[-1]
                    measurements.image_set_number = last_image_number
                    if not worker_runs_post_group:
                        self.pipeline.post_group(workspace, {})

                    workspace = cpw.Workspace(self.pipeline,
                                              None, None, None,
                                              measurements, None, None)
                    workspace.post_run_display_handler = \
                        self.post_run_display_handler
                    self.pipeline.post_run(workspace)
                    break

                measurements.flush()
                # not done, wait for more work
                with self.interface_work_cv:
                    while (self.paused or
                               ((not self.cancelled) and
                                    self.in_process_queue.empty() and
                                    self.finished_queue.empty() and
                                    self.received_measurements_queue.empty())):
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
                self.post_event(AnalysisFinished(measurements, was_cancelled))
            self.stop_workers()
        self.analysis_id = False  # this will cause the jobserver thread to exit

    def copy_recieved_measurements(self,
                                   recd_measurements,
                                   measurements,
                                   image_numbers):
        '''Copy the received measurements to the local process' measurements

        recd_measurements - measurements received from worker

        measurements - local measurements = destination for copy

        image_numbers - image numbers processed by worker
        '''
        measurements.copy_relationships(recd_measurements)
        for o in recd_measurements.get_object_names():
            if o == cpmeas.EXPERIMENT:
                continue  # Written during prepare_run / post_run
            elif o == cpmeas.IMAGE:
                # Some have been previously written. It's worth the time
                # to check values and only write changes
                for feature in recd_measurements.get_feature_names(o):
                    if not measurements.has_feature(cpmeas.IMAGE, feature):
                        f_image_numbers = image_numbers
                    else:
                        local_values = measurements[
                            cpmeas.IMAGE, feature, image_numbers]
                        remote_values = recd_measurements[
                            cpmeas.IMAGE, feature, image_numbers]
                        f_image_numbers = [
                            i for i, lv, rv in zip(
                                    image_numbers, local_values, remote_values)
                            if (np.any(rv != lv) if isinstance(lv, np.ndarray)
                                else lv != rv)]
                    if len(f_image_numbers) > 0:
                        measurements[o, feature, f_image_numbers] \
                            = recd_measurements[o, feature, f_image_numbers]
            else:
                for feature in recd_measurements.get_feature_names(o):
                    measurements[o, feature, image_numbers] \
                        = recd_measurements[o, feature, image_numbers]
        for image_set_number in image_numbers:
            measurements[cpmeas.IMAGE, self.STATUS, image_set_number] = self.STATUS_DONE

    def jobserver(self, analysis_id, start_signal):
        # this server subthread should be very lightweight, as it has to handle
        # all the requests from workers, of which there might be several.

        # start the zmqrequest Boundary
        request_queue = Queue.Queue()
        boundary = register_analysis(analysis_id,
                                     request_queue)
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
                    self.post_event(AnalysisPaused())
                    i_was_paused_before = True
                if self.paused or request_queue.empty():
                    self.jobserver_work_cv.wait(1)  # we timeout in order to keep announcing ourselves.
                    continue  # back to while... check that we're still running

            if i_was_paused_before:
                self.post_event(AnalysisResumed())
                i_was_paused_before = False

            try:
                req = request_queue.get(timeout=0.25)
            except Queue.Empty:
                continue

            if isinstance(req, PipelinePreferencesRequest):
                logger.debug("Received pipeline preferences request")
                req.reply(Reply(pipeline_blob=np.array(self.pipeline_as_string()),
                                preferences=cpprefs.preferences_as_dict()))
                logger.debug("Replied to pipeline preferences request")
            elif isinstance(req, InitialMeasurementsRequest):
                logger.debug("Received initial measurements request")
                req.reply(Reply(buf=self.initial_measurements_buf))
                logger.debug("Replied to initial measurements request")
            elif isinstance(req, WorkRequest):
                if not self.work_queue.empty():
                    logger.debug("Received work request")
                    job, worker_runs_post_group, wants_dictionary = \
                        self.work_queue.get()
                    req.reply(WorkReply(
                            image_set_numbers=job,
                            worker_runs_post_group=worker_runs_post_group,
                            wants_dictionary=wants_dictionary))
                    self.queue_dispatched_job(job)
                    logger.debug("Dispatched job: image sets=%s" %
                                 ",".join([str(i) for i in job]))
                else:
                    # there may be no work available, currently, but there
                    # may be some later.
                    req.reply(NoWorkReply())
            elif isinstance(req, ImageSetSuccess):
                # interface() is responsible for replying, to allow it to
                # request the shared_state dictionary if needed.
                logger.debug("Received ImageSetSuccess")
                self.queue_imageset_finished(req)
                logger.debug("Enqueued ImageSetSuccess")
            elif isinstance(req, SharedDictionaryRequest):
                logger.debug("Received shared dictionary request")
                req.reply(SharedDictionaryReply(dictionaries=self.shared_dicts))
                logger.debug("Sent shared dictionary reply")
            elif isinstance(req, MeasurementsReport):
                logger.debug("Received measurements report")
                self.queue_received_measurements(req.image_set_numbers,
                                                 req.buf)
                req.reply(Ack())
                logger.debug("Acknowledged measurements report")
            elif isinstance(req, AnalysisCancelRequest):
                # Signal the interface that we are cancelling
                logger.debug("Received analysis worker cancel request")
                with self.interface_work_cv:
                    self.cancelled = True
                    self.interface_work_cv.notify()
                req.reply(Ack())
            elif isinstance(req, (InteractionRequest, DisplayRequest,
                                  DisplayPostGroupRequest,
                                  ExceptionReport, DebugWaiting, DebugComplete,
                                  OmeroLoginRequest)):
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
        s = StringIO.StringIO()
        self.pipeline.savetxt(s)
        return s.getvalue()

    # Class methods for managing the worker pool
    @classmethod
    def start_workers(cls, num=None):
        if cls.workers:
            return

        try:
            num = multiprocessing.cpu_count() if num is None else num
        except NotImplementedError:
            num = 4

        cls.work_announce_address = get_announcer_address()
        logger.info("Starting workers on address %s" % cls.work_announce_address)
        if 'CP_DEBUG_WORKER' in os.environ:
            if os.environ['CP_DEBUG_WORKER'] == 'NOT_INPROC':
                return
            from cellprofiler.worker import \
                AnalysisWorker, NOTIFY_ADDR, NOTIFY_STOP
            from cellprofiler.pipeline import CancelledException

            class WorkerRunner(threading.Thread):
                def __init__(self, work_announce_address):
                    threading.Thread.__init__(self)
                    self.work_announce_address = work_announce_address
                    self.notify_socket = zmq.Context.instance().socket(zmq.PUB)
                    self.notify_socket.bind(NOTIFY_ADDR)

                def run(self):
                    with AnalysisWorker(self.work_announce_address) as aw:
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
            if sys.platform == 'darwin':
                close_all_on_exec()

            aw_args = ["--work-announce", cls.work_announce_address,
                       "--plugins-directory", cpprefs.get_plugin_directory(),
                       "--ij-plugins-directory", cpprefs.get_ij_plugin_directory()]
            jvm_arg = "%dm" % cpprefs.get_jvm_heap_mb()
            aw_args.append("--jvm-heap-size=%s" % jvm_arg)
            # stdin for the subprocesses serves as a deadman's switch.  When
            # closed, the subprocess exits.
            if hasattr(sys, 'frozen'):
                if sys.platform == 'darwin':
                    executable = os.path.join(
                            os.path.dirname(sys.executable), "cp")
                    args = ([executable] + aw_args)
                elif sys.platform.startswith('linux'):
                    aw_path = os.path.join(os.path.dirname(__file__),
                                           "worker.py")
                    args = [sys.executable, aw_path] + aw_args
                else:
                    args = [sys.executable] + aw_args

                worker = subprocess.Popen(args,
                                          env=find_worker_env(idx),
                                          stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT,
                                          close_fds=close_fds)
            else:
                worker = subprocess.Popen(
                        [find_python(),
                         '-u',  # unbuffered
                         find_analysis_worker_source()] + aw_args,
                        env=find_worker_env(idx),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        close_fds=close_fds)

            def run_logger(workR, widx):
                while True:
                    try:
                        line = workR.stdout.readline()
                        if not line:
                            break
                        logger.info("Worker %d: %s", widx, line.rstrip())
                    except:
                        break

            start_daemon_thread(target=run_logger, args=(worker, idx,), name='worker stdout logger')

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


def find_python():
    if hasattr(sys, 'frozen'):
        if sys.platform == "darwin":
            app_python = os.path.join(os.path.dirname(os.environ['ARGVZERO']), "python")
            return app_python
    return sys.executable


def find_worker_env(idx):
    '''Construct a command-line environment for the worker

    idx - index of the worker, e.g., 0 for the first, 1 for the second...
    '''
    newenv = os.environ.copy()
    root_dir = os.path.abspath(
            os.path.join(os.path.dirname(cellprofiler.__file__), '..'))
    added_paths = []
    if 'PYTHONPATH' in newenv:
        old_path = newenv['PYTHONPATH']
        if not any([root_dir == path
                    for path in old_path.split(os.pathsep)]):
            added_paths.append(root_dir)
    else:
        added_paths.append(root_dir)

    if hasattr(sys, 'frozen'):
        if sys.platform == "darwin":
            # http://mail.python.org/pipermail/pythonmac-sig/2005-April/013852.html
            added_paths += [p for p in sys.path if isinstance(p, basestring)]
    if 'PYTHONPATH' in newenv:
        added_paths.insert(0, newenv['PYTHONPATH'])
    newenv['PYTHONPATH'] = os.pathsep.join(
            [x.encode('utf-8') for x in added_paths])
    if "CP_JDWP_PORT" in newenv:
        del newenv["CP_JDWP_PORT"]
    if "AW_JDWP_PORT" in newenv:
        port = str(int(newenv["AW_JDWP_PORT"]) + idx)
        newenv["CP_JDWP_PORT"] = port
        del newenv["AW_JDWP_PORT"]
    for key in newenv:
        if isinstance(newenv[key], unicode):
            newenv[key] = newenv[key].encode('utf-8')
    return newenv


def find_analysis_worker_source():
    # import here to break circular dependency.
    import cellprofiler.analysis  # used to get the path to the code
    return os.path.join(os.path.dirname(cellprofiler.analysis.__file__), "worker.py")


def start_daemon_thread(target=None, args=(), kwargs=None, name=None):
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, name=name)
    thread.daemon = True
    thread.start()
    return thread


###############################
# Request, Replies, Events
###############################
class AnalysisStarted(object):
    pass


class AnalysisProgress(object):
    def __init__(self, counts):
        self.counts = counts


class AnalysisPaused(object):
    pass


class AnalysisResumed(object):
    pass


class AnalysisFinished(object):
    def __init__(self, measurements, cancelled):
        self.measurements = measurements
        self.cancelled = cancelled


class PipelinePreferencesRequest(AnalysisRequest):
    pass


class InitialMeasurementsRequest(AnalysisRequest):
    pass


class WorkRequest(AnalysisRequest):
    pass


class ImageSetSuccess(AnalysisRequest):
    def __init__(self, analysis_id, image_set_number=None):
        AnalysisRequest.__init__(self, analysis_id,
                                 image_set_number=image_set_number)


class ImageSetSuccessWithDictionary(ImageSetSuccess):
    def __init__(self, analysis_id, image_set_number, shared_dicts):
        ImageSetSuccess.__init__(self, analysis_id,
                                 image_set_number=image_set_number)
        self.shared_dicts = shared_dicts


class DictionaryReqRep(Reply):
    pass


class MeasurementsReport(AnalysisRequest):
    def __init__(self, analysis_id, buf, image_set_numbers=[]):
        AnalysisRequest.__init__(self, analysis_id,
                                 buf=buf,
                                 image_set_numbers=image_set_numbers)


class InteractionRequest(AnalysisRequest):
    pass


class AnalysisCancelRequest(AnalysisRequest):
    pass


class DisplayRequest(AnalysisRequest):
    pass


class DisplayPostRunRequest(object):
    '''Request a post-run display

    This is a message sent to the UI from the analysis worker'''

    def __init__(self, module_num, display_data):
        self.module_num = module_num
        self.display_data = display_data


class DisplayPostGroupRequest(AnalysisRequest):
    '''Request a post-group display

    This is a message sent to the UI from the analysis worker'''

    def __init__(self, analysis_id, module_num, display_data, image_set_number):
        AnalysisRequest.__init__(
                self, analysis_id,
                module_num=module_num,
                image_set_number=image_set_number,
                display_data=display_data)


class SharedDictionaryRequest(AnalysisRequest):
    def __init__(self, analysis_id, module_num=-1):
        AnalysisRequest.__init__(self, analysis_id, module_num=module_num)


class SharedDictionaryReply(Reply):
    def __init__(self, dictionaries=[{}]):
        Reply.__init__(self, dictionaries=dictionaries)


class ExceptionReport(AnalysisRequest):
    def __init__(self, analysis_id,
                 image_set_number, module_name,
                 exc_type, exc_message, exc_traceback,
                 filename, line_number):
        AnalysisRequest.__init__(self,
                                 analysis_id,
                                 image_set_number=image_set_number,
                                 module_name=module_name,
                                 exc_type=exc_type,
                                 exc_message=exc_message,
                                 exc_traceback=exc_traceback,
                                 filename=filename,
                                 line_number=line_number)

    def __str__(self):
        return "(Worker) %s: %s" % (self.exc_type, self.exc_message)


class ExceptionPleaseDebugReply(Reply):
    def __init__(self, disposition, verification_hash=None):
        Reply.__init__(self, disposition=disposition, verification_hash=verification_hash)


class DebugWaiting(AnalysisRequest):
    '''Communicate the debug port to the server and wait for server OK to attach'''

    def __init__(self, analysis_id, port):
        AnalysisRequest.__init__(self,
                                 analysis_id=analysis_id,
                                 port=port)


class DebugCancel(Reply):
    '''If sent in response to DebugWaiting, the user has changed his/her mind'''


class DebugComplete(AnalysisRequest):
    pass


class InteractionReply(Reply):
    pass


class WorkReply(Reply):
    pass


class NoWorkReply(Reply):
    pass


class ServerExited(UpstreamExit):
    pass


class OmeroLoginRequest(AnalysisRequest):
    pass


class OmeroLoginReply(Reply):
    def __init__(self, credentials):
        Reply.__init__(self, credentials=credentials)


class Ack(Reply):
    def __init__(self, message="THANKS"):
        Reply.__init__(self, message=message)


if sys.platform == "darwin":
    import fcntl


    def close_all_on_exec():
        '''Mark every file handle above 2 with CLOEXEC

        We don't want child processes inheret anything
        except for STDIN / STDOUT / STDERR. This should
        make it so in a horribly brute-force way.
        '''
        try:
            maxfd = os.sysconf('SC_OPEN_MAX')
        except:
            maxfd = 256
        for fd in range(3, maxfd):
            try:
                fcntl.fcntl(fd, fcntl.FD_CLOEXEC)
            except:
                pass

if __name__ == '__main__':
    import time
    import cellprofiler.pipeline
    import cellprofiler.preferences

    # This is an ugly hack, but it's necesary to unify the Request/Reply
    # classes above, so that regardless of whether this is the current module,
    # or a separately imported one, they see the same classes.
    import cellprofiler.analysis

    globals().update(cellprofiler.analysis.__dict__)

    AnalysisRunner.start_workers(2)
    AnalysisRunner.stop_workers()
