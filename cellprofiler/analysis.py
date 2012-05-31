"""analysis.py - Run pipelines on imagesets to produce measurements.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
from __future__ import with_statement

import subprocess
import multiprocessing
import logging
import threading
import Queue
import uuid
import numpy as np
import cStringIO as StringIO
import sys
import os
import os.path
import zmq
import gc
import collections

import cellprofiler
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.cpimage as cpimage
import cellprofiler.preferences as cpprefs
from cellprofiler.utilities.zmqrequest import Request, Reply, Boundary, BoundaryExited
import subimager.client


logger = logging.getLogger(__name__)

use_analysis = True

DEBUG = 'DEBUG'


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
        self.initial_measurements = cpmeas.Measurements(image_set_start=None,
                                                        copy=initial_measurements)
        self.output_path = measurements_filename
        self.debug_mode = False
        self.analysis_in_progress = False
        self.runner = None

        self.runner_lock = threading.Lock()  # defensive coding purposes

    def start(self, analysis_event_callback):
        with self.runner_lock:
            assert not self.analysis_in_progress
            self.analysis_in_progress = uuid.uuid1().hex

            self.runner = AnalysisRunner(self.analysis_in_progress,
                                         self.pipeline,
                                         self.initial_measurements,
                                         self.output_path,
                                         analysis_event_callback)
            self.runner.start()
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
    announce_queue = None

    # measurement status
    STATUS = "ProcessingStatus"
    STATUS_UNPROCESSED = "Unprocessed"
    STATUS_IN_PROCESS = "InProcess"
    STATUS_FINISHED_WAITING = "FinishedWaitingMeasurements"
    STATUS_DONE = "Done"
    STATUSES = [STATUS_UNPROCESSED, STATUS_IN_PROCESS, STATUS_FINISHED_WAITING, STATUS_DONE]

    def __init__(self, analysis_id, pipeline,
                 initial_measurements, output_path, event_listener):
        # for sending to workers
        self.initial_measurements = cpmeas.Measurements(image_set_start=None,
                                                        copy=initial_measurements)

        # for writing results - initialized in start()
        self.measurements = None
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
        self.returned_measurements_queue = Queue.Queue()

        self.shared_dicts = None

        self.interface_thread = None
        self.jobserver_thread = None

    # External control interfaces
    def start(self):
        '''start the analysis run'''

        # Although it would be nice to reuse the worker pool, I'm not entirely
        # sure they recover correctly from the user cancelling an analysis
        # (e.g., during an interaction request).  This should be handled by
        # zmqRequest.Boundary.stop(), but just in case, we stop the workers and
        # restart them.  Note that this creates a new announce port, so we
        # don't have to worry about old workers taking a job before noticing
        # that their stdin has closed.
        self.stop_workers()
        announce_queue = self.start_workers(2)  # start worker pool via class method (below)

        self.measurements = cpmeas.Measurements(image_set_start=None,
                                                filename=self.output_path,
                                                copy=self.initial_measurements)
        self.shared_dicts = [m.get_dictionary() for m in self.pipeline.modules()]
        self.interface_thread = start_daemon_thread(target=self.interface, name='AnalysisRunner.interface')
        self.jobserver_thread = start_daemon_thread(target=self.jobserver, args=(self.analysis_id, announce_queue), name='AnalysisRunner.jobserver')

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
        self.cancelled = True
        self.paused = False
        self.notify_threads()

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

    # XXX - catch and deal with exceptions in interface() and jobserver() threads
    def interface(self, image_set_start=1, image_set_end=None,
                     overwrite=True):
        '''Top-half thread for running an analysis.  Sets up grouping for jobs,
        deals with returned measurements, reports status periodically.

        image_set_start - beginning image set number to process
        image_set_end - last image set number to process
        overwrite - whether to recompute imagesets that already have data in initial_measurements.
        '''
        # listen for pipeline events, and pass them upstream
        self.pipeline.add_listener(lambda pipe, evt: self.post_event(evt))

        workspace = cpw.Workspace(self.pipeline, None, None, None,
                                  self.measurements, cpimage.ImageSetList())

        if image_set_end is None:
            image_set_end = len(self.measurements.get_image_numbers())
        image_sets_to_process = range(image_set_start, image_set_end + 1)

        self.post_event(AnalysisStarted())

        # reset the status of every image set that needs to be processed
        for image_set_number in image_sets_to_process:
            if (overwrite or
                (not self.measurements.has_measurements(cpmeas.IMAGE, self.STATUS, image_set_number)) or
                (self.measurements[cpmeas.IMAGE, self.STATUS, image_set_number] != self.STATUS_DONE)):
                self.measurements[cpmeas.IMAGE, self.STATUS, image_set_number] = self.STATUS_UNPROCESSED

        # Find image groups.  These are written into measurements prior to
        # analysis.  Groups are processed as a single job.
        if self.measurements.has_groups():
            worker_runs_post_group = True
            job_groups = {}
            for image_set_number in image_sets_to_process:
                group_number = self.measurements[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_set_number]
                group_index = self.measurements[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_set_number]
                job_groups[group_number] = job_groups.get(group_number, []) + [(group_index, image_set_number)]
            job_groups[group_number] = [[isn for _, isn in sorted(job_groups[group_number])] for group_number in job_groups]
            first_image_job_group = self.measurements[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_set_start]
        else:
            worker_runs_post_group = False  # prepare_group will be run in worker, but post_group is below.
            job_groups = [[image_set_number] for image_set_number in image_sets_to_process]
            first_image_job_group = 0
            for idx, image_set_number in enumerate(image_sets_to_process):
                self.initial_measurements[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_set_number] = 1
                self.initial_measurements[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_set_number] = idx + 1
            self.initial_measurements.flush()

        # XXX - check that any constructed groups are complete, i.e.,
        # image_set_start and image_set_end shouldn't carve them up.

        # put the first job in the queue, then wait for the first image to
        # finish (see the check of self.finish_queue below) to post the rest.
        # This ensures that any shared data from the first imageset is
        # available to later imagesets.
        self.work_queue.put((job_groups[first_image_job_group], worker_runs_post_group))
        del job_groups[first_image_job_group]

        waiting_for_first_imageset = True

        # We loop until every image is completed, or an outside event breaks the loop.
        while True:
            if self.cancelled:
                break

            # gather measurements
            while not self.returned_measurements_queue.empty():
                job, reported_measurements = self.returned_measurements_queue.get()
                for object in reported_measurements.get_object_names():
                    if object == cpmeas.EXPERIMENT:
                        continue  # XXX - who writes this, when?
                    for feature in reported_measurements.get_feature_names(object):
                        for imagenumber in job:
                            self.measurements[object, feature, imagenumber] = reported_measurements[object, feature, imagenumber]
                for image_set_number in job:
                    self.measurements[cpmeas.IMAGE, self.STATUS, int(image_set_number)] = self.STATUS_DONE

            # check for jobs in progress
            while not self.in_process_queue.empty():
                image_set_numbers = self.in_process_queue.get()
                for image_set_number in image_set_numbers:
                    self.measurements[cpmeas.IMAGE, self.STATUS, int(image_set_number)] = self.STATUS_IN_PROCESS

            # check for finished jobs that haven't returned measurements, yet
            while not self.finished_queue.empty():
                finished_req = self.finished_queue.get()
                self.measurements[cpmeas.IMAGE, self.STATUS, int(finished_req.image_set_number)] = self.STATUS_FINISHED_WAITING
                if waiting_for_first_imageset:
                    waiting_for_first_imageset = False
                    # request shared state dictionary.
                    # we use the nonstandard Request/Reply/Reply/Reply pattern.
                    dict_reply = finished_req.reply(DictionaryReqRep(), please_reply=True)
                    self.shared_dicts = dict_reply.shared_dicts
                    assert len(self.shared_dicts) == len(self.pipeline.modules())
                    dict_reply.reply(Ack())
                    # if we had jobs waiting for the first image set to finish,
                    # queue them now that the shared state is available.
                    for job in job_groups:
                        self.work_queue.put((job, worker_runs_post_group))
                else:
                    finished_req.reply(Ack())

            # check progress and report
            counts = collections.Counter(self.measurements[cpmeas.IMAGE, self.STATUS, image_set_number]
                                         for image_set_number in image_sets_to_process)
            self.post_event(AnalysisProgress(counts))

            # Are we finished?
            if counts[self.STATUS_DONE] == len(image_sets_to_process):
                if worker_runs_post_group:
                    self.pipeline.post_group(workspace, {})
                # XXX - revise pipeline.post_run to use the workspace
                self.pipeline.post_run(self.measurements, None, None)
                break

            # not done, wait for more work
            with self.interface_work_cv:
                while (self.paused or
                       ((not self.cancelled) and
                        self.in_process_queue.empty() and
                        self.finished_queue.empty() and
                        self.returned_measurements_queue.empty())):
                    self.interface_work_cv.wait()  # wait for a change of status or work to arrive

        self.post_event(AnalysisFinished(self.measurements, self.cancelled))
        self.stop_workers()
        self.measurements.close()
        self.measurements = None  # do not hang onto measurements
        self.analysis_id = False  # this will cause the jobserver thread to exit

    def jobserver(self, analysis_id, announce_queue):
        # this server subthread should be very lightweight, as it has to handle
        # all the requests from workers, of which there might be several.

        # start the zmqrequest Boundary
        request_queue = Queue.Queue()
        boundary = Boundary('tcp://127.0.0.1', request_queue, self.jobserver_work_cv, random_port=True)

        # XXX - is this just to keep from posting another AnalysisPaused event?
        # If so, probably better to simplify the code and keep sending them
        # (should be only one per second).
        i_was_paused_before = False

        # start serving work until the analysis is done (or changed)
        while self.analysis_id == analysis_id:
            # announce ourselves
            announce_queue.put([boundary.request_address, analysis_id])

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

            req = request_queue.get()
            if isinstance(req, PipelinePreferencesRequest):
                req.reply(Reply(pipeline_blob=np.array(self.pipeline_as_string()),
                                preferences=cpprefs.preferences_as_dict()))
            elif isinstance(req, InitialMeasurementsRequest):
                req.reply(Reply(path=self.initial_measurements.hdf5_dict.filename.encode('utf-8')))
            elif isinstance(req, WorkRequest):
                if not self.work_queue.empty():
                    job, worker_runs_post_group = self.work_queue.get()
                    req.reply(WorkReply(image_set_numbers=job, worker_runs_post_group=worker_runs_post_group))
                    self.queue_dispatched_job(job)
                else:
                    # there may be no work available, currently, but there
                    # may be some later.
                    req.reply(NoWorkReply())
            elif isinstance(req, ImageSetSuccess):
                # interface() is responsible for replying, to allow it to
                # request the shared_state dictionary if needed.
                self.queue_imageset_finished(req)
            elif isinstance(req, SharedDictionaryRequest):
                req.reply(SharedDictionaryReply(dictionaries=self.shared_dicts))
            elif isinstance(req, MeasurementsReport):
                # Measurements are available at location indicated
                measurements_path = req.path.decode('utf-8')
                successes = [int(s) for s in req.image_set_numbers.split(",")]
                try:
                    reported_measurements = cpmeas.load_measurements(measurements_path)
                    self.queue_received_measurements(successes, reported_measurements)
                except Exception:
                    raise
                    # XXX - report error, push back job
                req.reply(Ack())
            elif isinstance(req, (InteractionRequest, DisplayRequest, ExceptionReport)):
                # bump upward
                self.post_event(req)
            else:
                raise ValueError("Unknown request from worker: %s of type %s" % (req, type(req)))

        # announce that this job is done/cancelled
        announce_queue.put(['DONE', analysis_id])

        # stop the ZMQ-boundary thread - will also deal with any requests waiting on replies
        boundary.stop()

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

    def queue_received_measurements(self, image_set_numbers, measure_blob):
        self.returned_measurements_queue.put((image_set_numbers, measure_blob))
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
            return cls.annouce_queue # already started

        try:
            num = num or multiprocessing.cpu_count()
        except NotImplementedError:
            num = 4

        # Set up the work announcement PUB socket, and start the announce() thread
        cls.zmq_context = zmq.Context()
        work_announce_socket = cls.zmq_context.socket(zmq.PUB)
        work_announce_socket.setsockopt(zmq.LINGER, 0)
        work_announce_port = work_announce_socket.bind_to_random_port("tcp://127.0.0.1")
        cls.announce_queue = Queue.Queue()

        def announcer(announce_queue):
            while True:
                mesg = announce_queue.get()
                if mesg == "STOP":  # request to stop this thread
                    break
                work_announce_socket.send_multipart(mesg)
        start_daemon_thread(target=announcer, name='RunAnalysis.announcer', args=(cls.announce_queue,))

        # ensure subimager is started
        subimager.client.start_subimager()

        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(cellprofiler.__file__), '..') + ':' + os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(cellprofiler.__file__), '..')

        if 'CP_DEBUG_WORKER' in os.environ:
            logger.info("Announcing work on port %d" % work_announce_port)
            logger.info("Subimager port: %d" % subimager.client.port)
            logger.info("Please manually start a worker using the command-line:")
            logger.info(("python -u %s --work-announce tcp://127.0.0.1:%d "
                         "--subimager-port %d") % (
                             find_analysis_worker_source(),
                             work_announce_port,
                             subimager.client.port))
            return cls.announce_queue
                
        # start workers
        for idx in range(num):
            # stdin for the subprocesses serves as a deadman's switch.  When
            # closed, the subprocess exits.
            worker = subprocess.Popen([find_python(),
                                       '-u',  # unbuffered
                                       find_analysis_worker_source(),
                                       '--work-announce',
                                       'tcp://127.0.0.1:%d' % (work_announce_port),
                                       '--subimager-port',
                                       '%d' % subimager.client.port],
                                      env=find_worker_env(),
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)

            def run_logger(workR, widx):
                while(True):
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

        return cls.announce_queue

    @classmethod
    def stop_workers(cls):
        for deadman_switch in cls.deadman_switches:
            deadman_switch.close()
        if cls.announce_queue:
            cls.announce_queue.put("STOP")
        cls.workers = []
        cls.deadman_swtiches = []
        cls.announce_queue = None


def find_python():
    if hasattr(sys, 'frozen'):
        if sys.platform == "darwin":
            app_python = os.path.join(os.path.dirname(os.environ['ARGVZERO']), "python")
            return app_python
    return 'python'


def find_worker_env():
    if hasattr(sys, 'frozen'):
        if sys.platform == "darwin":
            newenv = os.environ.copy()
            # http://mail.python.org/pipermail/pythonmac-sig/2005-April/013852.html
            newenv['PYTHONPATH'] = ':'.join([p for p in sys.path if isinstance(p, basestring)])
            return newenv
    return os.environ


def find_analysis_worker_source():
    # import here to break circular dependency.
    import cellprofiler.analysis  # used to get the path to the code
    return os.path.join(os.path.dirname(cellprofiler.analysis.__file__), "analysis_worker.py")


def start_daemon_thread(target=None, args=(), name=None):
    thread = threading.Thread(target=target, args=args, name=name)
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


class PipelinePreferencesRequest(Request):
    pass


class InitialMeasurementsRequest(Request):
    pass


class WorkRequest(Request):
    pass


class ImageSetSuccess(Request):
    def __init__(self, image_set_number=None):
        Request.__init__(self, image_set_number=image_set_number)


class DictionaryReqRep(Reply):
    pass


class DictionaryReqRepRep(Reply):
    def __init__(self, shared_dicts=None):
        Reply.__init__(self, shared_dicts=shared_dicts)


class MeasurementsReport(Request):
    def __init__(self, path="", image_set_numbers=""):
        Request.__init__(self, path=path, image_set_numbers=image_set_numbers)


class InteractionRequest(Request):
    pass


class SharedDictionaryRequest(Request):
    def __init__(self, module_num=-1):
        Request.__init__(self, module_num=module_num)


class SharedDictionaryReply(Reply):
    def __init__(self, dictionaries=[{}]):
        Reply.__init__(self, dictionaries=dictionaries)


class DisplayRequest(Request):
    pass

class ExceptionReport(Request):
    def __init__(self,
                 image_set_number, module_name,
                 exc_type, exc_message, exc_traceback,
                 filename, line_number):
        Request.__init__(self,
                         image_set_number=image_set_number,
                         module_name=module_name,
                         exc_type=exc_type,
                         exc_message=exc_message,
                         exc_traceback=exc_traceback,
                         filename=filename,
                         line_number=line_number)

class ExceptionPleaseDebugReply(Reply):
    def __init__(self, disposition, verification_hash=None):
        Reply.__init__(self, disposition=disposition, verification_hash=verification_hash)

class DebugWaiting(Reply):
    def __init__(self, port):
        Reply.__init__(self, port=port)

class DebugComplete(Reply):
    pass


class InteractionReply(Reply):
    pass


class WorkReply(Reply):
    pass


class NoWorkReply(Reply):
    pass


class ServerExited(BoundaryExited):
    pass


class Ack(Reply):
    def __init__(self, message="THANKS"):
        Reply.__init__(self, message=message)


if __name__ == '__main__':
    import time
    import cellprofiler.pipeline
    import cellprofiler.preferences
    import cellprofiler.utilities.thread_excepthook

    # This is an ugly hack, but it's necesary to unify the Request/Reply
    # classes above, so that regardless of whether this is the current module,
    # or a separately imported one, they see the same classes.
    import cellprofiler.analysis
    globals().update(cellprofiler.analysis.__dict__)

    print "TESTING", WorkRequest is cellprofiler.analysis.WorkRequest
    print id(WorkRequest), id(cellprofiler.analysis.WorkRequest)

    cellprofiler.utilities.thread_excepthook.install_thread_sys_excepthook()

    cellprofiler.preferences.set_headless()
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(logging.StreamHandler())

    batch_data = sys.argv[1]
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.load(batch_data)
    measurements = cellprofiler.measurements.load_measurements(batch_data)
    analysis = Analysis(pipeline, 'test_out.h5', initial_measurements=measurements)

    keep_going = True

    def callback(event):
        global keep_going
        print "Pipeline Event", event
        if isinstance(event, AnalysisFinished):
            keep_going = False

    analysis.start(callback)
    while keep_going:
        time.sleep(0.25)
    del analysis
    gc.collect()
