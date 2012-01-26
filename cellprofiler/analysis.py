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
import zmq
import cStringIO as StringIO
import sys
import os
import os.path
import hashlib
import gc

import cellprofiler
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.cpimage as cpimage
import cellprofiler.analysis_worker  # used to get the path to the code
import subimager.client

logger = logging.getLogger(__name__)

use_analysis = False

SKIP = 'SKIP'
ABORT = 'ABORT'
DEBUG = 'DEBUG'

class Analysis(object):
    '''An Analysis is the application of a particular pipeline of modules to a
    set of images to produce measurements.

    Multiprocessing for analyses is handled by multiple layers of threads and
    processes, to keep the GUI responsive and simplify the code.  Threads and
    processes are organized as below.

    +------------------------------------------------+
    |           CellProfiler GUI/WX thread           |
    |                                                |
    +--Analysis methods() down, AnalysisEvent()s up--+
    |                                                |
    |       AnalysisRunner.interface() thread        |
    |                                                |
    +----------------  Queues  ----------------------+
    |                                                |
    |  AnalysisRunner.jobserver()/announce() threads |
    |                                                |
    +----------------------------------------------- +
              ZMQ sockets (other processes)
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
        self.measurements = cpmeas.Measurements(image_set_start=None,
                                                filename=measurements_filename,
                                                copy=initial_measurements)
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
                                         self.measurements,
                                         analysis_event_callback)
            self.runner.start()
            return self.analysis_in_progress

    def pause_analysis(self):
        with self.runner_lock:
            assert self.analysis_in_progress
            self.runner.pause()

    def resume_analysis(self):
        with self.runner_lock:
            assert self.analysis_in_progress
            self.runner.resume()

    def cancel_analysis(self):
        with self.runner_lock:
            assert self.analysis_in_progress
            self.analysis_in_progress = False
            self.runner.cancel()
            self.runner = None

    def check(self):
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

    interface() is responsible for communication upstream (e.g., between
    workers and the GUI), tracking job progress, integrating measurements
    returned from workers.  Upstream messages from interface() are
    AnalysisEvent()s.

    jobserver() is a lightweight thread that primarily acts as a switchboard
    between workers and interface().  It serves jobs and places messages into
    the proper queue for upstream work.

    workers are stored in AnalysisRunner.workers[], and are separate processes.
    They are expected to exit if their stdin() closes, such as if the parent
    process goes away.

    interface() and jobserver() communicate via Queues and using condition
    variables to get each other's attention.  Zeromq is used to communicate
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
    STATUS_DONE = "Done"


    def __init__(self, analysis_id, pipeline,
                 initial_measurements, event_listener):
        # for sending to workers
        self.initial_measurements = cpmeas.Measurements(image_set_start=None,
                                                        copy=initial_measurements)
        # for storing results locally - created in start()
        self.measurements = None

        self.analysis_id = analysis_id
        self.pipeline = pipeline.copy()
        self.event_listener = event_listener
        self.original_exception_events = {}

        self.interface_work_cv = threading.Condition()
        self.pause_cancel_cv = threading.Condition()
        self.paused = False
        self.cancelled = False

        self.work_queue = Queue.Queue()
        self.in_process_queue = Queue.Queue()
        self.display_request_queue = Queue.Queue()
        self.interaction_request_queue = Queue.Queue()
        self.interaction_reply_queue = Queue.Queue()
        self.receive_measurements_queue = Queue.Queue()

        self.debug_reply_queue = Queue.Queue()
        self.debug_port_callbacks = {}

        self.interface_thread = None
        self.jobserver_thread = None

        self.start_workers(2)  # start worker pool via class method

    # External control interfaces
    def start(self):
        '''start the analysis run'''
        workspace = cpw.Workspace(self.pipeline, None, None, None,
                                  self.initial_measurements, cpimage.ImageSetList())
        self.pipeline.prepare_run(workspace)
        self.initial_measurements.flush()  # Make sure file is valid before we start threads.
        self.measurements = cpmeas.Measurements(image_set_start=None,
                                                copy=self.initial_measurements)

        self.interface_thread = start_daemon_thread(target=self.interface, name='AnalysisRunner.interface')
        self.jobserver_thread = start_daemon_thread(target=self.jobserver, args=(self.analysis_id,), name='AnalysisRunner.jobserver')

    def check(self):
        return ((self.interface_thread is not None) and
                (self.jobserver_thread is not None) and
                self.interface_thread.is_alive() and
                self.jobserver_thread.is_alive())

    def cancel(self):
        '''cancel the analysis run'''
        with self.pause_cancel_cv:
            self.cancelled = True
            self.paused = False
            self.pause_cancel_cv.notifyAll()  # interface() and jobserver() threads
        with self.interface_work_cv:
            self.interface_work_cv.notify()  # in case interface is waiting

    def pause(self):
        '''pause the analysis run'''
        with self.pause_cancel_cv:
            self.paused = True
            self.pause_cancel_cv.notifyAll()  # interface() and jobserver() threads

    def resume(self):
        '''resume a paused analysis run'''
        with self.pause_cancel_cv:
            self.paused = False
            self.pause_cancel_cv.notifyAll()  # interface() and jobserver() threads

    # event posting
    def post_event(self, evt):
        self.event_listener(evt)

    # XXX - catch and deal with exceptions in interface() and jobserver() threads
    def interface(self, image_set_start=1, image_set_end=None,
                     overwrite=True):
        '''Top-half thread for running an analysis.  Interacts with GUI and
        actual job server thread.

        image_set_start - beginning image set number
        image_set_end - final image set number
        overwrite - whether to recompute imagesets that already have data in initial_measurements.
        '''
        # listen for pipeline events, and pass them upstream
        self.pipeline.add_listener(lambda pipe, evt: self.post_event(evt))

        workspace = cpw.Workspace(self.pipeline, None, None, None,
                                  self.measurements, cpimage.ImageSetList())

        if image_set_end is None:
            image_set_end = len(self.measurements.get_image_numbers())

        self.post_event(AnalysisStartedEvent())

        # reset the status of every image set that needs to be processed
        for image_set_number in range(image_set_start, image_set_end):
            if (overwrite or
                (not self.measurements.has_measurements(cpmeas.IMAGE, self.STATUS, image_set_number)) or
                (self.measurements[cpmeas.IMAGE, self.STATUS, image_set_number] != self.STATUS_DONE)):
                self.measurements[cpmeas.IMAGE, self.STATUS, image_set_number] = self.STATUS_UNPROCESSED

        # Find image groups.  These are written into measurements prior to
        # analysis.  Groups are processed as a single job.
        if self.measurements.has_groups():
            grouping_needed = True
            job_groups = {}
            for image_set_number in range(image_set_start, image_set_end):
                group_number = self.measurements[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_set_number]
                group_index = self.measurements[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_set_number]
                job_groups[group_number] = job_groups.get(group_number, []) + [(group_index, image_set_number)]
            job_groups[group_number] = [[isn for _, isn in sorted(job_groups[group_number])] for group_number in job_groups]
        else:
            grouping_needed = False
            job_groups = [[image_set_number] for image_set_number in range(image_set_start, image_set_end)]
            for idx, image_set_number in enumerate(range(image_set_start, image_set_end)):
                self.initial_measurements[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_set_number] = 0
                self.initial_measurements[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_set_number] = idx
            self.initial_measurements.flush()
            # As there's no grouping, we call prepare_group() once on the
            # pipeline (see pipeline.prepare_group()'s docstring)
            if not self.pipeline.prepare_group(workspace, {}, range(image_set_start, image_set_end)):
                # Exception in prepare group, and run was cancelled.
                self.cancel()
                del self.measurements
                self.analysis_id = False  # this will cause the jobserver thread to exit
                return

        # XXX - check that any constructed groups are complete, i.e.,
        # image_set_start and image_set_end shouldn't carve them up.

        # put the jobs in the queue
        for job in job_groups:
            self.work_queue.put((job, grouping_needed))

        # We loop until every image is completed, or an outside event breaks the loop.
        while True:
            # take the interface_work_cv lock to keep our interaction with the measurements atomic.
            with self.interface_work_cv:
                counts = dict((s, 0) for s in [self.STATUS_UNPROCESSED, self.STATUS_IN_PROCESS, self.STATUS_DONE])
                for image_set_number in range(image_set_start, image_set_end):
                    counts[self.measurements[cpmeas.IMAGE, self.STATUS, image_set_number]] += 1

                # report our progress
                self.post_event(AnalysisProgressEvent(counts))

                # Are we finished?
                if (counts[self.STATUS_IN_PROCESS] + counts[self.STATUS_UNPROCESSED]) == 0:
                    if not grouping_needed:
                        self.pipeline.post_group(workspace, {})
                    # XXX - revise pipeline.post_run to use the workspace
                    self.pipeline.post_run(self.measurements, None, None)
                    self.measurements.flush()
                    self.post_event(AnalysisFinishedEvent(self.analysis_id, self.measurements))
                    break

                # wait for an image set to finish or some sort of display or
                # interaction work that needs handling.
                self.interface_work_cv.wait()

            # check pause/cancel
            if not self.check_pause_and_cancel():
                break
            # display/interaction/measurements
            self.handle_queues()

        del self.measurements
        self.analysis_id = False  # this will cause the jobserver thread to exit
        self.post_event(AnalysisEndedEvent(self.cancelled))

    def check_pause_and_cancel(self):
        '''if paused, wait for unpause or cancel.  return True if work should
        continue, False if cancelled.
        '''
        # paused - wait for us to be unpaused
        with self.pause_cancel_cv:
            while self.paused:
                self.post_event(AnalysisPausedEvent())
                self.pause_cancel_cv.wait()
                if not self.paused:
                    self.post_event(AnalysisResumedEvent())
        if self.cancelled:
            return False
        return True

    def handle_queues(self):
        '''handle any UI-based work, including checking for display/interaction
        requests, and measurements that have been returned.
        '''
        # display requests
        while not self.display_request_queue.empty():
            image_set_number, display_request = self.display_request_queue.get()
            self.post_event(AnalysisDisplayEvent(image_set_number, display_request))
        # interaction requests
        while not self.interaction_request_queue.empty():
            reply_address, module_number, image_set_number, interaction_request = self.interaction_request_queue.get()

            def reply_cb(interaction_result, reply_address=reply_address):
                # this callback function will be called from another thread
                with self.pause_cancel_cv:
                    # cancellation during interaction is handled in jobserver().
                    if not self.cancelled:
                        self.interaction_reply_queue.put((reply_address, interaction_result))
                        with self.interface_work_cv:
                            self.interface_work_cv.notify()  # notify interface() there's a reply to be handled.

            self.post_event(AnalysisInteractionRequest(module_number, image_set_number, interaction_request, reply_cb))
        # notification that images are in process
        while not self.in_process_queue.empty():
            image_set_numbers = self.in_process_queue.get()
            for image_set_number in image_set_numbers:
                self.measurements[cpmeas.IMAGE, self.STATUS, int(image_set_number)] = self.STATUS_IN_PROCESS
        # measurements
        while not self.receive_measurements_queue.empty():
            # XXX - GROUPS
            returned_measurements, job = self.receive_measurements_queue.get()
            for image_set_number in job:
                self.measurements[cpmeas.IMAGE, self.STATUS, int(image_set_number)] = self.STATUS_DONE

    def jobserver(self, analysis_id):
        # this server subthread should be very lightweight, as it has to handle
        # all the requests from workers, of which there might be several.

        # set up the zmq.XREP socket we'll serve jobs from
        work_queue_socket = self.zmq_context.socket(zmq.XREP)
        work_queue_socket.setsockopt(zmq.LINGER, 0)  # Allow close without waiting for messages to be delivered.
        work_queue_port = work_queue_socket.bind_to_random_port('tcp://127.0.0.1')

        self.announce_queue.put(['tcp://127.0.0.1:%d' % work_queue_port, analysis_id])

        poller = zmq.Poller()
        poller.register(work_queue_socket, zmq.POLLIN)

        workers_waiting_on_interaction = []

        # start serving work until the analysis is done (or changed)
        while self.analysis_id == analysis_id:
            # check for pause
            if not self.check_pause_and_cancel():
                break  # exit

            # Check interactive_reply_queue for messages to be sent to
            # workers.
            while not self.interaction_reply_queue.empty():
                address, reply = self.interaction_reply_queue.get()
                work_queue_socket.send_multipart([address, '', reply])
                workers_waiting_on_interaction.remove(address)

            # Check for any debug replies that need to be forwarded.
            while not self.debug_reply_queue.empty():
                reply = self.debug_reply_queue.get()
                address, disposition = reply[:2]
                if disposition in (SKIP, ABORT):
                    if disposition == ABORT:
                        self.cancel()  # cancel the whole run
                    work_queue_socket.send_multipart([address, '',
                                                      'SKIP' if disposition == SKIP else 'ABORT'])
                else:
                    assert disposition == DEBUG
                    verification_string, port_callback = reply[2:]
                    work_queue_socket.send_multipart([address, '', 'DEBUG',
                                                      hashlib.sha1(verification_string).hexdigest()])
                    self.debug_port_callbacks[address] = port_callback

            # Check for activity from workers
            if not (poller.poll(timeout=250)):  # 1/4 second
                # timeout.  re-announce work queue, and try again.  (this
                # also handles the case where the workers start up slowly,
                # and miss the first announcement.)
                self.announce_queue.put(['tcp://127.0.0.1:%d' % work_queue_port, analysis_id])
                continue
            msg = work_queue_socket.recv_multipart()
            address, _, message_type = msg[:3]  # [address, '', message_type, optional job #, optional body]
            if message_type == 'PIPELINE':
                work_queue_socket.send_multipart([address, '', self.pipeline_as_string()])
            elif message_type == 'INITIAL_MEASUREMENTS':
                work_queue_socket.send_multipart([address, '', self.initial_measurements.hdf5_dict.filename.encode('utf-8')])
            elif message_type == 'WORK REQUEST':
                if not self.work_queue.empty():
                    job, grouping_needed = self.work_queue.get()
                    if grouping_needed:
                        work_queue_socket.send_multipart([address, '', 'GROUP'] + [str(j) for j in job])
                    else:
                        work_queue_socket.send_multipart([address, '', 'IMAGE', str(job[0])])
                    self.in_process_queue.put(job)
                else:
                    # there may be no work available, currently, but there
                    # may be some later.
                    work_queue_socket.send_multipart([address, '', 'NONE'])
            elif message_type == 'DISPLAY':
                self.queue_display_request(msg[3], msg[4])
                work_queue_socket.send_multipart([address, '', 'OK'])
            elif message_type == 'INTERACT':
                module_number, image_set_number, interaction_request_blob = msg[3:]
                self.queue_interaction_request(address,
                                               int(module_number), int(image_set_number),
                                               interaction_request_blob)
                # we don't reply immediately, as we have to wait for the
                # GUI to run.  We'll find the result on
                # self.interact_reply_queue sometime in the future.
                #
                # Note that there is some danger here, in that a worker will
                # hang in socket.recv() until it gets a reply, and if the
                # analysis is cancelled at this point, it will hang
                # indefinitely.  We put the worker's address into a list of
                # workers waiting for interactions and send them a special
                # result if we're cancelled before they get their answer.
                workers_waiting_on_interaction.append(address)
            elif message_type == 'MEASUREMENTS':
                # Measurements are available at location indicated
                measurements_path = msg[3].decode('utf-8')
                job = msg[4:]
                work_queue_socket.send_multipart([address, '', 'THANKS'])
                try:
                    reported_measurements = cpmeas.load_measurements(measurements_path)
                    self.queue_receive_measurements(reported_measurements, job)
                except Exception, e:
                    raise
                    # XXX - report error, push back job
            elif message_type == 'EXCEPTION':
                def disposition_callback(disposition, *args):
                    assert disposition in (SKIP, ABORT, DEBUG)
                    self.debug_reply_queue.put((address, disposition) + args)
                where = msg[3]
                if where == 'PIPELINE':
                    image_set_number, module_name, exc_type, exc_message, exc_traceback, filename, line_number = msg[4:]
                    event = AnalysisPipelineExceptionEvent(int(image_set_number), module_name,
                                                           exc_type, exc_message, exc_traceback,
                                                           filename, int(line_number),
                                                           disposition_callback)
                else:
                    exc_type, exc_message, exc_traceback, filename, line_number = msg[4:]
                    event = AnalysisWorkerExceptionEvent(exc_type, exc_message, exc_traceback,
                                                         filename, int(line_number),
                                                         disposition_callback)
                self.original_exception_events[address] = event
                self.post_event(event)
            elif message_type == 'DEBUG WAITING':
                work_queue_socket.send_multipart([address, '', 'ACK'])  # ACK
                self.debug_port_callbacks[address](int(msg[3]))
            elif message_type == 'DEBUG COMPLETE':
                # original callback will reply to message.
                self.post_event(AnalysisDebugCompleteEvent(self.original_exception_events[address]))
            else:
                raise ValueError("Unknown message from worker: %s", message_type)

        # announce that this job is done/cancelled
        self.announce_queue.put(['DONE', analysis_id])

        # make sure any workers waiting on interaction get notified
        for address in workers_waiting_on_interaction:
            # special: 2 parts to body instead of just 1
            work_queue_socket.send_multipart([address, '', '', 'CANCELLED'])

    def queue_job(self, image_set_number):
        self.work_queue.put(image_set_number)

    def queue_display_request(self, jobnum, display_blob):
        self.display_request_queue.put((jobnum, display_blob))
        # notify interface thread
        with self.interface_work_cv:
            self.interface_work_cv.notify()

    def queue_interaction_request(self, address, module_number, image_set_number, interaction_request_blob):
        '''queue an interaction request, to be handled by the GUI.  Results are
        returned from the GUI via the interaction_reply_queue.'''
        self.interaction_request_queue.put((address, module_number, image_set_number, interaction_request_blob))
        # notify interface thread
        with self.interface_work_cv:
            self.interface_work_cv.notify()

    def queue_receive_measurements(self, image_set_number, measure_blob):
        self.receive_measurements_queue.put((image_set_number, measure_blob))
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
            return  # already started

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

        def announcer():
            while True:
                mesg = cls.announce_queue.get()
                work_announce_socket.send_multipart(mesg)
        start_daemon_thread(target=announcer, name='RunAnalysis.announcer')

        # ensure subimager is started
        subimager.client.start_subimager()

        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(cellprofiler.__file__), '..') + ':' + os.environ['PYTHONPATH']
        else:
            os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(cellprofiler.__file__), '..')

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
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)

            def run_logger(workR, widx):
                while(True):
                    try:
                        line = workR.stdout.readline().rstrip()
                        if line:
                            logger.info("Worker %d: %s", widx, line)
                    except:
                        logger.info("stdout of Worker %d closed" % widx)
                        break
            start_daemon_thread(target=run_logger, args=(worker, idx,), name='worker stdout logger')

            cls.workers += [worker]
            cls.deadman_switches += [worker.stdin]  # closing stdin will kill subprocess

    @classmethod
    def stop_workers(cls):
        for deadman_switch in cls.deadman_swtiches:
            deadman_switch.close()
        cls.workers = []
        cls.deadman_swtiches = []


def find_python():
    return 'python'


def find_analysis_worker_source():
    return cellprofiler.analysis_worker.__file__


def start_daemon_thread(target=None, args=(), name=None):
    thread = threading.Thread(target=target, args=args, name=name)
    thread.daemon = True
    thread.start()
    return thread


class AbstractAnalysisEvent(object):
    "Something happened in an Analysis."
    pass


class AnalysisStartedEvent(AbstractAnalysisEvent):
    "Event indicating the analysis has started running."
    pass


class AnalysisProgressEvent(AbstractAnalysisEvent):
    """Progress was made in the Analysis.
    AnalysisProgressEvent.counts - dictionary with count of #images at each step of analysis.
    """
    def __init__(self, counts):
        self.counts = counts


class AnalysisFinishedEvent(AbstractAnalysisEvent):
    """Analysis finished successfully.
    AnalysisFinishedEvent.analysis_id - the ID of the finished analysis.
    AnalysisFinishedEvent.measurements - the resulting measurements.
    """
    def __init__(self, analysis_id, measurements):
        self.analysis_id = analysis_id
        self.measurements = measurements


class AnalysisEndedEvent(AbstractAnalysisEvent):
    """Analysis has terminated (finished or cancelled).
    AnalysisEndedEvent.cancelled - whether the analysis was cancelled.
    """
    def __init__(self, cancelled):
        self.cancelled = cancelled


class AnalysisDisplayEvent(AbstractAnalysisEvent):
    """Display request from an analysis.
    AnalysisDisplayEvent.image_set_number - image set number requesting display.
    AnalysisDisplayEvent.display_request - the display request data.
    """
    def __init__(self, image_set_number, display_request):
        self.image_set_number = image_set_number
        self.display_request = self.display_request


class AnalysisInteractionRequest(AbstractAnalysisEvent):
    """Interaction request from an analysis.
    AnalysisInteractionRequest.module_number - one-based module number requesting the interactions.
    AnalysisInteractionRequest.image_set_number - image set number requesting interaction.
    AnalysisInteractionRequest.interaction_request_blob - the interaction request data.
    AnalysisInteractionRequest.reply_cb - callback function for the reply.
    """
    def __init__(self, module_number, image_set_number, interaction_request_blob, reply_cb):
        self.module_number = module_number
        self.image_set_number = image_set_number
        self.interaction_request_blob = interaction_request_blob
        self.reply_cb = reply_cb


class AnalysisPausedEvent(AbstractAnalysisEvent):
    """Analysis has been paused."""
    pass


class AnalysisResumedEvent(AbstractAnalysisEvent):
    """Analysis has been resumed."""
    pass


class AnalysisPipelineExceptionEvent(AbstractAnalysisEvent):
    """An exception occurred in the pipeline during an analysis.
    AnalysisPipelineExceptionEvent.image_set_number - image set number that caused the exception.
    AnalysisPipelineExceptionEvent.module_name - module's name where exception happened.
    AnalysisPipelineExceptionEvent.exc_type - exception type (a string).
    AnalysisPipelineExceptionEvent.exc_message - the exception message.
    AnalysisPipelineExceptionEvent.exc_traceback - the traceback (a string).
    AnalysisPipelineExceptionEvent.filename - filename of the exception
    AnalysisPipelineExceptionEvent.line_number - line_number of the exception
    AnalysisPipelineExceptionEvent.disposition_cb - callback function for how to handle this exception.
             Callback arguments are: (disposition=SKIP/ABORT/DEBUG, verification_string=None, port_callback=None)
             verification_string and port_callback must be set if first argument is DEBUG.
             port_callback will be called when the remote debugger is available with the port as an argument.
    """
    def __init__(self, image_set_number, module_name, exc_type, exc_message, exc_traceback, filename, line_number, disposition_callback):
        self.image_set_number = image_set_number
        self.module_name = module_name
        self.exc_type = exc_type
        self.exc_message = exc_message
        self.exc_traceback = exc_traceback
        self.filename = filename
        self.line_number = line_number
        self.disposition_callback = disposition_callback


class AnalysisWorkerExceptionEvent(AbstractAnalysisEvent):
    """An exception occurred in the worker during an analysis, outside of the pipeline.
    AnalysisWorkerExceptionEvent.exc_type - exception type (a string).
    AnalysisWorkerExceptionEvent.exc_message - the exception message.
    AnalysisWorkerExceptionEvent.exc_traceback - the traceback (a string).
    AnalysisWorkerExceptionEvent.filename - filename of the exception
    AnalysisWorkerExceptionEvent.line_number - line_number of the exception
    AnalysisWorkerExceptionEvent.disposition_cb - callback function for how to handle this exception.
             Callback arguments are: (disposition=SKIP/ABORT/DEBUG, verification_string=None, port_callback=None)
             verification_string and port_callback must be set if first argument is DEBUG.
             ABORT = terminate the worker, SKIP = continue processing.
             port_callback will be called when the remote debugger is available with the port as an argument.
    """
    def __init__(self, exc_type, exc_message, exc_traceback, filename, line_number, disposition_callback):
        self.exc_type = exc_type
        self.exc_message = exc_message
        self.exc_traceback = exc_traceback
        self.filename = filename
        self.line_number = line_number
        self.disposition_callback = disposition_callback


class AnalysisDebugCompleteEvent(AbstractAnalysisEvent):
    """Remote debugging of a pipeline or worker has completed.
    AnalysisDebugCompleteEvent.original_event - the original AnalysisPipelineExceptionEvent or AnalysisWorkerExceptionEvent.

    The handler of this event should treat it (programmatically) in the same
    way as the original event, calling original_event.disposition_callback()
    (possibly debugging again, even).
    """
    def __init__(self, original_event):
        self.original_event = original_event


if __name__ == '__main__':
    import time
    import cellprofiler.pipeline
    import cellprofiler.preferences

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
        if isinstance(event, AnalysisEndedEvent):
            keep_going = False

    analysis.start_analysis(callback, None)  # no exception handling, yet.
    while keep_going:
        time.sleep(0.25)
    del analysis
    gc.collect()
