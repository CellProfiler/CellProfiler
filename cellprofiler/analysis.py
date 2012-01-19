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
import gc

import cellprofiler
import cellprofiler.measurements as cpmeas
import cellprofiler.analysis_worker  # used to get the path

logger = logging.getLogger(__name__)


class Analysis(object):
    '''An Analysis is the application of a particular pipeline of modules to a
    set of images to produce measurements.

    Multiprocessing for analyses is handled by multiple layers of threads and
    processes, to keep the GUI responsive and simplify the code.  Threads and
    processes are organized as below.

    +---------------------------------------------+
    |           CellProfiler GUI/WX thread        |
    +----------calls to Analysis methods()--------+
    |       AnalysisRunner.interface() thread     |
    +------------------  Queues  -----------------+
    |AnalysisRunner.jobserver()/announce() threads|
    +---------------------------------------------+
                     ZMQ sockets
    +-------------+----------------+--------------+
    |   Worker    |     Worker     |   Worker     |
    +-------------+----------------+--------------+

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

    # XXX - should we replace callbacks with the add_listener()/notify_listeners() pattern?
    def start_analysis(self, display_callback, interaction_callback):
        with self.runner_lock:
            assert not self.analysis_in_progress
            self.analysis_in_progress = uuid.uuid1().hex

            def finished_cb(analysis_id, measurement_results):
                if self.analysis_in_progress == analysis_id:
                    # XXX merge measurements_results into self.measurements
                    self.analysis_in_progress = False
                    self.runner = None
            self.runner = AnalysisRunner(self.analysis_in_progress,
                                         self.pipeline,
                                         display_callback,
                                         interaction_callback,
                                         self.measurements,
                                         finished_cb)
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


class AnalysisRunner(object):
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
                 display_callback, interaction_callback,
                 initial_measurements, finished_cb):
        # for sending to workers
        self.initial_measurements = cpmeas.Measurements(image_set_start=None,
                                                        copy=initial_measurements)
        # for writing results
        self.measurements = cpmeas.Measurements(image_set_start=None,
                                                copy=initial_measurements)
        self.analysis_id = analysis_id
        self.pipeline = pipeline.copy()
        self.display_callback = display_callback
        self.interaction_callback = interaction_callback
        self.finished_cb = finished_cb

        self.interface_work_cv = threading.Condition()
        self.pause_cancel_cv = threading.Condition()
        self.paused = False
        self.cancelled = False

        self.work_queue = Queue.Queue()
        self.display_request_queue = Queue.Queue()
        self.interaction_request_queue = Queue.Queue()
        self.interaction_reply_queue = Queue.Queue()
        self.receive_measurements_queue = Queue.Queue()

        self.start_workers(8)  # start worker pool via class method

    # External control interfaces
    def start(self):
        '''start the analysis run'''
        self.start_interface_thread()
        self.start_jobserver_thread(self.analysis_id)

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

    def resume(self):
        '''resume a paused analysis run'''
        with self.pause_cancel_cv:
            self.paused = False
            self.pause_cancel_cv.notifyAll()  # interface() and jobserver() threads

    def start_interface_thread(self):
        start_daemon_thread(target=self.interface, name='AnalysisRunner.interface')

    def start_jobserver_thread(self, analysis_id):
        start_daemon_thread(target=self.jobserver, args=(analysis_id,), name='AnalysisRunner.jobserver')

    def interface(self, image_set_start=1, image_set_end=None,
                     overwrite=False):
        '''Top-half thread for running an analysis.  Interacts with GUI and
        actual job server thread.

        image_set_start - beginning image set number
        image_set_end - final image set number
        overwrite - whether to recompute imagesets with data in initial_measurements.
        '''
        if image_set_end is None:
            image_set_end = len(self.measurements.get_image_numbers())

        def get_status(image_set_number):
            return self.measurements.get_measurement(cpmeas.IMAGE, self.STATUS, image_set_number)

        # reset the status of every image set that needs to be processed
        for image_set_number in range(image_set_start, image_set_end):
            if (overwrite or
                (not self.measurements.has_measurements(cpmeas.IMAGE, self.STATUS, image_set_number)) or
                (get_status(image_set_number) != self.STATUS_DONE)):
                self.measurements.add_measurement(cpmeas.IMAGE, self.STATUS, self.STATUS_UNPROCESSED, image_set_number=image_set_number)

        # Find image groups.  These are written into measurements prior to
        # analysis.  If the pipeline aggregates data or requires images in
        # order, we have to process individual groups as a single job.
        if self.pipeline.aggregates_data() or self.pipeline.needs_images_in_order():
            grouping_needed = True
            job_groups = {}
            for image_set_number in range(image_set_start, image_set_end):
                group_number = self.measurements.get_measurement(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_set_number)
                group_index = self.measurements.get_measurement(cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_set_number)
                job_groups[group_number] = job_groups.get(group_number, []) + [(group_index, image_set_number)]
            job_groups[group_number] = [[isn for _, isn in sorted(job_groups[group_number])] for group_number in job_groups]
        else:
            grouping_needed = False
            # no need for grouping
            job_groups = [[image_set_number] for image_set_number in range(image_set_start, image_set_end)]
            # XXX - we need to call prepare_group() here, based on its docstring in cpmodule.py, right?

        # XXX - check that any constructed groups are complete, i.e.,
        # image_set_start and image_set_end shouldn't carve them up.

        # put the jobs in the queue
        for job in job_groups:
            self.work_queue.put((job, grouping_needed))

        # We loop until every image is completed, or an outside event breaks the loop.
        while True:
            # take the interface_work_cv lock to keep our interaction with the meaurements atomic.
            with self.interface_work_cv:
                counts = dict((s, 0) for s in [self.STATUS_UNPROCESSED, self.STATUS_IN_PROCESS, self.STATUS_DONE])
                for image_set_number in range(image_set_start, image_set_end):
                    status = get_status(image_set_number)
                    counts[status] += 1

                # report our progress
                self.progress(counts[self.STATUS_UNPROCESSED], counts[self.STATUS_IN_PROCESS], counts[self.STATUS_DONE])

                # Are we finished?
                if (counts[self.STATUS_QUEUED] + counts[self.STATUS_UNPROCESSED]) == 0:
                    break

                # wait for an image set to finish or some sort of display or
                # interaction work that needs handling.
                self.interface_work_cv.wait()

            # check pause/cancel
            if not self.check_pause_and_cancel():
                break
            # display/interaction/measurements
            self.handle_queues()

        self.measurements.flush()
        self.finished_cb(self.analysis_id, self.measurements)
        del self.measurements
        self.analysis_id = False  # this will cause the jobserver thread to exit

    def progress(self, unprocessed, queued, done):
        print "PROGRESS", unprocessed, queued, done

    def check_pause_and_cancel(self):
        '''if paused, wait for unpause or cancel.  return True if work should
        continue, False if cancelled.
        '''
        # paused - wait for us to be unpaused
        with self.pause_cancel_cv:
            while self.paused:
                self.pause_cancel_cv.wait()
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
            self.display_callback(image_set_number, display_request)
        # interaction requests
        while not self.interaction_request_queue.empty():
            reply_address, image_set_number, interaction_request = self.interaction_request_queue.get()

            def reply_cb(interaction_reply, reply_address=reply_address):
                # this callback function will be called from another thread
                self.interaction_reply_queue.put((reply_address, interaction_reply))
                self.interface_work_cv.notify()  # notify interface() there's a reply to be handled.

            self.interaction_callback(image_set_number, interaction_request, reply_cb)
        # measurements
        while not self.receive_measurements_queue.empty():
            image_set_number, measure_blob = self.receive_measurements_queue.get()
            self.measurements.add_measurement(cpmeas.IMAGE, self.STATUS, self.STATUS_DONE, image_set_number=int(image_set_number))

    def jobserver(self, analysis_id):
        # this server subthread should be very thin and light, as it has to
        # handle all the requests from workers, of which there might be
        # several.

        # set up the zmq.XREP socket we'll serve jobs from
        work_queue_socket = self.zmq_context.socket(zmq.XREP)
        work_queue_socket.setsockopt(zmq.LINGER, 0)
        work_queue_port = work_queue_socket.bind_to_random_port('tcp://127.0.0.1')

        self.announce_queue.put(['tcp://127.0.0.1:%d' % work_queue_port, analysis_id])

        poller = zmq.Poller()
        poller.register(work_queue_socket, zmq.POLLIN)

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
            if message_type == 'INITIAL_MEASUREMENTS':
                work_queue_socket.send_multipart([address, '', self.initial_measurements.hdf5_dict.filename])
            elif message_type == 'WORK REQUEST':
                if not self.work_queue.empty():
                    job, grouping_needed = self.work_queue.get()
                    if grouping_needed:
                        work_queue_socket.send_multipart([address, '', 'GROUP'] + [str(j) for j in job])
                    else:
                        work_queue_socket.send_multipart([address, '', 'IMAGE', str(job[0])])
                else:
                    # there may be no work available, currently, but there
                    # may be some later.
                    work_queue_socket.send_multipart([address, '', 'NONE'])
            elif message_type == 'DISPLAY':
                self.queue_display_request(msg[3], msg[4])
                work_queue_socket.send_multipart([address, '', 'OK'])
            elif message_type == 'INTERACT':
                self.queue_interaction_request(address, msg[3], msg[4])
                # we don't reply immediately, as we have to wait for the
                # GUI to run.  We'll find the result on
                # self.interact_reply_queue sometime in the future.
            elif message_type == 'MEASUREMENTS':
                # Measurements are available at location indicated
                measurements_path = msg[3]
                job = msg[4:]
                try:
                    reported_measurements = cpmeas.load_measurements(measurements_path)
                    self.queue_receive_measurements(reported_measurements, msg[4])
                    work_queue_socket.send_multipart([address, '', 'THANKS'])
                except Exception, e:
                    # XXX - report error, push back job
                    pass

        # announce that this job is done/cancelled
        self.announce_queue.put(['DONE', analysis_id])

    def queue_job(self, image_set_number):
        self.work_queue.put(image_set_number)

    def queue_display_request(self, jobnum, display_blob):
        self.display_request_queue.put((jobnum, display_blob))
        # notify interface thread
        with self.interface_work_cv:
            self.interface_work_cv.notify()

    def queue_interaction_request(self, address, jobnum, interact_blob):
        '''queue an interaction request, to be handled by the GUI.  Results are
        returned from the GUI via the interaction_reply_queue.'''
        self.interaction_request_queue.put((address, jobnum, interact_blob))
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

        # start workers
        for idx in range(num):
            # stdin for the subprocesses serves as a deadman's switch.  When
            # closed, the subprocess exits.
            worker = subprocess.Popen([find_python(),
                                       '-u',  # unbuffered
                                       find_analysis_worker_source(),
                                       '--work-announce',
                                       'tcp://127.0.0.1:%d' % (work_announce_port)],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)

            def run_logger(workR, widx):
                while(True):
                    try:
                        line = workR.stdout.readline().strip()
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

    def display_callback(image_set_number, display_blob):
        print "DISPLAYING", image_set_number, display_blob

    def interact_callback(image_set_number, interaction_blob, reply_callback):
        print "INTERACTING", image_set_number, interaction_blob
        reply_callback("REPLY %s" % image_set_number)

    analysis.start_analysis(display_callback, interact_callback)
    while analysis.analysis_in_progress:
        time.sleep(0.25)
    del analysis
    gc.collect()
