"""analysis_worker.py - Run pipelines on imagesets to produce measurements.

The analysis worker listens on a ZMQ port for work announcements. It then
requests jobs from the announcer and executes them. As an application,
the analysis worker runs three threads:

* Main thread - spawns the worker and monitor threads and enters a run loop.
                The run loop is needed on OS/X in order to process the UI.
                The UI is needed by ImageJ 1.0 which starts AWT. The main thread
                issues a stop notification to the worker thread after exiting
                the run loop.
                
* Worker thread - listens for jobs and processes them until it receives a stop
                  notification from the main thread.
                  
* Monitor thread - reads from STDIN. If the parent process closes STDIN,
                   the read call throws an exception and the monitor thread
                   stops the main thread's run loop.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import sys
import os
import logging
logger = logging.getLogger(__name__)

'''Set the log level through the environment by specifying AW_LOG_LEVEL'''
AW_LOG_LEVEL = "AW_LOG_LEVEL"

work_announce_address = None
def aw_parse_args():
    '''Parse the application arguments into setup parameters'''
    from cellprofiler.preferences import \
         set_headless, set_awt_headless, \
         set_plugin_directory, set_ij_plugin_directory
    import optparse
    global work_announce_address
    set_headless()
    set_awt_headless(False)
    parser = optparse.OptionParser()
    parser.add_option("--work-announce",
                      dest="work_announce_address",
                      help="ZMQ port where work announcements are published",
                      default=None)
    parser.add_option("--log-level",
                      dest="log_level",
                      help="Logging level for logger: DEBUG, INFO, WARNING, ERROR",
                      default=os.environ.get(AW_LOG_LEVEL, logging.INFO))
    parser.add_option("--plugins-directory",
                      dest="plugins_directory",
                      help="Folder containing the CellProfiler plugin modules needed by client pipelines",
                      default=None)
    parser.add_option("--ij-plugins-directory",
                      dest="ij_plugins_directory",
                      help="Folder containing the ImageJ plugin .jar and .class files needed bby client pipelines",
                      default=None)
    parser.add_option("--jvm-heap-size",
                      dest="jvm_heap_size",
                      default=None,
                      help=("This is the amount of memory reserved for the "
                            "Java Virtual Machine (similar to the java -Xmx switch)."
                            "Example formats: 512000k, 512m, 1g"))

    options, args = parser.parse_args()
    if options.jvm_heap_size is not None:
        from cellprofiler.preferences import set_jvm_heap_mb
        set_jvm_heap_mb(options.jvm_heap_size, False)
    logging.root.setLevel(options.log_level)
    if len(logging.root.handlers) == 0:
        logging.root.addHandler(logging.StreamHandler())

    if not options.work_announce_address:
        parser.print_help()
        sys.exit(1)
    work_announce_address = options.work_announce_address
    #
    # Set up the headless plugins and ij plugins directories before doing
    # anything so loading will get them
    #
    if options.plugins_directory is not None:
        set_plugin_directory(options.plugins_directory, globally=False)
    else:
        logger.warning("Plugins directory not set")
    if options.ij_plugins_directory is not None:
        logger.debug("Using %s as IJ plugins directory" % options.ij_plugins_directory)
        set_ij_plugin_directory(options.ij_plugins_directory, globally=False)
    else:
        logger.debug("IJ plugins directory not set")

if __name__=="__main__":
    if "CP_DEBUG_WORKER" not in os.environ:
        #
        # Sorry to put ugliness so early:
        #     The process inherits file descriptors from the parent. Windows doesn't
        #     let you selectively inherit file descriptors, so we close them here.
        #
        try:
            maxfd = os.sysconf('SC_OPEN_MAX')
        except:
            maxfd = 256
        os.closerange(3, maxfd)
    if not hasattr(sys, 'frozen'):
        # In the development version, maybe the bioformats package is installed?
        # Add the root to the pythonpath
        root = os.path.split(os.path.split(__file__)[0])[0]
        sys.path.insert(0, root)

    aw_parse_args()
    
import time
import threading
import thread
import random
import zmq
import cStringIO as StringIO
import gc
import traceback
from weakref import WeakSet

import cellprofiler.workspace as cpw
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
from cellprofiler.gui.errordialog import ED_STOP, ED_SKIP
from cellprofiler.analysis import \
     PipelinePreferencesRequest, InitialMeasurementsRequest, WorkRequest, \
     NoWorkReply, MeasurementsReport, InteractionRequest, DisplayRequest, \
     DisplayPostGroupRequest, AnalysisCancelRequest, \
     ExceptionReport, DebugWaiting, DebugComplete, InteractionReply, \
     ServerExited, ImageSetSuccess, ImageSetSuccessWithDictionary, \
     SharedDictionaryRequest, Ack, UpstreamExit, ANNOUNCE_DONE,  \
     OmeroLoginRequest, OmeroLoginReply
import javabridge as J
from cellprofiler.utilities.rpdb import Rpdb
from cellprofiler.utilities.run_loop import enter_run_loop, stop_run_loop
#
# CellProfiler expects NaN as a result during calculation
#
import numpy as np
np.seterr(all='ignore')


# to guarantee closing of measurements, we store all of them in a WeakSet, and
# close them on exit.
all_measurements = WeakSet()

DEADMAN_START_ADDR = "inproc://deadmanstart"
DEADMAN_START_MSG = "STARTED"
NOTIFY_ADDR = "inproc://notify"
NOTIFY_STOP = "STOP"

the_zmq_context = zmq.Context.instance()

def main():
    #
    # For Windows build with Ilastik, look for site-packages
    # in order to find Ilastik sources.
    #
    if hasattr(sys, 'frozen') and sys.platform == "win32":
        root = os.path.split(sys.argv[0])[0]
        if len(root) == 0:
            root = os.curdir
        root = os.path.abspath(root)
        site_packages = os.path.join(root, 'site-packages').encode('utf-8')
        if os.path.exists(site_packages) and os.path.isdir(site_packages):
            import site
            site.addsitedir(site_packages)
    #
    # For OS/X set up the UI elements that users expect from
    # an app.
    #
    if sys.platform == "darwin":
        from cellprofiler.icons import get_builtin_images_path
        
        icon_path = os.path.join(get_builtin_images_path(), "CellProfilerIcon.png")
        os.environ["APP_NAME_%d" % os.getpid()] = "CellProfilerWorker"
        os.environ["APP_ICON_%d" % os.getpid()] = icon_path
    
    # Start the JVM
    from cellprofiler.utilities.cpjvm import cp_start_vm
    cp_start_vm()
    
    deadman_start_socket = the_zmq_context.socket(zmq.PAIR)
    deadman_start_socket.bind(DEADMAN_START_ADDR)
    
    # Start the deadman switch thread.
    start_daemon_thread(target=exit_on_stdin_close, 
                        name="exit_on_stdin_close")
    deadman_start_socket.recv()
    deadman_start_socket.close()
        
    with AnalysisWorker(work_announce_address) as worker:
        worker_thread = threading.Thread(target = worker.run, 
                                         name="WorkerThread")
        worker_thread.setDaemon(True)
        worker_thread.start()
        enter_run_loop()
        worker_thread.join()
            
    #
    # Shutdown - need to handle some global cleanup here
    #
    try:
        from ilastik.core.jobMachine import GLOBAL_WM
        GLOBAL_WM.stopWorkers()
    except:
        logger.warn("Failed to stop Ilastik")
    try:
        from imagej.imagej2 import allow_quit
        allow_quit()
    except:
        logger.warn("Failed to signal ImageJ to stop")
    try:
        J.kill_vm()
    except:
        logger.warn("Failed to stop the Java VM")
            
        
class AnalysisWorker(object):
    '''An analysis worker processing work at a given address
    
    '''
    def __init__(self, work_announce_address, with_stop_run_loop=True):
        from bioformats.formatreader import set_omero_login_hook
        self.work_announce_address = work_announce_address
        self.cancelled = False
        self.with_stop_run_loop = with_stop_run_loop
        self.current_analysis_id = False
        set_omero_login_hook(self.omero_login_handler)
        
    def __enter__(self):
        # (analysis_id -> (pipeline, preferences dictionary))
        self.pipelines_and_preferences = {}

        # initial measurements (analysis_id -> measurements)
        self.initial_measurements = {}
        self.cancelled = False
        self.current_analysis_id = None

        # pipeline listener object
        self.pipeline_listener = PipelineEventListener(self.handle_exception)
        return self
        
    def __exit__(self, type, value, traceback):
        for m in self.initial_measurements.values():
            m.close()
        self.initial_measurements = {}
     
    class AnalysisWorkerThreadObject(object):
        '''Provide the scope needed by the analysis worker thread
        
        '''
        def __init__(self, worker):
            self.worker = worker
            
        def __enter__(self):
            self.worker.enter_thread()
            
        def __exit__(self, type, value, tb):
            if type is not None:
                traceback.print_exception(type, value, tb)
            self.worker.exit_thread()
        
    def enter_thread(self):
        J.attach()
        if not cpprefs.get_awt_headless():
            J.activate_awt()
        self.notify_socket = the_zmq_context.socket(zmq.SUB)
        self.notify_socket.setsockopt(zmq.SUBSCRIBE, "")
        self.notify_socket.connect(NOTIFY_ADDR)
        
    def exit_thread(self):
        from bioformats.formatreader import clear_image_reader_cache
        self.notify_socket.close()
        clear_image_reader_cache()
        J.deactivate_awt()
        J.detach()
        if self.with_stop_run_loop:
            stop_run_loop()
        
    def run(self):
        from cellprofiler.pipeline import CancelledException
        t0 = 0
        with self.AnalysisWorkerThreadObject(self):
            while not self.cancelled:
                try:
                    self.current_analysis_id, \
                        self.work_request_address = self.get_announcement()
                    if t0 is None or time.time() - t0 > 30:
                        logger.debug("Connecting at address %s" % 
                                     self.work_request_address)
                        t0 = time.time()
                    self.work_socket = the_zmq_context.socket(zmq.REQ)
                    self.work_socket.connect(self.work_request_address)
                    # fetch a job 
                    job = self.send(WorkRequest(self.current_analysis_id))
        
                    if isinstance(job, NoWorkReply):
                        time.sleep(0.25)  # avoid hammering server
                        # no work, currently.
                        continue
                    self.do_job(job)
                except CancelledException:
                    break
                finally:
                    self.work_socket.close()
    
    def do_job(self, job):
        '''Handle a work request to its completion
        
        job - WorkRequest
        '''
        import cellprofiler.pipeline as cpp
        job_measurements = []
        try:
            send_dictionary = job.wants_dictionary
    
            logger.info("Starting job")
            # Fetch the pipeline and preferences for this analysis if we don't have it
            current_pipeline, current_preferences = \
                self.pipelines_and_preferences.get(
                    self.current_analysis_id, (None, None))
            if not current_pipeline:
                logger.debug("Fetching pipeline and preferences")
                rep = self.send(PipelinePreferencesRequest(
                    self.current_analysis_id))
                logger.debug("Received pipeline and preferences response")
                preferences_dict = rep.preferences
                # update preferences to match remote values
                cpprefs.set_preferences_from_dict(preferences_dict)
 
                logger.debug("Loading pipeline")
                pipeline_blob = rep.pipeline_blob.tostring()
                current_pipeline = cpp.Pipeline()
                current_pipeline.loadtxt(StringIO.StringIO(pipeline_blob), 
                                         raise_on_error=True)
                logger.debug("Pipeline loaded")
                current_pipeline.add_listener(
                    self.pipeline_listener.handle_event)
                current_preferences = rep.preferences
                self.pipelines_and_preferences[self.current_analysis_id] = (
                    current_pipeline, current_preferences)
            else:
                # update preferences to match remote values
                cpprefs.set_preferences_from_dict(current_preferences)
            
            # Reset the listener's state
            self.pipeline_listener.reset()
            logger.debug("Getting initial measurements")
            # Fetch the path to the intial measurements if needed.
            current_measurements = self.initial_measurements.get(
                self.current_analysis_id)
            if current_measurements is None:
                logger.debug("Sending initial measurements request")
                rep = self.send(InitialMeasurementsRequest(
                    self.current_analysis_id))
                logger.debug("Got initial measurements")
                current_measurements = \
                    self.initial_measurements[self.current_analysis_id] = \
                    cpmeas.load_measurements_from_buffer(rep.buf)
            else:
                logger.debug("Has initial measurements")
            # Make a copy of the measurements for writing during this job
            current_measurements = cpmeas.Measurements(copy=current_measurements)
            all_measurements.add(current_measurements)
            job_measurements.append(current_measurements)
        
            successful_image_set_numbers = []
            image_set_numbers = job.image_set_numbers
            worker_runs_post_group = job.worker_runs_post_group
            logger.info("Doing job: " + ",".join(map(str, image_set_numbers)))
        
            self.pipeline_listener.image_set_number = image_set_numbers[0]
        
            if not worker_runs_post_group:
                # Get the shared state from the first imageset in this run.
                shared_dicts = self.send(
                    SharedDictionaryRequest(self.current_analysis_id)).dictionaries
                assert len(shared_dicts) == len(current_pipeline.modules())
                for module, new_dict in zip(current_pipeline.modules(), 
                                            shared_dicts):
                    module.set_dictionary_for_worker(new_dict)
        
            # Run prepare group if this is the first image in the group.  We do
            # this here (even if there's no grouping in the pipeline) to ensure
            # that any changes to the modules' shared state dictionaries get
            # propagated correctly.
            should_process = True
            if current_measurements[cpmeas.IMAGE, 
                                    cpmeas.GROUP_INDEX, 
                                    image_set_numbers[0]] == 1:
                workspace = cpw.Workspace(current_pipeline, None, None, None,
                                          current_measurements, None, None)
                if not current_pipeline.prepare_group(
                    workspace, 
                    current_measurements.get_grouping_keys(), 
                    image_set_numbers):
                    # exception handled elsewhere, possibly cancelling this run.
                    should_process = False
                del workspace
        
            # process the images
            if should_process:
                abort = False
                for image_set_number in image_set_numbers:
                    gc.collect()
                    try:
                        self.pipeline_listener.image_set_number = image_set_number
                        current_pipeline.run_image_set(
                            current_measurements,
                            image_set_number,
                            self.interaction_handler,
                            self.display_handler,
                            self.cancel_handler)
                        if self.pipeline_listener.should_abort:
                            abort = True
                            break
                        elif self.pipeline_listener.should_skip:
                            # Report skipped image sets as successful so that
                            # analysis can complete.
                            # Report their measurements because some modules
                            # may have provided measurements before skipping.
                            pass
                        successful_image_set_numbers.append(image_set_number)
                        # Send an indication that the image set finished successfully.
                        if send_dictionary:
                            # The jobserver would like a copy of our modules' 
                            # run_state dictionaries.
                            ws = cpw.Workspace(current_pipeline, None, None, None,
                                               current_measurements, None, None)
                            dicts = [m.get_dictionary_for_worker() 
                                     for m in current_pipeline.modules()]
                            req = ImageSetSuccessWithDictionary(
                                self.current_analysis_id,
                                image_set_number=image_set_number,
                                shared_dicts = dicts)
                        else:
                            req = ImageSetSuccess(
                                self.current_analysis_id,
                                image_set_number = image_set_number)
                        rep = self.send(req)
                    except cpp.CancelledException:
                        logging.info("Aborting job after cancellation")
                        abort = True
                    except Exception:
                        try:
                            logging.error("Error in pipeline", exc_info=True)
                            if self.handle_exception(
                                image_set_number=image_set_number) == ED_STOP:
                                abort = True
                                break
                        except:
                            logging.error("Error in handling of pipeline exception", exc_info=True)
                            # this is bad.  We can't handle nested exceptions
                            # remotely so we just fail on this run.
                            abort = True
        
                if abort:
                    current_measurements.close()
                    job_measurements.remove(current_measurements)
                    return
        
                if worker_runs_post_group:
                    workspace = cpw.Workspace(current_pipeline, None, 
                                              current_measurements, None,
                                              current_measurements, None, None)
                    workspace.interaction_handler = self.interaction_handler
                    workspace.cancel_handler = self.cancel_handler
                    workspace.post_group_display_handler = \
                        self.post_group_display_handler
                    # There might be an exception in this call, but it will be
                    # handled elsewhere, and there's nothing we can do for it
                    # here.
                    current_pipeline.post_group(
                        workspace, 
                        current_measurements.get_grouping_keys())
        
            # send measurements back to server
            req = MeasurementsReport(self.current_analysis_id,
                                     buf=current_measurements.file_contents(),
                                     image_set_numbers=image_set_numbers)
            rep = self.send(req)
        
        except cpp.CancelledException:
            # Main thread received shutdown signal
            raise
        
        except Exception:
            logging.error("Error in worker", exc_info=True)
            if self.handle_exception() == ED_STOP:
                raise cpp.CancelledException("Cancelling after user-requested stop")
        finally:
            # Clean up any measurements owned by us
            for m in job_measurements:
                m.close()
        
    def interaction_handler(self, module, *args, **kwargs):
        '''handle interaction requests by passing them to the jobserver and wait for the reply.'''
        # we write args and kwargs into the InteractionRequest to allow
        # more complex data to be sent by the underlying zmq machinery.
        arg_kwarg_dict = dict([('arg_%d' % idx, v) for idx, v in enumerate(args)] +
                              [('kwarg_%s' % name, v) for (name, v) in kwargs.items()])
        req = InteractionRequest(
            self.current_analysis_id,
            module_num=module.module_num,
            num_args=len(args),
            kwargs_names=kwargs.keys(),
            **arg_kwarg_dict)
        rep = self.send(req)
        return rep.result
    
    def cancel_handler(self):
        '''Handle a cancel request by sending AnalysisCancelRequest
        
        '''
        self.send(AnalysisCancelRequest(self.current_analysis_id))
        
    def display_handler(self, module, display_data, image_set_number):
        '''handle display requests'''
        req = DisplayRequest(self.current_analysis_id,
                             module_num=module.module_num,
                             display_data_dict=display_data.__dict__,
                             image_set_number=image_set_number)
        rep = self.send(req)
        
    def post_group_display_handler(self, module, display_data, image_set_number):
        req = DisplayPostGroupRequest(
            self.current_analysis_id,
            module.module_num, display_data.__dict__, image_set_number)
        rep = self.send(req)
        
    def omero_login_handler(self):
        '''Handle requests for an Omero login'''
        from bioformats.formatreader import use_omero_credentials
        req = OmeroLoginRequest(self.current_analysis_id)
        rep = self.send(req)
        use_omero_credentials(rep.credentials)
        
    def send(self, req, work_socket = None):
        '''Send a request and receive a reply
        
        req - request to send
        
        socket - socket to use for send. Default is current work socket
        
        returns a reply on success. If cancelled, throws a CancelledException
        '''
        if self.current_analysis_id is None:
            from cellprofiler.pipeline import CancelledException
            raise CancelledException("Can't send after cancelling")
        if work_socket is None:
            work_socket = self.work_socket
        poller = zmq.Poller()
        poller.register(self.notify_socket, zmq.POLLIN)
        poller.register(work_socket, zmq.POLLIN)
        req.send_only(work_socket)
        response = None
        while response is None:
            for socket, state in poller.poll():
                if socket == self.notify_socket and state == zmq.POLLIN:
                    notify_msg = self.notify_socket.recv()
                    if notify_msg == NOTIFY_STOP:
                        self.cancelled = True
                        self.raise_cancel(
                            "Received stop notification while waiting for "
                            "response from %s" % str(req))
                if socket == work_socket and state == zmq.POLLIN:
                    response = req.recv(work_socket)
        if isinstance(response, UpstreamExit):
            self.raise_cancel(
                "Received UpstreamExit for analysis %s during request %s" %
                (self.current_analysis_id, str(req)))
        return response
    
    def raise_cancel(self, msg="Cancelling analysis"):
        '''Handle the cleanup after some proximate cause of cancellation
        
        msg - reason for cancellation
        
        This should only be called upon detection of a server-driven
        cancellation of analysis: either UpstreamExit or a stop notification
        from the deadman thread.
        '''
        from cellprofiler.pipeline import CancelledException
        logger.debug(msg)
        self.cancelled = True
        if self.current_analysis_id in self.initial_measurements:
            self.initial_measurements[self.current_analysis_id].close()
            del self.initial_measurements[self.current_analysis_id]
        if self.current_analysis_id in self.pipelines_and_preferences:
            del self.pipelines_and_preferences[self.current_analysis_id]
        self.current_analysis_id = None
        raise CancelledException(msg)
        
    def get_announcement(self):
        '''Connect to the announcing socket and get an analysis announcement
        
        returns an analysis_id / worker_request address pair
        
        raises a CancelledException if we detect cancellation.
        '''
        poller = zmq.Poller()
        poller.register(self.notify_socket, zmq.POLLIN)
        announce_socket = the_zmq_context.socket(zmq.SUB)
        announce_socket.setsockopt(zmq.SUBSCRIBE, "")
        announce_socket.connect(self.work_announce_address)
        try:
            poller.register(announce_socket, zmq.POLLIN)
            while True:
                for socket, state in poller.poll():
                    if socket == self.notify_socket and state == zmq.POLLIN:
                        msg = self.notify_socket.recv()
                        if msg == NOTIFY_STOP:
                            from cellprofiler.pipeline import CancelledException
                            self.cancelled = True
                            raise CancelledException()
                    elif socket == announce_socket and state == zmq.POLLIN:
                        announcement = dict(announce_socket.recv_json())
                        if len(announcement) == 0:
                            threading._sleep(0.25)
                            continue
                        if self.current_analysis_id in announcement:
                            analysis_id = self.current_analysis_id
                        else:
                            analysis_id = random.choice(announcement.keys())
                        return analysis_id, announcement[analysis_id]
        finally:
            announce_socket.close()

    def handle_exception(self, image_set_number=None, 
                         module_name=None, exc_info=None):
        '''report and handle an exception, possibly by remote debugging, returning
        how to proceed (skip or abort).
    
        A new socket is created for each exception report, to allow us to sidestep
        any REP/REQ state in the worker.
        '''
        if self.current_analysis_id is None:
            # Analysis has been cancelled - don't initiate server interactions
            return ED_STOP
        if exc_info is None:
            t, exc, tb = sys.exc_info()
        else:
            t, exc, tb = exc_info
        filename, line_number, _, _ = traceback.extract_tb(tb, 1)[0]
        report_socket = the_zmq_context.socket(zmq.REQ)
        try:
            report_socket.connect(self.work_request_address)
        except:
            return ED_STOP  # nothing to do but give up
        try:
            req = ExceptionReport(
                self.current_analysis_id,
                image_set_number,
                module_name,
                exc_type=t.__name__,
                exc_message=str(exc),
                exc_traceback="".join(traceback.format_exception(t, exc, tb)),
                filename=filename, line_number=line_number)
            reply = self.send(req, report_socket)
            while True:
                if reply.disposition == 'DEBUG':
                    # 
                    # Send DebugWaiting after we know the port #
                    #
                    debug_reply = [None]
                    def pc(port):
                        print "GOT PORT ", port
                        debug_reply[0] = self.send(
                            DebugWaiting(self.current_analysis_id, port),
                            report_socket)
                    print  "HASH", reply.verification_hash
                    try:
                        rpdb = Rpdb(verification_hash=reply.verification_hash, 
                                    port_callback=pc)
                    except:
                        return ED_STOP
                    rpdb.verify()
                    rpdb.post_mortem(tb)
                    # We get a new reply at the end, which might be "DEBUG" again.
                    reply = self.send(DebugComplete(self.current_analysis_id),
                                      report_socket)
                else:
                    return reply.disposition
        finally:
            report_socket.close()


class PipelineEventListener(object):
    """listen for pipeline events, communicate them as necessary to the
    analysis manager."""
    def __init__(self, handle_exception_fn):
        self.handle_exception_fn = handle_exception_fn
        self.image_set_number = 0
        self.should_abort = False
        self.should_skip = False

    def reset(self):
        self.should_abort = False
        self.should_skip = False

    def handle_event(self, pipeline, event):
        from cellprofiler.pipeline import RunExceptionEvent
        if isinstance(event, RunExceptionEvent):
            disposition = self.handle_exception_fn(
                image_set_number=self.image_set_number,
                module_name=event.module.module_name,
                exc_info=(type(event.error), event.error, event.tb))
            if disposition == ED_STOP:
                self.should_abort = True
                event.cancel_run = True
            elif disposition == ED_SKIP:
                self.should_skip = True
                event.cancel_run = False
                event.skip_thisset = True

__the_notify_pub_socket = None

def get_the_notify_pub_socket():
    '''Get the socket used to publish the worker stop message'''
    global __the_notify_pub_socket
    if __the_notify_pub_socket is None or __the_notify_pub_socket.closed:
        __the_notify_pub_socket = the_zmq_context.socket(zmq.PUB)
        __the_notify_pub_socket.bind(NOTIFY_ADDR)
    return __the_notify_pub_socket
        
def exit_on_stdin_close():
    '''Read until EOF, then exit, possibly without cleanup.'''
    notify_pub_socket = get_the_notify_pub_socket()
    deadman_socket = the_zmq_context.socket(zmq.PAIR)
    deadman_socket.connect(DEADMAN_START_ADDR)
    deadman_socket.send(DEADMAN_START_MSG)
    deadman_socket.close()
    
    # If sys.stdin closes, either our parent has closed it (indicating we
    # should exit), or our parent has died.  Attempt to exit cleanly via main
    # thread, but if that takes too long (hung filesystem or socket, perhaps),
    # use a hard os._exit() instead.
    stdin = sys.stdin
    try:
        while stdin.read():
            pass
    except:
        pass
    finally:
        print "Cancelling worker"
        notify_pub_socket.send(NOTIFY_STOP)
        notify_pub_socket.close()
        # hard exit after 10 seconds unless app exits
        time.sleep(10)
        for m in all_measurements:
            try:
                m.close()
            except:
                pass
        os._exit(0)

def start_daemon_thread(target=None, args=(), name=None):
    thread = threading.Thread(target=target, args=args, name=name)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    main()
