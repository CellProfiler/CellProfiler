from __future__ import with_statement
import imaging.eukaryote.analysis.analysis_runner as analysis_runner
import imaging.eukaryote.measurements.measurements as cpmeas
import threading
import uuid


class Analysis(object):
    """
    An Analysis is the application of a particular pipeline of modules to a
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
    """

    def __init__(self, pipeline, measurements_filename, initial_measurements=None):
        """
        create an Analysis applying pipeline to a set of images, writing out
        to measurements_filename, optionally starting with previous
        measurements.
        """
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
        """
        Start the analysis runner

        analysis_event_callback - callback from runner to UI thread for
                                  event progress and UI handlers

        num_workers - # of worker processes to instantiate, default is # of cores

        overwrite - True (default) to process all image sets, False to only
                    process incomplete ones (or incomplete groups if grouping)
        """
        with self.runner_lock:
            assert not self.analysis_in_progress
            self.analysis_in_progress = uuid.uuid1().hex
            self.runner = analysis_runner.AnalysisRunner(self.analysis_in_progress, self.pipeline, self.initial_measurements_buf, self.output_path, analysis_event_callback)
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
        """
        Verify that an analysis is running, allowing the GUI to recover even
        if the AnalysisRunner fails in some way.

        Returns True if analysis is still running (threads are still alive).
        """
        with self.runner_lock:
            if self.analysis_in_progress:
                return self.runner.check()
            return False
