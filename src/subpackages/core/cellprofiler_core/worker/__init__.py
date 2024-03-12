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
"""

import logging
import os
import sys
import psutil

import importlib.resources

from cellprofiler_core.constants.worker import (
    DEADMAN_START_ADDR,
    DEADMAN_START_MSG,
    NOTIFY_STOP,
    all_measurements, NOTIFY_ADDR,
)
from cellprofiler_core.preferences import set_always_continue, set_conserve_memory
from cellprofiler_core.worker._worker import Worker
from cellprofiler_core.utilities.logging import set_log_level


LOGGER = logging.getLogger(__name__)

"""Set the log level through the environment by specifying AW_LOG_LEVEL"""
AW_LOG_LEVEL = "AW_LOG_LEVEL"

knime_bridge_address = None
notify_address = None
analysis_id = None
work_server_address = None


def aw_parse_args():
    """Parse the application arguments into setup parameters"""
    from cellprofiler_core.preferences import (
        set_headless,
        set_awt_headless,
        set_plugin_directory,
    )
    import argparse

    global analysis_id
    global work_server_address
    global notify_address
    global knime_bridge_address

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-id",
        dest="analysis_id",
        help="Unique ID for the analysis to be performed",
        default=None,
    )
    parser.add_argument(
        "--work-server",
        dest="work_server_address",
        help="ZMQ port where work requests should be sent",
        default=None,
    )
    parser.add_argument(
        "--notify-server",
        dest="notify_address",
        help="ZMQ port where continue/shutdown notifications are published",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=os.environ.get(AW_LOG_LEVEL, logging.INFO),
        help=(
            "Set the verbosity for logging messages: "
            + ("%d or %s for debugging, " % (logging.DEBUG, "DEBUG"))
            + ("%d or %s for informational, " % (logging.INFO, "INFO"))
            + ("%d or %s for warning, " % (logging.WARNING, "WARNING"))
            + ("%d or %s for error, " % (logging.ERROR, "ERROR"))
            + ("%d or %s for critical, " % (logging.CRITICAL, "CRITICAL"))
            + ("%d or %s for fatal." % (logging.FATAL, "FATAL"))
            + " Otherwise, the argument is interpreted as the file name of a log configuration file (see http://docs.python.org/library/logging.config.html for file format)"
        ),
    )
    parser.add_argument(
        "--plugins-directory",
        dest="plugins_directory",
        help="Folder containing the CellProfiler plugin modules needed by client pipelines",
        default=None,
    )
    parser.add_argument(
        "--conserve-memory",
        dest="conserve_memory",
        default=None,
        help="CellProfiler will attempt to release unused memory after each image set.",
    )
    parser.add_argument(
        "--jvm-heap-size",
        dest="jvm_heap_size",
        default=None,
        help=(
            "This is the amount of memory reserved for the "
            "Java Virtual Machine (similar to the java -Xmx switch)."
            "Example formats: 512000k, 512m, 1g"
        ),
    )
    parser.add_argument(
        "--knime-bridge-address",
        dest="knime_bridge_address",
        help="Open up a port to handle the Knime bridge protocol",
        default=None,
    )
    parser.add_argument(
        "--always-continue",
        dest="always_continue",
        help="Don't stop the analysis when an image set raises an error",
        default=None
    )

    args = parser.parse_args()
    
    set_log_level(args.log_level, subprocess=True)

    set_awt_headless(True)
    set_headless()

    # NOTE: don't call fill_readers here,
    # woker needs to do PipelinePreferences req then set_preferences_from_dict,
    # which it does later in worker.do_job

    if not args.work_server_address and args.work_server_address and \
            args.analysis_id:
        parser.print_help()
        sys.exit(1)
    analysis_id = args.analysis_id
    notify_address = args.notify_address
    work_server_address = args.work_server_address
    knime_bridge_address = args.knime_bridge_address

    #
    # Set up the headless plugins directories before doing
    # anything so loading will get them
    #
    if args.plugins_directory is not None:
        set_plugin_directory(args.plugins_directory, globally=False)
    if args.conserve_memory is not None:
        set_conserve_memory(args.conserve_memory, globally=False)
    if args.always_continue is not None:
        set_always_continue(args.always_continue, globally=False)
    else:
        LOGGER.warning("Plugins directory not set")


if __name__ == "__main__":
    #
    # Sorry to put ugliness so early: The process inherits file descriptors
    # from the parent. Windows doesn't let you selectively inherit file
    # descriptors, so we close them here.
    #
    try:
        maxfd = os.sysconf("SC_OPEN_MAX")
    except:
        maxfd = 256

    proc = psutil.Process()

    # This is a hacky solution to an annoying problem with vscode debugging:
    # we want to set breakpoints in the debugger,
    # but one of the file descriptors inhertited by the child
    # corresponds to a TCP/IPv4 connection attached to the debugger,
    # so closing it would cause the debugger to detach from the subprocess.
    # AFAICT the debugger fd is always 3 or 4, and is not associated with
    # any device + inode combo (both 0, ie null), so
    # if we find that, skip closing fd 4, else close all
    try:
        stat3 = os.fstat(3)
        stat4 = os.fstat(4)
        if stat3.st_ino == 0 and stat3.st_dev == 0:
            os.closerange(4, maxfd)
        elif stat4.st_ino == 0 and stat4.st_dev == 0:
            os.close(3)
            os.closerange(5, maxfd)
        else:
            os.closerange(3, maxfd)
    except OSError:
        os.closerange(3, maxfd)

    if not hasattr(sys, "frozen"):
        # In the development version, maybe the bioformats package is installed?
        # Add the root to the pythonpath
        root = os.path.split(os.path.split(__file__)[0])[0]
        sys.path.insert(0, root)

    aw_parse_args()

import time
import threading
import zmq

#
# CellProfiler expects NaN as a result during calculation
#
import numpy as np

np.seterr(all="ignore")


# to guarantee closing of measurements, we store all of them in a WeakSet, and
# close them on exit.


def main():
    #
    # For OS/X set up the UI elements that users expect from
    # an app.
    #
    if sys.platform == "darwin":
        import os.path

        icon_path = str(importlib.resources.files("cellprofiler").joinpath(
            "data", "icons", "CellProfiler.png"
        ))
        os.environ["APP_NAME_%d" % os.getpid()] = "CellProfilerWorker"
        os.environ["APP_ICON_%d" % os.getpid()] = icon_path

    # Start the JVM
    from cellprofiler_core.utilities.java import start_java, stop_java
    with zmq.Context() as the_zmq_context:
        the_zmq_context.setsockopt(zmq.LINGER, 0)
        deadman_start_socket = the_zmq_context.socket(zmq.PAIR)
        deadman_start_socket.bind(DEADMAN_START_ADDR)

        # Start the deadman switch thread.
        monitor_thread = threading.Thread(
            target=monitor_keepalive,
            args=(the_zmq_context, notify_address, ),
            name="heartbeat_monitor thread",
            daemon=True
        )
        monitor_thread.start()

        deadman_start_socket.recv()
        deadman_start_socket.close()

        from cellprofiler.knime_bridge import KnimeBridgeServer

        with Worker(the_zmq_context, analysis_id, work_server_address, notify_address) as worker:
            worker_thread = threading.Thread(
                target=worker.run,
                name="WorkerThread",
                daemon=True,
            )
            worker_thread.start()
            with KnimeBridgeServer(
                the_zmq_context, knime_bridge_address, NOTIFY_ADDR, NOTIFY_STOP
            ):
                worker_thread.join()
            the_zmq_context.destroy(linger=0)
        LOGGER.debug("Worker thread joined")
        #
        # Shutdown - need to handle some global cleanup here
        #
        try:
            stop_java()
        except:
            LOGGER.warning("Failed to stop the JVM", exc_info=True)


def monitor_keepalive(context, keepalive_address):
    """The keepalive socket should send a regular heartbeat telling workers
    to stay alive. This will stop if the parent process crashes.
    If we see no keepalive message in 15 seconds we take that as a bad
    omen and attempt to gracefully shut down the worker processes.

    The same socket also broadcasts the shutdown notification, so we watch
    for that too. The main worker thread should pick that up and shut down
    the next time it makes a request, but this can take a while. Therefore we
    also apply a timeout and cease work manually if the processes don't
    respond to the kill signal quickly enough.

    Note that if CellProfiler is paused in debug mode during an analysis, this
    can cause the workers to exit. Use test mode for testing or only debug on
    the worker threads."""
    keepalive_socket = context.socket(zmq.SUB)
    keepalive_socket.connect(keepalive_address)
    keepalive_socket.setsockopt(zmq.SUBSCRIBE, b"")

    # Send a message back when we've set this thread up
    deadman_socket = context.socket(zmq.PAIR)
    deadman_socket.connect(DEADMAN_START_ADDR)
    deadman_socket.send(DEADMAN_START_MSG)
    deadman_socket.close()

    missed = 0
    while missed < 3:
        event = keepalive_socket.poll(5000)
        if not event:
            missed += 1
            LOGGER.warning(f"Worker failed to receive communication for"
                            f" {5 * missed} seconds")
        else:
            missed = 0
            msg = keepalive_socket.recv()
            if msg == NOTIFY_STOP:
                break
    # Stop has been called, we must manually close our sockets before the
    # main thread can exit.
    keepalive_socket.close()
    if missed >= 3:
        # Parent is dead, hard kill
        LOGGER.critical("Worker stopped receiving communication from "
                         "CellProfiler, shutting down now")
    else:
        # Stop signal captured, give some time to shut down gracefully.
        time.sleep(10)
    LOGGER.info("Cancelling worker")
    # hard exit after 10 seconds unless app exits

    for m in all_measurements:
        try:
            m.close()
        except:
            pass
    LOGGER.error("Worker failed to stop gracefully, forcing exit now")
    os._exit(0)


if __name__ == "__main__":
    main()
    sys.exit(0)
