"""analysis.py - Run pipelines on imagesets to produce measurements.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import sys
import os
import time
import optparse
import threading
import thread
import random
import zmq
import cStringIO as StringIO
import gc

import cellprofiler.pipeline


def main():
    parser = optparse.OptionParser()
    parser.add_option("--work-announce",
                      dest="work_announce_address",
                      help="ZMQ port where work announcements are published",
                      default=None)
    options, args = parser.parse_args()

    if not options.work_announce_address:
        parser.print_help()
        sys.exit(1)

    # Start the deadman switch thread.
    start_daemon_thread(target=exit_on_stdin_close)

    # known work servers (analysis_id -> socket_address)
    work_servers = {}
    # their pipelines (analysis_id -> pipeline)
    pipelines = {}
    # lock for work_server and pipelines
    work_server_lock = threading.Lock()

    start_daemon_thread(target=listen_for_announcements,
                        args=(options.work_announce_address,
                              work_servers, pipelines,
                              work_server_lock))

    zmq_context = zmq.Context()

    # Loop until exit
    while True:
        gc.collect()

        # no servers = no work
        if len(work_servers) == 0:
            time.sleep(.1)  # wait for work servers
            continue

        # connect to a randomly chosen work_server
        with work_server_lock:
            if len(work_servers) == 0:
                continue
            current_analysis_id = random.choice(work_servers.keys())
            current_server = work_servers[current_analysis_id]
        work_socket = zmq_context.socket(zmq.REQ)
        try:
            work_socket.connect(current_server)
        except:
            # give up on this server.  We'll get a new announcement if it comes back.
            with work_server_lock:
                del work_servers[current_analysis_id]
            continue

        # Fetch the pipeline for this analysis if we don't have it
        current_pipeline = pipelines.get(current_analysis_id, None)
        if not current_pipeline:
            work_socket.send("PIPELINE")
            pipeline_blob = work_socket.recv()
            pipeline = cellprofiler.pipeline.Pipeline()
            pipeline.loadtxt(StringIO.StringIO(pipeline_blob))
            # make sure the server hasn't finished or quit since we fetched the pipeline
            with work_server_lock:
                if current_analysis_id in work_servers:
                    current_pipeline = pipelines[current_analysis_id] = pipeline
                else:
                    continue

        # fetch a job
        work_socket.send("WORK REQUEST")
        job = work_socket.recv()
        if job == 'NONE':
            # no work, currently.
            continue

        print "Doing job", job

        # possibly send a display packet
        if not random.randint(0, 5):
            print "DISPLAYING"
            work_socket.send_multipart(["DISPLAY", job, "display blob"])
            work_socket.recv()  # get ACK

        # possibly send an interactive request
        if not random.randint(0, 5):
            print "INTERACTING"
            work_socket.send_multipart(["INTERACT", job, "interact request blob"])
            print "INTERACT RESULT", work_socket.recv()  # get result back

        # fake doing work, return measurements
        print "RUNNING PIPELINE", current_pipeline.settings_hash()
        time.sleep(2)
        work_socket.send_multipart(["MEASUREMENTS", job, "None"])
        work_socket.recv()  # get ACK


def exit_on_stdin_close():
    '''Read until EOF, then exit, possibly without cleanup.'''
    # If sys.stdin closes, either our parent has closed it (indicating we
    # should exit), or our parent has died.  Attempt to exit cleanly via main
    # thread, but if that takes too long (hung filesystem or socket, perhaps),
    # use a hard os._exit() instead.
    try:
        while sys.stdin.read():
            pass
    except:
        pass
    finally:
        thread.interrupt_main()  # try a soft shutdown
        # hard exit after 10 seconds
        time.sleep(10)
        os._exit(0)


def listen_for_announcements(work_announce_address,
                             work_servers, pipelines,
                             work_server_lock):
    zmq_context = zmq.Context()
    work_announce_socket = zmq_context.socket(zmq.SUB)
    work_announce_socket.setsockopt(zmq.SUBSCRIBE, '')
    work_announce_socket.connect(work_announce_address)

    while True:
        work_queue_address, analysis_id = work_announce_socket.recv_multipart()
        with work_server_lock:
            if work_queue_address == 'DONE':
                # remove this work server
                if analysis_id in work_servers:
                    del work_servers[analysis_id]
                if analysis_id in pipelines:
                    del pipelines[analysis_id]
            else:
                work_servers[analysis_id] = work_queue_address


def start_daemon_thread(target=None, args=(), name=None):
    thread = threading.Thread(target=target, args=args, name=name)
    thread.daemon = True
    thread.start()


if __name__ == "__main__":
    import cellprofiler.preferences
    cellprofiler.preferences.set_headless()
    main()
