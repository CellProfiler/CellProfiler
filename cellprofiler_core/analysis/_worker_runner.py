import logging
import threading

import zmq

from cellprofiler_core.pipeline import CancelledException
from ..worker import Worker
from ..constants.worker import NOTIFY_ADDR, NOTIFY_STOP


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
                logging.info("Exiting debug worker thread")

    def wait(self):
        self.notify_socket.send(NOTIFY_STOP)
        self.join()
