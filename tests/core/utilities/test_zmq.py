import threading

import pytest
import zmq

from cellprofiler_core.utilities.zmq._boundary import Boundary


@pytest.mark.parametrize("fails", [False, True])
def test_boundary_closes_notification_socket_in_owning_thread(fails):
    boundary = Boundary.__new__(Boundary)
    boundary.threadlocal = threading.local()
    context = zmq.Context()
    boundary.threadlocal.notify_socket = context.socket(zmq.PUB)
    parent_socket = boundary.threadlocal.notify_socket
    sockets = []
    outcomes = []

    def work():
        socket = context.socket(zmq.PUB)
        boundary.threadlocal.notify_socket = socket
        sockets.append(socket)
        if fails:
            raise RuntimeError("thread failed")
        return 19

    def run():
        try:
            outcomes.append(boundary.run_in_thread(work))
        except RuntimeError as error:
            outcomes.append(str(error))

    try:
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        assert outcomes == (["thread failed"] if fails else [19])
        assert len(sockets) == 1 and sockets[0].closed
        assert not parent_socket.closed
        boundary.close_thread_socket()
        assert parent_socket.closed
    finally:
        context.destroy(linger=0)
