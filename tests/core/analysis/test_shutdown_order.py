import threading
from unittest.mock import Mock

from cellprofiler_core.analysis._runner import Runner


def make_runner():
    runner = object.__new__(Runner)
    runner.boundary = Mock()
    runner.workers = [Mock()]
    runner.interface_thread = Mock()
    runner.jobserver_thread = Mock()
    runner.interface_work_cv = threading.Condition()
    runner.jobserver_work_cv = threading.Condition()
    runner.cancelled = False
    runner.paused = True
    return runner


def test_cancel_joins_socket_owners_before_destroying_context():
    runner = make_runner()
    boundary = runner.boundary
    interface = runner.interface_thread
    jobserver = runner.jobserver_thread
    joined = set()
    interface.join.side_effect = lambda: joined.add("interface")
    jobserver.join.side_effect = lambda: joined.add("jobserver")

    def destroy(linger):
        assert joined == {"interface", "jobserver"}
        assert linger == 0

    boundary.zmq_context.destroy.side_effect = destroy
    runner.cancel()
    boundary.join.assert_called_once_with()
    boundary.zmq_context.destroy.assert_called_once_with(0)
    assert runner.interface_thread is None
    assert runner.jobserver_thread is None
    assert runner.boundary is None
    assert runner.cancelled
    assert not runner.paused


def test_stop_workers_preserves_default_context_cleanup():
    runner = make_runner()
    boundary = runner.boundary
    worker = runner.workers[0]
    assert runner.stop_workers() is None
    boundary.join.assert_called_once_with()
    boundary.zmq_context.destroy.assert_called_once_with(0)
    worker.wait.assert_called_once_with()
    assert runner.boundary is None
    assert not runner.workers
