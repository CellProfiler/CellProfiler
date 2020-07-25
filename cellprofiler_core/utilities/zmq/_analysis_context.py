from .communicable import Communicable
from .communicable.reply.upstream_exit import BoundaryExited


class AnalysisContext:
    """The analysis context holds the pieces needed to route analysis requests"""

    def __init__(self, analysis_id, upq, lock):
        self.lock = lock
        self.analysis_id = analysis_id
        self.upq = upq
        self.cancelled = False
        # A map of requests pending to the closure that can be used to
        # reply to the request
        self.reqs_pending = set()

    def reply(self, req, rep):
        """Reply to a AnalysisRequest with this analysis ID

        rep - the intended reply

        Returns True if the intended reply was sent, returns False
        if BoundaryExited was sent instead.

        Always executed on the boundary thread.
        """
        with self.lock:
            if self.cancelled:
                return False
            if req in self.reqs_pending:
                Communicable.reply(req, rep)
                self.reqs_pending.remove(req)
            return True

    def enqueue(self, req):
        """Enqueue a request on the upward queue

        req - request to be enqueued. The enqueue should be done before
              req.reply is replaced.

        returns True if the request was enqueued, False if the analysis
        has been cancelled. It is up to the caller to send a BoundaryExited
        reply to the request.

        Always executes on the boundary thread.
        """
        with self.lock:
            if not self.cancelled:
                assert req not in self.reqs_pending
                self.reqs_pending.add(req)
                self.upq.put(req)
                return True
            else:
                Communicable.reply(req, BoundaryExited())
                return False

    def cancel(self):
        """Cancel this analysis

        All analysis requests will receive BoundaryExited() after this
        method returns.
        """
        with self.lock:
            if self.cancelled:
                return
            self.cancelled = True
            self.upq = None

    def handle_cancel(self):
        """Handle a cancel in the boundary thread.

        Take care of workers expecting replies.
        """
        with self.lock:
            for req in list(self.reqs_pending):
                Communicable.reply(req, BoundaryExited())
            self.reqs_pending = set()
