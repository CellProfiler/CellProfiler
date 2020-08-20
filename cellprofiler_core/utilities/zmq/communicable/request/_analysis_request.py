from cellprofiler_core.utilities.zmq.communicable.request._request import Request


class AnalysisRequest(Request):
    """A request associated with an analysis

    Every analysis request is made with an analysis ID. The Boundary
    will reply with BoundaryExited if the analysis associated with the
    analysis ID has been cancelled.
    """

    def __init__(self, analysis_id, **kwargs):
        Request.__init__(self, **kwargs)
        self.analysis_id = analysis_id

    def __lt__(self, other):
        return True
