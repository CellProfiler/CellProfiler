import cellprofiler_core.utilities.zmq.communicable.request._analysis_request


class DebugWaiting(
    cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest
):
    """Communicate the debug port to the server and wait for server OK to attach"""

    def __init__(self, analysis_id, port):
        cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest.__init__(
            self, analysis_id=analysis_id, port=port
        )
