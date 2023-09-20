from ...utilities.zmq.communicable.request import AnalysisRequest


class DebugWaiting(AnalysisRequest):
    """Communicate the debug port to the server and wait for server OK to attach"""

    def __init__(self, analysis_id, port):
        AnalysisRequest.__init__(self, analysis_id=analysis_id, port=port)
