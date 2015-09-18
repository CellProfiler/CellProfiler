from __future__ import with_statement
from imaging.utilities.zmqrequest import AnalysisRequest, Request, Reply, UpstreamExit


class DebugWaiting(AnalysisRequest):
    def __init__(self, analysis_id, port):
        AnalysisRequest.__init__(self, analysis_id=analysis_id, port=port)
