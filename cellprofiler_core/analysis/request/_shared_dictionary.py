import cellprofiler_core.utilities.zmq.communicable.request._analysis_request


class SharedDictionary(
    cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest
):
    def __init__(self, analysis_id, module_num=-1):
        cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest.__init__(
            self, analysis_id, module_num=module_num
        )
