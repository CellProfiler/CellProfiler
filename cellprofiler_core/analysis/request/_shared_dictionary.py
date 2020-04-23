import cellprofiler_core.utilities.zmqrequest


class SharedDictionary(cellprofiler_core.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, module_num=-1):
        cellprofiler_core.utilities.zmqrequest.AnalysisRequest.__init__(
            self, analysis_id, module_num=module_num
        )
