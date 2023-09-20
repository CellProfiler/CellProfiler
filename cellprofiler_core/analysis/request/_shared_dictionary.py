from ...utilities.zmq.communicable.request import AnalysisRequest


class SharedDictionary(AnalysisRequest):
    def __init__(self, analysis_id, module_num=-1):
        AnalysisRequest.__init__(self, analysis_id, module_num=module_num)
