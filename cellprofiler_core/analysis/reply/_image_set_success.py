from ...utilities.zmq.communicable.request import AnalysisRequest


class ImageSetSuccess(AnalysisRequest):
    def __init__(self, analysis_id, image_set_number=None):
        AnalysisRequest.__init__(self, analysis_id, image_set_number=image_set_number)
