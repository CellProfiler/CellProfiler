from __future__ import with_statement
from imaging.utilities.zmqrequest import AnalysisRequest


class ImageSetSuccess(AnalysisRequest):
    def __init__(self, analysis_id, image_set_number=None):
        AnalysisRequest.__init__(self, analysis_id, image_set_number=image_set_number)
