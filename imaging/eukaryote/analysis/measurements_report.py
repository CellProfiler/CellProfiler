from __future__ import with_statement
from imaging.utilities.zmqrequest import AnalysisRequest


class MeasurementsReport(AnalysisRequest):
    def __init__(self, analysis_id, buf, image_set_numbers=[]):
        AnalysisRequest.__init__(self, analysis_id, buf=buf, image_set_numbers=image_set_numbers)
