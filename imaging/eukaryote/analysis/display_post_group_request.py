from __future__ import with_statement
from imaging.utilities.zmqrequest import AnalysisRequest


class DisplayPostGroupRequest(AnalysisRequest):
    def __init__(self, analysis_id, module_num, display_data, image_set_number):
        AnalysisRequest.__init__(self, analysis_id, module_num=module_num, image_set_number=image_set_number, display_data=display_data)
