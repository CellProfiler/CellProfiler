from ...utilities.zmq.communicable.request import AnalysisRequest


class DisplayPostGroup(AnalysisRequest):
    """Request a post-group display

    This is a message sent to the UI from the analysis worker"""

    def __init__(self, analysis_id, module_num, display_data, image_set_number):
        AnalysisRequest.__init__(
            self,
            analysis_id,
            module_num=module_num,
            image_set_number=image_set_number,
            display_data=display_data,
        )
