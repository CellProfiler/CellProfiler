from ...utilities.zmq.communicable.request import AnalysisRequest


class MeasurementsReport(AnalysisRequest):
    def __init__(self, analysis_id, buf, image_set_numbers=None, debug_i_am_report=None):
        self._debug_i_am_report = debug_i_am_report
        AnalysisRequest.__init__(
            self, analysis_id, buf=buf, image_set_numbers=image_set_numbers
        )
        if image_set_numbers is None:
            image_set_numbers = []
