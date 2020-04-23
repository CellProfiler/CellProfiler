import cellprofiler_core.utilities.zmq.communicable.request._analysis_request


class MeasurementsReport(
    cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest
):
    def __init__(self, analysis_id, buf, image_set_numbers=None):
        cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest.__init__(
            self, analysis_id, buf=buf, image_set_numbers=image_set_numbers
        )
        if image_set_numbers is None:
            image_set_numbers = []
