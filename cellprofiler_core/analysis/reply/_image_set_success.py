import cellprofiler_core.utilities.zmq.communicable.request._analysis_request


class ImageSetSuccess(
    cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest
):
    def __init__(self, analysis_id, image_set_number=None):
        cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest.__init__(
            self, analysis_id, image_set_number=image_set_number
        )
