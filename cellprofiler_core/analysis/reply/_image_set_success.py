import cellprofiler_core.utilities.zmqrequest


class ImageSetSuccess(cellprofiler_core.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, image_set_number=None):
        cellprofiler_core.utilities.zmqrequest.AnalysisRequest.__init__(
            self, analysis_id, image_set_number=image_set_number
        )
