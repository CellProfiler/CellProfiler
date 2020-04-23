import cellprofiler_core.utilities.zmqrequest


class MeasurementsReport(cellprofiler_core.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, buf, image_set_numbers=None):
        cellprofiler_core.utilities.zmqrequest.AnalysisRequest.__init__(
            self, analysis_id, buf=buf, image_set_numbers=image_set_numbers
        )
        if image_set_numbers is None:
            image_set_numbers = []
