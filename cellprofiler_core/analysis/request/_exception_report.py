import cellprofiler_core.utilities.zmq.communicable.request._analysis_request


class ExceptionReport(
    cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest
):
    def __init__(
        self,
        analysis_id,
        image_set_number,
        module_name,
        exc_type,
        exc_message,
        exc_traceback,
        filename,
        line_number,
    ):
        cellprofiler_core.utilities.zmq.communicable.request._analysis_request.AnalysisRequest.__init__(
            self,
            analysis_id,
            image_set_number=image_set_number,
            module_name=module_name,
            exc_type=exc_type,
            exc_message=exc_message,
            exc_traceback=exc_traceback,
            filename=filename,
            line_number=line_number,
        )

    def __str__(self):
        return "(Worker) %s: %s" % (self.exc_type, self.exc_message)
