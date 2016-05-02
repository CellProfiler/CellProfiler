import cellprofiler.utilities.zmqrequest


class AnalysisCancel(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class DebugComplete(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class DebugWaiting(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    """
    Communicate the debug port to the server and wait for server OK to attach
    """

    def __init__(self, analysis_id, port):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id=analysis_id, port=port)


class Display(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class DisplayPostGroup(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    """
    Request a post-group display

    This is a message sent to the UI from the analysis worker
    """
    def __init__(self, analysis_id, module_num, display_data, image_set_number):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, module_num=module_num, image_set_number=image_set_number, display_data=display_data)


class DisplayPostRun(object):
    """
    Request a post-run display

    This is a message sent to the UI from the analysis worker
    """
    def __init__(self, module_num, display_data):
        self.module_num = module_num
        self.display_data = display_data


class ExceptionReport(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, image_set_number, module_name, exc_type, exc_message, exc_traceback, filename, line_number):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, image_set_number=image_set_number, module_name=module_name, exc_type=exc_type, exc_message=exc_message, exc_traceback=exc_traceback, filename=filename, line_number=line_number)

    def __str__(self):
        return "(Worker) %s: %s" % (self.exc_type, self.exc_message)


class ImageSetSuccess(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, image_set_number=None):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, image_set_number=image_set_number)


class ImageSetSuccessWithDictionary(ImageSetSuccess):
    def __init__(self, analysis_id, image_set_number, shared_dicts):
        ImageSetSuccess.__init__(self, analysis_id, image_set_number=image_set_number)

        self.shared_dicts = shared_dicts


class InitialMeasurements(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class Interaction(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class MeasurementsReport(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, buf, image_set_numbers=None):
        if image_set_numbers is None:
            image_set_numbers = []

        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, buf=buf, image_set_numbers=image_set_numbers)

class OMEROLogin(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class PipelinePreferences(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class SharedDictionary(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, module_num=-1):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, module_num=module_num)


class Work(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass
