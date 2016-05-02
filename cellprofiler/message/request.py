import cellprofiler.utilities.zmqrequest


class PipelinePreferencesRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class InitialMeasurementsRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class WorkRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class ImageSetSuccess(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, image_set_number=None):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, image_set_number=image_set_number)


class ImageSetSuccessWithDictionary(ImageSetSuccess):
    def __init__(self, analysis_id, image_set_number, shared_dicts):
        ImageSetSuccess.__init__(self, analysis_id,
                                 image_set_number=image_set_number)
        self.shared_dicts = shared_dicts


class MeasurementsReport(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, buf, image_set_numbers=[]):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, buf=buf, image_set_numbers=image_set_numbers)


class InteractionRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class AnalysisCancelRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class DisplayRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class DisplayPostRunRequest(object):
    '''Request a post-run display

    This is a message sent to the UI from the analysis worker'''

    def __init__(self, module_num, display_data):
        self.module_num = module_num
        self.display_data = display_data


class DisplayPostGroupRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    '''Request a post-group display

    This is a message sent to the UI from the analysis worker'''

    def __init__(self, analysis_id, module_num, display_data, image_set_number):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(
                self, analysis_id,
                module_num=module_num,
                image_set_number=image_set_number,
                display_data=display_data)


class SharedDictionaryRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id, module_num=-1):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self, analysis_id, module_num=module_num)


class ExceptionReport(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    def __init__(self, analysis_id,
                 image_set_number, module_name,
                 exc_type, exc_message, exc_traceback,
                 filename, line_number):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self,
                                                                   analysis_id,
                                                                   image_set_number=image_set_number,
                                                                   module_name=module_name,
                                                                   exc_type=exc_type,
                                                                   exc_message=exc_message,
                                                                   exc_traceback=exc_traceback,
                                                                   filename=filename,
                                                                   line_number=line_number)

    def __str__(self):
        return "(Worker) %s: %s" % (self.exc_type, self.exc_message)


class DebugWaiting(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    '''Communicate the debug port to the server and wait for server OK to attach'''

    def __init__(self, analysis_id, port):
        cellprofiler.utilities.zmqrequest.AnalysisRequest.__init__(self,
                                                                   analysis_id=analysis_id,
                                                                   port=port)


class DebugComplete(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass


class OmeroLoginRequest(cellprofiler.utilities.zmqrequest.AnalysisRequest):
    pass