from cellprofiler.utilities.zmqrequest import Request, Reply, BoundaryExited


class PipelineRequest(Request):
    pass


class InitialMeasurementsRequest(Request):
    pass


class WorkRequest(Request):
    pass


class MeasurementsReport(Request):
    def __init__(self, path="", image_set_numbers=""):
        Request.__init__(self, path=path, image_set_numbers=image_set_numbers)


class InteractionRequest(Request):
    pass


class DisplayRequest(Request):
    pass


class ExceptionReport(Request):
    pass


class DebugWaiting(Request):
    pass


class DebugComplete(Request):
    pass


class InteractionReply(Reply):
    pass


class WorkReply(Reply):
    pass


class ServerExited(BoundaryExited):
    pass
