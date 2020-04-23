import cellprofiler_core.utilities.zmqrequest


class OmeroLogin(cellprofiler_core.utilities.zmqrequest.Reply):
    def __init__(self, credentials):
        cellprofiler_core.utilities.zmqrequest.Reply.__init__(
            self, credentials=credentials
        )
