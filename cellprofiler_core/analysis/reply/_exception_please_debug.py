import cellprofiler_core.utilities.zmqrequest


class ExceptionPleaseDebug(cellprofiler_core.utilities.zmqrequest.Reply):
    def __init__(self, disposition, verification_hash=None):
        cellprofiler_core.utilities.zmqrequest.Reply.__init__(
            self, disposition=disposition, verification_hash=verification_hash
        )
