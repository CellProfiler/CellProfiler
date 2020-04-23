from cellprofiler_core.utilities.zmqrequest import Reply


class ExceptionPleaseDebug(Reply):
    def __init__(self, disposition, verification_hash=None):
        Reply.__init__(
            self, disposition=disposition, verification_hash=verification_hash
        )
