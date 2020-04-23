import cellprofiler_core.utilities.zmq.communicable.reply._reply


class ExceptionPleaseDebug(
    cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply
):
    def __init__(self, disposition, verification_hash=None):
        cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply.__init__(
            self, disposition=disposition, verification_hash=verification_hash
        )
