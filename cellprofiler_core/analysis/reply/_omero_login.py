import cellprofiler_core.utilities.zmq.communicable.reply._reply


class OmeroLogin(cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply):
    def __init__(self, credentials):
        cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply.__init__(
            self, credentials=credentials
        )
