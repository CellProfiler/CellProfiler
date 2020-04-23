import cellprofiler_core.utilities.zmq.communicable.reply._reply
import cellprofiler_core.utilities.zmq


class Ack(cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply):
    def __init__(self, message="THANKS"):
        cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply.__init__(
            self, message=message
        )
