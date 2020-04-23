import cellprofiler_core.utilities.zmqrequest


class Ack(cellprofiler_core.utilities.zmqrequest.Reply):
    def __init__(self, message="THANKS"):
        cellprofiler_core.utilities.zmqrequest.Reply.__init__(self, message=message)
