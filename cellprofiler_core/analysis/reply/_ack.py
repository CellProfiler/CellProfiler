from cellprofiler_core.utilities.zmqrequest import Reply


class Ack(Reply):
    def __init__(self, message="THANKS"):
        Reply.__init__(self, message=message)
