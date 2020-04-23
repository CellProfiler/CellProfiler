import cellprofiler_core.utilities.zmq.communicable.reply._reply


class SharedDictionary(cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply):
    def __init__(self, dictionaries=None):
        cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply.__init__(
            self, dictionaries=dictionaries
        )
        if dictionaries is None:
            dictionaries = [{}]
