import cellprofiler_core.utilities.zmqrequest


class SharedDictionary(cellprofiler_core.utilities.zmqrequest.Reply):
    def __init__(self, dictionaries=None):
        cellprofiler_core.utilities.zmqrequest.Reply.__init__(
            self, dictionaries=dictionaries
        )
        if dictionaries is None:
            dictionaries = [{}]
