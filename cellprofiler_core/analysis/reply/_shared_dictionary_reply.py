from cellprofiler.utilities.zmqrequest import Reply


class SharedDictionaryReply(Reply):
    def __init__(self, dictionaries=None):
        Reply.__init__(self, dictionaries=dictionaries)
        if dictionaries is None:
            dictionaries = [{}]
