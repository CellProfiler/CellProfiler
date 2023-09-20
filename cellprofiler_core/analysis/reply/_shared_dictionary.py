from ...utilities.zmq.communicable.reply import Reply


class SharedDictionary(Reply):
    def __init__(self, dictionaries=None):
        Reply.__init__(self, dictionaries=dictionaries)

        if dictionaries is None:
            dictionaries = [{}]
