import cellprofiler.utilities.zmqrequest


class Ack(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, message="THANKS"):
        cellprofiler.utilities.zmqrequest.Reply.__init__(self, message=message)


class DebugCancel(cellprofiler.utilities.zmqrequest.Reply):
    """
    If sent in response to DebugWaiting, the user has changed his/her mind
    """
    pass


class DictionaryRequest(cellprofiler.utilities.zmqrequest.Reply):
    pass


class ExceptionPleaseDebug(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, disposition, verification_hash=None):
        cellprofiler.utilities.zmqrequest.Reply.__init__(self, disposition=disposition, verification_hash=verification_hash)


class Interaction(cellprofiler.utilities.zmqrequest.Reply):
    pass


class NoWork(cellprofiler.utilities.zmqrequest.Reply):
    pass


class OMEROLogin(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, credentials):
        cellprofiler.utilities.zmqrequest.Reply.__init__(self, credentials=credentials)


class ServerExited(cellprofiler.utilities.zmqrequest.UpstreamExit):
    pass


class SharedDictionary(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, dictionaries=None):
        if dictionaries is None:
            dictionaries = [{}]

        cellprofiler.utilities.zmqrequest.Reply.__init__(self, dictionaries=dictionaries)


class Work(cellprofiler.utilities.zmqrequest.Reply):
    pass
