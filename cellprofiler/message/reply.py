import cellprofiler.utilities.zmqrequest


class DictionaryReqRep(cellprofiler.utilities.zmqrequest.Reply):
    pass


class SharedDictionaryReply(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, dictionaries=None):
        if dictionaries is None:
            dictionaries = [{}]

        cellprofiler.utilities.zmqrequest.Reply.__init__(self, dictionaries=dictionaries)


class InteractionReply(cellprofiler.utilities.zmqrequest.Reply):
    pass


class WorkReply(cellprofiler.utilities.zmqrequest.Reply):
    pass


class NoWorkReply(cellprofiler.utilities.zmqrequest.Reply):
    pass


class OmeroLoginReply(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, credentials):
        cellprofiler.utilities.zmqrequest.Reply.__init__(self, credentials=credentials)


class Ack(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, message="THANKS"):
        cellprofiler.utilities.zmqrequest.Reply.__init__(self, message=message)


class ExceptionPleaseDebugReply(cellprofiler.utilities.zmqrequest.Reply):
    def __init__(self, disposition, verification_hash=None):
        cellprofiler.utilities.zmqrequest.Reply.__init__(self, disposition=disposition, verification_hash=verification_hash)


class DebugCancel(cellprofiler.utilities.zmqrequest.Reply):
    '''If sent in response to DebugWaiting, the user has changed his/her mind'''
    pass


class ServerExited(cellprofiler.utilities.zmqrequest.UpstreamExit):
    pass
