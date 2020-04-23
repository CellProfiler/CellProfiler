from cellprofiler.utilities.zmqrequest import Reply


class OmeroLogin(Reply):
    def __init__(self, credentials):
        Reply.__init__(self, credentials=credentials)
