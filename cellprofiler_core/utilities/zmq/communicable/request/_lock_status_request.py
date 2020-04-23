from cellprofiler_core.utilities.zmq.communicable.request._request import Request


class LockStatusRequest(Request):
    """A request for the status on some locked file

    uid - the unique ID stored inside the file's lock
    """

    def __init__(self, uid, **kwargs):
        self.uid = uid
        Request.__init__(self, **kwargs)
