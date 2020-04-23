from cellprofiler_core.utilities.zmq.communicable.reply._reply import Reply


class LockStatusReply(Reply):
    """A reply to the LockStatusRequest

    self.uid - the unique ID of the locked file
    self.locked - true if locked, false if not
    """

    def __init__(self, uid, locked, **kwargs):
        Reply.__init__(self, **kwargs)
        self.uid = uid
        self.locked = locked
