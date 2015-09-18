from __future__ import with_statement
from imaging.utilities.zmqrequest import Reply


class ExceptionPleaseDebugReply(Reply):
    def __init__(self, disposition, verification_hash=None):
        Reply.__init__(self, disposition=disposition, verification_hash=verification_hash)
