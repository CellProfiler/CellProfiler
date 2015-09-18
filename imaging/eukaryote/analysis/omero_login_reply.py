from __future__ import with_statement
from imaging.utilities.zmqrequest import Reply


class OmeroLoginReply(Reply):
    def __init__(self, credentials):
        Reply.__init__(self, credentials=credentials)
