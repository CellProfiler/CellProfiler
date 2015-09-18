from __future__ import with_statement
from imaging.utilities.zmqrequest import Reply


class SharedDictionaryReply(Reply):
    def __init__(self, dictionaries=[{}]):
        Reply.__init__(self, dictionaries=dictionaries)
