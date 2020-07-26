from ...utilities.zmq.communicable.reply import Reply


class Ack(Reply):
    def __init__(self, message="THANKS"):
        Reply.__init__(self, message=message)
