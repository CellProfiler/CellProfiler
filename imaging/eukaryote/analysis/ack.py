import imaging.utilities.zmqrequest as zmqrequest


class Ack(zmqrequest.Reply):
    def __init__(self, message="THANKS"):
        zmqrequest.Reply.__init__(self, message=message)
