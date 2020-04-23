from cellprofiler_core.utilities.zmq.communicable._communicable import Communicable


class Reply(Communicable):
    """The counterpart to a Request.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    """

    def __init__(self, **kwargs):
        # all keywords become attributes
        self.__dict__.update(kwargs)
