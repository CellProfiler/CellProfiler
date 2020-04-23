from cellprofiler.utilities.zmqrequest import Reply


class DebugCancel(Reply):
    """If sent in response to DebugWaiting, the user has changed his/her mind"""
