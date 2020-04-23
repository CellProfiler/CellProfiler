import cellprofiler_core.utilities.zmqrequest


class DebugCancel(cellprofiler_core.utilities.zmqrequest.Reply):
    """If sent in response to DebugWaiting, the user has changed his/her mind"""
