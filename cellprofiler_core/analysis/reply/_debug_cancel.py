import cellprofiler_core.utilities.zmq.communicable.reply._reply


class DebugCancel(cellprofiler_core.utilities.zmq.communicable.reply._reply.Reply):
    """If sent in response to DebugWaiting, the user has changed his/her mind"""
