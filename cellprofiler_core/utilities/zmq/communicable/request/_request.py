from cellprofiler_core.utilities.zmq.communicable._communicable import Communicable
from cellprofiler_core.utilities.zmq.communicable.reply._reply import Reply


class Request(Communicable):
    """A class for making requests and receiving replies across a zmq REQ
    socket.  When communicated through a socket, the class will self-serialize
    any non-callable attributes that do not start with an underscore.

    When received, reply() can be used to send a Reply object to the original
    Request.

    Note that both sides of the connection must have imported the same module
    under the same name defining any objects that will be communicated.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    """

    def __init__(self, **kwargs):
        # all keywords become attributes
        self.__dict__.update(kwargs)
        self._boundary = None

    def send(self, socket):
        Communicable.send(self, socket)
        return Communicable.recv(socket)

    def send_only(self, socket):
        """Send the request but don't perform the .recv

        socket - send on this socket

        First part of a two-part client-side request: send the request
        with an expected .recv, possibly after polling to make the .recv
        non-blocking.
        """
        Communicable.send(self, socket)

    def set_boundary(self, boundary):
        """Set the boundary object to use when sending the reply

        boundary - the reply will be enqueued on this boundary's transmit thread
        """
        self._boundary = boundary

    def reply(self, reply_obj, please_reply=False):
        """send a reply to a request.  If please_reply is True, wait for and
        return a reply to the reply.  Note that that reply should be treated
        like a Request object, i.e., it should be replied to."""
        assert isinstance(
            reply_obj, Reply
        ), "send_reply() called with something other than a Reply object!"
        if self._boundary is None:
            return Communicable.reply(self, reply_obj, please_reply)
        else:
            self._boundary.enqueue_reply(self, reply_obj)
