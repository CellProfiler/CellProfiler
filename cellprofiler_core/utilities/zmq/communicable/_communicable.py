import sys

import cellprofiler_core.utilities.zmq


class Communicable:
    """Base class for Requests and Replies.

    All subclasses must accept keyword arguments to __init__() corresponding to
    their attributes.
    """

    def send(self, socket, routing=None):
        if routing is None:
            routing = []
        if hasattr(self, "_remote"):
            assert not self._remote, "send() called on a non-local Communicable object."
        json_str, buffers = cellprofiler_core.utilities.zmq.json_encode(self.__dict__)
        json_str = json_str.encode("utf-8")
        message_parts = (
            routing
            + [
                self.__class__.__module__.encode("utf-8"),
                self.__class__.__name__.encode("utf-8"),
            ]
            + [json_str]
        )
        socket.send_multipart(
            message_parts + buffers, copy=False,
        )

    class MultipleReply(RuntimeError):
        pass

    @classmethod
    def recv(cls, socket, routed=False):
        message = socket.recv_multipart()
        if routed:
            split = message.index(b"") + 1
            routing = message[:split]
            message = message[split:]
        else:
            routing = []
        module, classname = message[:2]
        module = module.decode("unicode_escape")
        classname = classname.decode("unicode_escape")
        buffers = message[3:]
        attribute_dict = cellprofiler_core.utilities.zmq.json_decode(
            message[2], buffers
        )
        try:
            instance = sys.modules[module].__dict__[classname](**attribute_dict)
        except:
            print(
                "Communicable could not instantiate %s from module %s with kwargs %s"
                % (module, classname, attribute_dict)
            )
            raise
        instance._remote = True
        instance._routing = routing
        instance._socket = socket
        instance._replied = False
        return instance

    def routing(self):
        return "/".join(self._routing)

    def reply(self, reply_obj, please_reply=False):
        assert self._remote, "Replying to a local Communicable!"
        if self._replied:
            raise self.MultipleReply("Can't reply to a Communicable more than once!")
        Communicable.send(reply_obj, self._socket, self._routing)
        self._replied = True
        if please_reply:
            raise NotImplementedError(
                "Req / rep / rep / rep pattern is no longer supported"
            )
