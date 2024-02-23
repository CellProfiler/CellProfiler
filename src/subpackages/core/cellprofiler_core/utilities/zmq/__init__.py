import errno
import json
import logging
import os
import queue
import sys
import threading
import uuid

import numpy
import zmq

import cellprofiler_core.utilities.grid
from cellprofiler_core.utilities.zmq._boundary import Boundary
from cellprofiler_core.utilities.zmq.communicable.reply import LockStatusReply, Reply
from cellprofiler_core.utilities.zmq.communicable.request import (
    LockStatusRequest,
    Request,
)
from ._event import PollTimeoutException


LOGGER = logging.getLogger(__name__)

NOTIFY_SOCKET_ADDR = "inproc://BoundaryNotifications"
SD_KEY_DICT = "__keydict__"


class CPJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            else:
                return item

        return super().encode(hint_tuples(obj))

def make_CP_fallback_encoder(buffers):
    """create an encoder for CellProfiler data and numpy arrays (which will be
    stored in the input argument)"""

    def encoder(data, buffers=buffers):
        if isinstance(data, numpy.ndarray):
            #
            # Maybe it's nice to save memory by converting 64-bit to 32-bit
            # but the purpose here is to fix a bug on the Mac where a
            # 32-bit worker gets a 64-bit array or unsigned 32-bit array,
            # tries to use it for indexing and fails because the integer
            # is wider than a 32-bit pointer
            #
            info32 = numpy.iinfo(numpy.int32)
            if (
                data.dtype.kind == "i"
                and data.dtype.itemsize > 4
                or data.dtype.kind == "u"
                and data.dtype.itemsize >= 4
            ):
                if numpy.prod(data.shape) == 0 or (
                    numpy.min(data) >= info32.min and numpy.max(data) <= info32.max
                ):
                    data = data.astype(numpy.int32)
            idx = len(buffers)
            buffers.append(numpy.ascontiguousarray(data))
            dtype = str(data.dtype)
            return {
                "__ndarray__": True,
                "dtype": str(data.dtype),
                "shape": data.shape,
                "idx": idx,
            }
        if isinstance(data, numpy.generic):
            # http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
            return data.astype(object)
        if isinstance(data, cellprofiler_core.utilities.grid.Grid):
            d = data.serialize()
            d["__CPGridInfo__"] = True
            return d
        if isinstance(data, memoryview):
            # arbitrary data
            idx = len(buffers)
            buffers.append(data)
            return {"__buffer__": True, "idx": idx}
        raise TypeError("%r of type %r is not JSON serializable" % (data, type(data)))

    return encoder


def make_CP_decoder(buffers):
    def decoder(dct, buffers=buffers):
        if '__tuple__' in dct:
            return tuple(dct['items'])
        if "__ndarray__" in dct:
            buf = memoryview(buffers[dct["idx"]])
            shape = dct["shape"]
            dtype = dct["dtype"]
            if numpy.prod(shape) == 0:
                return numpy.zeros(shape, dtype)
            return numpy.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        if "__buffer__" in dct:
            return memoryview(buffers[dct["idx"]])
        if "__CPGridInfo__" in dct:
            grid = cellprofiler_core.utilities.grid.Grid()
            grid.deserialize(dct)
            return grid
        return dct

    return decoder


def make_sendable_dictionary(d):
    """Make a dictionary that passes muster with JSON"""
    result = {}
    fake_key_idx = 1
    for k, v in list(d.items()):
        if (isinstance(k, str) and k.startswith("_")) or callable(d[k]):
            continue
        if isinstance(v, dict):
            v = make_sendable_dictionary(v)
        elif isinstance(v, (list, tuple)):
            v = make_sendable_sequence(v)
        if not isinstance(k, str):
            if SD_KEY_DICT not in result:
                result[SD_KEY_DICT] = {}
            fake_key = "__%d__" % fake_key_idx
            fake_key_idx += 1
            result[SD_KEY_DICT][fake_key] = k
            result[fake_key] = v
        else:
            result[k] = v
    return result


def make_sendable_sequence(l):
    """Make a list that passes muster with JSON"""
    result = []
    for v in l:
        if isinstance(v, (list, tuple)):
            result.append(make_sendable_sequence(v))
        elif isinstance(v, dict):
            result.append(make_sendable_dictionary(v))
        else:
            result.append(v)
    if isinstance(l, tuple):
        return tuple(result)
    else: # else return as list
        return result


def decode_sendable_dictionary(d):
    """Decode the dictionary encoded by make_sendable_dictionary"""
    result = {}
    for k, v in list(d.items()):
        if k == SD_KEY_DICT:
            continue
        if isinstance(v, dict):
            v = decode_sendable_dictionary(v)
        elif isinstance(v, list):
            v = decode_sendable_sequence(v, list)
        if k.startswith("__") and k.endswith("__"):
            k = d[SD_KEY_DICT][k]
            if isinstance(k, list):
                k = decode_sendable_sequence(k, tuple)
        result[k] = v
    return result


def decode_sendable_sequence(l, desired_type):
    """Decode a tuple encoded by make_sendable_sequence"""
    result = []
    for v in l:
        if isinstance(v, dict):
            result.append(decode_sendable_dictionary(v))
        elif isinstance(v, (list, tuple)):
            result.append(decode_sendable_sequence(v, desired_type))
        else:
            result.append(v)
    return result if isinstance(result, desired_type) else desired_type(result)


def json_encode(o):
    """Encode an object as a JSON string

    o - object to encode

    returns a 2-tuple of json-encoded object + buffers of binary stuff
    """
    sendable_dict = make_sendable_dictionary(o)

    # replace each buffer with its metadata, and send it separately
    buffers = []
    fallback_encoder = make_CP_fallback_encoder(buffers)
    json_str = json.dumps(sendable_dict, cls=CPJSONEncoder, default=fallback_encoder)
    return json_str, buffers


def json_decode(json_str, buffers):
    """Decode a JSON-encoded string

    json_str - the JSON string

    buffers - buffers of binary data to feed into the decoder of special cases

    return the decoded dictionary
    """
    decoder = make_CP_decoder(buffers)
    attribute_dict = json.loads(json_str, object_hook=decoder)
    return decode_sendable_dictionary(attribute_dict)


the_boundary = None


def start_boundary():
    global the_boundary
    if the_boundary is None:
        the_boundary = Boundary("tcp://127.0.0.1")
    return the_boundary


def get_announcer_address():
    return start_boundary().announce_address


def register_analysis(analysis_id, upward_queue):
    """Register for all analysis request messages with the given ID

    analysis_id - the analysis ID present in every AnalysisRequest

    upward_queue - requests are placed on this queue

    upward_cv - the condition variable used to signal the queue's thread

    returns the boundary singleton.
    """
    global the_boundary
    start_boundary()
    the_boundary.register_analysis(analysis_id, upward_queue)
    return the_boundary


def join_to_the_boundary():
    """Send a stop signal to the boundary thread and join to it"""
    global the_boundary
    if the_boundary is not None:
        the_boundary.join()
        the_boundary = None


__lock_queue: queue.Queue = queue.Queue()
__lock_thread = None
LOCK_REQUEST = "Lock request"
UNLOCK_REQUEST = "Unlock request"
UNLOCK_OK = "OK"


def start_lock_thread():
    """Start the thread that handles file locking"""
    global __lock_thread
    if __lock_thread is not None:
        return
    the_boundary.register_request_class(LockStatusRequest, __lock_queue)

    def lock_thread_fn():
        global __lock_thread
        locked_uids = {}
        locked_files = {}
        while True:
            msg = __lock_queue.get()
            boundary = msg[0]
            if msg[1] == Boundary.NOTIFY_STOP:
                msg[2].put(__lock_thread)
                break
            elif msg[1] == Boundary.NOTIFY_REQUEST:
                request = msg[2]
                assert isinstance(request, LockStatusRequest)
                assert isinstance(boundary, Boundary)
                LOGGER.info("Received lock status request for %s" % request.uid)
                reply = LockStatusReply(request.uid, request.uid in locked_uids)
                if reply.locked:
                    LOGGER.info(
                        "Denied lock request for %s" % locked_uids[request.uid]
                    )
                boundary.enqueue_reply(request, reply)
            elif msg[1] == LOCK_REQUEST:
                uid, path = msg[2]
                locked_uids[uid] = path
                locked_files[path] = uid
                msg[3].put("OK")
            elif msg[1] == UNLOCK_REQUEST:
                try:
                    uid = locked_files[msg[2]]
                    del locked_uids[uid]
                    del locked_files[msg[2]]
                    msg[3].put("OK")
                except Exception as e:
                    msg[3].put(e)
        __lock_thread = None
        LOGGER.info("Exiting the lock thread")

    __lock_thread = threading.Thread(target=lock_thread_fn, name="FileLock thread")
    __lock_thread.start()


def get_lock_path(path):
    """Return the path to the lockfile"""
    pathpart, filepart = os.path.split(path)
    return os.path.join(pathpart, "." + filepart + ".lock")


def lock_file(path, timeout=3):
    """Lock a file

    path - path to the file

    timeout - timeout in seconds when waiting for announcement

    returns True if we obtained the lock, False if the file is already owned.
    """
    lock_path = get_lock_path(path)
    start_boundary()
    uid = uuid.uuid4().hex
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        with os.fdopen(fd, "a") as f:
            f.write(the_boundary.external_request_address + "\n" + uid)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        LOGGER.info("Lockfile for %s already exists - contacting owner" % path)
        with open(lock_path, "r") as f:
            remote_address = f.readline().strip()
            remote_uid = f.readline().strip()
        if len(remote_address) > 0 and len(remote_uid) > 0:
            LOGGER.info("Owner is %s" % remote_address)
            request_socket = the_boundary.zmq_context.socket(zmq.REQ)
            request_socket.setsockopt(zmq.LINGER, 0)
            assert isinstance(request_socket, zmq.Socket)
            request_socket.connect(remote_address)

            lock_request = LockStatusRequest(remote_uid)
            lock_request.send_only(request_socket)
            poller = zmq.Poller()
            poller.register(request_socket, zmq.POLLIN)
            keep_polling = True
            while keep_polling:
                keep_polling = False
                for socket, status in poller.poll(timeout * 1000):
                    keep_polling = True
                    if socket == request_socket and status == zmq.POLLIN:
                        lock_response = lock_request.recv(socket)
                        if isinstance(lock_response, LockStatusReply):
                            if lock_response.locked:
                                LOGGER.info("%s is locked" % path)
                                return False
                            keep_polling = False
        #
        # Fall through if we believe that the other end is dead
        #
        with open(lock_path, "w") as f:
            f.write(the_boundary.request_address + "\n" + uid)
    #
    # The coast is clear to lock
    #
    q = queue.Queue()
    start_lock_thread()
    __lock_queue.put((None, LOCK_REQUEST, (uid, path), q))
    q.get()
    return True


def unlock_file(path):
    """Unlock the file at the given path"""
    if the_boundary is None:
        return
    q = queue.Queue()
    start_lock_thread()
    __lock_queue.put((None, UNLOCK_REQUEST, path, q))
    result = q.get()
    if result != UNLOCK_OK:
        raise result
    lock_path = get_lock_path(path)
    os.remove(lock_path)


context = zmq.Context()


def subproc():
    address = sys.argv[sys.argv.index("subproc") + 1]
    mysock = context.socket(zmq.REQ)
    mysock.connect(address)
    req = Request(
        this="is", a="test", b=5, c=1.3, d=numpy.arange(10), e=[{"q": numpy.arange(5)}],
    )
    rep = req.send(mysock)
    print("subproc received", rep, rep.__dict__)
    rep = rep.reply(Reply(msg="FOO"), please_reply=True)
    print("subproc received", rep, rep.__dict__)
