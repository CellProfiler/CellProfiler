"""knime_bridge.py - server support for knime bridge

The knime bridge supports a ZMQ protocol that lets a single client
run an analysis worker to get pipeline metadata and run a pipeline on
an image set.
"""
import json
import logging
import threading
import uuid
from io import StringIO

import numpy
import zmq
from cellprofiler_core.constants.measurement import (
    GROUP_NUMBER,
    GROUP_INDEX,
    OBJECT_NUMBER,
    EXPERIMENT,
    MCA_AVAILABLE_POST_RUN,
    COLTYPE_FLOAT,
    COLTYPE_INTEGER,
    COLTYPE_VARCHAR,
    IMAGE_NUMBER,
)
from cellprofiler_core.constants.workspace import DISPOSITION_SKIP, DISPOSITION_CANCEL
from cellprofiler_core.image import Image
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.setting.text import ImageName

if not hasattr(zmq, "Frame"):
    # Apparently, not in some versions of ZMQ?
    #
    def ZmqFrame(data=""):
        return data

    zmq.Frame = ZmqFrame

import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

CONNECT_REQ_1 = "connect-request-1"
CONNECT_REPLY_1 = "connect-reply-1"
PIPELINE_INFO_REQ_1 = "pipeline-info-req-1"
PIPELINE_INFO_REPLY_1 = "pipeline-info-reply-1"
RUN_REQ_1 = "run-request-1"
RUN_GROUP_REQ_1 = "run-group-request-1"
RUN_REPLY_1 = "run-reply-1"
CELLPROFILER_EXCEPTION_1 = "cellprofiler-exception-1"
PIPELINE_EXCEPTION_1 = "pipeline-exception-1"
CLEAN_PIPELINE_REQ_1 = "clean-pipeline-request-1"
CLEAN_PIPELINE_REPLY_1 = "clean-pipeline-reply-1"

LOGGER = logging.getLogger(__name__)


class KnimeBridgeServer(threading.Thread):
    """The server maintains the port and hands off the requests to workers

    example of use:

    context = zmq.Context()
    with KnimeBridgeServer(
        context,
        "tcp://127.0.0.1:4000", # server address
        "inproc://Notify        # kill notify address
        ) as server:
        notify_socket = context.socket(zmq.PUB)
        notify_socket.bind("inproc://Notify")
        ....
        notify_socket.send("Stop")
    """

    def __init__(self, context, address, notify_address, notify_stop, **kwargs):
        super(KnimeBridgeServer, self).__init__(**kwargs)
        self.setDaemon(True)
        self.setName("Knime bridge server")
        self.address = address
        self.context = context
        self.notify_addr = notify_address
        self.stop_msg = notify_stop
        self.dispatch = {
            CONNECT_REQ_1: self.connect,
            CLEAN_PIPELINE_REQ_1: self.clean_pipeline,
            PIPELINE_INFO_REQ_1: self.pipeline_info,
            RUN_REQ_1: self.run_request,
            RUN_GROUP_REQ_1: self.run_group_request,
        }
        self.start_addr = "inproc://" + uuid.uuid4().hex
        self.start_socket = context.socket(zmq.PAIR)
        self.start_socket.bind(self.start_addr)

    def __enter__(self):
        if self.address is not None:
            self.start()

    def __exit__(self, exc_type, value, tb):
        if self.address is not None:
            self.join()

    def start(self):
        super(KnimeBridgeServer, self).start()
        self.start_socket.recv()
        self.start_socket.close()

    def run(self):
        try:
            LOGGER.info("Binding Knime bridge server to %s" % self.address)
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(self.address)
            poller = zmq.Poller()
            poller.register(self.socket, flags=zmq.POLLIN)
            if self.notify_addr is not None:
                self.notify_socket = self.context.socket(zmq.SUB)
                self.notify_socket.setsockopt(zmq.SUBSCRIBE, "")
                self.notify_socket.connect(self.notify_addr)
                poller.register(self.notify_socket, flags=zmq.POLLIN)
            else:
                self.notify_socket = None
            start_socket = self.context.socket(zmq.PAIR)
            start_socket.connect(self.start_addr)
            start_socket.send("OK")
            start_socket.close()
            try:
                while True:
                    for socket, event in poller.poll(1):
                        if socket == self.notify_socket:
                            msg = self.notify_socket.recv_string()
                            if msg == self.stop_msg:
                                break
                        elif socket == self.socket:
                            msg = self.socket.recv_multipart(copy=False)
                            session_id_frame = msg.pop(0)
                            session_id = str(session_id_frame.bytes)
                            # should be None
                            wrapper = msg.pop(0)
                            message_type = msg.pop(0).bytes
                            if message_type not in self.dispatch:
                                self.raise_cellprofiler_exception(
                                    session_id,
                                    "Unhandled message type: %s" % message_type,
                                )
                            else:
                                try:
                                    self.dispatch[message_type](
                                        session_id, message_type, msg
                                    )
                                except Exception as e:
                                    LOGGER.warning(e)
                                    self.raise_cellprofiler_exception(session_id, e)
                    else:
                        continue
                    break
            finally:
                if self.notify_socket:
                    self.notify_socket.close()
                self.socket.close()
        except Exception as e:
            LOGGER.error("Could not bind Knime bridge server")

    def connect(self, session_id, message_type, message):
        """Handle the connect message"""
        self.socket.send_multipart(
            [zmq.Frame(session_id), zmq.Frame(), zmq.Frame(CONNECT_REPLY_1)]
        )

    def pipeline_info(self, session_id, message_type, message):
        """Handle the pipeline info message"""
        LOGGER.info("Handling pipeline info request")
        pipeline_txt = message.pop(0).bytes
        pipeline = cellprofiler_core.pipeline.Pipeline()
        try:
            pipeline.loadtxt(StringIO(pipeline_txt))
        except Exception as e:
            LOGGER.warning("Failed to load pipeline: sending pipeline exception")
            self.raise_pipeline_exception(session_id, str(e))
            return
        input_modules, other_modules = self.split_pipeline(pipeline)
        channels = self.find_channels(input_modules)
        type_names, measurements = self.find_measurements(other_modules, pipeline)
        body = json.dumps([channels, type_names, measurements])
        msg_out = [
            zmq.Frame(session_id),
            zmq.Frame(),
            zmq.Frame(PIPELINE_INFO_REPLY_1),
            zmq.Frame(body),
        ]
        self.socket.send_multipart(msg_out)

    def clean_pipeline(self, session_id, message_type, message):
        """Handle the clean pipeline request message"""
        LOGGER.info("Handling clean pipeline request")
        pipeline_txt = message.pop(0).bytes
        module_names = json.loads(message.pop(0).bytes)
        pipeline = cellprofiler_core.pipeline.Pipeline()
        try:
            pipeline.loadtxt(StringIO(pipeline_txt))
        except Exception as e:
            LOGGER.warning("Failed to load pipeline: sending pipeline exception")
            self.raise_pipeline_exception(session_id, str(e))
            return
        to_remove = []
        for module in pipeline.modules(exclude_disabled=False):
            if module.module_name in module_names:
                to_remove.insert(0, module)
        for module in to_remove:
            pipeline.remove_module(module.module_num)
        pipeline_fd = StringIO()
        pipeline.dump(pipeline_fd, save_image_plane_details=False)
        msg_out = [
            zmq.Frame(session_id),
            zmq.Frame(),
            zmq.Frame(CLEAN_PIPELINE_REPLY_1),
            zmq.Frame(pipeline_fd.getvalue()),
        ]
        self.socket.send_multipart(msg_out)

    def run_request(self, session_id, message_type, message):
        """Handle the run request message"""
        pipeline, m, object_set = self.prepare_run(message, session_id)
        if pipeline is None:
            return
        m["Image", GROUP_NUMBER,] = 1
        m["Image", GROUP_INDEX,] = 1
        input_modules, other_modules = self.split_pipeline(pipeline)
        for module in other_modules:
            workspace = cellprofiler_core.workspace.Workspace(
                pipeline, module, m, None, m, None
            )
            module.prepare_run(workspace)
        for module in other_modules:
            workspace = cellprofiler_core.workspace.Workspace(
                pipeline, module, m, object_set, m, None
            )
            try:
                LOGGER.info(
                    "Running module # %d: %s" % (module.module_num, module.module_name)
                )
                pipeline.run_module(module, workspace)
                if workspace.disposition in (DISPOSITION_SKIP, DISPOSITION_CANCEL,):
                    break
            except Exception as e:
                msg = 'Encountered error while running module, "%s": %s' % (
                    module.module_name,
                    e,
                )
                LOGGER.warning(msg)
                self.raise_cellprofiler_exception(session_id, msg)
                return
        type_names, feature_dict = self.find_measurements(other_modules, pipeline)

        double_features = []
        double_data = []
        float_features = []
        float_data = []
        int_features = []
        int_data = []
        string_features = []
        string_data = []
        metadata = [double_features, float_features, int_features, string_features]

        no_data = ()

        for object_name, features in list(feature_dict.items()):
            df = []
            double_features.append((object_name, df))
            ff = []
            float_features.append((object_name, ff))
            intf = []
            int_features.append((object_name, intf))
            sf = []
            string_features.append((object_name, sf))
            for feature, data_type in features:
                if not m.has_feature(object_name, feature):
                    data = no_data
                else:
                    data = numpy.atleast_1d(m[object_name, feature])
                if type_names[data_type] == "java.lang.Double":
                    df.append((feature, len(data)))
                    if len(data) > 0:
                        double_data.append(data.astype("<f8"))
                elif type_names[data_type] == "java.lang.Float":
                    ff.append((feature, len(data)))
                    if len(data) > 0:
                        float_data.append(data.astype("<f4"))
                elif type_names[data_type] == "java.lang.Integer":
                    intf.append((feature, len(data)))
                    if len(data) > 0:
                        int_data.append(data.astype("<i4"))
                elif type_names[data_type] == "java.lang.String":
                    if len(data) == 0:
                        sf.append((feature, 0))
                    else:
                        s = data[0]
                        if isinstance(s, str):
                            s = s
                        else:
                            s = str(s)
                        string_data.append(numpy.frombuffer(s, numpy.uint8))
        data = numpy.hstack(
            [
                numpy.frombuffer(numpy.hstack(ditem).data, numpy.uint8)
                for ditem in (double_data, float_data, int_data, string_data)
                if len(ditem) > 0
            ]
        )
        self.socket.send_multipart(
            [
                zmq.Frame(session_id),
                zmq.Frame(),
                zmq.Frame(RUN_REPLY_1),
                zmq.Frame(json.dumps(metadata)),
                zmq.Frame(bytes(data.data)),
            ]
        )

    def run_group_request(self, session_id, message_type, message):
        """Handle a run-group request message"""
        pipeline = cellprofiler_core.pipeline.Pipeline()
        m = Measurements()
        image_group = m.hdf5_dict.hdf5_file.create_group("ImageData")
        if len(message) < 2:
            self.raise_cellprofiler_exception(
                session_id, "Missing run request sections"
            )
            return
        pipeline_txt = message.pop(0).bytes
        image_metadata = message.pop(0).bytes
        n_image_sets = None
        try:
            image_metadata = json.loads(image_metadata)
            channel_names = []
            for channel_name, channel_metadata in image_metadata:
                channel_names.append(channel_name)
                if len(message) < 1:
                    self.raise_cellprofiler_exception(
                        session_id, "Missing binary data for channel %s" % channel_name
                    )
                    return None, None, None
                pixel_data = self.decode_image(
                    channel_metadata, message.pop(0).bytes, grouping_allowed=True
                )
                if pixel_data.ndim < 3:
                    self.raise_cellprofiler_exception(
                        session_id,
                        "The image for channel %s does not have a Z or T dimension",
                    )
                    return
                if n_image_sets is None:
                    n_image_sets = pixel_data.shape[0]
                elif n_image_sets != pixel_data.shape[0]:
                    self.raise_cellprofiler_exception(
                        session_id,
                        "The images passed have different numbers of Z or T planes",
                    )
                    return
                image_group.create_dataset(channel_name, data=pixel_data)
        except Exception as e:
            self.raise_cellprofiler_exception(session_id, e)
            return None, None, None
        try:
            pipeline.loadtxt(StringIO(pipeline_txt))
        except Exception as e:
            LOGGER.warning("Failed to load pipeline: sending pipeline exception")
            self.raise_pipeline_exception(session_id, str(e))
            return

        image_numbers = numpy.arange(1, n_image_sets + 1)
        for image_number in image_numbers:
            m["Image", GROUP_NUMBER, image_number,] = 1
            m["Image", GROUP_INDEX, image_number,] = image_number
        input_modules, other_modules = self.split_pipeline(pipeline)
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline, None, m, None, m, None
        )
        LOGGER.info("Preparing group")
        for module in other_modules:
            module.prepare_group(
                workspace,
                dict([("image_number", i) for i in image_numbers]),
                image_numbers,
            )

        for image_index in range(n_image_sets):
            object_set = cellprofiler_core.object.ObjectSet()
            m.next_image_set(image_index + 1)
            for channel_name in channel_names:
                dataset = image_group[channel_name]
                pixel_data = dataset[image_index]
                m.add(channel_name, Image(pixel_data))

            for module in other_modules:
                workspace = cellprofiler_core.workspace.Workspace(
                    pipeline, module, m, object_set, m, None
                )
                try:
                    LOGGER.info(
                        "Running module # %d: %s"
                        % (module.module_num, module.module_name)
                    )
                    pipeline.run_module(module, workspace)
                    if workspace.disposition in (DISPOSITION_SKIP, DISPOSITION_CANCEL,):
                        break
                except Exception as e:
                    msg = 'Encountered error while running module, "%s": %s' % (
                        module.module_name,
                        e,
                    )
                    LOGGER.warning(msg)
                    self.raise_cellprofiler_exception(session_id, msg)
                    return
            else:
                continue
            if workspace.disposition == DISPOSITION_CANCEL:
                break
        for module in other_modules:
            module.post_group(
                workspace, dict([("image_number", i) for i in image_numbers])
            )
        LOGGER.info("Finished group")

        type_names, feature_dict = self.find_measurements(other_modules, pipeline)

        double_features = []
        double_data = []
        float_features = []
        float_data = []
        int_features = []
        int_data = []
        string_features = []
        string_data = []
        metadata = [double_features, float_features, int_features, string_features]

        for object_name, features in list(feature_dict.items()):
            df = []
            double_features.append((object_name, df))
            ff = []
            float_features.append((object_name, ff))
            intf = []
            int_features.append((object_name, intf))
            sf = []
            string_features.append((object_name, sf))
            if object_name == "Image":
                object_counts = [] * n_image_sets
            else:
                object_numbers = m[
                    object_name, OBJECT_NUMBER, image_numbers,
                ]
                object_counts = [len(x) for x in object_numbers]
            for feature, data_type in features:
                if data_type == "java.lang.String":
                    continue
                if not m.has_feature(object_name, feature):
                    data = numpy.zeros(numpy.sum(object_counts))
                else:
                    data = m[object_name, feature, image_numbers]
                temp = []
                for i, (di, count) in enumerate(zip(data, object_counts)):
                    if count == 0:
                        continue
                    di = numpy.atleast_1d(di)
                    if len(di) > count:
                        di = di[:count]
                    elif len(di) == count:
                        temp.append(di)
                    else:
                        temp += [di + numpy.zeros(len(di) - count)]
                if len(temp) > 0:
                    data = numpy.hstack(temp)

                if type_names[data_type] == "java.lang.Double":
                    df.append((feature, len(data)))
                    if len(data) > 0:
                        double_data.append(data.astype("<f8"))
                elif type_names[data_type] == "java.lang.Float":
                    ff.append((feature, len(data)))
                    if len(data) > 0:
                        float_data.append(data.astype("<f4"))
                elif type_names[data_type] == "java.lang.Integer":
                    intf.append((feature, len(data)))
                    if len(data) > 0:
                        int_data.append(data.astype("<i4"))
        data = numpy.hstack(
            [
                numpy.frombuffer(
                    numpy.ascontiguousarray(numpy.hstack(ditem)).data, numpy.uint8
                )
                for ditem in (double_data, float_data, int_data)
                if len(ditem) > 0
            ]
        )
        data = numpy.ascontiguousarray(data)
        self.socket.send_multipart(
            [
                zmq.Frame(session_id),
                zmq.Frame(),
                zmq.Frame(RUN_REPLY_1),
                zmq.Frame(json.dumps(metadata)),
                zmq.Frame(data),
            ]
        )

    def prepare_run(self, message, session_id, grouping_allowed=False):
        """Prepare a pipeline and measurements to run

        message - the run-request or run-groups-request message
        session_id - the session ID for the session
        grouping_allowed - true to allow grouped images
        """
        pipeline = cellprofiler_core.pipeline.Pipeline()
        m = Measurements()
        object_set = cellprofiler_core.object.ObjectSet()
        if len(message) < 2:
            self.raise_cellprofiler_exception(
                session_id, "Missing run request sections"
            )
            return
        pipeline_txt = message.pop(0).bytes
        image_metadata = message.pop(0).bytes
        try:
            image_metadata = json.loads(image_metadata)
            for channel_name, channel_metadata in image_metadata:
                if len(message) < 1:
                    self.raise_cellprofiler_exception(
                        session_id, "Missing binary data for channel %s" % channel_name
                    )
                    return None, None, None
                pixel_data = self.decode_image(
                    channel_metadata,
                    message.pop(0).bytes,
                    grouping_allowed=grouping_allowed,
                )
                m.add(channel_name, cellprofiler_core.image.Image(pixel_data))
        except Exception as e:
            LOGGER.warning("Failed to decode message")
            self.raise_cellprofiler_exception(session_id, e)
            return None, None, None
        try:
            pipeline.loadtxt(StringIO(pipeline_txt))
        except Exception as e:
            LOGGER.warning("Failed to load pipeline: sending pipeline exception")
            self.raise_pipeline_exception(session_id, str(e))
            return None, None, None

        return pipeline, m, object_set

    def raise_pipeline_exception(self, session_id, message):
        if isinstance(message, str):
            message = message
        else:
            message = str(message)
        self.socket.send_multipart(
            [
                zmq.Frame(session_id),
                zmq.Frame(),
                zmq.Frame(PIPELINE_EXCEPTION_1),
                zmq.Frame(message),
            ]
        )

    def raise_cellprofiler_exception(self, session_id, message):
        if isinstance(message, str):
            message = message
        else:
            message = str(message)
        self.socket.send_multipart(
            [
                zmq.Frame(session_id),
                zmq.Frame(),
                zmq.Frame(CELLPROFILER_EXCEPTION_1),
                zmq.Frame(message),
            ]
        )

    def split_pipeline(self, pipeline):
        """Split the pipeline into input modules and everything else

        pipeline - the pipeline to be split

        returns a two-tuple of input modules and other
        """
        input_modules = []
        other_modules = []
        for module in pipeline.modules():
            if module.is_load_module() or module.is_input_module():
                input_modules.append(module)
            else:
                other_modules.append(module)
        return input_modules, other_modules

    def find_channels(self, input_modules):
        """Find image providers in the input modules"""
        channels = []
        for module in input_modules:
            for setting in module.visible_settings():
                if isinstance(setting, ImageName):
                    channels.append(setting.value)
        return channels

    def find_measurements(self, modules, pipeline):
        """Scan the modules for features

        modules - modules to scan for features
        pipeline - the pipeline they came from

        returns a two tuple of
            Java types, e.g., "java.lang.Integer"
            A dictionary whose key is the object name and whose
            value is a list of two-tuples of feature name and index into
            the java types array.
        """
        jtypes = ["java.lang.Integer"]
        features = {}
        for module in modules:
            assert isinstance(module, Module)
            for column in module.get_measurement_columns(pipeline):
                objects, name, dbtype = column[:3]
                qualifiers = {} if len(column) < 4 else column[3]
                if (
                    objects == EXPERIMENT
                    and qualifiers.get(MCA_AVAILABLE_POST_RUN, False) == True
                ):
                    continue
                if dbtype == COLTYPE_FLOAT:
                    jtype = "java.lang.Double"
                elif dbtype == COLTYPE_INTEGER:
                    jtype = "java.lang.Integer"
                elif dbtype.startswith(COLTYPE_VARCHAR):
                    jtype = "java.lang.String"
                else:
                    continue
                if jtype in jtypes:
                    type_idx = jtypes.index(jtype)
                else:
                    type_idx = len(jtypes)
                    jtypes.append(jtype)
                if objects not in features:
                    ofeatures = features[objects] = {}
                else:
                    ofeatures = features[objects]
                if name not in ofeatures:
                    ofeatures[name] = type_idx
        for key in features:
            features[key][IMAGE_NUMBER] = 0
        features_out = dict([(k, list(v.items())) for k, v in list(features.items())])
        return jtypes, features_out

    def decode_image(self, channel_metadata, buf, grouping_allowed=False):
        """Decode an image sent via the wire format

        channel_metadata: sequence of 3 tuples of axis name, dimension and stride
        buf: byte-buffer, low-endian representation of doubles
        grouping_allowed: true if we can accept images grouped by X or T

        returns numpy array in y, x indexing format.
        """
        pixel_data = numpy.frombuffer(buf, "<f8")
        strides_out = [None] * len(channel_metadata)
        dimensions_out = [None] * len(channel_metadata)
        grouping = False
        for axis_name, dim, stride in channel_metadata:
            if axis_name.lower() == "y":
                if strides_out[0] is not None:
                    raise RuntimeError("Y dimension doubly specified")
                strides_out[0] = stride * 8
                dimensions_out[0] = dim
            elif axis_name.lower() == "x":
                if strides_out[1] is not None:
                    raise RuntimeError("X dimension doubly specified")
                strides_out[1] = stride * 8
                dimensions_out[1] = dim
            elif axis_name.lower() == "channel":
                if strides_out[2] is not None:
                    raise RuntimeError("Channel doubly specified")
                strides_out[2] = stride * 8
                dimensions_out[2] = dim
            elif grouping == False and grouping_allowed:
                grouping = True
                strides_out[-1] = stride * 8
                dimensions_out[-1] = dim
            else:
                raise RuntimeError("Unknown dimension: " + axis_name)
        if grouping:
            strides_out[0], strides_out[1:] = strides_out[-1], strides_out[:-1]
            dimensions_out[0], dimensions_out[1:] = (
                dimensions_out[-1],
                dimensions_out[:-1],
            )
        pixel_data.shape = tuple(dimensions_out)
        pixel_data.strides = tuple(strides_out)
        return pixel_data


__all__ = ['KnimeBridgeServer']
#
# For testing only
#
__all__ += [
    CONNECT_REQ_1,
    CONNECT_REPLY_1,
    PIPELINE_INFO_REQ_1,
    PIPELINE_INFO_REPLY_1,
    RUN_REQ_1,
    RUN_GROUP_REQ_1,
    RUN_REPLY_1,
    PIPELINE_EXCEPTION_1,
    CELLPROFILER_EXCEPTION_1,
]
