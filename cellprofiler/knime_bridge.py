"""knime_bridge.py - server support for knime bridge

The knime bridge supports a ZMQ protocol that lets a single client
run an analysis worker to get pipeline metadata and run a pipeline on
an image set.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import logging
logger = logging.getLogger(__name__)

from cStringIO import StringIO
import json
import javabridge
import numpy as np
import threading
import zmq

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw

CONNECT_REQ_1 = "connect-request-1"
CONNECT_REPLY_1 = "connect-reply-1"
PIPELINE_INFO_REQ_1 = "pipeline-info-req-1"
PIPELINE_INFO_REPLY_1 = "pipeline-info-reply-1"
RUN_REQ_1 = "run-request-1"
RUN_REPLY_1 = "run-reply-1"
CELLPROFILER_EXCEPTION_1 = "cellprofiler-exception-1"
PIPELINE_EXCEPTION_1 = "pipeline-exception-1"

class KnimeBridgeServer(threading.Thread):
    '''The server maintains the port and hands off the requests to workers
    
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
    '''
    
    def __init__(self, context, address, notify_address, **kwargs):
        super(KnimeBridgeServer, self).__init__(**kwargs)
        self.setDaemon(True)
        self.setName("Knime bridge server")
        self.address = address
        self.context = context
        self.notify_addr = notify_address
        self.dispatch = {
            CONNECT_REQ_1: self.connect,
            PIPELINE_INFO_REQ_1: self.pipeline_info,
            RUN_REQ_1: self.run_request
        }
        
    def __enter__(self):
        if self.address is not None:
            self.start()
    def __exit__(self, exc_type, value, tb):
        if self.address is not None:
            self.join()
            
    def run(self):
        javabridge.attach()
        try:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(self.address)
            logger.info("Binding Knime bridge server to %s" % self.address)
            poller = zmq.Poller()
            poller.register(self.socket, flags=zmq.POLLIN)
            if self.notify_addr != None:
                self.notify_socket = self.context.socket(zmq.SUB)
                self.notify_socket.setsockopt(zmq.SUBSCRIBE, "")
                self.notify_socket.connect(self.notify_addr)
                poller.register(self.notify_socket, flags=zmq.POLLIN)
            else:
                self.notify_socket = None
            try:
                while True:
                    for socket, event in poller.poll():
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
                                    "Unhandled message type: %s" % message_type)
                            else:
                                self.dispatch[message_type](
                                    session_id, message_type, msg)
                            
            finally:
                if self.notify_socket:
                    self.notify_socket.close()
                self.socket.close()
        finally:
            javabridge.detach()
            
    def connect(self, session_id, message_type, message):
        '''Handle the connect message'''
        self.socket.send_multipart(
            [zmq.Frame(session_id),
             zmq.Frame(),
             zmq.Frame(CONNECT_REPLY_1)])
    
    def pipeline_info(self, session_id, message_type, message):
        '''Handle the pipeline info message'''
        logger.info("Handling pipeline info request")
        pipeline_txt = message.pop(0).bytes
        pipeline = cpp.Pipeline()
        try:
            pipeline.loadtxt(StringIO(pipeline_txt))
        except Exception, e:
            logger.warning(
                "Failed to load pipeline: sending pipeline exception", 
                exc_info=1)
            self.raise_pipeline_exception(session_id, str(e))
            return
        input_modules, other_modules = self.split_pipeline(pipeline)
        channels = self.find_channels(input_modules)
        type_names, measurements = self.find_measurements(
            other_modules, pipeline)
        body = json.dumps([channels, type_names, measurements])
        msg_out = [
            zmq.Frame(session_id),
            zmq.Frame(),
            zmq.Frame(PIPELINE_INFO_REPLY_1),
            zmq.Frame(body)]
        self.socket.send_multipart(msg_out)
    
    def run_request(self, session_id, message_type, message):
        '''Handle the run request message'''
        pipeline = cpp.Pipeline()
        m = cpmeas.Measurements()
        object_set = cpo.ObjectSet()
        if len(message) < 2:
            self.raise_cellprofiler_exception(
                session_id, "Missing run request sections")
            return
        pipeline_txt = message.pop(0).bytes
        image_metadata = message.pop(0).bytes
        try:
            image_metadata = json.loads(image_metadata)
            for channel_name, channel_metadata in image_metadata:
                if len(message) < 1:
                    self.raise_cellprofiler_exception(
                        session_id,
                        "Missing binary data for channel %s" % channel_name)
                    return
                pixel_data = self.decode_image(
                    channel_metadata, message.pop(0).bytes)
                m.add(channel_name, cpi.Image(pixel_data))
        except Exception, e:
            self. raise_cellprofiler_exception(
                session_id, e.message)
        try:
            pipeline.loadtxt(StringIO(pipeline_txt))
        except Exception, e:
            logger.warning(
                "Failed to load pipeline: sending pipeline exception", 
                exc_info=1)
            self.raise_pipeline_exception(session_id, str(e))
            return
        
        input_modules, other_modules = self.split_pipeline(pipeline)
        for module in other_modules:
            workspace = cpw.Workspace(
                pipeline, module, m, object_set, m, None)
            try:
                logger.info(
                    "Running module # %d: %s" % 
                    (module.module_num, module.module_name))
                module.run(workspace)
            except Exception, e:
                msg = "Encountered error while running module, \"%s\": %s" % (
                    module.module_name, e.message)
                logger.warning(msg, exc_info=1)
                self.raise_cellprofiler_exception(session_id, msg)
                return
        type_names, feature_dict = self.find_measurements(
            other_modules, pipeline)
                
        double_features = []
        double_data = []
        float_features = []
        float_data = []
        int_features = []
        int_data = []
        string_features = []
        string_data = []
        metadata = [double_features, float_features, 
                    int_features, string_features]
        
        no_data = ()
        
        for object_name, features in feature_dict.items():
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
                    data = np.atleast_1d(m[object_name, feature])
                if type_names[data_type] == 'java.lang.Double':
                    df.append((feature, len(data)))
                    if len(data) > 0:
                        double_data.append(data.astype("<f8"))
                elif type_names[data_type] == 'java.lang.Float':
                    ff.append((feature, len(data)))
                    if len(data) > 0:
                        float_data.append(data.astype('<f4'))
                elif type_names[data_type] == 'java.lang.Integer':
                    intf.append((feature, len(data)))
                    if len(data) > 0:
                        int_data.append(data.astype('<i4'))
                elif type_names[data_type] == 'java.lang.String':
                    if len(data) == 0:
                        sf.append((feature, 0))
                    else:
                        s = data[0]
                        if isinstance(s, unicode):
                            s = s.encode("utf-8")
                        else:
                            s = str(s)
                        string_data.append(np.array(s).data)
        data = np.hstack([np.hstack(ditem).data for ditem in
                          double_data, float_data, int_data, string_data
                          if len(ditem) > 0])
        self.socket.send_multipart(
            [zmq.Frame(session_id),
             zmq.Frame(),
             zmq.Frame(RUN_REPLY_1),
             zmq.Frame(json.dumps(metadata)),
             zmq.Frame(data.data)])
                            
    def raise_pipeline_exception(self, session_id, message):
        self.socket.send_multipart(
            [zmq.Frame(session_id),
             zmq.Frame(),
             zmq.Frame(PIPELINE_EXCEPTION_1),
             zmq.Frame(message)])
        
    def raise_cellprofiler_exception(self, session_id, message):
        self.socket.send_multipart(
            [zmq.Frame(session_id),
             zmq.Frame(),
             zmq.Frame(CELLPROFILER_EXCEPTION_1),
             zmq.Frame(message)])
        
    def split_pipeline(self, pipeline):
        '''Split the pipeline into input modules and everything else
        
        pipeline - the pipeline to be split
        
        returns a two-tuple of input modules and other
        '''
        input_modules = []
        other_modules = []
        for module in pipeline.modules():
            if module.is_load_module() or module.is_input_module():
                input_modules.append(module)
            else:
                other_modules.append(module)
        return input_modules, other_modules
    
    def find_channels(self, input_modules):
        '''Find image providers in the input modules'''
        channels = []
        for module in input_modules:
            for setting in module.visible_settings():
                if isinstance(setting, cps.ImageNameProvider):
                    channels.append(setting.value)
        return channels
    
    def find_measurements(self, modules, pipeline):
        '''Scan the modules for features
        
        modules - modules to scan for features
        pipeline - the pipeline they came from
        
        returns a two tuple of
            Java types, e.g. "java.lang.Integer"
            A dictionary whose key is the object name and whose
            value is a list of two-tuples of feature name and index into
            the java types array.
        '''
        jtypes = []
        features = {}
        for module in modules:
            assert isinstance(module, cpm.CPModule)
            for column in module.get_measurement_columns(pipeline):
                objects, name, dbtype = column[:3]
                if dbtype == cpmeas.COLTYPE_FLOAT:
                    jtype = "java.lang.Double"
                elif dbtype == cpmeas.COLTYPE_INTEGER:
                    jtype = "java.lang.Integer"
                elif dbtype.startswith(cpmeas.COLTYPE_VARCHAR):
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
        features_out = dict([(k, v.items()) for k, v in features.items()])
        return jtypes, features_out
    
    def decode_image(self, channel_metadata, buf):
        '''Decode an image sent via the wire format
        
        channel_metadata: sequence of 3 tuples of axis name, dimension and stride
        buf: byte-buffer, low-endian representation of doubles
        
        returns numpy array in y, x indexing format.
        '''
        pixel_data = np.frombuffer(buf, "<f8")
        strides_out = [None] * len(channel_metadata)
        dimensions_out = [None] * len(channel_metadata)
        for axis_name, dim, stride in channel_metadata:
            if axis_name.lower() == "y":
                if strides_out[0] != None:
                    raise RuntimeError("Y dimension doubly specified")
                strides_out[0] = stride * 8
                dimensions_out[0] = dim
            elif axis_name.lower() == "x":
                if strides_out[1] != None:
                    raise RuntimeError("X dimension doubly specified")
                strides_out[1] = stride * 8
                dimensions_out[1] = dim
            elif axis_name.lower() == "channel":
                if strides_out[2] != None:
                    raise RuntimeError("Channel doubly specified")
                strides_out[2] = stride * 8
                dimensions_out[2] = dim
            else:
                raise RuntimeError("Unknown dimension: " + axis_name)

        pixel_data.shape = tuple(dimensions_out)
        pixel_data.strides = tuple(strides_out)
        return pixel_data
                
