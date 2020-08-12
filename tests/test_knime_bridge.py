"""test_knime_bridge.py - test the Knime bridge"""

import pytest
import unittest
import uuid

import zmq

import cellprofiler.knime_bridge
import cellprofiler.modules.flagimage
import cellprofiler.modules.identifyprimaryobjects
import cellprofiler.modules.measureobjectsizeshape
import cellprofiler.modules.saveimages
import cellprofiler.modules.threshold
import cellprofiler_core.worker


@pytest.mark.skip("What is this even supposed to do")
class TestKnimeBridge(unittest.TestCase):
    def setUp(self):
        context = zmq.Context.instance()
        self.notify_addr = "inproc://" + uuid.uuid4().hex
        self.socket_addr = "inproc://" + uuid.uuid4().hex
        self.kill_pub = context.socket(zmq.PUB)
        self.kill_pub.bind(self.notify_addr)
        self.server = cellprofiler.knime_bridge.KnimeBridgeServer(
            context,
            self.socket_addr,
            self.notify_addr,
            cellprofiler_core.worker.NOTIFY_STOP,
        )
        self.server.start()
        self.session_id = uuid.uuid4().hex
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.socket_addr)

    def tearDown(self):
        self.kill_pub.send(cellprofiler_core.worker.NOTIFY_STOP)
        self.server.join()
        self.kill_pub.close()
        self.socket.close()

    def test_01_01_do_nothing(self):
        # test KB thread lifecycle
        pass

    def test_01_02_connect(self):
        message = [
            zmq.Frame(self.session_id),
            zmq.Frame(),
            zmq.Frame(cellprofiler.knime_bridge.CONNECT_REQ_1),
        ]
        self.socket.send_multipart(message)
        reply = self.socket.recv_multipart()
        self.assertEqual(reply.pop(0), self.session_id)
        self.assertEqual(reply.pop(0), "")
        self.assertEqual(reply.pop(0), cellprofiler.knime_bridge.CONNECT_REPLY_1)

    # FIXME: wxPython 4 PR
    # def test_02_01_pipeline_info(self):
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     load_images = cellprofiler_core.modules.loadimages.LoadImages()
    #     load_images.module_num = 1
    #     load_images.add_imagecb()
    #     load_images.images[0].channels[0].image_name.value = "Foo"
    #     load_images.images[1].channels[0].image_name.value = "Bar"
    #     pipeline.add_module(load_images)
    #     identify = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    #     identify.module_num = 2
    #     identify.x_name.value = "Foo"
    #     identify.y_name.value = "dizzy"
    #     pipeline.add_module(identify)
    #
    #     pipeline_txt = six.moves.StringIO()
    #     pipeline.savetxt(pipeline_txt)
    #     message = [zmq.Frame(self.session_id),
    #                zmq.Frame(),
    #                zmq.Frame(cellprofiler.knime_bridge.PIPELINE_INFO_REQ_1),
    #                zmq.Frame(pipeline_txt.getvalue())]
    #     self.socket.send_multipart(message)
    #     message = self.socket.recv_multipart()
    #     self.assertEqual(message.pop(0), self.session_id)
    #     self.assertEqual(message.pop(0), "")
    #     self.assertEqual(message.pop(0), cellprofiler.knime_bridge.PIPELINE_INFO_REPLY_1)
    #     body = json.loads(message.pop(0))
    #     self.assertEqual(len(body), 3)
    #     channels, type_names, measurements = body
    #     self.assertTrue("Foo" in channels)
    #     self.assertTrue("Bar" in channels)
    #     self.assertTrue("dizzy" in measurements)
    #     found_location = False
    #     found_object_number = False
    #     for feature, idx in measurements['dizzy']:
    #         if feature == "Location_Center_X":
    #             self.assertEqual('java.lang.Double', type_names[idx])
    #             found_location = True
    #         elif feature == "Number_Object_Number":
    #             self.assertEqual('java.lang.Integer', type_names[idx])
    #             found_object_number = True
    #     self.assertTrue(found_location)
    #     self.assertTrue(found_object_number)

    def test_02_02_bad_pipeline(self):
        message = [
            zmq.Frame(self.session_id),
            zmq.Frame(),
            zmq.Frame(cellprofiler.knime_bridge.PIPELINE_INFO_REQ_1),
            zmq.Frame("Freckles is a good dog but a bad pipeline"),
        ]
        self.socket.send_multipart(message)
        message = self.socket.recv_multipart()
        self.assertEqual(message.pop(0), self.session_id)
        self.assertEqual(message.pop(0), "")
        self.assertEqual(message.pop(0), cellprofiler.knime_bridge.PIPELINE_EXCEPTION_1)

    # FIXME: wxPython 4 PR
    # def test_02_03_clean_pipeline(self):
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     load_images = cellprofiler_core.modules.loadimages.LoadImages()
    #     load_images.module_num = 1
    #     load_images.add_imagecb()
    #     load_images.images[0].channels[0].image_name.value = "Foo"
    #     load_images.images[1].channels[0].image_name.value = "Bar"
    #     pipeline.add_module(load_images)
    #     identify = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    #     identify.module_num = 2
    #     identify.x_name.value = "Foo"
    #     identify.y_name.value = "dizzy"
    #     pipeline.add_module(identify)
    #     saveimages = cellprofiler.modules.saveimages.SaveImages()
    #     saveimages.module_num = 3
    #     saveimages.image_name.value = "Foo"
    #     pipeline.add_module(saveimages)
    #     measureobjectsizeshape = cellprofiler.modules.measureobjectsizeshape.MeasureObjectSizeShape()
    #     measureobjectsizeshape.module_num = 4
    #     measureobjectsizeshape.object_groups[0].name.value = "dizzy"
    #     pipeline.add_module(measureobjectsizeshape)
    #     pipeline_txt = six.moves.StringIO()
    #     pipeline.savetxt(pipeline_txt)
    #     module_names = json.dumps([cellprofiler.modules.saveimages.SaveImages.module_name])
    #     message = [
    #         zmq.Frame(self.session_id),
    #         zmq.Frame(),
    #         zmq.Frame(cellprofiler.knime_bridge.CLEAN_PIPELINE_REQ_1),
    #         zmq.Frame(pipeline_txt.getvalue()),
    #         zmq.Frame(module_names)]
    #     self.socket.send_multipart(message)
    #     message = self.socket.recv_multipart()
    #     self.assertEqual(message.pop(0), self.session_id)
    #     self.assertEqual(message.pop(0), "")
    #     self.assertEqual(message.pop(0), cellprofiler.knime_bridge.CLEAN_PIPELINE_REPLY_1)
    #     pipeline_txt = message.pop(0)
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     pipeline.loadtxt(six.moves.StringIO(pipeline_txt))
    #     self.assertEqual(len(pipeline.modules()), 3)
    #     self.assertIsInstance(pipeline.modules()[0], cellprofiler_core.modules.loadimages.LoadImages)
    #     self.assertIsInstance(pipeline.modules()[1], cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects)
    #     self.assertIsInstance(pipeline.modules()[2], cellprofiler.modules.measureobjectsizeshape.MeasureObjectSizeShape)

    # FIXME: wxPython 4 PR
    # def test_03_01_run_something(self):
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     load_images = cellprofiler_core.modules.loadimages.LoadImages()
    #     load_images.module_num = 1
    #     load_images.images[0].channels[0].image_name.value = "Foo"
    #     pipeline.add_module(load_images)
    #     identify = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    #     identify.use_advanced.value = True
    #     identify.module_num = 2
    #     identify.x_name.value = "Foo"
    #     identify.y_name.value = "dizzy"
    #     identify.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    #     identify.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    #     identify.threshold.manual_threshold.value = .5
    #     identify.exclude_size.value = False
    #     pipeline.add_module(identify)
    #
    #     pipeline_txt = six.moves.StringIO()
    #     pipeline.savetxt(pipeline_txt)
    #
    #     image = numpy.zeros((11, 17))
    #     image[2:-2, 2:-2] = 1
    #
    #     image_metadata = [
    #         ["Foo",
    #          [["Y", image.shape[0], image.strides[0] / 8],
    #           ["X", image.shape[1], image.strides[1] / 8]]]]
    #     message = [
    #         zmq.Frame(self.session_id),
    #         zmq.Frame(),
    #         zmq.Frame(cellprofiler.knime_bridge.RUN_REQ_1),
    #         zmq.Frame(pipeline_txt.getvalue()),
    #         zmq.Frame(json.dumps(image_metadata)),
    #         zmq.Frame(image)]
    #     self.socket.send_multipart(message)
    #     response = self.socket.recv_multipart()
    #     self.assertEqual(response.pop(0), self.session_id)
    #     self.assertEqual(response.pop(0), "")
    #     self.assertEqual(response.pop(0), cellprofiler.knime_bridge.RUN_REPLY_1)
    #     metadata = json.loads(response.pop(0))
    #     data = response.pop(0)
    #     measurements = self.decode_measurements(metadata, data)
    #     self.assertEqual(measurements[IMAGE]["Count_dizzy"][0], 1)
    #     self.assertEqual(measurements["dizzy"]["Location_Center_Y"][0], 5)

    # FIXME: wxPython 4 PR
    # def test_03_02_bad_cellprofiler(self):
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     load_images = cellprofiler_core.modules.loadimages.LoadImages()
    #     load_images.module_num = 1
    #     load_images.images[0].channels[0].image_name.value = "Foo"
    #     pipeline.add_module(load_images)
    #     identify = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    #     identify.module_num = 2
    #     identify.x_name.value = "Foo"
    #     identify.y_name.value = "dizzy"
    #     identify.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    #     identify.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    #     identify.threshold.manual_threshold.value = .5
    #     identify.exclude_size.value = False
    #     pipeline.add_module(identify)
    #
    #     pipeline_txt = six.moves.StringIO()
    #     pipeline.savetxt(pipeline_txt)
    #
    #     image = numpy.zeros((11, 17))
    #     image[2:-2, 2:-2] = 1
    #
    #     # Get the strides wrong (I broke it accidentally this way before...)
    #     image_metadata = [
    #         ["Foo",
    #          [["Y", image.shape[0], image.strides[0]],
    #           ["X", image.shape[1], image.strides[1]]]]]
    #     message = [
    #         zmq.Frame(self.session_id),
    #         zmq.Frame(),
    #         zmq.Frame(cellprofiler.knime_bridge.RUN_REQ_1),
    #         zmq.Frame(pipeline_txt.getvalue()),
    #         zmq.Frame(json.dumps(image_metadata)),
    #         zmq.Frame(image)]
    #     self.socket.send_multipart(message)
    #     response = self.socket.recv_multipart()
    #     self.assertEqual(response.pop(0), self.session_id)
    #     self.assertEqual(response.pop(0), "")
    #     self.assertEqual(response.pop(0), cellprofiler.knime_bridge.CELLPROFILER_EXCEPTION_1)

    # FIXME: wxPython 4 PR
    # def test_03_03_run_missing_measurement(self):
    #     # Regression test of knime-bridge issue #6
    #     #
    #     # Missing measurement causes exception
    #     #
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     load_images = cellprofiler_core.modules.loadimages.LoadImages()
    #     load_images.module_num = 1
    #     load_images.images[0].channels[0].image_name.value = "Foo"
    #     pipeline.add_module(load_images)
    #     identify = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    #     identify.module_num = 2
    #     identify.use_advanced.value = True
    #     identify.x_name.value = "Foo"
    #     identify.y_name.value = "dizzy"
    #     identify.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    #     identify.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    #     identify.threshold.manual_threshold.value = .5
    #     identify.exclude_size.value = False
    #     pipeline.add_module(identify)
    #
    #     flag_module = cellprofiler.modules.flagimage.FlagImage()
    #     flag_module.set_module_num(3)
    #     flag = flag_module.flags[0]
    #     flag.wants_skip.value = True
    #     criterion = flag.measurement_settings[0]
    #     criterion.source_choice.value = cellprofiler.modules.flagimage.S_IMAGE
    #     criterion.measurement.value = "Count_dizzy"
    #     criterion.wants_minimum.value = True
    #     criterion.minimum_value.value = 1000
    #     pipeline.add_module(flag_module)
    #
    #     measureobjectsizeshape = cellprofiler.modules.measureobjectsizeshape.MeasureObjectSizeShape()
    #     measureobjectsizeshape.module_num = 4
    #     measureobjectsizeshape.object_groups[0].name.value = "dizzy"
    #     pipeline.add_module(measureobjectsizeshape)
    #
    #     pipeline_txt = six.moves.StringIO()
    #     pipeline.savetxt(pipeline_txt)
    #
    #     image = numpy.zeros((11, 17))
    #     image[2:-2, 2:-2] = 1
    #
    #     image_metadata = [
    #         ["Foo",
    #          [["Y", image.shape[0], image.strides[0] / 8],
    #           ["X", image.shape[1], image.strides[1] / 8]]]]
    #     message = [
    #         zmq.Frame(self.session_id),
    #         zmq.Frame(),
    #         zmq.Frame(cellprofiler.knime_bridge.RUN_REQ_1),
    #         zmq.Frame(pipeline_txt.getvalue()),
    #         zmq.Frame(json.dumps(image_metadata)),
    #         zmq.Frame(image)]
    #     self.socket.send_multipart(message)
    #     response = self.socket.recv_multipart()
    #     self.assertEqual(response.pop(0), self.session_id)
    #     self.assertEqual(response.pop(0), "")
    #     self.assertEqual(response.pop(0), cellprofiler.knime_bridge.RUN_REPLY_1)
    #     metadata = json.loads(response.pop(0))
    #     data = response.pop(0)
    #     measurements = self.decode_measurements(metadata, data)
    #     self.assertEqual(measurements[IMAGE]["Count_dizzy"][0], 1)
    #     self.assertEqual(measurements["dizzy"]["Location_Center_Y"][0], 5)
    #     self.assertEqual(len(measurements["dizzy"]["AreaShape_Area"]), 0)

    # FIXME: wxPython 4 PR
    # def test_04_01_run_group(self):
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     load_images = cellprofiler_core.modules.loadimages.LoadImages()
    #     load_images.module_num = 1
    #     load_images.images[0].channels[0].image_name.value = "Foo"
    #     pipeline.add_module(load_images)
    #     identify = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    #     identify.use_advanced.value = True
    #     identify.module_num = 2
    #     identify.x_name.value = "Foo"
    #     identify.y_name.value = "dizzy"
    #     identify.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    #     identify.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    #     identify.threshold.manual_threshold.value = .5
    #     identify.exclude_size.value = False
    #     pipeline.add_module(identify)
    #
    #     pipeline_txt = six.moves.StringIO()
    #     pipeline.savetxt(pipeline_txt)
    #
    #     image = numpy.zeros((2, 11, 17))
    #     image[0, 2:-2, 2:-2] = 1
    #     image[1, 2:-2, 2:7] = 1
    #     image[1, 2:-2, 10:-2] = 1
    #
    #     image_metadata = [
    #         ["Foo",
    #          [["Z", image.shape[0], image.strides[0] / 8],
    #           ["Y", image.shape[1], image.strides[1] / 8],
    #           ["X", image.shape[2], image.strides[2] / 8]]]]
    #     message = [
    #         zmq.Frame(self.session_id),
    #         zmq.Frame(),
    #         zmq.Frame(cellprofiler.knime_bridge.RUN_GROUP_REQ_1),
    #         zmq.Frame(pipeline_txt.getvalue()),
    #         zmq.Frame(json.dumps(image_metadata)),
    #         zmq.Frame(image)]
    #     self.socket.send_multipart(message)
    #     response = self.socket.recv_multipart()
    #     self.assertEqual(response.pop(0), self.session_id)
    #     self.assertEqual(response.pop(0), "")
    #     self.assertEqual(response.pop(0), cellprofiler.knime_bridge.RUN_REPLY_1)
    #     metadata = json.loads(response.pop(0))
    #     data = response.pop(0)
    #     measurements = self.decode_measurements(metadata, data)
    #     self.assertEqual(len(measurements[IMAGE][IMAGE_NUMBER]), 2)
    #     self.assertEqual(measurements[IMAGE]["Count_dizzy"][0], 1)
    #     self.assertEqual(measurements[IMAGE]["Count_dizzy"][1], 2)
    #     self.assertEqual(measurements["dizzy"]["Location_Center_Y"][0], 5)

    # FIXME: wxPython 4 PR
    # def test_04_02_bad_cellprofiler(self):
    #     pipeline = cellprofiler_core.pipeline.Pipeline()
    #     load_images = cellprofiler_core.modules.loadimages.LoadImages()
    #     load_images.module_num = 1
    #     load_images.images[0].channels[0].image_name.value = "Foo"
    #     pipeline.add_module(load_images)
    #     identify = cellprofiler.modules.identifyprimaryobjects.IdentifyPrimaryObjects()
    #     identify.module_num = 2
    #     identify.x_name.value = "Foo"
    #     identify.y_name.value = "dizzy"
    #     identify.threshold.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    #     identify.threshold.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    #     identify.threshold.manual_threshold.value = .5
    #     identify.exclude_size.value = False
    #     pipeline.add_module(identify)
    #
    #     pipeline_txt = six.moves.StringIO()
    #     pipeline.savetxt(pipeline_txt)
    #
    #     image = numpy.zeros((11, 17))
    #     image[2:-2, 2:-2] = 1
    #
    #     # Get the strides wrong (I broke it accidentally this way before...)
    #     # And there's more wrong in this one.
    #     image_metadata = [
    #         ["Foo",
    #          [["Y", image.shape[0], image.strides[0]],
    #           ["X", image.shape[1], image.strides[1]]]]]
    #     message = [
    #         zmq.Frame(self.session_id),
    #         zmq.Frame(),
    #         zmq.Frame(cellprofiler.knime_bridge.RUN_GROUP_REQ_1),
    #         zmq.Frame(pipeline_txt.getvalue()),
    #         zmq.Frame(json.dumps(image_metadata)),
    #         zmq.Frame(image)]
    #     self.socket.send_multipart(message)
    #     response = self.socket.recv_multipart()
    #     self.assertEqual(response.pop(0), self.session_id)
    #     self.assertEqual(response.pop(0), "")
    #     self.assertEqual(response.pop(0), cellprofiler.knime_bridge.CELLPROFILER_EXCEPTION_1)
    #
    # def decode_measurements(self, metadata, data):
    #     offset = 0
    #     ddata = {}
    #     self.assertEqual(len(metadata), 4)
    #     for object_name, md in metadata[0]:
    #         items = {}
    #         ddata[object_name] = items
    #         for feature, count in md:
    #             next_offset = offset + count * 8
    #             items[feature] = numpy.frombuffer(
    #                     data[offset:next_offset], numpy.float64)
    #             offset = next_offset
    #     for object_name, md in metadata[1]:
    #         if object_name not in ddata:
    #             items = {}
    #             ddata[object_name] = items
    #         else:
    #             items = ddata[object_name]
    #         for feature, count in md:
    #             next_offset = offset + count * 4
    #             items[feature] = numpy.frombuffer(
    #                     data[offset:next_offset], numpy.float32)
    #             offset = next_offset
    #     for object_name, md in metadata[2]:
    #         if object_name not in ddata:
    #             items = {}
    #             ddata[object_name] = items
    #         else:
    #             items = ddata[object_name]
    #         for feature, count in md:
    #             next_offset = offset + count * 4
    #             items[feature] = numpy.frombuffer(
    #                     data[offset:next_offset], numpy.int32)
    #             offset = next_offset
    #     return ddata
