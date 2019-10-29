import io
import os
import tempfile

import nucleus.measurement
import nucleus.modules.images
import nucleus.pipeline
import nucleus.pipeline.event._load_exception
import nucleus.workspace


class TestImages:
    def setup_method(self):
        # The Images module needs a workspace and the workspace needs
        # an HDF5 file.
        #
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.measurements = nucleus.measurement.Measurements(
            filename=self.temp_filename
        )
        os.close(self.temp_fd)

    def teardown_method(self):
        self.measurements.close()
        os.unlink(self.temp_filename)
        assert not os.path.exists(self.temp_filename)

    def test_load_v1(self):
        with open("./tests/data/modules/images/v1.pipeline", "r") as fd:
            data = fd.read()

        pipeline = nucleus.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(event,
                                  nucleus.pipeline.event._load_exception.LoadException)

        pipeline.add_listener(callback)
        pipeline.load(io.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, nucleus.modules.images.Images)
        assert module.filter_choice == nucleus.modules.images.FILTER_CHOICE_CUSTOM
        assert (
            module.filter.value
            == 'or (directory does startwith "foo") (file does contain "bar")'
        )

    def test_load_v2(self):
        with open("./tests/data/modules/images/v2.pipeline", "r") as fd:
            data = fd.read()

        for fc, fctext in (
            (nucleus.modules.images.FILTER_CHOICE_CUSTOM, "Custom"),
            (nucleus.modules.images.FILTER_CHOICE_IMAGES, "Images only"),
            (nucleus.modules.images.FILTER_CHOICE_NONE, "No filtering"),
        ):
            pipeline = nucleus.pipeline.Pipeline()

            def callback(caller, event):
                assert not isinstance(event,
                                      nucleus.pipeline.event._load_exception.LoadException)

            pipeline.add_listener(callback)
            pipeline.load(io.StringIO(data % fctext))
            assert len(pipeline.modules()) == 1
            module = pipeline.modules()[0]
            assert isinstance(module, nucleus.modules.images.Images)
            assert module.filter_choice == fc
            assert (
                module.filter.value
                == 'or (directory does startwith "foo") (file does contain "bar")'
            )

    def test_filter_url(self):
        module = nucleus.modules.images.Images()
        module.filter_choice.value = nucleus.modules.images.FILTER_CHOICE_CUSTOM
        for url, filter_value, expected in (
            (
                "file:/TestImages/NikonTIF.tif",
                'and (file does startwith "Nikon") (extension does istif)',
                True,
            ),
            (
                "file:/TestImages/NikonTIF.tif",
                'or (file doesnot startwith "Nikon") (extension doesnot istif)',
                False,
            ),
            (
                "file:/TestImages/003002000.flex",
                'and (directory does endwith "ges") (directory doesnot contain "foo")',
                True,
            ),
            (
                "file:/TestImages/003002000.flex",
                'or (directory doesnot endwith "ges") (directory does contain "foo")',
                False,
            ),
        ):
            module.filter.value = filter_value
            self.check(module, url, expected)

    def check(self, module, url, expected):
        """Check filtering of one URL using the module as configured"""
        pipeline = nucleus.pipeline.Pipeline()
        pipeline.add_urls([url])
        module.set_module_num(1)
        pipeline.add_module(module)
        m = nucleus.measurement.Measurements()
        workspace = nucleus.workspace.Workspace(pipeline, module, None, None, m, None)
        file_list = pipeline.get_filtered_file_list(workspace)
        if expected:
            assert len(file_list) == 1
            assert file_list[0] == url
        else:
            assert len(file_list) == 0

    def test_filter_standard(self):
        module = nucleus.modules.images.Images()
        module.filter_choice.value = nucleus.modules.images.FILTER_CHOICE_IMAGES
        for url, expected in (
            ("file:/TestImages/NikonTIF.tif", True),
            ("file:/foo/.bar/baz.tif", False),
            ("file:/TestImages/foo.bar", False),
        ):
            self.check(module, url, expected)
