import io
import os
import tempfile

import cellprofiler_core.constants.modules.images
import cellprofiler_core.measurement
import cellprofiler_core.modules.images
import cellprofiler_core.pipeline
import cellprofiler_core.pipeline.event._load_exception
import cellprofiler_core.workspace


def get_data_directory():
    folder = os.path.dirname(cellprofiler_core.workspace.__file__)
    return os.path.abspath(os.path.join(folder, "../..", "tests/data/"))


class TestImages:
    def setup_method(self):
        # The Images module needs a workspace and the workspace needs
        # an HDF5 file.
        #
        self.temp_fd, self.temp_filename = tempfile.mkstemp(".h5")
        self.measurements = cellprofiler_core.measurement.Measurements(
            filename=self.temp_filename
        )
        os.close(self.temp_fd)

    def teardown_method(self):
        self.measurements.close()
        os.unlink(self.temp_filename)
        assert not os.path.exists(self.temp_filename)

    def test_load_v1(self):
        pipeline_file = os.path.join(get_data_directory(), "modules/images/v1.pipeline")
        with open(pipeline_file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(
                event, cellprofiler_core.pipeline.event._load_exception.LoadException
            )

        pipeline.add_listener(callback)
        pipeline.load(io.StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, cellprofiler_core.modules.images.Images)
        assert (
            module.filter_choice.value
            == cellprofiler_core.constants.modules.images.FILTER_CHOICE_CUSTOM
        )
        assert (
            module.filter.value
            == 'or (directory does startwith "foo") (file does contain "bar")'
        )

    def test_load_v2(self):
        pipeline_file = os.path.join(get_data_directory(), "modules/images/v2.pipeline")
        with open(pipeline_file, "r") as fd:
            data = fd.read()

        for fc, fctext in (
            (cellprofiler_core.constants.modules.images.FILTER_CHOICE_CUSTOM, "Custom"),
            (
                cellprofiler_core.constants.modules.images.FILTER_CHOICE_IMAGES,
                "Images only",
            ),
            (
                cellprofiler_core.constants.modules.images.FILTER_CHOICE_NONE,
                "No filtering",
            ),
        ):
            pipeline = cellprofiler_core.pipeline.Pipeline()

            def callback(caller, event):
                assert not isinstance(
                    event,
                    cellprofiler_core.pipeline.event._load_exception.LoadException,
                )

            pipeline.add_listener(callback)
            pipeline.load(io.StringIO(data % fctext))
            assert len(pipeline.modules()) == 1
            module = pipeline.modules()[0]
            assert isinstance(module, cellprofiler_core.modules.images.Images)
            assert module.filter_choice == fc
            assert (
                module.filter.value
                == 'or (directory does startwith "foo") (file does contain "bar")'
            )

    def test_filter_url(self):
        module = cellprofiler_core.modules.images.Images()
        module.filter_choice.value = (
            cellprofiler_core.constants.modules.images.FILTER_CHOICE_CUSTOM
        )
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
        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.add_urls([url])
        module.set_module_num(1)
        pipeline.add_module(module)
        m = cellprofiler_core.measurement.Measurements()
        workspace = cellprofiler_core.workspace.Workspace(
            pipeline, module, None, None, m, None
        )
        file_list = pipeline.get_filtered_file_list(workspace)
        if expected:
            assert len(file_list) == 1
            assert file_list[0] == url
        else:
            assert len(file_list) == 0

    def test_filter_standard(self):
        module = cellprofiler_core.modules.images.Images()
        module.filter_choice.value = (
            cellprofiler_core.constants.modules.images.FILTER_CHOICE_IMAGES
        )
        for url, expected in (
            ("file:/TestImages/NikonTIF.tif", True),
            ("file:/foo/.bar/baz.tif", False),
            ("file:/TestImages/foo.bar", False),
        ):
            self.check(module, url, expected)
