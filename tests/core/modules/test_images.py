import io
import os
import tempfile

import cellprofiler_core.constants.modules.images
import cellprofiler_core.measurement
import cellprofiler_core.pipeline
import cellprofiler_core.pipeline.event._load_exception
import cellprofiler_core.workspace
import cellprofiler_core.modules.images
import cellprofiler_core.modules.metadata
from cellprofiler_core.utilities.pathname import url2pathname, pathname2url

import tests.core


def get_data_directory():
    folder = os.path.dirname(tests.core.__file__)
    return os.path.abspath(os.path.join(folder, "data/"))


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

    def test_load_CP4_extraction_pipeline(self):
        """
        In CP5 the metadata extraction from file headers was moved from Metadata to Images.
        Loaded pipelines from older versions should enable extraction if the method was removed
        from Metadata.
        """
        pipeline_file = os.path.join(get_data_directory(), "modules/images/migration.pipeline")
        with open(pipeline_file, "r") as fd:
            data = fd.read()

        pipeline = cellprofiler_core.pipeline.Pipeline()

        def callback(caller, event):
            assert not isinstance(
                event,
                cellprofiler_core.pipeline.event._load_exception.LoadException,
            )

        pipeline.add_listener(callback)
        pipeline.load(io.StringIO(data))
        assert len(pipeline.modules()) == 2
        images_module, metadata_module = pipeline.modules()
        assert isinstance(images_module, cellprofiler_core.modules.images.Images)
        assert isinstance(metadata_module, cellprofiler_core.modules.metadata.Metadata)
        assert images_module.want_split.value
        # Confirm that the header extraction method was removed from Metadata
        assert metadata_module.extraction_method_count.value == 1

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

    def check(self, module, url, expected, extract=False):
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
            assert file_list[0].url == url
        else:
            assert len(file_list) == 0
        if extract:
            # When not in the GUI pipeline and workspace are disconnected.
            # Since we're manually extracting metadata we'll keep them aligned.
            hdf5_list = workspace.file_list
            true_file_list = hdf5_list.get_filelist()
            assert len(true_file_list) == len(file_list) == 1
            assert true_file_list[0] == file_list[0].url
            file_object = file_list[0]
            assert not file_object.extracted
            old_meta, series_names = hdf5_list.get_metadata(url)
            assert all([x == -1 for x in old_meta])
            assert series_names == [""]
            module.prepare_run(workspace)
            assert file_object.extracted
            plane_meta = file_object.metadata
            assert plane_meta['SizeS'] == 1
            expected_meta = [3, 1, 1, 21, 31]
            plane_metadata = [plane_meta[k][0] for k in ('SizeC', 'SizeZ', 'SizeT', 'SizeY', 'SizeX')]
            stored_metadata, stored_names = hdf5_list.get_metadata(url)
            assert len(stored_metadata) == len(expected_meta)
            assert len(stored_names) == len(expected_meta) // 5
            for expected, obj, hdf5 in zip(expected_meta, plane_metadata, stored_metadata):
                assert expected == obj == hdf5

    def test_filter_standard(self):
        module = cellprofiler_core.modules.images.Images()
        module.filter_choice.value = (
            cellprofiler_core.constants.modules.images.FILTER_CHOICE_IMAGES
        )
        for url, expected in (
            ("file:/TestImages/NikonTIF.tif", True),
            ("file:/foo/.bar/baz.tif", True),
            ("file:/TestImages/foo.bar", False),
        ):
            self.check(module, url, expected)

    def test_extract_planes(self):
        module = cellprofiler_core.modules.images.Images()
        module.want_split.value = True
        path = tests.core.modules.maybe_download_example_image(
            ["ExampleColorToGray"], "nt_03_01_color.tif", (21, 31, 3)
        )
        url = pathname2url(path)

        self.check(module, url, True, extract=True)
        # We need to use the same image later with different shape params.
        os.remove(path)
