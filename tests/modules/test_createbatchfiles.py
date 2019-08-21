"""test_createbatchfiles - test the CreateBatchFiles module
"""

import base64
import os
import sys
import tempfile
import unittest
import zlib
from six.moves import StringIO

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps

import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.createbatchfiles as C
import tests.modules as T
import pytest


class TestCreateBatchFiles(unittest.TestCase):
    def test_01_00_test_load_version_9_please(self):
        assert C.CreateBatchFiles.variable_revision_number == 8

    def test_01_07_load_v7(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150713184605
GitHash:2f7b3b9
ModuleCount:1
HasImagePlaneDetails:False

CreateBatchFiles:[module_num:19|svn_version:\'Unknown\'|variable_revision_number:7|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Store batch files in default output folder?:Yes
    Output folder path:C\x3A\\\\foo\\\\bar
    Are the cluster computers running Windows?:No
    Hidden\x3A in batch mode:No
    Hidden\x3A in distributed mode:No
    Hidden\x3A default input folder at time of save:C\x3A\\\\bar\\\\baz
    Hidden\x3A revision number:0
    Hidden\x3A from old matlab:No
    Launch BatchProfiler:Yes
    Local root path:\\\\\\\\argon-cifs\\\\imaging_docs
    Cluster root path:/imaging/docs
"""
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, C.CreateBatchFiles)
        assert module.wants_default_output_directory
        assert module.custom_output_directory == r"C:\foo\bar"
        assert not module.remote_host_is_windows
        assert not module.distributed_mode
        assert module.default_image_directory == r"C:\bar\baz"
        assert module.revision == 0
        assert not module.from_old_matlab
        assert len(module.mappings) == 1
        mapping = module.mappings[0]
        assert mapping.local_directory == r"\\argon-cifs\imaging_docs"
        assert mapping.remote_directory == r"/imaging/docs"

    def test_01_08_load_v8(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150713184605
GitHash:2f7b3b9
ModuleCount:1
HasImagePlaneDetails:False

CreateBatchFiles:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:8|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Store batch files in default output folder?:Yes
    Output folder path:/Users/cellprofiler
    Are the cluster computers running Windows?:No
    Hidden\x3A in batch mode:No
    Hidden\x3A in distributed mode:No
    Hidden\x3A default input folder at time of save:/Users/mcquin/Pictures
    Hidden\x3A revision number:0
    Hidden\x3A from old matlab:No
    Local root path:/Users/cellprofiler/Pictures
    Cluster root path:/Remote/cellprofiler/Pictures
"""
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert module.wants_default_output_directory.value
        assert module.custom_output_directory.value == "/Users/cellprofiler"
        assert not module.remote_host_is_windows.value
        assert (
            module.mappings[0].local_directory.value == "/Users/cellprofiler/Pictures"
        )
        assert (
            module.mappings[0].remote_directory.value == "/Remote/cellprofiler/Pictures"
        )

    def test_02_01_module_must_be_last(self):
        """Make sure that the pipeline is invalid if CreateBatchFiles is not last"""
        #
        # First, make sure that a naked CPModule tests valid
        #
        pipeline = cpp.Pipeline()
        module = cpm.Module()
        module.set_module_num(len(pipeline.modules()) + 1)
        pipeline.add_module(module)
        pipeline.test_valid()
        #
        # Make sure that CreateBatchFiles on its own tests valid
        #
        pipeline = cpp.Pipeline()
        module = C.CreateBatchFiles()
        module.set_module_num(len(pipeline.modules()) + 1)
        pipeline.add_module(module)
        pipeline.test_valid()

        module = cpm.Module()
        module.set_module_num(len(pipeline.modules()) + 1)
        pipeline.add_module(module)
        with pytest.raises(cps.ValidationError):
            pipeline.test_valid()

    def test_04_01_alter_path(self):
        module = C.CreateBatchFiles()
        module.mappings[0].local_directory.value = "foo"
        module.mappings[0].remote_directory.value = "bar"

        assert module.alter_path("foo/bar") == "bar/bar"
        assert module.alter_path("baz/bar") == "baz/bar"

    def test_04_02_alter_path_regexp(self):
        module = C.CreateBatchFiles()
        module.mappings[0].local_directory.value = "foo"
        module.mappings[0].remote_directory.value = "bar"

        assert module.alter_path("foo/bar", regexp_substitution=True) == "bar/bar"
        assert module.alter_path("baz/bar", regexp_substitution=True) == "baz/bar"

        module.mappings[0].local_directory.value = r"\foo\baz"
        module.remote_host_is_windows.value = True
        assert (
            module.alter_path(r"\\foo\\baz\\bar", regexp_substitution=True)
            == r"bar\\bar"
        )

    if sys.platform == "win32":

        def test_04_03_alter_path_windows(self):
            module = C.CreateBatchFiles()
            module.mappings[0].local_directory.value = "\\foo"
            module.mappings[0].remote_directory.value = "\\bar"

            assert module.alter_path("\\foo\\bar") == "/bar/bar"
            assert module.alter_path("\\FOO\\bar") == "/bar/bar"
            assert module.alter_path("\\baz\\bar") == "/baz/bar"

        def test_04_04_alter_path_windows_regexp(self):
            module = C.CreateBatchFiles()
            module.mappings[0].local_directory.value = "foo"
            module.mappings[0].remote_directory.value = "bar"

            assert (
                module.alter_path("\\\\foo\\\\bar", regexp_substitution=True)
                == "/foo/bar"
            )
            assert (
                module.alter_path("\\\\foo\\g<bar>", regexp_substitution=True)
                == "/foo\\g<bar>"
            )
