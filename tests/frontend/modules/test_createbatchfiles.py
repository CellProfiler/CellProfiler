import sys

import pytest
import six.moves

import cellprofiler_core.module
import cellprofiler.modules.createbatchfiles
import cellprofiler_core.pipeline
import cellprofiler_core.setting

import tests.frontend.modules


def test_test_load_version_9_please():
    assert (
        cellprofiler.modules.createbatchfiles.CreateBatchFiles.variable_revision_number
        == 8
    )


@pytest.mark.skip(reason="Outdated pipeline")
def test_load_v7():
    file = tests.frontend.modules.get_test_resources_directory("createbatchfiles/v7.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.loadtxt(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.createbatchfiles.CreateBatchFiles)
    assert module.wants_default_output_directory
    assert module.custom_output_directory.value == r"C:\foo\bar"
    assert not module.remote_host_is_windows
    assert not module.distributed_mode
    assert module.default_image_directory == r"C:\bar\baz"
    assert module.revision == 0
    assert not module.from_old_matlab
    assert len(module.mappings) == 1
    mapping = module.mappings[0]
    assert mapping.local_directory == r"\\argon-cifs\imaging_docs"
    assert mapping.remote_directory == r"/imaging/docs"


def test_load_v8():
    file = tests.frontend.modules.get_test_resources_directory("createbatchfiles/v8.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.loadtxt(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert module.wants_default_output_directory.value
    assert module.custom_output_directory.value == "/Users/cellprofiler"
    assert not module.remote_host_is_windows.value
    assert module.mappings[0].local_directory.value == "/Users/cellprofiler/Pictures"
    assert module.mappings[0].remote_directory.value == "/Remote/cellprofiler/Pictures"


def test_module_must_be_last():
    """Make sure that the pipeline is invalid if CreateBatchFiles is not last"""
    #
    # First, make sure that a naked CPModule tests valid
    #
    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler_core.module.Module()
    module.set_module_num(len(pipeline.modules()) + 1)
    pipeline.add_module(module)
    pipeline.test_valid()
    #
    # Make sure that CreateBatchFiles on its own tests valid
    #
    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.createbatchfiles.CreateBatchFiles()
    module.set_module_num(len(pipeline.modules()) + 1)
    pipeline.add_module(module)
    pipeline.test_valid()

    module = cellprofiler_core.module.Module()
    module.set_module_num(len(pipeline.modules()) + 1)
    pipeline.add_module(module)
    with pytest.raises(cellprofiler_core.setting.ValidationError):
        pipeline.test_valid()


def test_alter_path():
    module = cellprofiler.modules.createbatchfiles.CreateBatchFiles()
    module.mappings[0].local_directory.value = "foo"
    module.mappings[0].remote_directory.value = "bar"

    assert module.alter_path("foo/bar") == "bar/bar"
    assert module.alter_path("baz/bar") == "baz/bar"


def test_alter_path_regexp():
    module = cellprofiler.modules.createbatchfiles.CreateBatchFiles()
    module.mappings[0].local_directory.value = "foo"
    module.mappings[0].remote_directory.value = "bar"

    assert module.alter_path("foo/bar", regexp_substitution=True) == "bar/bar"
    assert module.alter_path("baz/bar", regexp_substitution=True) == "baz/bar"

    module.mappings[0].local_directory.value = r"\foo\baz"
    module.remote_host_is_windows.value = True
    assert (
        module.alter_path(r"\\foo\\baz\\bar", regexp_substitution=True) == r"bar\\bar"
    )


if sys.platform == "win32":

    def test_alter_path_windows():
        module = cellprofiler.modules.createbatchfiles.CreateBatchFiles()
        module.mappings[0].local_directory.value = "\\foo"
        module.mappings[0].remote_directory.value = "\\bar"

        assert module.alter_path("\\foo\\bar") == "/bar/bar"
        assert module.alter_path("\\FOO\\bar") == "/bar/bar"
        assert module.alter_path("\\baz\\bar") == "/baz/bar"

    def test_alter_path_windows_regexp():
        module = cellprofiler.modules.createbatchfiles.CreateBatchFiles()
        module.mappings[0].local_directory.value = "foo"
        module.mappings[0].remote_directory.value = "bar"

        assert (
            module.alter_path("\\\\foo\\\\bar", regexp_substitution=True) == "/foo/bar"
        )
        assert (
            module.alter_path("\\\\foo\\g<bar>", regexp_substitution=True)
            == "/foo\\g<bar>"
        )
