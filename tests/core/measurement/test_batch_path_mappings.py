"""Batch path tests use real path codecs, not a native Windows runtime."""

import json
import ntpath
import nturl2path
import os
import pathlib
import posixpath
import urllib.parse
import urllib.request
from types import SimpleNamespace

import cellprofiler_core.measurement._measurements as measurements_module
import pytest
from cellprofiler_core.constants.measurement import (
    K_CASE_SENSITIVE,
    K_LOCAL_SEPARATOR,
    K_PATH_MAPPINGS,
    K_URL2PATHNAME_PACKAGE_NAME,
    M_PATH_MAPPINGS,
)
from cellprofiler_core.measurement import Measurements

LOCAL_ROOT = r"D:\$PROJECT\images"
REMOTE_ROOT = "/home/ubuntu/bucket/projects/$PROJECT/$BATCH/images"
FILENAME = "café 50% #1 $sample.tif"


@pytest.fixture
def measurements(tmp_path):
    m = Measurements(filename=str(tmp_path / "batch.h5"))
    try:
        yield m
    finally:
        m.close()


def set_mappings(m, mappings, *, windows=True, package=None):
    metadata = {
        K_CASE_SENSITIVE: not windows,
        K_LOCAL_SEPARATOR: "\\" if windows else "/",
        K_PATH_MAPPINGS: mappings,
        K_URL2PATHNAME_PACKAGE_NAME: package
        or ("nturl2path" if windows else "urllib.request"),
    }
    m.add_experiment_measurement(M_PATH_MAPPINGS, json.dumps(metadata))


@pytest.fixture(params=["posix", "windows"])
def target(request, monkeypatch):
    """Only replace the measurement module's target OS, not the test host OS."""
    windows = request.param == "windows"
    path = ntpath if windows else posixpath
    encode = nturl2path.pathname2url if windows else urllib.parse.quote

    # Use the actual POSIX decoder on POSIX hosts. On Windows, provide its
    # file-URI semantics without borrowing the Windows urllib.request codec.
    def posix_decode(url_path):
        if url_path.startswith("///"):
            url_path = url_path[2:]
        elif url_path.startswith("//localhost/"):
            url_path = url_path[11:]
        return urllib.parse.unquote(url_path)

    posix_decode = urllib.request.url2pathname if os.name == "posix" else posix_decode
    decode = nturl2path.url2pathname if windows else posix_decode
    monkeypatch.setattr(measurements_module, "os", SimpleNamespace(path=path))
    monkeypatch.setattr(urllib.request, "pathname2url", encode)
    # The old code's decoder alias must also represent the selected target.
    monkeypatch.setattr(measurements_module, "os_url2pathname", decode, raising=False)
    root = r"Z:\cluster\$PROJECT\$BATCH\images" if windows else REMOTE_ROOT
    return SimpleNamespace(path=path, encode=encode, decode=decode, root=root)


@pytest.mark.parametrize(
    "local_root, source_path",
    [
        (LOCAL_ROOT, LOCAL_ROOT + "\\" + FILENAME),
        (LOCAL_ROOT.lower(), LOCAL_ROOT + "\\" + FILENAME),
        ("d:/$project/images", LOCAL_ROOT + "\\" + FILENAME),
        (LOCAL_ROOT, "D:/$PROJECT/images/" + FILENAME),
        (r"D:\Batch_1_CellGrowth\images", r"D:\Batch_1_CellGrowth\images\image.tif"),
        (r"\\server\share\images", r"\\SERVER\share\images" + "\\" + FILENAME),
    ],
)
def test_deferred_windows_mapping(measurements, target, local_root, source_path):
    set_mappings(measurements, [(local_root, target.root)])
    url = "file:" + nturl2path.pathname2url(source_path)
    expected_path = target.path.join(target.root, ntpath.basename(source_path))
    result = measurements.alter_url_post_create_batch(url)
    assert result == "file:" + target.encode(expected_path)
    assert target.decode(result[5:]) == expected_path
    assert measurements.alter_url_post_create_batch(result) == result


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize(
    "package", [None, "urllib", "urllib.request", "untrusted.module"]
)
def test_legacy_and_unknown_codec(measurements, target, windows, package):
    root = LOCAL_ROOT if windows else "/source/images"
    source_path = root + ("\\" if windows else "/") + FILENAME
    encode = nturl2path.pathname2url if windows else urllib.parse.quote
    set_mappings(measurements, [(root, target.root)], windows=windows)
    metadata = json.loads(measurements.get_experiment_measurement(M_PATH_MAPPINGS))
    if package is None:
        del metadata[K_URL2PATHNAME_PACKAGE_NAME]
    else:
        metadata[K_URL2PATHNAME_PACKAGE_NAME] = package
    measurements.add_experiment_measurement(M_PATH_MAPPINGS, json.dumps(metadata))
    url = "file:" + encode(source_path)
    result = measurements.alter_url_post_create_batch(url)
    # Python 3.14 records urllib.request on Windows as well as POSIX.
    # The saved separator disambiguates the generic codec name.
    if package == "untrusted.module":
        assert result == url
    else:
        assert result == "file:" + target.encode(
            target.path.join(target.root, FILENAME)
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///E:/unmapped/image%20%2520.tif",
        "file://other-server/share/image%20one.tif",
        "file:relative/image%20one.tif",
        "localpath/image.tif",
        "https://example.org/image%20one.tif?signature=a%2Fb",
        "s3://bucket/$PROJECT/image%20one.tif",
        "file:///home/ubuntu/bucket/projects/%24PROJECT/%24BATCH/images/image.tif",
    ],
)
def test_unmapped_urls_unchanged(measurements, target, url):
    assert measurements.alter_url_post_create_batch(url) == url
    set_mappings(measurements, [(LOCAL_ROOT, target.root)])
    assert measurements.alter_url_post_create_batch(url) == url
    set_mappings(measurements, [])
    assert measurements.alter_url_post_create_batch(url) == url


@pytest.mark.parametrize("url_prefix", ["file:", "file://"])
def test_deferred_posix_source(measurements, target, url_prefix):
    set_mappings(measurements, [("/source/images", target.root)], windows=False)
    filename = FILENAME
    url = url_prefix + urllib.parse.quote("/source/images/" + filename)
    result = measurements.alter_url_post_create_batch(url)
    assert result == "file:" + target.encode(target.path.join(target.root, filename))


def test_write_path_mappings_records_origin(measurements, monkeypatch):
    mappings = [(LOCAL_ROOT, REMOTE_ROOT)]
    measurements.write_path_mappings(mappings)
    metadata = json.loads(measurements.get_experiment_measurement(M_PATH_MAPPINGS))
    assert metadata == {
        K_CASE_SENSITIVE: os.path.normcase("A") != os.path.normcase("a"),
        K_LOCAL_SEPARATOR: os.path.sep,
        K_PATH_MAPPINGS: [list(mappings[0])],
        K_URL2PATHNAME_PACKAGE_NAME: urllib.request.url2pathname.__module__,
    }
    with monkeypatch.context() as patch:
        patch.setattr(measurements_module, "os", SimpleNamespace(path=ntpath))
        patch.setattr(urllib.request, "url2pathname", nturl2path.url2pathname)
        measurements.write_path_mappings(mappings)
    metadata = json.loads(measurements.get_experiment_measurement(M_PATH_MAPPINGS))
    assert metadata[K_URL2PATHNAME_PACKAGE_NAME] == "nturl2path"
    assert metadata[K_LOCAL_SEPARATOR] == "\\"
    assert metadata[K_CASE_SENSITIVE] is False
    assert metadata[K_PATH_MAPPINGS] == [list(mappings[0])]


@pytest.mark.parametrize("caller", ["measurements", "namesandtypes", "loaddata"])
@pytest.mark.parametrize("is_image", [False, True])
def test_eager_windows_to_posix(measurements, monkeypatch, caller, is_image):
    from cellprofiler_core.modules import loaddata
    from cellprofiler_core.modules.loaddata import LoadData
    from cellprofiler_core.modules.namesandtypes import (
        ASSIGN_RULES,
        LOAD_AS_OBJECTS,
        NamesAndTypes,
    )
    from cellprofiler_core.utilities import pathname

    name = "DNA"
    categories = (
        ("URL", "PathName", "FileName")
        if is_image
        else ("ObjectsURL", "ObjectsPathName", "ObjectsFileName")
    )
    source_path = LOCAL_ROOT + "\\" + FILENAME
    values = ("file:" + nturl2path.pathname2url(source_path), LOCAL_ROOT, FILENAME)
    for category, value in zip(categories, values):
        measurements.add_all_measurements("Image", category + "_" + name, [value])
    workspace = SimpleNamespace(measurements=measurements)

    if caller == "namesandtypes":
        module = NamesAndTypes()
        module.assignment_method.value = ASSIGN_RULES
        group = module.assignments[0]
        if is_image:
            group.image_name.value = name
        else:
            group.load_as_choice.value = LOAD_AS_OBJECTS
            group.object_name.value = name
    elif caller == "loaddata":
        module = LoadData()
        monkeypatch.setattr(
            module, "get_image_names", lambda: [name] if is_image else []
        )
        monkeypatch.setattr(
            module, "get_object_names", lambda: [] if is_image else [name]
        )

    seen = []

    def alter_path(path, **kwargs):
        seen.append(path)
        if path.lower().startswith(LOCAL_ROOT.lower()):
            path = REMOTE_ROOT + path[len(LOCAL_ROOT) :]
        return path.replace("\\", "/")

    # Emulate the author's Windows codecs and path classification without
    # replacing os.name globally or requiring Windows filesystem access.
    with monkeypatch.context() as patch:
        patch.setattr(urllib.request, "url2pathname", nturl2path.url2pathname)
        patch.setattr(urllib.request, "pathname2url", nturl2path.pathname2url)
        patch.setattr(
            pathname, "pathlib", SimpleNamespace(Path=pathlib.PureWindowsPath)
        )
        patch.setattr(loaddata, "os", SimpleNamespace(path=ntpath))
        if caller == "measurements":
            measurements.alter_path_for_create_batch(name, is_image, alter_path)
        else:
            module.prepare_to_create_batch(workspace, alter_path)
    assert source_path in seen
    expected_path = REMOTE_ROOT + "/" + FILENAME
    result = urllib.parse.urlsplit(measurements["Image", categories[0] + "_" + name])
    assert result.netloc == ""
    assert urllib.parse.unquote(result.path) == expected_path
    assert measurements["Image", categories[1] + "_" + name] == REMOTE_ROOT
    assert measurements["Image", categories[2] + "_" + name] == FILENAME


@pytest.mark.parametrize("target", ["posix"], indirect=True)
def test_posix_literal_backslash(measurements, target):
    set_mappings(measurements, [("/source/images", target.root)], windows=False)
    filename = "literal\\escape %20.tif"
    result = measurements.alter_url_post_create_batch(
        "file:" + urllib.parse.quote("/source/images/" + filename)
    )
    assert result == "file:" + target.encode(target.root + "/" + filename)
    # POSIX mappings remain case-sensitive.
    url = "file:" + urllib.parse.quote("/Source/images/" + filename)
    assert measurements.alter_url_post_create_batch(url) == url


def test_saved_windows_metadata_after_reopen(tmp_path, target, monkeypatch):
    filename = str(tmp_path / "saved-batch.h5")
    source_url = "file:" + nturl2path.pathname2url(LOCAL_ROOT + "\\" + FILENAME)
    m = Measurements(filename=filename)
    try:
        m.add_all_measurements("Image", "URL_DNA", [source_url])
        with monkeypatch.context() as patch:
            patch.setattr(measurements_module, "os", SimpleNamespace(path=ntpath))
            patch.setattr(urllib.request, "url2pathname", nturl2path.url2pathname)
            m.write_path_mappings([(LOCAL_ROOT, target.root)])
    finally:
        m.close()
    loaded = Measurements(filename=filename, mode="r")
    try:
        result = loaded.alter_url_post_create_batch(loaded["Image", "URL_DNA"])
        assert result == "file:" + target.encode(
            target.path.join(target.root, FILENAME)
        )
        assert json.loads(loaded.get_experiment_measurement(M_PATH_MAPPINGS))[
            K_PATH_MAPPINGS
        ] == [[LOCAL_ROOT, target.root]]
    finally:
        loaded.close()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Arbitrary POSIX filename bytes have no Windows filename equivalent",
)
def test_posix_undecodable_filename_bytes(measurements, tmp_path):
    filename = pathlib.Path(os.fsdecode(os.fsencode(tmp_path) + b"/image\xff.tif"))
    set_mappings(measurements, [("/source", str(tmp_path))], windows=False)
    result = measurements.alter_url_post_create_batch("file:/source/image%FF.tif")
    assert result == "file:" + urllib.request.pathname2url(str(filename))
    mapped_path = pathlib.Path(urllib.request.url2pathname(result[5:]))
    assert os.fsencode(mapped_path) == os.fsencode(tmp_path) + b"/image\xff.tif"


@pytest.mark.parametrize(
    "windows", [False, True], ids=["posix-source", "windows-source"]
)
@pytest.mark.parametrize("rule_type", ["overlap", "cascade"])
def test_ordered_rules_match_current_path(measurements, target, windows, rule_type):
    source_root = LOCAL_ROOT if windows else "/source/images"
    source_separator = "\\" if windows else "/"
    source_path = (
        source_root + source_separator + "sub" + source_separator + "image.tif"
    )
    encode = nturl2path.pathname2url if windows else urllib.parse.quote
    staging_root = target.path.join(target.root, "stage")
    final_root = target.path.join(target.root, "final")
    second_source = (
        source_root + source_separator + "sub"
        if rule_type == "overlap"
        else staging_root
    )
    set_mappings(
        measurements,
        [(source_root, staging_root), (second_source, final_root)],
        windows=windows,
    )
    result = measurements.alter_url_post_create_batch("file:" + encode(source_path))
    expected_root = staging_root if rule_type == "overlap" else final_root
    expected_path = target.path.join(expected_root, "sub", "image.tif")
    assert result == "file:" + target.encode(expected_path)
