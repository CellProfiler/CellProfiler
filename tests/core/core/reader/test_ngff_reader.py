import functools
import http.server
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import zarr
from lxml import etree

from cellprofiler_core.constants.image import (
    MD_SERIES_NAME,
    MD_SIZE_C,
    MD_SIZE_S,
    MD_SIZE_T,
    MD_SIZE_X,
    MD_SIZE_Y,
    MD_SIZE_Z,
)
from cellprofiler_core.readers.ngff_reader import NGFFReader


@pytest.fixture(autouse=True)
def clear_cache():
    NGFFReader.clear_cached_readers()
    yield
    NGFFReader.clear_cached_readers()


@pytest.fixture(params=[2, 3], ids=["format2", "format3"])
def zarr_format(request):
    return request.param


def make_reader(path):
    url = path.resolve().as_uri() if isinstance(path, Path) else path
    return NGFFReader(SimpleNamespace(url=url))


def make_image(group, data, namespaced=False, axes=None):
    group.create_array(
        "0", data=data, chunks=tuple(min(4, size) for size in data.shape)
    )
    multiscales = [
        {"axes": axes or list("tczyx"[-data.ndim :]), "datasets": [{"path": "0"}]}
    ]
    if namespaced:
        group.attrs["ome"] = {"version": "0.5", "multiscales": multiscales}
    else:
        group.attrs["multiscales"] = multiscales


def exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in path.rglob("*")
        if file.is_file()
    }


def write_xml(
    path,
    images=(),
    maximum="4095",
    namespace="http://www.openmicroscopy.org/Schemas/OME/2016-06",
):
    document = etree.Element(f"{{{namespace}}}OME", nsmap={None: namespace})
    for identifier, name in images:
        attributes = {"ID": f"Image:{identifier}"}
        if name is not None:
            attributes["Name"] = name
        etree.SubElement(document, f"{{{namespace}}}Image", **attributes)
    annotation = etree.SubElement(document, f"{{{namespace}}}XMLAnnotation")
    value = etree.SubElement(annotation, f"{{{namespace}}}Value")
    original = etree.SubElement(value, f"{{{namespace}}}OriginalMetadata")
    etree.SubElement(original, f"{{{namespace}}}Key").text = "MaxSampleValue"
    etree.SubElement(original, f"{{{namespace}}}Value").text = maximum
    target = path / "OME" / "METADATA.ome.xml"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(etree.tostring(document))


@pytest.mark.parametrize("shape", [(7, 9), (3, 7, 9), (2, 3, 7, 9), (2, 2, 3, 7, 9)])
@pytest.mark.parametrize("consolidated", [False, True])
def test_planes_volumes_and_metadata(tmp_path, zarr_format, shape, consolidated):
    path = tmp_path / "image with spaces.zarr"
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    make_image(group, data, axes=[{"name": name} for name in "tczyx"[-len(shape) :]])
    if consolidated:
        zarr.consolidate_metadata(group.store)
    before = snapshot(path)
    reader = make_reader(path)
    metadata = reader.get_series_metadata()
    assert metadata == {
        MD_SIZE_S: 1,
        MD_SIZE_X: [9],
        MD_SIZE_Y: [7],
        MD_SIZE_Z: [shape[-3] if len(shape) >= 3 else 1],
        MD_SIZE_C: [shape[-4] if len(shape) >= 4 else 1],
        MD_SIZE_T: [shape[-5] if len(shape) == 5 else 1],
        MD_SERIES_NAME: [""],
    }
    cases = [({}, data), ({"xywh": (2, 1, 4, 3)}, data[..., 1:4, 2:6])]
    if len(shape) == 3:
        cases += [
            ({"z": 1}, data[1]),
            ({"z": 1, "xywh": (2, 1, 4, 3)}, data[1, 1:4, 2:6]),
        ]
    if len(shape) == 4:
        cases[0] = ({}, np.moveaxis(data, 0, -1))
        cases[1] = ({"xywh": (2, 1, 4, 3)}, np.moveaxis(data[..., 1:4, 2:6], 0, -1))
        cases += [
            ({"c": 1, "z": 1}, data[1, 1]),
            ({"c": 1}, data[1]),
            ({"z": 1}, np.moveaxis(data[:, 1], 0, -1)),
        ]
    if len(shape) == 5:
        cases[0] = ({}, np.moveaxis(data, 0, -1))
        cases[1] = ({"xywh": (2, 1, 4, 3)}, np.moveaxis(data[..., 1:4, 2:6], 0, -1))
        cases += [
            ({"t": 1, "c": 1, "z": 1}, data[1, 1, 1]),
            ({"t": 1, "c": 1}, data[1, 1]),
            ({"t": 1, "z": 1}, np.moveaxis(data[1, :, 1], 0, -1)),
        ]
    for parameters, expected in cases:
        exact(reader.read(**parameters), expected)
    volume = data if len(shape) < 4 else np.moveaxis(data, 0, -1)
    exact(reader.read_volume(z=1), volume)
    assert reader.get_reader().store.read_only
    reader.close()
    NGFFReader.clear_cached_readers()
    assert snapshot(path) == before


@pytest.mark.parametrize(
    "dtype", ["bool", "uint8", "uint16", "int16", "float32", "float64"]
)
def test_dtype_and_special_values(tmp_path, zarr_format, dtype):
    path = tmp_path / "values.zarr"
    data = np.arange(35).reshape(5, 7).astype(dtype)
    if np.issubdtype(data.dtype, np.floating):
        data[0, :5] = [0.0, -0.0, np.nan, np.inf, -np.inf]
    make_image(zarr.open_group(path, mode="w", zarr_format=zarr_format), data)
    exact(make_reader(path).read(), data)


@pytest.mark.parametrize("namespaced", [False, True])
def test_multiscales_select_full_resolution(tmp_path, zarr_format, namespaced):
    path = tmp_path / "scales.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    data = np.arange(63, dtype=np.uint16).reshape(7, 9)
    make_image(group, data, namespaced=namespaced)
    group.create_array("1", data=data[::2, ::2])
    metadata = dict(group.attrs["ome"]) if namespaced else dict(group.attrs)
    metadata["multiscales"][0]["datasets"].append({"path": "1"})
    group.attrs["ome" if namespaced else "multiscales"] = (
        metadata if namespaced else metadata["multiscales"]
    )
    reader = make_reader(path)
    assert reader.get_series_metadata()[MD_SIZE_S] == 1
    exact(reader.read(), data)
    scales = reader.find_arrays(reader.get_reader(), multiscales=True)
    assert [array.path for array, _ in scales] == ["0", "1"]
    exact(scales[1][0][:], data[::2, ::2])


def test_inner_group_cache_and_repeat_reads(tmp_path, zarr_format):
    path = tmp_path / "inner.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    first = np.full((5, 7), 11, dtype=np.uint16)
    second = np.full((5, 7), 29, dtype=np.uint16)
    make_image(group.create_group("a"), first)
    make_image(group.create_group("b"), second)
    a = make_reader(path.as_uri() + "/a")
    b = make_reader(path.as_uri() + "/b")
    for reader, expected in [
        (a, first),
        (b, second),
        (make_reader(path.as_uri() + "/a"), first),
    ]:
        assert reader.get_series_metadata()[MD_SIZE_S] == 1
        exact(reader.read(), expected)
        exact(reader.read(), expected)
    a.close()
    exact(a.read(), first)
    exact(b.read(), second)


def test_array_selection(tmp_path, zarr_format):
    path = tmp_path / "array.zarr"
    data = np.arange(35, dtype=np.float32).reshape(5, 7)
    zarr.create_array(store=str(path), data=data, zarr_format=zarr_format)
    reader = make_reader(path)
    assert reader.get_series_metadata()[MD_SIZE_S] == 1
    exact(reader.read(), data)
    child_path = tmp_path / "child.zarr"
    group = zarr.open_group(child_path, mode="w", zarr_format=zarr_format)
    group.create_array("pixels", data=data)
    selected = make_reader(child_path.as_uri() + "/pixels")
    exact(selected.read(), data)


def test_partial_read_does_not_truncate_metadata(tmp_path, zarr_format):
    path = tmp_path / "series.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    for name, value in [("2", 2), ("10", 10)]:
        make_image(group.create_group(name), np.full((5, 7), value, dtype=np.uint16))
    reader = make_reader(path)
    exact(reader.read(), np.full((5, 7), 10, dtype=np.uint16))
    assert reader.get_series_metadata()[MD_SERIES_NAME] == ["10", "2"]
    exact(reader.read(series=1), np.full((5, 7), 2, dtype=np.uint16))
    fresh = make_reader(path)
    exact(fresh.read(series=1), np.full((5, 7), 2, dtype=np.uint16))


@pytest.mark.parametrize(
    "maximum,expected", [("4095", 4095), ("not a number", None), (None, None)]
)
def test_xml_series_names_and_rescaling(
    tmp_path, zarr_format, maximum, expected, caplog
):
    path = tmp_path / "xml.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    for name in ["0", "1"]:
        make_image(
            group.create_group(name), np.full((5, 7), int(name) + 7, dtype=np.uint16)
        )
    group.create_group("OME").attrs["series"] = ["0", "1"]
    write_xml(
        path, [("1", "Second image"), ("0", None), ("9", "Missing")], maximum=maximum
    )
    before = snapshot(path)
    reader = make_reader(path)
    assert reader.get_series_metadata()[MD_SERIES_NAME] == ["Second image", "0"]
    pixels, scale = reader.read(wants_metadata_rescale=True)
    exact(pixels, np.full((5, 7), 8, dtype=np.uint16))
    assert scale == ((0, expected) if expected is not None else None)
    assert "was absent" in caplog.text
    assert snapshot(path) == before


def test_ome05_series_order(tmp_path):
    path = tmp_path / "ome05.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.attrs["ome"] = {"version": "0.5", "bioformats2raw.layout": 3}
    for name in ["0", "1"]:
        make_image(
            group.create_group(name),
            np.full((5, 7), int(name), dtype=np.uint16),
            namespaced=True,
        )
    group.create_group("OME").attrs["ome"] = {"version": "0.5", "series": ["1", "0"]}
    reader = make_reader(path)
    assert reader.get_series_metadata()[MD_SIZE_S] == 2
    exact(reader.read(), np.ones((5, 7), dtype=np.uint16))
    exact(reader.read(series=1), np.zeros((5, 7), dtype=np.uint16))


@pytest.mark.parametrize("namespaced", [False, True])
def test_plate_and_well_order(tmp_path, zarr_format, namespaced):
    path = tmp_path / "plate.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    fields = ["last", "first"] if namespaced else ["0", "1"]
    wells = [
        {
            "path": "B/2",
            "rowIndex" if namespaced else "row_index": 1,
            "columnIndex" if namespaced else "column_index": 1,
        },
        {
            "path": "A/1",
            "rowIndex" if namespaced else "row_index": 0,
            "columnIndex" if namespaced else "column_index": 0,
        },
    ]
    plate = {
        "wells": wells,
        "rows": [{"name": "A"}, {"name": "B"}],
        "columns": [{"name": "1"}, {"name": "2"}],
    }
    if not namespaced:
        plate["field_count"] = 2
    group.attrs["ome" if namespaced else "plate"] = (
        {"version": "0.5", "plate": plate} if namespaced else plate
    )
    for well_index, entry in enumerate(wells):
        well = group.require_group(entry["path"])
        well_info = {"images": [{"path": field} for field in fields]}
        well.attrs["ome" if namespaced else "well"] = (
            {"version": "0.5", "well": well_info} if namespaced else well_info
        )
        for field_index, field in enumerate(fields):
            make_image(
                well.create_group(field),
                np.full((5, 7), 10 * well_index + field_index, dtype=np.uint16),
                namespaced=namespaced,
            )
    reader = make_reader(path)
    assert reader.get_series_metadata()[MD_SERIES_NAME] == [
        f"r{row}c{column}f{field}"
        for row, column in [("B", "2"), ("A", "1")]
        for field in fields
    ]
    for series, value in enumerate([0, 1, 10, 11]):
        exact(reader.read(series=series), np.full((5, 7), value, dtype=np.uint16))


def test_cache_cleanup_and_open_failure(tmp_path, zarr_format, monkeypatch):
    path = tmp_path / "cache.zarr"
    data = np.arange(35, dtype=np.uint16).reshape(5, 7)
    make_image(zarr.open_group(path, mode="w", zarr_format=zarr_format), data)
    reader = make_reader(path)
    store = reader.get_reader().store
    calls = []
    close = store.close
    monkeypatch.setattr(store, "close", lambda: (calls.append(True), close())[-1])
    reader.close()
    assert not calls
    NGFFReader.clear_cached_readers()
    assert calls == [True]
    exact(reader.read(), data)
    assert reader.get_reader().store is not store
    bad = tmp_path / "bad.zarr"
    bad.mkdir()
    with pytest.raises(zarr.errors.GroupNotFoundError):
        make_reader(bad).get_reader()
    assert str(bad.as_uri()) not in NGFFReader.ZARR_READER_CACHE


def test_errors_and_missing_xml(tmp_path, zarr_format):
    with pytest.raises(FileNotFoundError):
        make_reader(tmp_path / "missing.zarr").read()
    path = tmp_path / "empty.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    reader = make_reader(path)
    assert reader.get_series_metadata()[MD_SIZE_S] == 0
    with pytest.raises(IOError, match="no valid series"):
        reader.read()
    make_image(group, np.zeros((5, 7), dtype=np.uint8))
    NGFFReader.clear_cached_readers()
    pixels, scale = reader.read(wants_metadata_rescale=True)
    assert scale is None
    exact(pixels, np.zeros((5, 7), dtype=np.uint8))
    with pytest.raises(KeyError):
        make_reader(path.as_uri() + "/missing").read()
    with pytest.raises(IndexError):
        reader.read(series=3)


@pytest.mark.parametrize("invalid_xml", [b"<OME", b'<OME xmlns="wrong"/>'])
def test_invalid_xml_is_visible(tmp_path, zarr_format, invalid_xml):
    path = tmp_path / "invalid.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    make_image(group.create_group("0"), np.zeros((5, 7), dtype=np.uint8))
    target = path / "OME"
    target.mkdir()
    (target / "METADATA.ome.xml").write_bytes(invalid_xml)
    with pytest.raises((etree.XMLSyntaxError, ValueError)):
        make_reader(path).get_series_metadata()


@pytest.fixture
def http_store(tmp_path):
    class StorageHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            query = parse_qs(request.query)
            if query.get("list-type") != ["2"]:
                return super().do_GET()
            bucket = tmp_path / request.path.strip("/")
            prefix = query.get("prefix", [""])[0]
            document = etree.Element(
                "ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/"
            )
            etree.SubElement(document, "Name").text = bucket.name
            etree.SubElement(document, "Prefix").text = prefix
            etree.SubElement(document, "IsTruncated").text = "false"
            keys = [
                path
                for path in bucket.rglob("*")
                if path.is_file()
                and path.relative_to(bucket).as_posix().startswith(prefix)
            ]
            etree.SubElement(document, "KeyCount").text = str(len(keys))
            for path in keys:
                item = etree.SubElement(document, "Contents")
                etree.SubElement(item, "Key").text = path.relative_to(bucket).as_posix()
                etree.SubElement(item, "Size").text = str(path.stat().st_size)
                etree.SubElement(item, "ETag").text = '"ngff-fixture"'
                etree.SubElement(item, "LastModified").text = "2026-01-01T00:00:00.000Z"
                etree.SubElement(item, "StorageClass").text = "STANDARD"
            payload = etree.tostring(document)
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    handler = functools.partial(StorageHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.parametrize("consolidated", [False, True])
def test_remote_http_data_xml_and_read_only(http_store, zarr_format, consolidated):
    directory, url = http_store
    path = directory / "remote.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    data = np.arange(105, dtype=np.uint16).reshape(3, 5, 7)
    make_image(group.create_group("0"), data)
    write_xml(path, [("0", "Remote series")])
    if consolidated:
        zarr.consolidate_metadata(group.store)
    before = snapshot(path)
    reader = make_reader(url + "/remote.zarr")
    assert reader.get_series_metadata()[MD_SERIES_NAME] == ["Remote series"]
    pixels, maximum = reader.read(z=1, wants_metadata_rescale=True)
    exact(pixels, data[1])
    assert maximum == (0, 4095)
    exact(reader.read_volume(), data)
    assert reader.get_reader().store.read_only
    reader.close()
    NGFFReader.clear_cached_readers()
    assert snapshot(path) == before


def test_remote_missing_metadata(http_store, zarr_format):
    directory, url = http_store
    path = directory / "scan.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    data = np.arange(35, dtype=np.uint8).reshape(5, 7)
    make_image(group.create_group("0"), data)
    reader = make_reader(url + "/scan.zarr")
    assert reader.get_series_metadata()[MD_SERIES_NAME] == ["0"]
    pixels, maximum = reader.read(wants_metadata_rescale=True)
    exact(pixels, data)
    assert maximum is None


@pytest.mark.parametrize("dimensions", [1, 6])
def test_unsupported_dimensions(tmp_path, zarr_format, dimensions):
    path = tmp_path / "dimensions.zarr"
    zarr.create_array(
        store=str(path),
        data=np.zeros((2,) * dimensions, dtype=np.uint8),
        zarr_format=zarr_format,
    )
    with pytest.raises(NotImplementedError, match="Unsupported dimensionality"):
        make_reader(path).read()


def test_s3_backend_with_local_endpoint(http_store, zarr_format, monkeypatch):
    import s3fs

    directory, endpoint = http_store
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", endpoint)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ngff-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ngff-test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    s3fs.S3FileSystem.clear_instance_cache()
    path = directory / "bucket" / "image.zarr"
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    data = np.arange(35, dtype=np.uint16).reshape(5, 7)
    make_image(group.create_group("0"), data)
    write_xml(path, [("0", "S3 image")])
    zarr.consolidate_metadata(group.store)
    before = snapshot(path)
    try:
        reader = make_reader("s3://bucket/image.zarr")
        assert reader.get_series_metadata()[MD_SERIES_NAME] == ["S3 image"]
        pixels, maximum = reader.read(wants_metadata_rescale=True)
        exact(pixels, data)
        assert maximum == (0, 4095)
        assert snapshot(path) == before
    finally:
        NGFFReader.clear_cached_readers()
        s3fs.S3FileSystem.clear_instance_cache()


@pytest.mark.parametrize("metadata_rescale", [False, True])
@pytest.mark.parametrize("volume", [False, True])
def test_image_provider_rescaling(
    tmp_path, zarr_format, metadata_rescale, volume, monkeypatch
):
    from cellprofiler_core.constants.reader import AVAILABLE_READERS
    from cellprofiler_core.image import URLImage

    monkeypatch.setitem(AVAILABLE_READERS, NGFFReader.reader_name, NGFFReader)
    path = tmp_path / "provider.zarr"
    data = (
        np.arange(105, dtype=np.uint16).reshape(3, 5, 7) * 39
        if volume
        else np.arange(35, dtype=np.uint16).reshape(5, 7) * 113
    )
    make_image(zarr.open_group(path, mode="w", zarr_format=zarr_format), data)
    write_xml(path)
    provider = URLImage(
        "Input", path.as_uri(), metadata_rescale=metadata_rescale, volume=volume
    )
    try:
        actual = provider.provide_image(None).pixel_data
        expected = (
            data.astype(np.float64) / (4095 if metadata_rescale else 65535)
        ).astype(np.float32)
        exact(actual, expected)
    finally:
        provider.release_memory()
