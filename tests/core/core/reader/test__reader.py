import pytest
import numpy
import skimage.io as io
from cellprofiler_core.reader import fill_readers, filter_active_readers
from cellprofiler_core.constants.reader import ALL_READERS, AVAILABLE_READERS
from cellprofiler_core.image import MonochromeImage
from cellprofiler_core.utilities.pathname import pathname2url


@pytest.fixture(scope="class")
def readers():
    # init readers
    fill_readers()
    # do not test GCS Reader
    filter_active_readers(
        [x for x in ALL_READERS.keys() if "Google Cloud Storage" not in x],
        by_module_name=False)
    # post-filtered readers
    return AVAILABLE_READERS

def create_image(reader, img_data, path, rescale=True):
    io.imsave(path, img_data)

    provider = MonochromeImage(
        path.stem, # name
        pathname2url(str(path)), # url
        0, # series
        None, # index
        None, # channel
        rescale=rescale,
        volume=False,
        spacing=None,
        z=None,
        t=None
    )

    provider._FileImage__preferred_reader = reader

    # check if image reader supports ext
    image_file = provider.get_image_file()
    if ALL_READERS[reader].supports_format(image_file) == -1:
        return None

    return provider.provide_image(None) # img.pixel_data for array

# used to name the test something legible
# https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures
def idfn(fixture_value):
    return fixture_value["img_name"]

@pytest.fixture(scope="class", ids=idfn, params=[
    {
        "img_ext": "tiff",
        "img_name": "uint8_full_range",
        "dtype": "uint8",
        "divisor": numpy.float32(2**8-1),
        "shift": 0,
        "start": numpy.iinfo("uint8").min,
        "stop": numpy.iinfo("uint8").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint8_partial_range",
        "dtype": "uint8",
        "divisor": numpy.float32(2**8-1),
        "shift": 0,
        "start": 64,
        "stop": 128,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint8_smallest_range",
        "dtype": "uint8",
        "divisor": numpy.float32(2**8-1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "int8_full_range",
        "dtype": "int8",
        "divisor": numpy.float32(2**8-1),
        "shift": numpy.iinfo("int8").min,
        "start": numpy.iinfo("int8").min,
        "stop": numpy.iinfo("int8").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "int8_partial_range",
        "dtype": "int8",
        "divisor": numpy.float32(2**8-1),
        "shift": numpy.iinfo("int8").min,
        "start": -10,
        "stop": 10,
    },
    {
        "img_ext": "tiff",
        "img_name": "int8_smallest_range",
        "dtype": "int8",
        "divisor": numpy.float32(2**8-1),
        "shift": numpy.iinfo("int8").min,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint16_full_range",
        "dtype": "uint16",
        "divisor": numpy.float32(2**16-1),
        "shift": 0,
        "start": numpy.iinfo("uint16").min,
        "stop": numpy.iinfo("uint16").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint16_partial_range",
        "dtype": "uint16",
        "divisor": numpy.float32(2**16-1),
        "shift": 0,
        "start": 255,
        "stop": 10_000,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint16_smallest_range",
        "dtype": "uint16",
        "divisor": numpy.float32(2**16-1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "int16_full_range",
        "dtype": "int16",
        "divisor": numpy.float32(2**16-1),
        "shift": numpy.iinfo("int16").min,
        "start": numpy.iinfo("int16").min,
        "stop": numpy.iinfo("int16").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "int16_partial_range",
        "dtype": "int16",
        "divisor": numpy.float32(2**16-1),
        "shift": numpy.iinfo("int16").min,
        "start": -10,
        "stop": 10,
    },
    {
        "img_ext": "tiff",
        "img_name": "int16_smallest_range",
        "dtype": "int16",
        "divisor": numpy.float32(2**16-1),
        "shift": numpy.iinfo("int16").min,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint32_full_range",
        "dtype": "uint32",
        "divisor": numpy.float32(2**32-1),
        "shift": 0,
        "start": numpy.iinfo("uint32").min,
        "stop": numpy.iinfo("uint32").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint32_partial_range",
        "dtype": "uint32",
        "divisor": numpy.float32(2**32-1),
        "shift": 0,
        "start": 65535,
        "stop": 1_000_000_000,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint32_smallest_range",
        "dtype": "uint32",
        "divisor": numpy.float32(2**32-1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "int32_full_range",
        "dtype": "int32",
        "divisor": numpy.float32(2**32-1),
        "shift": numpy.iinfo("int32").min,
        "start": numpy.iinfo("int32").min,
        "stop": numpy.iinfo("int32").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "int32_partial_range",
        "dtype": "int32",
        "divisor": numpy.float32(2**32-1),
        "shift": numpy.iinfo("int32").min,
        "start": -65_536,
        "stop": 65_536,
    },
    {
        "img_ext": "tiff",
        "img_name": "int32_smallest_range",
        "dtype": "int32",
        "divisor": numpy.float32(2**32-1),
        "shift": numpy.iinfo("int32").min,
        "start": 0,
        "stop": 1,
    },
])
def monochrome_image(request):
    shift = request.param["shift"]
    start = request.param["start"]
    stop = request.param["stop"]
    dtype = request.param["dtype"]
    divisor = request.param["divisor"]
    img_name = request.param["img_name"]
    img_ext = request.param["img_ext"]

    img_data = numpy.linspace(start, stop, num=4, dtype=numpy.dtype(dtype)).reshape(2,2)

    return img_data, img_ext, img_name, dtype, divisor, shift, start, stop


class TestReaders:
    def test_integer_types(self, monochrome_image, readers, tmp_path):
        img_data, img_ext, img_name, dtype, divisor, shift, start, stop = monochrome_image

        for reader in readers:

            test_img = create_image(reader, img_data, tmp_path / f"{img_name}.{img_ext}", rescale=True)

            # ext not supported, skip testing reader for this image
            if test_img == None:
                continue

            test_img_data = test_img.pixel_data
            ref_img_data = ((img_data - shift) / divisor).astype("float32")

            assert test_img_data.dtype == numpy.dtype("float32"), "dtype mismatch"
            assert test_img_data.shape == ref_img_data.shape, "shape mismatch"
            assert test_img_data.min() == numpy.float32((start - shift) / divisor), "min mismatch"
            assert test_img_data.max() == numpy.float32((stop - shift) / divisor), "max mismatch"
            assert numpy.all(test_img_data == ref_img_data), "data mismatch"
            assert numpy.all((test_img_data * divisor + shift).astype(dtype) == img_data), "precision lost"
