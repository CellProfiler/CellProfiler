import pytest
import numpy
import skimage.io as io
from cellprofiler_core.reader import fill_readers, filter_active_readers
from cellprofiler_core.constants.reader import ALL_READERS, AVAILABLE_READERS
from cellprofiler_core.image import MonochromeImage
from cellprofiler_core.utilities.pathname import pathname2url


def readers():
    # init readers
    fill_readers()
    # do not test GCS Reader
    filter_active_readers(
        [x for x in ALL_READERS.keys() if "Google Cloud Storage" not in x],
        by_module_name=False)
    # post-filtered readers
    return AVAILABLE_READERS

# used to name the test something legible
# https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures
def idfn(fixture_value):
    return fixture_value["img_name"]

@pytest.fixture(scope="function", ids=idfn, params=[
    # 8 bit images
    {
        "img_ext": "tiff",
        "img_name": "uint8_full_range",
        "dtype": "uint8",
        "divisor": numpy.float64(2**8-1),
        "shift": 0,
        "start": numpy.iinfo("uint8").min,
        "stop": numpy.iinfo("uint8").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint8_partial_range",
        "dtype": "uint8",
        "divisor": numpy.float64(2**8-1),
        "shift": 0,
        "start": 64,
        "stop": 128,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint8_smallest_range",
        "dtype": "uint8",
        "divisor": numpy.float64(2**8-1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "int8_full_range",
        "dtype": "int8",
        "divisor": numpy.float64(2**8-1),
        "shift": numpy.iinfo("int8").min,
        "start": numpy.iinfo("int8").min,
        "stop": numpy.iinfo("int8").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "int8_partial_range",
        "dtype": "int8",
        "divisor": numpy.float64(2**8-1),
        "shift": numpy.iinfo("int8").min,
        "start": -10,
        "stop": 10,
    },
    {
        "img_ext": "tiff",
        "img_name": "int8_smallest_range",
        "dtype": "int8",
        "divisor": numpy.float64(2**8-1),
        "shift": numpy.iinfo("int8").min,
        "start": 0,
        "stop": 1,
    },
    # 16 bit integer images
    {
        "img_ext": "tiff",
        "img_name": "uint16_full_range",
        "dtype": "uint16",
        "divisor": numpy.float64(2**16-1),
        "shift": 0,
        "start": numpy.iinfo("uint16").min,
        "stop": numpy.iinfo("uint16").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint16_partial_range",
        "dtype": "uint16",
        "divisor": numpy.float64(2**16-1),
        "shift": 0,
        "start": 255,
        "stop": 10_000,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint16_smallest_range",
        "dtype": "uint16",
        "divisor": numpy.float64(2**16-1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "int16_full_range",
        "dtype": "int16",
        "divisor": numpy.float64(2**16-1),
        "shift": numpy.iinfo("int16").min,
        "start": numpy.iinfo("int16").min,
        "stop": numpy.iinfo("int16").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "int16_partial_range",
        "dtype": "int16",
        "divisor": numpy.float64(2**16-1),
        "shift": numpy.iinfo("int16").min,
        "start": -10,
        "stop": 10,
    },
    {
        "img_ext": "tiff",
        "img_name": "int16_smallest_range",
        "dtype": "int16",
        "divisor": numpy.float64(2**16-1),
        "shift": numpy.iinfo("int16").min,
        "start": 0,
        "stop": 1,
    },
    # 32 bit integer images
    {
        "img_ext": "tiff",
        "img_name": "uint32_full_range",
        "dtype": "uint32",
        "divisor": numpy.float64(2**32-1),
        "shift": 0,
        "start": numpy.iinfo("uint32").min,
        "stop": numpy.iinfo("uint32").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint32_partial_range",
        "dtype": "uint32",
        "divisor": numpy.float64(2**32-1),
        "shift": 0,
        "start": 65535,
        "stop": 1_000_000_000,
    },
    {
        "img_ext": "tiff",
        "img_name": "uint32_smallest_range",
        "dtype": "uint32",
        "divisor": numpy.float64(2**32-1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "int32_full_range",
        "dtype": "int32",
        "divisor": numpy.float64(2**32-1),
        "shift": numpy.iinfo("int32").min,
        "start": numpy.iinfo("int32").min,
        "stop": numpy.iinfo("int32").max,
    },
    {
        "img_ext": "tiff",
        "img_name": "int32_partial_range",
        "dtype": "int32",
        "divisor": numpy.float64(2**32-1),
        "shift": numpy.iinfo("int32").min,
        "start": -65_536,
        "stop": 65_536,
    },
    {
        "img_ext": "tiff",
        "img_name": "int32_smallest_range",
        "dtype": "int32",
        "divisor": numpy.float64(2**32-1),
        "shift": numpy.iinfo("int32").min,
        "start": 0,
        "stop": 1,
    },
    # 16 bit float types
    {
        "img_ext": "tiff",
        "img_name": "float16_zero_one_full_range",
        "dtype": "float16",
        "divisor": numpy.float64(1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "float16_zero_one_partial_range",
        "dtype": "float16",
        "divisor": numpy.float64(1),
        "shift": 0,
        "start": 0.25,
        "stop": 0.75,
    },
    {
        "img_ext": "tiff",
        "img_name": "float16_zero_one_smallest_range",
        "dtype": "float16",
        "divisor": numpy.float64(1),
        "shift": 0,
        "start": 0.,
        "stop": numpy.finfo('float16').smallest_subnormal, # smallest positive float16 = 2**(-24)
    },
    {
        "img_ext": "tiff",
        "img_name": "float16_neg_one_pos_one",
        "dtype": "float16",
        "divisor": numpy.float64(2),
        "shift": -1,
        "start": -1.,
        "stop": 1.,
    },
    {
        "img_ext": "tiff",
        "img_name": "float16_uintlike_full_range",
        "dtype": "float16",
        "divisor": numpy.float64(2**16-1),
        "shift": 0,
        "start": 0.,
        "stop": float(numpy.finfo("float16").max),
    },
    {
        "img_ext": "tiff",
        "img_name": "float16_intlike_partial_range",
        "dtype": "float16",
        "divisor": numpy.float64(2**16-1),
        "shift": int(numpy.iinfo("int16").min),
        "start": -10000.,
        "stop": 10000.
    },
    # 32 bit float types
    {
        "img_ext": "tiff",
        "img_name": "float32_zero_one_full_range",
        "dtype": "float32",
        "divisor": numpy.float64(1),
        "shift": 0,
        "start": 0,
        "stop": 1,
    },
    {
        "img_ext": "tiff",
        "img_name": "float32_zero_one_partial_range",
        "dtype": "float32",
        "divisor": numpy.float64(1),
        "shift": 0,
        "start": 0.25,
        "stop": 0.75,
    },
    {
        "img_ext": "tiff",
        "img_name": "float32_zero_one_smallest_range",
        "dtype": "float32",
        "divisor": numpy.float64(1),
        "shift": 0,
        "start": 0.,
        "stop": numpy.finfo('float32').smallest_subnormal # smallest positive float32 = 2**(-149)
    },
    {
        "img_ext": "tiff",
        "img_name": "float32_neg_one_pos_one",
        "dtype": "float32",
        "divisor": numpy.float64(2),
        "shift": -1,
        "start": -1.,
        "stop": 1.,
    },
    {
        "img_ext": "tiff",
        "img_name": "float32_uintlike_full_range",
        "dtype": "float32",
        "divisor": numpy.float64(2**32-1),
        "shift": 0,
        "start": 0.,
        # float32 goes way beyond the max uint32 value
        # but it starts to get so sparse, we don't define values for the
        # higher ranges of uintlike
        "stop": 4000000000.,
    },
    {
        "img_ext": "tiff",
        "img_name": "float32_intlike_partial_range",
        "dtype": "float32",
        "divisor": numpy.float64(2**32-1),
        "shift": numpy.iinfo("int32").min,
        "start": -2000000000.,
        "stop": 2000000000.,
    },
])
def img_details(request, tmp_path):
    shift = request.param["shift"]
    start = request.param["start"]
    stop = request.param["stop"]
    dtype = request.param["dtype"]
    divisor = request.param["divisor"]
    img_name = request.param["img_name"]
    img_ext = request.param["img_ext"]

    n_data_root = 20
    if dtype.startswith("float") and ("intlike" in img_name or "uintlike" in img_name):
        img_data = numpy.trunc(
            numpy.linspace(
                start, stop, num=n_data_root**2, dtype=numpy.dtype(dtype)
            ).reshape(n_data_root,n_data_root)
        )
    else:
        img_data = numpy.linspace(
            start, stop, num=n_data_root**2, dtype=numpy.dtype(dtype)
        ).reshape(n_data_root,n_data_root)

    path = tmp_path / f"{img_name}.{img_ext}"

    io.imsave(path, img_data)

    yield img_data, path, dtype, divisor, shift, start, stop

    # nothing to cleanup - pytest only keeps last 3 tmpt_pth directories

def create_image(reader, path, metadata_rescale=False, rescale_range=None):
    provider = MonochromeImage(
        path.stem, # name
        pathname2url(str(path)), # url
        0, # series
        None, # index
        None, # channel
        metadata_rescale=metadata_rescale,
        rescale_range=rescale_range,
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

class TestReaders:
    @pytest.mark.parametrize("reader", readers())
    def test_autoscale(self, img_details, reader):
        img_data, path, dtype, divisor, shift, start, stop = img_details

        # Bio-Formats only supports FLOAT (float32) and DOUBLE (float64)
        # it will think float16 is FLOAT, and try to process it as such
        if dtype == "float16" and reader == "Bio-Formats":
            return

        test_img = create_image(reader, path, metadata_rescale=False, rescale_range=None)

        # ext not supported, skip testing reader for this image
        if test_img == None:
            return

        test_img_data = test_img.pixel_data
        # must cast up to float64 to avoid overflow
        # e.g. if img_data is int8, and shift is 255, then result of img_data - shift stays int8
        # and you have negative values in result, which is what we're trying to avoid
        ref_img_data = ((img_data.astype("float64") - shift) / divisor).astype("float32")
        ref_img_min = numpy.float32((start - shift) / divisor)
        ref_img_max = numpy.float32((stop - shift) / divisor)


        assert test_img_data.dtype == numpy.dtype("float32"), "dtype mismatch"
        assert test_img_data.shape == ref_img_data.shape, "shape mismatch"
        assert test_img_data.min() == ref_img_min, "min mismatch"
        assert test_img_data.max() == ref_img_max, "max mismatch"
        assert numpy.all(test_img_data == ref_img_data), "data mismatch"

        unscaled_test_img_data = (test_img_data.astype('float64') * divisor + shift) # .astype(dtype)
        if numpy.issubdtype(dtype, numpy.integer):
            assert numpy.allclose(
                unscaled_test_img_data,
                # cast up to float64 to avoid overflow
                img_data.astype('float64'), atol=128., rtol=2**(-24)
                # img_data.astype('float64'), atol=1, rtol=2**(-18)
            ), "precision mismatch"

    @pytest.mark.parametrize("reader", readers())
    def test_manualscale(self, img_details, reader):
        img_data, path, dtype, divisor, shift, start, stop = img_details
        start = float(start)
        stop = float(stop)
        shift = start
        divisor = stop - start

        # Bio-Formats only supports FLOAT (float32) and DOUBLE (float64)
        # it will think float16 is FLOAT, and try to process it as such
        if dtype == "float16" and reader == "Bio-Formats":
            return

        test_img = create_image(reader, path, metadata_rescale=False, rescale_range=(float(start), float(stop)))

        # ext not supported, skip testing reader for this image
        if test_img == None:
            return

        test_img_data = test_img.pixel_data
        # must cast up to float64 to avoid overflow
        # e.g. if img_data is int8, and shift is 255, then result of img_data - shift stays int8
        # and you have negative values in result, which is what we're trying to avoid
        ref_img_data = ((img_data.astype("float64") - shift) / divisor).astype("float32")
        ref_img_min = numpy.float32((start - shift) / divisor)
        ref_img_max = numpy.float32((stop - shift) / divisor)


        assert test_img_data.dtype == numpy.dtype("float32"), "dtype mismatch"
        assert test_img_data.shape == ref_img_data.shape, "shape mismatch"
        assert test_img_data.min() == ref_img_min, "min mismatch"
        assert test_img_data.max() == ref_img_max, "max mismatch"
        assert numpy.all(test_img_data == ref_img_data), "data mismatch"

        unscaled_test_img_data = (test_img_data.astype('float64') * divisor + shift) # .astype(dtype)
        if numpy.issubdtype(dtype, numpy.integer):
            assert numpy.allclose(
                unscaled_test_img_data,
                # cast up to float64 to avoid overflow
                img_data.astype('float64'), atol=128., rtol=2**(-24)
                # img_data.astype('float64'), atol=1, rtol=2**(-18)
            ), "precision mismatch"

