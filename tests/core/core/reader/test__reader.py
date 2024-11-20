import pytest
import numpy
import skimage.io as io
from cellprofiler_core.reader import fill_readers, filter_active_readers
from cellprofiler_core.constants.reader import ALL_READERS, AVAILABLE_READERS
from cellprofiler_core.image import MonochromeImage
from cellprofiler_core.utilities.pathname import pathname2url


@pytest.fixture(scope="module")
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


class TestReaders:
    def test_uint8_tif_full_range(self, readers, tmp_path):
        img_ext = "tiff"
        img_name = "uint8_full_range"
        img_data = numpy.array([0,128,255,0], dtype=numpy.dtype("uint8")).reshape(2,2)

        for reader in readers:

            test_img = create_image(reader, img_data, tmp_path / f"{img_name}.{img_ext}", rescale=True)

            # ext not supported, skip testing reader for this image
            if test_img == None:
                continue

            test_img_data = test_img.pixel_data
            ref_img_data = (img_data / (2**8-1)).astype("float32")

            assert test_img_data.dtype == numpy.dtype("float32"), "dtype mismatch"
            assert test_img_data.shape == ref_img_data.shape, "shape mismatch"
            assert test_img_data.min() == ref_img_data.min(), "min mismatch"
            assert test_img_data.max() == ref_img_data.max(), "max mismatch"
            # test_img intensity values should range from 0 to 1
            assert numpy.all(test_img_data == ref_img_data), "data mismatch"

    def test_uint8_tif_partial_range(self, readers, tmp_path):
        img_ext = "tiff"
        img_name = "uint8_partial_range"
        img_data = numpy.array([64,65,127,128], dtype=numpy.dtype("uint8")).reshape(2,2)

        for reader in readers:

            test_img = create_image(reader, img_data, tmp_path / f"{img_name}.{img_ext}", rescale=True)

            # ext not supported, skip testing reader for this image
            if test_img == None:
                continue

            test_img_data = test_img.pixel_data
            ref_img_data = (img_data / (2**8-1)).astype("float32")

            assert test_img_data.dtype == numpy.dtype("float32"), "dtype mismatch"
            assert test_img_data.shape == ref_img_data.shape, "shape mismatch"
            assert test_img_data.min() == ref_img_data.min(), "min mismatch"
            assert test_img_data.max() == ref_img_data.max(), "max mismatch"
            # test_img intensity values should range from 64/255 to 128/255
            assert numpy.all(test_img_data == ref_img_data), "data mismatch"
