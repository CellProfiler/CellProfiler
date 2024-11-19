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

def create_provider(img_data, path, rescale=True):
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
    return provider

class TestReaders:
    def test_simple(self, readers, tmp_path):
        img_data = numpy.array([0,0,0,0], dtype=numpy.dtype("uint8")).reshape(2,2)
        provider = create_provider(img_data, tmp_path / "uint8_all_zero.tiff", rescale=True)
        img = provider.provide_image(None) # img.pixel_data for array
        print("Did the thing")