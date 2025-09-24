import numpy
from pydantic import ValidationError, validate_call, ConfigDict, Field
from typing import Annotated
from cellprofiler_library.types import Image2DColor

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def sum_up_im(im: Annotated[Image2DColor, Field(description="2D image with multiple channels of type float32")]) -> float:
    return im.sum()

def test_input_type_valid_throws_no_error():
    try:
        sum_up_im(numpy.ones((10, 10, 10)).astype(numpy.float32))
        assert True
    except Exception:
        assert False


def test_input_type_invalid_throws_error():
    try:
        sum_up_im(numpy.ones((10, 10)).astype(numpy.float32))
        assert False
    except ValidationError as e:
        assert e.errors()[0]['ctx']['error'].message == 'CellProfiler Input Validation Error: Expected a 3D array (cyx),got 2D', 'Error message is not as expected'