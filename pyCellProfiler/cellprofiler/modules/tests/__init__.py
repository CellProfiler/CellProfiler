"""Directory for tests of individual modules

"""
import base64
import os
import unittest
import tempfile

import scipy.io.matlab.mio

def example_images_directory():
    fyle = os.path.abspath(__file__)
    d = os.path.split(fyle)[0] # CellProfiler.Modules.tests
    d = os.path.split(d)[0]        # CellProfiler.Modules
    d = os.path.split(d)[0]        # CellProfiler
    d = os.path.split(d)[0]        # pyCellProfiler
    d = os.path.split(d)[0]        # CellProfiler
    d = os.path.split(d)[0]        # either trunk or build directory
    for imagedir in ["CP-CPEXAMPLEIMAGES","ExampleImages"]:
        path = os.path.join(d,imagedir)
        if os.path.exists(path):
            return path
    return None

class testExampleImagesDirectory(unittest.TestCase):
    def test_00_00_got_something(self):
        self.assertTrue(example_images_directory(), "You need to have the example images checked out to run these tests")

def load_pipeline(test_case, encoded_data):
    """Load a pipeline from base-64 encoded data
    
    test_case - an instance of unittest.TestCase
    encoded_data - a pipeline encoded using base-64
    The magic incantation to do the above is the following:
    import base64
    fd = open('my_PIPE.mat')
    bindata = fd.read()
    fd.close()
    b64data = base64.b64encode(bindata)
    """
    import cellprofiler.pipeline

    (matfd,matpath) = tempfile.mkstemp('.mat')
    matfh = os.fdopen(matfd,'wb')
    try:
        data = base64.b64decode(encoded_data)
        matfh.write(data)
        matfh.flush()
        pipeline = cellprofiler.pipeline.Pipeline()
        handles=scipy.io.matlab.mio.loadmat(matpath, struct_as_record=True)
    finally:
        matfh.close()
    def blowup(pipeline,event):
        if isinstance(event,cellprofiler.pipeline.RunExceptionEvent):
            test_case.assertFalse(event.error.message)
    pipeline.add_listener(blowup)
    pipeline.create_from_handles(handles)
    return pipeline
    
if __name__ == "__main__":
    import nose
    
    nose.main()
