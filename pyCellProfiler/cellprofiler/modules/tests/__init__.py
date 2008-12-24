"""Directory for tests of individual modules

"""
import os
import unittest

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
    
if __name__ == "__main__":
    import nose
    
    nose.main()
