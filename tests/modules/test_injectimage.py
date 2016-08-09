import unittest

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.injectimage
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

class testInjectImage(unittest.TestCase):
    def test_00_00_init(self):
        image = numpy.zeros((10, 10), dtype=float)
        x = cellprofiler.modules.injectimage.InjectImage("my_image", image)

    def test_01_01_get_from_image_set(self):
        image = numpy.zeros((10, 10), dtype=float)
        ii = cellprofiler.modules.injectimage.InjectImage("my_image", image)
        pipeline = cellprofiler.pipeline.Pipeline()
        measurements = cellprofiler.measurement.Measurements()
        workspace = cellprofiler.workspace.Workspace(pipeline, ii, measurements, None,
                                                     measurements,
                                                     cellprofiler.image.ImageSetList())
        ii.prepare_run(workspace)
        ii.prepare_group(workspace, {}, [1])
        ii.run(workspace)
        image_set = workspace.image_set
        self.assertTrue(image_set, "No image set returned from ImageSetList.GetImageSet")
        my_image = image_set.get_image("my_image")
        self.assertTrue(my_image, "No image returned from ImageSet.GetImage")
        self.assertEqual(my_image.image.shape[0], 10, "Wrong image shape")


if __name__ == "main":
    unittest.main()
