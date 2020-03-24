import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.modules.injectimage
import cellprofiler_core.pipeline
import cellprofiler_core.workspace


def test_init():
    image = numpy.zeros((10, 10), dtype=float)
    x = cellprofiler_core.modules.injectimage.InjectImage("my_image", image)


def test_get_from_image_set():
    image = numpy.zeros((10, 10), dtype=float)
    ii = cellprofiler_core.modules.injectimage.InjectImage("my_image", image)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    measurements = cellprofiler_core.measurement.Measurements()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, ii, measurements, None, measurements, cellprofiler_core.image.Image()
    )
    ii.prepare_run(workspace)
    ii.prepare_group(workspace, {}, [1])
    ii.run(workspace)
    image_set = workspace.image_set
    assert image_set, "No image set returned from ImageSetList.GetImageSet"
    my_image = image_set.get_image("my_image")
    assert my_image, "No image returned from ImageSet.GetImage"
    assert my_image.image.shape[0] == 10, "Wrong image shape"
