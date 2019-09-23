# import numpy
#
# import nucleus.image
# import nucleus.measurement
# import nucleus.modules.injectimage
# import nucleus.pipeline
# import nucleus.workspace
#
#
# def test_init():
#     image = numpy.zeros((10, 10), dtype=float)
#     x = nucleus.modules.injectimage.InjectImage("my_image", image)
#
#
# def test_get_from_image_set():
#     image = numpy.zeros((10, 10), dtype=float)
#     ii = nucleus.modules.injectimage.InjectImage("my_image", image)
#     pipeline = nucleus.pipeline.Pipeline()
#     measurements = nucleus.measurement.Measurements()
#     workspace = nucleus.workspace.Workspace(
#         pipeline, ii, measurements, None, measurements, nucleus.image()
#     )
#     ii.prepare_run(workspace)
#     ii.prepare_group(workspace, {}, [1])
#     ii.run(workspace)
#     image_set = workspace.image_set
#     assert image_set, "No image set returned from ImageSetList.GetImageSet"
#     my_image = image_set.get_image("my_image")
#     assert my_image, "No image returned from ImageSet.GetImage"
#     assert my_image.image.shape[0] == 10, "Wrong image shape"
