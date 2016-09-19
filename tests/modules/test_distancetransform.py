import cellprofiler.modules.distancetransform
import numpy.testing
import scipy.ndimage

instance = cellprofiler.modules.distancetransform.DistanceTransform()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "DistanceTransform"

    module.run(workspace)

    actual = image_set.get_image("DistanceTransform")

    desired = scipy.ndimage.distance_transform_edt(image.pixel_data)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
