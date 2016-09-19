import cellprofiler.modules.closing
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.closing.Closing()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "closing"

    disk = skimage.morphology.disk(1)

    module.structuring_element.value = disk

    module.run(workspace)

    actual = image_set.get_image("closing")

    desired = skimage.morphology.closing(image.pixel_data, disk)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
