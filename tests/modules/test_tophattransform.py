import cellprofiler.modules.tophattransform
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.tophattransform.TopHatTransform()


def test_black(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "TopHatTransform"

    module.operation_name.value = "Black"

    disk = skimage.morphology.disk(1)

    module.structuring_element.value = disk

    module.run(workspace)

    actual = image_set.get_image("TopHatTransform")

    desired = skimage.morphology.black_tophat(image.pixel_data, disk)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)


def test_white(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "TopHatTransform"

    module.operation_name.value = "White"

    disk = skimage.morphology.disk(1)

    module.structuring_element.value = disk

    module.run(workspace)

    actual = image_set.get_image("TopHatTransform")

    desired = skimage.morphology.white_tophat(image.pixel_data, disk)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
