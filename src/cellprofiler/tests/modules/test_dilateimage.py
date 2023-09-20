import numpy
import numpy.testing
import skimage.morphology

import cellprofiler.modules.dilateimage

instance = cellprofiler.modules.dilateimage.DilateImage()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "dilation"

    if image.dimensions == 3 or image.multichannel:
        # test 3d structuring element
        module.structuring_element.shape = "ball"

        footprint = skimage.morphology.ball(1)

        module.run(workspace)

        actual = image_set.get_image("dilation")

        desired = skimage.morphology.dilation(image.pixel_data, footprint)

        numpy.testing.assert_array_equal(actual.pixel_data, desired)

        # test planewise
        footprint = skimage.morphology.disk(1)

        module.structuring_element.shape = "disk"

        module.run(workspace)

        actual = image_set.get_image("dilation")

        desired = numpy.zeros_like(image.pixel_data)

        for index, plane in enumerate(image.pixel_data):
            desired[index] = skimage.morphology.dilation(plane, footprint)

        numpy.testing.assert_array_equal(actual.pixel_data, desired)

    else:
        footprint = skimage.morphology.disk(1)

        module.run(workspace)

        actual = image_set.get_image("dilation")

        desired = skimage.morphology.dilation(image.pixel_data, footprint)

        numpy.testing.assert_array_equal(actual.pixel_data, desired)
