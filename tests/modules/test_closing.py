import cellprofiler.modules.closing
import numpy
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.closing.Closing()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "closing"

    if image.dimensions == 3 or image.multichannel:
        # test 3d structuring element
        module.structuring_element.shape = "ball"

        selem = skimage.morphology.ball(1)

        module.run(workspace)

        actual = image_set.get_image("closing")

        desired = skimage.morphology.closing(image.pixel_data, selem)

        numpy.testing.assert_array_equal(actual.pixel_data, desired)

        # test planewise
        selem = skimage.morphology.disk(1)

        module.structuring_element.shape = "disk"

        module.run(workspace)

        actual = image_set.get_image("closing")

        desired = numpy.zeros_like(image.pixel_data)

        for index, plane in enumerate(image.pixel_data):

            desired[index] = skimage.morphology.closing(plane, selem)

        numpy.testing.assert_array_equal(actual.pixel_data, desired)

    else:
        selem = skimage.morphology.disk(1)

        module.run(workspace)

        actual = image_set.get_image("closing")

        desired = skimage.morphology.closing(image.pixel_data, selem)

        numpy.testing.assert_array_equal(actual.pixel_data, desired)
