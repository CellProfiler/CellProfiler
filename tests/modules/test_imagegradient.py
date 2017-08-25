import cellprofiler.image
import cellprofiler.modules.imagegradient
import numpy
import numpy.testing
import skimage.filters.rank
import skimage.morphology

instance = cellprofiler.modules.imagegradient.ImageGradient()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "ImageGradient"

    if image.dimensions == 3:
        module.structuring_element.shape = "ball"

    module.run(workspace)

    actual = image_set.get_image("ImageGradient")

    data = image.pixel_data

    data = skimage.img_as_uint(data)

    disk = skimage.morphology.disk(1)

    if image.dimensions == 3 or image.multichannel:
        expected_data = numpy.zeros_like(data)

        for z, img in enumerate(data):
            expected_data[z] = skimage.filters.rank.gradient(img, disk)
    else:
        expected_data = skimage.filters.rank.gradient(data, disk)

    # CellProfiler converts Image data according to MatLab standards. Remove this and test against
    # expected_data once MatLab support is removed. Until then, use the Image constructro to convert the data
    # a la MatLab before comparison.
    expected = cellprofiler.image.Image(
        image=expected_data,
        dimensions=3
    )

    numpy.testing.assert_array_equal(expected.pixel_data, actual.pixel_data)
