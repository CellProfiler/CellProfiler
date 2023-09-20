import numpy.testing
import skimage.color
import skimage.morphology

import cellprofiler_core.image
import cellprofiler.modules.medialaxis

instance = cellprofiler.modules.medialaxis.MedialAxis()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "MedialAxis"

    module.run(workspace)

    actual = image_set.get_image("MedialAxis")

    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    if image.dimensions == 3:
        expected_data = numpy.zeros_like(data)

        for z, img in enumerate(data):
            expected_data[z] = skimage.morphology.medial_axis(img)
    else:
        expected_data = skimage.morphology.medial_axis(data)

    expected = cellprofiler_core.image.Image(
        image=expected_data, dimensions=image.dimensions
    )

    numpy.testing.assert_array_equal(actual.pixel_data, expected.pixel_data)
