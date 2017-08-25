import numpy.testing
import skimage.morphology

import cellprofiler.modules.removeholes


instance = cellprofiler.modules.removeholes.RemoveHoles()


def test_run(image, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "output"

    module.size.value = 3.0

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    if image.dimensions == 2:
        factor = ((3.0 / 2.0) ** 2)
    else:
        factor = (4.0 / 3.0) * ((3.0 / 2.0) ** 3)

    data = skimage.img_as_bool(image.pixel_data)

    expected = skimage.morphology.remove_small_holes(data, numpy.pi * factor)

    numpy.testing.assert_array_equal(actual, expected)
