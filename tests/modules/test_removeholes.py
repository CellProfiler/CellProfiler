import numpy
import skimage.morphology

import cellprofiler_core.image
import cellprofiler.modules.removeholes
import cellprofiler_core.workspace

instance = cellprofiler.modules.removeholes.RemoveHoles()


def test_run(image, module, workspace):
    module.x_name.value = "example"

    module.y_name.value = "output"

    module.size.value = 3.0

    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    if image.dimensions == 2:
        factor = (3.0 / 2.0) ** 2
    else:
        factor = (4.0 / 3.0) * ((3.0 / 2.0) ** 3)

    data = skimage.img_as_bool(image.pixel_data)

    expected = skimage.morphology.remove_small_holes(data, numpy.pi * factor)

    numpy.testing.assert_array_equal(actual, expected)


# https://github.com/CellProfiler/CellProfiler/issues/3369
def test_run_label_image(module):
    data = numpy.zeros((10, 10), dtype=numpy.uint8)
    data[3:8, 3:8] = 1
    data[5, 5] = 0

    image = cellprofiler_core.image.Image(image=data, convert=False)

    image_set_list = cellprofiler_core.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)
    image_set.add("example", image)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline=None,
        module=module,
        image_set=image_set,
        object_set=None,
        measurements=None,
        image_set_list=image_set_list,
    )

    module.x_name.value = "example"
    module.y_name.value = "output"
    module.run(workspace)

    actual = workspace.image_set.get_image("output").pixel_data

    expected = numpy.zeros((10, 10), dtype=bool)
    expected[3:8, 3:8] = True

    numpy.testing.assert_array_equal(actual, expected)
