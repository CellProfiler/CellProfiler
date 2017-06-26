import numpy
import numpy.testing
import skimage.color
import skimage.feature
import skimage.filters
import skimage.filters.rank
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform
import skimage.util

import cellprofiler.image
import cellprofiler.modules.clearborder

instance = cellprofiler.modules.clearborder.ClearBorder()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "segmentation"

    module.y_name.value = "clearborder"

    segmentation = numpy.array(
        [[0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    image_set.add(
        "segmentation",
        cellprofiler.image.Image(
            convert=False,
            dimensions=image.dimensions,
            image=segmentation
        )
    )

    module.run(workspace)

    expected = numpy.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    )

    expected = skimage.measure.label(expected)

    actual = workspace.get_objects("clearborder")

    numpy.testing.assert_array_equal(expected, actual.segmented)
