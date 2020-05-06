import numpy.testing
import pytest
import skimage.data
import skimage.morphology

import cellprofiler_core.image
import cellprofiler.modules.morphologicalskeleton

instance = cellprofiler.modules.morphologicalskeleton.MorphologicalSkeleton()


@pytest.fixture(
    scope="module",
    params=[
        (skimage.data.camera()[0:128, 0:128], 2),
        (numpy.tile(skimage.data.camera()[0:32, 0:32], (2, 1)).reshape(2, 32, 32), 3),
    ],
    ids=["grayscale_image", "grayscale_volume"],
)
def image(request):
    data, dimensions = request.param

    return cellprofiler_core.image.Image(image=data, dimensions=dimensions)


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "MorphologicalSkeleton"

    module.run(workspace)

    actual = image_set.get_image("MorphologicalSkeleton")

    if image.volumetric:
        desired = skimage.morphology.skeletonize_3d(image.pixel_data)
    else:
        desired = skimage.morphology.skeletonize(image.pixel_data)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
