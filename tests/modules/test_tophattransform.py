import cellprofiler.image
import cellprofiler.modules.tophattransform
import numpy.testing
import pytest
import skimage.data
import skimage.morphology

instance = cellprofiler.modules.tophattransform.TopHatTransform()


@pytest.fixture(
    scope="module",
    params=[
        (skimage.data.camera()[0:128, 0:128], 2),
        (numpy.tile(skimage.data.camera()[0:32, 0:32], (2, 1)).reshape(2, 32, 32), 3)
    ],
    ids=[
        "grayscale_image",
        "grayscale_volume"
    ]
)
def image(request):
    data, dimensions = request.param

    return cellprofiler.image.Image(image=data, dimensions=dimensions)


def test_run_black_tophat(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "TopHatTransform"

    module.operation_name.value = "Black top-hat transform"

    if image.volumetric:
        module.structuring_element.value = "ball,1"

        structure = skimage.morphology.ball(1)
    else:
        module.structuring_element.value = "disk,1"

        structure = skimage.morphology.disk(1)

    module.run(workspace)

    actual = image_set.get_image("TopHatTransform")

    desired = skimage.morphology.black_tophat(image.pixel_data, structure)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)


def test_run_white_tophat(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "TopHatTransform"

    module.operation_name.value = "White top-hat transform"

    if image.volumetric:
        module.structuring_element.value = "ball,1"

        structure = skimage.morphology.ball(1)
    else:
        module.structuring_element.value = "disk,1"

        structure = skimage.morphology.disk(1)

    module.run(workspace)

    actual = image_set.get_image("TopHatTransform")

    desired = skimage.morphology.white_tophat(image.pixel_data, structure)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
