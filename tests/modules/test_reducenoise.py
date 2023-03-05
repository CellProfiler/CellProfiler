import numpy.testing
import skimage.restoration

import cellprofiler.modules.reducenoise

instance = cellprofiler.modules.reducenoise.ReduceNoise()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "ReduceNoise"

    module.size.value = 7

    module.distance.value = 11

    module.cutoff_distance.value = 0.1

    module.run(workspace)

    actual = image_set.get_image("ReduceNoise")

    desired = skimage.restoration.denoise_nl_means(
        fast_mode=True,
        h=0.1,
        image=image.pixel_data,
        channel_axis=2 if image.multichannel else None,
        patch_distance=11,
        patch_size=7,
    )

    numpy.testing.assert_array_almost_equal(actual.pixel_data, desired)
