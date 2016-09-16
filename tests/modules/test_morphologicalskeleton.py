import cellprofiler.modules.morphologicalskeleton
import numpy.testing
import skimage.morphology

instance = cellprofiler.modules.morphologicalskeleton.MorphologicalSkeleton()


def test_run(image, module, image_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "MorphologicalSkeleton"

    module.run(workspace)

    actual = image_set.get_image("MorphologicalSkeleton")

    desired = skimage.morphology.skeletonize_3d(image.pixel_data)

    numpy.testing.assert_array_equal(actual.pixel_data, desired)
