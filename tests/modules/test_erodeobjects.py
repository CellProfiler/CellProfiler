import numpy
import numpy.testing
import skimage.morphology

import cellprofiler.modules.erodeobjects

instance = cellprofiler.modules.erodeobjects.ErodeObjects()


def test_run(object_with_data, module, object_set_with_data, workspace_with_data):
    module.x_name.value = "InputObjects"

    module.y_name.value = "OutputObjects"

    if object_with_data.dimensions == 3:
        # test 3d structuring element
        module.structuring_element.shape = "ball"

        selem = skimage.morphology.ball(1)

        module.run(workspace_with_data)

        actual = object_set_with_data.get_objects("OutputObjects")

        desired = skimage.morphology.erosion(object_with_data.segmented, selem)

        numpy.testing.assert_array_equal(actual.segmented, desired)

        # test planewise
        selem = skimage.morphology.disk(1)

        module.structuring_element.shape = "disk"

        module.y_name.value = "OutputObjectsPlane"

        module.run(workspace_with_data)

        actual = object_set_with_data.get_objects("OutputObjectsPlane")

        desired = numpy.zeros_like(object_with_data.segmented)

        for index, plane in enumerate(object_with_data.segmented):
            desired[index] = skimage.morphology.erosion(plane, selem)

        numpy.testing.assert_array_equal(actual.segmented, desired)

    else:
        selem = skimage.morphology.disk(1)

        module.preserve_midpoints.value = False

        module.relabel_objects.value = False

        module.run(workspace_with_data)

        actual = object_set_with_data.get_objects("OutputObjects")

        desired = skimage.morphology.erosion(object_with_data.segmented, selem)

        numpy.testing.assert_array_equal(actual.segmented, desired)
