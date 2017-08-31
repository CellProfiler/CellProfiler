import numpy
import numpy.testing
import skimage.color
import skimage.filters
import skimage.measure

import cellprofiler.image
import cellprofiler.modules.overlayobjects
import cellprofiler.object

instance = cellprofiler.modules.overlayobjects.OverlayObjects()


def _create_objects(image):
    data = image.pixel_data

    if image.multichannel:
        data = skimage.color.rgb2gray(data)

    binary = data > skimage.filters.threshold_li(data)

    labels = skimage.measure.label(binary)

    objects = cellprofiler.object.Objects()

    objects.segmented = labels

    return objects


def test_run(image, module, image_set, object_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "OverlayObjects"

    module.objects.value = "DNA"

    module.opacity.value = 0.3

    objects = _create_objects(image)

    object_set.add_objects(objects, "DNA")

    module.run(workspace)

    actual_image = image_set.get_image("OverlayObjects")

    assert actual_image.pixel_data.shape == objects.shape + (3,)
