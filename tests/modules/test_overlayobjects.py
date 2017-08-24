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


# https://github.com/CellProfiler/CellProfiler/issues/2751
def test_run_issues_2751(image_set_list, measurements, module, object_set, pipeline):
    data = numpy.zeros((9, 9, 9))

    labels = numpy.zeros_like(data, dtype=numpy.uint8)
    labels[:3, :3, :3] = 1
    labels[:, 3:-3, 3:-3] = 2
    labels[-3:, -3:, -3:] = 3

    image = cellprofiler.image.Image(
        data,
        dimensions=3
    )

    image_set = image_set_list.get_image_set(0)
    image_set.add("example", image)

    objects = cellprofiler.object.Objects()
    objects.segmented = labels

    object_set.add_objects(objects, "DNA")

    workspace = cellprofiler.workspace.Workspace(
        image_set=image_set,
        image_set_list=image_set_list,
        measurements=measurements,
        module=module,
        object_set=object_set,
        pipeline=pipeline
    )

    module.x_name.value = "example"
    module.y_name.value = "OverlayObjects"
    module.objects.value = "DNA"

    module.run(workspace)

    overlay_image = image_set.get_image("OverlayObjects")
    overlay_pixel_data = overlay_image.pixel_data

    overlay_region_1 = overlay_pixel_data[:3, :3, :3]
    assert numpy.all(overlay_region_1 == overlay_region_1[0, 0, 0])

    overlay_region_2 = overlay_pixel_data[:, 3:-3, 3:-3]
    assert numpy.all(overlay_region_2 == overlay_region_2[0, 0, 0])

    overlay_region_3 = overlay_pixel_data[-3:, -3:, -3:]
    assert numpy.all(overlay_region_3 == overlay_region_3[0, 0, 0])

    assert not numpy.all(overlay_region_1[0, 0, 0] == overlay_region_2[0, 0, 0])
    assert not numpy.all(overlay_region_1[0, 0, 0] == overlay_region_3[0, 0, 0])
    assert not numpy.all(overlay_region_2[0, 0, 0] == overlay_region_3[0, 0, 0])
