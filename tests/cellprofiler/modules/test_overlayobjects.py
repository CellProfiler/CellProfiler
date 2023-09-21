import cellprofiler.modules.overlayobjects

instance = cellprofiler.modules.overlayobjects.OverlayObjects()


def test_run(image, module, image_set, object_with_data, object_set, workspace):
    module.x_name.value = "example"

    module.y_name.value = "OverlayObjects"

    module.objects.value = "DNA"

    module.opacity.value = 0.3

    object_set.add_objects(object_with_data, "DNA")

    module.run(workspace)

    actual_image = image_set.get_image("OverlayObjects")

    assert actual_image.pixel_data.shape == object_with_data.shape + (3,)
