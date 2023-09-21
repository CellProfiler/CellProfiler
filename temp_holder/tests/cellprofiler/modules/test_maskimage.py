import numpy as np

import cellprofiler_core.image
import cellprofiler_core.measurement


import cellprofiler.modules.maskimage
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

MASKING_IMAGE_NAME = "maskingimage"
MASKED_IMAGE_NAME = "maskedimage"
IMAGE_NAME = "image"
OBJECTS_NAME = "objects"


def test_mask_with_objects():
    labels = np.zeros((10, 15), int)
    labels[2:5, 3:8] = 1
    labels[5:8, 10:14] = 2
    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, OBJECTS_NAME)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    np.random.seed(0)
    pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
    image_set.add(IMAGE_NAME, cellprofiler_core.image.Image(pixel_data))

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.maskimage.MaskImage()
    module.source_choice.value = cellprofiler.modules.maskimage.IO_OBJECTS
    module.object_name.value = OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.masked_image_name.value = MASKED_IMAGE_NAME
    module.invert_mask.value = False
    module.set_module_num(1)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
    assert isinstance(masked_image, cellprofiler_core.image.Image)
    assert np.all(masked_image.pixel_data[labels > 0] == pixel_data[labels > 0])
    assert np.all(masked_image.pixel_data[labels == 0] == 0)
    assert np.all(masked_image.mask == (labels > 0))
    assert np.all(masked_image.masking_objects.segmented == labels)


def test_mask_invert():
    labels = np.zeros((10, 15), int)
    labels[2:5, 3:8] = 1
    labels[5:8, 10:14] = 2
    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, OBJECTS_NAME)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    np.random.seed(0)
    pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
    image_set.add(IMAGE_NAME, cellprofiler_core.image.Image(pixel_data))

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.maskimage.MaskImage()
    module.source_choice.value = cellprofiler.modules.maskimage.IO_OBJECTS
    module.object_name.value = OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.masked_image_name.value = MASKED_IMAGE_NAME
    module.invert_mask.value = True
    module.set_module_num(1)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
    assert isinstance(masked_image, cellprofiler_core.image.Image)
    assert np.all(masked_image.pixel_data[labels == 0] == pixel_data[labels == 0])
    assert np.all(masked_image.pixel_data[labels > 0] == 0)
    assert np.all(masked_image.mask == (labels == 0))
    assert np.all(masked_image.masking_objects.segmented == labels)


def test_double_mask():
    labels = np.zeros((10, 15), int)
    labels[2:5, 3:8] = 1
    labels[5:8, 10:14] = 2
    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, OBJECTS_NAME)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    np.random.seed(0)
    pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
    mask = np.random.uniform(size=(10, 15)) > 0.5
    image_set.add(IMAGE_NAME, cellprofiler_core.image.Image(pixel_data, mask))

    expected_mask = mask & (labels > 0)

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.maskimage.MaskImage()
    module.source_choice.value = cellprofiler.modules.maskimage.IO_OBJECTS
    module.object_name.value = OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.masked_image_name.value = MASKED_IMAGE_NAME
    module.invert_mask.value = False
    module.set_module_num(1)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
    assert isinstance(masked_image, cellprofiler_core.image.Image)
    assert np.all(masked_image.pixel_data[expected_mask] == pixel_data[expected_mask])
    assert np.all(masked_image.pixel_data[~expected_mask] == 0)
    assert np.all(masked_image.mask == expected_mask)
    assert np.all(masked_image.masking_objects.segmented == labels)


def test_binary_mask():
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    np.random.seed(0)
    pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
    image_set.add(IMAGE_NAME, cellprofiler_core.image.Image(pixel_data))

    masking_image = np.random.uniform(size=(10, 15)) > 0.5
    image_set.add(MASKING_IMAGE_NAME, cellprofiler_core.image.Image(masking_image))

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.maskimage.MaskImage()
    module.source_choice.value = cellprofiler.modules.maskimage.IO_IMAGE
    module.object_name.value = OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.masking_image_name.value = MASKING_IMAGE_NAME
    module.masked_image_name.value = MASKED_IMAGE_NAME
    module.invert_mask.value = False
    module.set_module_num(1)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
    assert isinstance(masked_image, cellprofiler_core.image.Image)
    assert np.all(masked_image.pixel_data[masking_image] == pixel_data[masking_image])
    assert np.all(masked_image.pixel_data[~masking_image] == 0)
    assert np.all(masked_image.mask == masking_image)
    assert not masked_image.has_masking_objects


def test_gray_mask():
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    np.random.seed(0)
    pixel_data = np.random.uniform(size=(10, 15)).astype(np.float32)
    image_set.add(IMAGE_NAME, cellprofiler_core.image.Image(pixel_data))

    masking_image = np.random.uniform(size=(10, 15))
    image_set.add(MASKING_IMAGE_NAME, cellprofiler_core.image.Image(masking_image))
    masking_image = masking_image > 0.5

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.maskimage.MaskImage()
    module.source_choice.value = cellprofiler.modules.maskimage.IO_IMAGE
    module.object_name.value = OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.masking_image_name.value = MASKING_IMAGE_NAME
    module.masked_image_name.value = MASKED_IMAGE_NAME
    module.invert_mask.value = False
    module.set_module_num(1)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
    assert isinstance(masked_image, cellprofiler_core.image.Image)
    assert np.all(masked_image.pixel_data[masking_image] == pixel_data[masking_image])
    assert np.all(masked_image.pixel_data[~masking_image] == 0)
    assert np.all(masked_image.mask == masking_image)
    assert not masked_image.has_masking_objects


def test_color_mask():
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    np.random.seed(0)
    pixel_data = np.random.uniform(size=(10, 15, 3)).astype(np.float32)
    image_set.add(IMAGE_NAME, cellprofiler_core.image.Image(pixel_data))

    masking_image = np.random.uniform(size=(10, 15))

    image_set.add(MASKING_IMAGE_NAME, cellprofiler_core.image.Image(masking_image))
    expected_mask = masking_image > 0.5

    pipeline = cellprofiler_core.pipeline.Pipeline()
    module = cellprofiler.modules.maskimage.MaskImage()
    module.source_choice.value = cellprofiler.modules.maskimage.IO_IMAGE
    module.object_name.value = OBJECTS_NAME
    module.image_name.value = IMAGE_NAME
    module.masking_image_name.value = MASKING_IMAGE_NAME
    module.masked_image_name.value = MASKED_IMAGE_NAME
    module.invert_mask.value = False
    module.set_module_num(1)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    masked_image = workspace.image_set.get_image(MASKED_IMAGE_NAME)
    assert isinstance(masked_image, cellprofiler_core.image.Image)
    assert np.all(
        masked_image.pixel_data[expected_mask, :] == pixel_data[expected_mask, :]
    )
    assert np.all(masked_image.pixel_data[~expected_mask, :] == 0)
    assert np.all(masked_image.mask == expected_mask)
    assert not masked_image.has_masking_objects
