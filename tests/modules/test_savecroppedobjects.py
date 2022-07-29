import glob
import os.path

import numpy.testing
import pytest
import skimage
import skimage.measure
import skimage.morphology

import cellprofiler_core.image
import cellprofiler.modules.savecroppedobjects
import cellprofiler_core.object
import cellprofiler_core.preferences
import cellprofiler_core.setting

import tempfile

instance = cellprofiler.modules.savecroppedobjects.SaveCroppedObjects()

def test_extract_file_name(module, workspace, object_set, tmpdir):
    """
    Trying to independently add another image (here, img) to the workspace, 
    independently of the pytest decorator in conftest.

    Error returned from this test is 
    AssertionError: Feature FileName_example for Image does not exist
    as a result of the source_file_name_feature method in SaveCroppedObjects
    """
    directory = str(tmpdir.mkdir("example"))

    img = cellprofiler_core.image.Image(image = skimage.data.camera()[0:128, 0:128])

    segmented = skimage.measure.label(img.pixel_data > 0.5)

    workspace.image_set.add("example", img)

    obj = cellprofiler_core.object.Objects()

    obj.segmented = segmented

    object_set.add_objects(obj, "example")

    module.file_image_name.value = "example"

    module.image_name.value = "example"

    module.export_option.value = "Images"

    module.objects_name.value = "example"

    module.directory.value = "{}|{}".format(
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME, directory
    )

    module.run(workspace)

    unique_labels = numpy.unique(obj.segmented)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    filenames = glob.glob(os.path.join(directory, "example_*.tiff"))

    for label in unique_labels:
        print(label)
        mask_in = obj.segmented == label

        properties = skimage.measure.regionprops(
            mask_in.astype(int), intensity_image=image.pixel_data
        )

        mask = properties[0].intensity_image

        t = os.path.join(directory, "example_{}.tiff".format(label))

        filename = glob.glob(os.path.join(directory, "*example_{}.tiff".format(label)))[0]

        numpy.testing.assert_array_equal(
            skimage.io.imread(filename), skimage.img_as_ubyte(mask)
        )


def test_extract_file_name_different_img_loader(module, workspace, object_set, tmpdir):
    """
    Trying to use something like core.FileImage to load an image

    Results in: 
    AssertionError: Feature FileName_example for Image does not exist
    """
    directory = str(tmpdir.mkdir("example"))

    img = skimage.data.camera()

    segmented = skimage.measure.label(img > 0.5)

    handle, path = tempfile.mkstemp(suffix=".png")
    fd = os.fdopen(handle, "wb")
    fd.write(img)
    fd.close()

    fname = os.path.split(path)[-1].split(".")[0]

    image = cellprofiler_core.image.FileImage(name=os.path.splitext(fname)[0], pathname = path, filename=fname)

    workspace.image_set.add("example", image)

    obj = cellprofiler_core.object.Objects()

    obj.segmented = segmented

    object_set.add_objects(obj, "example")

    module.file_image_name.value = "example"

    module.image_name.value = "example"

    module.export_option.value = "Images"

    module.objects_name.value = "example"

    module.directory.value = "{}|{}".format(
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME, directory
    )

    module.run(workspace)

    unique_labels = numpy.unique(obj.segmented)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    filenames = glob.glob(os.path.join(directory, "example_*.tiff"))

    for label in unique_labels:
        print(label)
        mask_in = obj.segmented == label

        properties = skimage.measure.regionprops(
            mask_in.astype(int), intensity_image=image.pixel_data
        )

        mask = properties[0].intensity_image

        t = os.path.join(directory, "example_{}.tiff".format(label))

        filename = glob.glob(os.path.join(directory, "*example_{}.tiff".format(label)))[0]

        numpy.testing.assert_array_equal(
            skimage.io.imread(filename), skimage.img_as_ubyte(mask)
        )



def test_run_images(image, module, image_set, workspace, object_set, tmpdir):
    directory = str(tmpdir.mkdir("example"))

    segmented = skimage.measure.label(image.pixel_data > 0.5)

    obj = cellprofiler_core.object.Objects()

    obj.segmented = segmented

    object_set.add_objects(obj, "example")

    module.image_name.value = "example"

    module.export_option.value = "Images"

    module.objects_name.value = "example"

    module.directory.value = "{}|{}".format(
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME, directory
    )

    module.run(workspace)

    unique_labels = numpy.unique(obj.segmented)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    filenames = glob.glob(os.path.join(directory, "example_*.tiff"))

    for label in unique_labels:
        mask_in = obj.segmented == label

        properties = skimage.measure.regionprops(
            mask_in.astype(int), intensity_image=image.pixel_data
        )

        mask = properties[0].intensity_image

        filename = glob.glob(os.path.join(directory, "example_{}.tiff".format(label)))[
            0
        ]

        numpy.testing.assert_array_equal(
            skimage.io.imread(filename), skimage.img_as_ubyte(mask)
        )


def test_defaults(module):
    module.create_settings()

    assert (
        module.directory.get_dir_choice()
        == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
    )


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(
            cellprofiler_core.image.Image(numpy.random.rand(100, 100)),
            id="grayscale_image",
        )
    ],
)
def test_run_masks(image, module, image_set, workspace, object_set, tmpdir):
    directory = str(tmpdir.mkdir("example"))

    segmented = skimage.measure.label(image.pixel_data > 0.5)

    obj = cellprofiler_core.object.Objects()

    obj.segmented = segmented

    object_set.add_objects(obj, "example")

    module.export_option.value = "Masks"

    module.objects_name.value = "example"

    module.directory.value = "{}|{}".format(
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME, directory
    )

    module.run(workspace)

    unique_labels = numpy.unique(obj.segmented)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    filenames = glob.glob(os.path.join(directory, "example_*.tiff"))

    for label in unique_labels:
        mask = obj.segmented == label

        mask = skimage.img_as_ubyte(mask)

        filename = glob.glob(os.path.join(directory, "example_{}.tiff".format(label)))[
            0
        ]

        numpy.testing.assert_array_equal(skimage.io.imread(filename), mask)


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(
            cellprofiler_core.image.Image(numpy.random.rand(100, 100)),
            id="grayscale_image",
        )
    ],
)
def test_create_subfolders(image, module, image_set, workspace, object_set, tmpdir):
    directory = str(tmpdir.mkdir("example"))

    segmented = skimage.measure.label(image.pixel_data > 0.5)

    obj = cellprofiler_core.object.Objects()

    obj.segmented = segmented

    object_set.add_objects(obj, "example")

    module.export_option.value = "Masks"

    module.objects_name.value = "example"

    module.directory.value = "{}|{}".format(
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
        os.path.join(directory, "subdirectory"),
    )

    module.run(workspace)

    unique_labels = numpy.unique(obj.segmented)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    filenames = glob.glob(os.path.join(directory, "subdirectory", "example_*.tiff"))

    for label in unique_labels:
        mask = obj.segmented == label

        mask = skimage.img_as_ubyte(mask)

        filename = glob.glob(
            os.path.join(directory, "subdirectory", "example_{}.tiff".format(label))
        )[0]

        numpy.testing.assert_array_equal(skimage.io.imread(filename), mask)


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(
            cellprofiler_core.image.Image(numpy.random.rand(100, 100)),
            id="grayscale_image",
        )
    ],
)
def test_create_subfolders_from_metadata(
    image, module, image_set, workspace, object_set, tmpdir
):
    directory = str(tmpdir.mkdir("example"))

    segmented = skimage.measure.label(image.pixel_data > 0.5)

    obj = cellprofiler_core.object.Objects()

    obj.segmented = segmented

    object_set.add_objects(obj, "example")

    workspace.measurements.add_measurement("Image", "Metadata_Plate", "002")

    workspace.measurements.add_measurement("Image", "Metadata_Well", "D")

    module.objects_name.value = "example"

    module.export_option.value = "Masks"

    module.directory.value = "{}|{}".format(
        cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME,
        directory + r"/\g<Plate>/\g<Well>",
    )

    module.run(workspace)

    unique_labels = numpy.unique(obj.segmented)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    filenames = glob.glob(os.path.join(directory, "002", "D", "example_*.tiff"))

    assert len(filenames) > 0

    for label in unique_labels:
        mask = obj.segmented == label

        mask = skimage.img_as_ubyte(mask)

        filename = glob.glob(
            os.path.join(directory, "002", "D", "example_{}.tiff".format(label))
        )[0]

        numpy.testing.assert_array_equal(skimage.io.imread(filename), mask)
