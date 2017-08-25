import glob
import os.path

import numpy.testing
import skimage.measure
import skimage.morphology

import cellprofiler.modules.cropobjects
import cellprofiler.object
import cellprofiler.setting

instance = cellprofiler.modules.cropobjects.CropObjects()


def test_run(image, module, image_set, workspace, object_set, tmpdir):
    directory = str(tmpdir.mkdir("example"))

    segmented = skimage.measure.label(image.pixel_data > 0.5)

    obj = cellprofiler.object.Objects()

    obj.segmented = segmented

    object_set.add_objects(obj, "example")

    module.objects_name.value = "example"

    module.directory.value = "{}|{}".format(cellprofiler.setting.ABSOLUTE_FOLDER_NAME, directory)

    module.run(workspace)

    unique_labels = numpy.unique(obj.segmented)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    filenames = glob.glob(os.path.join(directory, "example_*.tiff"))

    for label in unique_labels:
        mask = obj.segmented == label

        mask = skimage.img_as_ubyte(mask)

        filename = glob.glob(os.path.join(directory, "example_{:04d}_*.tiff".format(label)))[0]

        numpy.testing.assert_array_equal(skimage.io.imread(filename), mask)
