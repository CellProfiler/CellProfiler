import numpy
import numpy.testing
import pytest
import skimage.morphology

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT, COLTYPE_INTEGER


import cellprofiler.modules.measureimageintensity
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace


@pytest.fixture(scope="function")
def image():
    return cellprofiler_core.image.Image()


@pytest.fixture(scope="function")
def objects(image):
    objects = cellprofiler_core.object.Objects()

    objects.parent_image = image

    return objects


@pytest.fixture(scope="function")
def measurements():
    return cellprofiler_core.measurement.Measurements()


@pytest.fixture(scope="function")
def module():
    module = cellprofiler.modules.measureimageintensity.MeasureImageIntensity()

    module.images_list.value = "image"

    module.objects_list.value = "objects"

    module.wants_percentiles.value = True
    module.percentiles.value = "10, 50, 90"

    return module


@pytest.fixture(scope="function")
def workspace(image, measurements, module, objects):
    image_set_list = cellprofiler_core.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)

    image_set.add("image", image)

    object_set = cellprofiler_core.object.ObjectSet()

    object_set.add_objects(objects, "objects")

    return cellprofiler_core.workspace.Workspace(
        cellprofiler_core.pipeline.Pipeline(),
        module,
        image_set,
        object_set,
        measurements,
        image_set_list,
    )


def test_volume_zeros(image, measurements, module, workspace):
    image.pixel_data = numpy.zeros((10, 10, 10))

    image.dimensions = 3

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image": 0.0,
        "Intensity_MeanIntensity_image": 0.0,
        "Intensity_MedianIntensity_image": 0.0,
        "Intensity_StdIntensity_image": 0.0,
        "Intensity_MADIntensity_image": 0.0,
        "Intensity_MaxIntensity_image": 0.0,
        "Intensity_MinIntensity_image": 0.0,
        "Intensity_TotalArea_image": 1000.0,
        "Intensity_PercentMaximal_image": 100.0,
        "Intensity_UpperQuartileIntensity_image": 0.0,
        "Intensity_LowerQuartileIntensity_image": 0.0,
        "Intensity_Percentile_10_image": 0.0,
        "Intensity_Percentile_50_image": 0.0,
        "Intensity_Percentile_90_image": 0.0,
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_current_measurement(
            "Image", feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume(image, measurements, module, workspace):
    image.set_image(skimage.morphology.ball(3), convert=False)

    image.dimensions = 3

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image": 123.0,
        "Intensity_MeanIntensity_image": 0.358600583090379,
        "Intensity_MedianIntensity_image": 0.0,
        "Intensity_StdIntensity_image": 0.47958962134059907,
        "Intensity_MADIntensity_image": 0.0,
        "Intensity_MaxIntensity_image": 1.0,
        "Intensity_MinIntensity_image": 0.0,
        "Intensity_TotalArea_image": 343.0,
        "Intensity_PercentMaximal_image": 35.8600583090379,
        "Intensity_UpperQuartileIntensity_image": 1.0,
        "Intensity_LowerQuartileIntensity_image": 0.0,
        "Intensity_Percentile_10_image": 0.0,
        "Intensity_Percentile_50_image": 0.0,
        "Intensity_Percentile_90_image": 1.0,
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_current_measurement(
            "Image", feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume_and_mask(image, measurements, module, workspace):
    mask = skimage.morphology.ball(3, dtype=bool)

    image.set_image(numpy.ones_like(mask, dtype=numpy.uint8), convert=False)

    image.mask = mask

    image.dimensions = 3

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image": 123.0,
        "Intensity_MeanIntensity_image": 1.0,
        "Intensity_MedianIntensity_image": 1.0,
        "Intensity_StdIntensity_image": 0.0,
        "Intensity_MADIntensity_image": 0.0,
        "Intensity_MaxIntensity_image": 1.0,
        "Intensity_MinIntensity_image": 1.0,
        "Intensity_TotalArea_image": 123.0,
        "Intensity_PercentMaximal_image": 100.0,
        "Intensity_UpperQuartileIntensity_image": 1.0,
        "Intensity_LowerQuartileIntensity_image": 1.0,
        "Intensity_Percentile_10_image": 1.0,
        "Intensity_Percentile_50_image": 1.0,
        "Intensity_Percentile_90_image": 1.0,

    }

    for feature, value in list(expected.items()):
        actual = measurements.get_current_measurement(
            "Image", feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume_and_objects(image, measurements, module, objects, workspace):
    object_data = skimage.morphology.ball(3, dtype=numpy.uint8)

    image.set_image(numpy.ones_like(object_data, dtype=numpy.uint8), convert=False)

    image.dimensions = 3

    objects.segmented = object_data

    module.wants_objects.value = True

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image_objects": 123.0,
        "Intensity_MeanIntensity_image_objects": 1.0,
        "Intensity_MedianIntensity_image_objects": 1.0,
        "Intensity_StdIntensity_image_objects": 0.0,
        "Intensity_MADIntensity_image_objects": 0.0,
        "Intensity_MaxIntensity_image_objects": 1.0,
        "Intensity_MinIntensity_image_objects": 1.0,
        "Intensity_TotalArea_image_objects": 123.0,
        "Intensity_PercentMaximal_image_objects": 100.0,
        "Intensity_UpperQuartileIntensity_image_objects": 1.0,
        "Intensity_LowerQuartileIntensity_image_objects": 1.0,
        "Intensity_Percentile_10_image_objects": 1.0,
        "Intensity_Percentile_50_image_objects": 1.0,
        "Intensity_Percentile_90_image_objects": 1.0,
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_current_measurement(
            "Image", feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_volume_and_objects_and_mask(image, measurements, module, objects, workspace):
    mask = skimage.morphology.ball(3, dtype=bool)

    image.set_image(numpy.ones_like(mask, dtype=numpy.uint8), convert=False)

    image.mask = mask

    image.dimensions = 3

    object_data = numpy.ones_like(mask, dtype=numpy.uint8)

    objects.segmented = object_data

    module.wants_objects.value = True

    module.run(workspace)

    expected = {
        "Intensity_TotalIntensity_image_objects": 123.0,
        "Intensity_MeanIntensity_image_objects": 1.0,
        "Intensity_MedianIntensity_image_objects": 1.0,
        "Intensity_StdIntensity_image_objects": 0.0,
        "Intensity_MADIntensity_image_objects": 0.0,
        "Intensity_MaxIntensity_image_objects": 1.0,
        "Intensity_MinIntensity_image_objects": 1.0,
        "Intensity_TotalArea_image_objects": 123.0,
        "Intensity_PercentMaximal_image_objects": 100.0,
        "Intensity_UpperQuartileIntensity_image_objects": 1.0,
        "Intensity_LowerQuartileIntensity_image_objects": 1.0,
        "Intensity_Percentile_10_image_objects": 1.0,
        "Intensity_Percentile_50_image_objects": 1.0,
        "Intensity_Percentile_90_image_objects": 1.0,
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_current_measurement(
            "Image", feature
        )

        assert actual == value, "{} expected {}, got {}".format(feature, value, actual)


def test_zeros(image, measurements, module, workspace):
    """Test operation on a completely-masked image"""
    image.pixel_data = numpy.zeros((10, 10))

    image.mask = numpy.zeros((10, 10), bool)

    module.run(workspace)

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_TotalArea_image"
        )
        == 0
    )

    assert len(measurements.get_object_names()) == 1

    assert measurements.get_object_names()[0] == "Image"

    columns = module.get_measurement_columns(workspace.pipeline)

    features = measurements.get_feature_names("Image")

    assert len(columns) == len(features)

    for column in columns:
        assert column[1] in features


def test_image(image, measurements, module, workspace):
    """Test operation on a single unmasked image"""
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * 0.99

    pixels[0:2, 0:2] = 1

    image.pixel_data = pixels

    module.run(workspace)

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_TotalArea_image"
        )
        == 100
    )

    assert measurements.get_current_measurement(
        "Image", "Intensity_TotalIntensity_image"
    ) == numpy.sum(pixels)

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_MeanIntensity_image"
        )
        == numpy.sum(pixels) / 100.0
    )

    assert measurements.get_current_image_measurement(
        "Intensity_MinIntensity_image"
    ) == numpy.min(pixels)

    assert measurements.get_current_image_measurement(
        "Intensity_MaxIntensity_image"
    ) == numpy.max(pixels)

    assert (
        measurements.get_current_image_measurement("Intensity_PercentMaximal_image")
        == 4.0
    )


def test_image_and_mask(image, measurements, module, workspace):
    """Test operation on a masked image"""
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * 0.99

    pixels[1:3, 1:3] = 1

    mask = numpy.zeros((10, 10), bool)

    mask[1:9, 1:9] = True

    image.pixel_data = pixels

    image.mask = mask

    module.run(workspace)

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_TotalArea_image"
        )
        == 64
    )

    assert measurements.get_current_measurement(
        "Image", "Intensity_TotalIntensity_image"
    ) == numpy.sum(pixels[1:9, 1:9])

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_MeanIntensity_image"
        )
        == numpy.sum(pixels[1:9, 1:9]) / 64.0
    )

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_PercentMaximal_image"
        )
        == 400.0 / 64.0
    )


def test_image_and_objects(image, measurements, module, objects, workspace):
    """Test operation on an image masked by objects"""
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * 0.99

    pixels[1:3, 1:3] = 1

    image.pixel_data = pixels

    labels = numpy.zeros((10, 10), int)

    labels[1:9, 1:5] = 1

    labels[1:9, 5:9] = 2

    objects.segmented = labels

    module.wants_objects.value = True

    module.run(workspace)

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_TotalArea_image_objects"
        )
        == 64
    )

    assert measurements.get_current_measurement(
        "Image", "Intensity_TotalIntensity_image_objects"
    ) == numpy.sum(pixels[1:9, 1:9])

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_MeanIntensity_image_objects"
        )
        == numpy.sum(pixels[1:9, 1:9]) / 64.0
    )

    numpy.testing.assert_almost_equal(
        measurements.get_current_measurement(
            "Image",
            "Intensity_PercentMaximal_image_objects",
        ),
        400.0 / 64.0,
    )

    assert len(measurements.get_object_names()) == 1

    assert measurements.get_object_names()[0] == "Image"

    columns = module.get_measurement_columns(workspace.pipeline)

    features = measurements.get_feature_names("Image")

    assert len(columns) == len(features)

    for column in columns:
        assert column[1] in features


def test_image_and_objects_and_mask(image, measurements, module, objects, workspace):
    """Test operation on an image masked by objects and a mask"""
    numpy.random.seed(0)

    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)

    mask = numpy.zeros((10, 10), bool)

    mask[1:9, :9] = True

    image.pixel_data = pixels

    image.mask = mask

    labels = numpy.zeros((10, 10), int)

    labels[1:9, 1:5] = 1

    labels[1:9, 5:] = 2

    objects.segmented = labels

    module.wants_objects.value = True

    module.run(workspace)

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_TotalArea_image_objects"
        )
        == 64
    )

    assert measurements.get_current_measurement(
        "Image", "Intensity_TotalIntensity_image_objects"
    ) == numpy.sum(pixels[1:9, 1:9])

    assert (
        measurements.get_current_measurement(
            "Image", "Intensity_MeanIntensity_image_objects"
        )
        == numpy.sum(pixels[1:9, 1:9]) / 64.0
    )


def test_get_measurement_columns_whole_image_mode(module):
    image_names = ["image%d" % i for i in range(3)]

    module.wants_objects.value = False

    expected_suffixes = []

    for image_name in image_names:
        im = module.images_list.value[-1]

        module.images_list.value.append(image_name)

        expected_suffixes.append(image_name)

    columns = module.get_measurement_columns(None)

    assert all([column[0] == "Image" for column in columns])

    for expected_suffix in expected_suffixes:
        for feature, coltype in (
            (
                cellprofiler.modules.measureimageintensity.F_TOTAL_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MEAN_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MIN_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MAX_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_TOTAL_AREA,
                COLTYPE_INTEGER,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_PERCENT_MAXIMAL,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MAD_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_LOWER_QUARTILE,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_UPPER_QUARTILE,
                COLTYPE_FLOAT,
            ),
        ):
            # feature names are now formatting strings
            feature_name = feature % expected_suffix

            assert any(
                [
                    (column[1] == feature_name and column[2] == coltype)
                    for column in columns
                ]
            )


def test_get_measurement_columns_object_mode(module):
    image_names = ["image%d" % i for i in range(3)]

    object_names = ["object%d" % i for i in range(3)]

    module.wants_objects.value = True

    expected_suffixes = []

    for image_name in image_names:
        module.images_list.value.append(image_name)

        for object_name in object_names:
            module.objects_list.value.append(object_name)

            expected_suffixes.append("%s_%s" % (image_name, object_name))

    columns = module.get_measurement_columns(None)

    assert all([column[0] == "Image" for column in columns])

    for expected_suffix in expected_suffixes:
        for feature, coltype in (
            (
                cellprofiler.modules.measureimageintensity.F_TOTAL_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MEAN_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MIN_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MAX_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_TOTAL_AREA,
                COLTYPE_INTEGER,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_PERCENT_MAXIMAL,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MAD_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_LOWER_QUARTILE,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_UPPER_QUARTILE,
                COLTYPE_FLOAT,
            ),
        ):
            # feature names are now formatting strings
            feature_name = feature % expected_suffix
            assert any(
                [
                    (column[1] == feature_name and column[2] == coltype)
                    for column in columns
                ]
            )

def test_get_measurement_columns_percentile_mode(module):
    image_names = ["image%d" % i for i in range(3)]

    module.wants_objects.value = False
    module.wants_percentiles.value = True
    module.percentiles.value = "5,10,15"

    expected_suffixes = []

    for image_name in image_names:
        im = module.images_list.value[-1]

        module.images_list.value.append(image_name)

        expected_suffixes.append(image_name)

    columns = module.get_measurement_columns(None)

    assert all([column[0] == "Image" for column in columns])

    for expected_suffix in expected_suffixes:
        for feature, coltype in (
            (
                cellprofiler.modules.measureimageintensity.F_TOTAL_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MEAN_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MIN_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MAX_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_TOTAL_AREA,
                COLTYPE_INTEGER,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_PERCENT_MAXIMAL,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_MAD_INTENSITY,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_LOWER_QUARTILE,
                COLTYPE_FLOAT,
            ),
            (
                cellprofiler.modules.measureimageintensity.F_UPPER_QUARTILE,
                COLTYPE_FLOAT,
            ),
            (
                "Intensity_Percentile_5_%s",
                COLTYPE_FLOAT,
            ),
            (
                "Intensity_Percentile_10_%s",
                COLTYPE_FLOAT,
            ),
            (
                "Intensity_Percentile_15_%s",
                COLTYPE_FLOAT,
            ),
        ):
            # feature names are now formatting strings
            feature_name = feature % expected_suffix

            assert any(
                [
                    (column[1] == feature_name and column[2] == coltype)
                    for column in columns
                ]
            )