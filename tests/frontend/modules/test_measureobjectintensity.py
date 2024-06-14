import math

import centrosome.outline
import numpy
import numpy.testing
import pytest
import skimage.measure
import skimage.segmentation

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import EXPERIMENT, COLTYPE_FLOAT, C_LOCATION


import cellprofiler.modules.measureobjectintensity
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

IMAGE_NAME = "MyImage"
OBJECT_NAME = "MyObjects"


@pytest.fixture(scope="function")
def image():
    return cellprofiler_core.image.Image()


@pytest.fixture(scope="function")
def measurements():
    return cellprofiler_core.measurement.Measurements()


@pytest.fixture(scope="function")
def module():
    module = cellprofiler.modules.measureobjectintensity.MeasureObjectIntensity()

    module.images_list.value = IMAGE_NAME

    module.objects_list.value = OBJECT_NAME

    return module


@pytest.fixture(scope="function")
def objects(image):
    objects = cellprofiler_core.object.Objects()

    objects.parent_image = image

    return objects


@pytest.fixture(scope="function")
def volume(image):
    data = numpy.zeros((11, 11, 20))

    k, i, j = numpy.mgrid[-5:6, -5:6, -5:15]
    data[k ** 2 + i ** 2 + j ** 2 <= 25] = 0.5
    data[k ** 2 + i ** 2 + j ** 2 <= 16] = 0.25
    data[k ** 2 + i ** 2 + j ** 2 == 0] = 1

    k, i, j = numpy.mgrid[-5:6, -5:6, -15:5]
    data[k ** 2 + i ** 2 + j ** 2 <= 9] = 0.5
    data[k ** 2 + i ** 2 + j ** 2 <= 4] = 0.75
    data[k ** 2 + i ** 2 + j ** 2 == 0] = 1

    image.set_image(data, convert=False)

    image.dimensions = 3

    return image


@pytest.fixture(scope="function")
def workspace(image, measurements, module, objects):
    image_set_list = cellprofiler_core.image.ImageSetList()

    image_set = image_set_list.get_image_set(0)

    image_set.add(IMAGE_NAME, image)

    object_set = cellprofiler_core.object.ObjectSet()

    object_set.add_objects(objects, OBJECT_NAME)

    return cellprofiler_core.workspace.Workspace(
        cellprofiler_core.pipeline.Pipeline(),
        module,
        image_set,
        object_set,
        measurements,
        image_set_list,
    )


def assert_features_and_columns_match(measurements, module):
    object_names = [
        x
        for x in measurements.get_object_names()
        if x
        not in (
            "Image",
            EXPERIMENT,
        )
    ]

    features = [
        [f for f in measurements.get_feature_names(object_name) if f != "Exit_Status"]
        for object_name in object_names
    ]

    columns = module.get_measurement_columns(None)

    assert sum([len(f) for f in features]) == len(columns)

    for column in columns:
        index = object_names.index(column[0])

        assert column[1] in features[index]

        assert column[2] == COLTYPE_FLOAT


def test_supplied_measurements(module):
    """Test the get_category / get_measurements, get_measurement_images functions"""
    module.images_list.value = "MyImage"

    module.objects_list.value = "MyObjects1, MyObjects2"

    expected_categories = tuple(
        sorted(
            [
                cellprofiler.modules.measureobjectintensity.INTENSITY,
                C_LOCATION,
            ]
        )
    )

    assert (
        tuple(sorted(module.get_categories(None, "MyObjects1"))) == expected_categories
    )

    assert module.get_categories(None, "Foo") == []

    measurements = module.get_measurements(
        None, "MyObjects1", cellprofiler.modules.measureobjectintensity.INTENSITY
    )

    assert len(measurements) == len(
        cellprofiler.modules.measureobjectintensity.ALL_MEASUREMENTS
    )

    measurements = module.get_measurements(
        None, "MyObjects1", C_LOCATION
    )

    assert len(measurements) == len(
        cellprofiler.modules.measureobjectintensity.ALL_LOCATION_MEASUREMENTS
    )

    assert all(
        [
            m in cellprofiler.modules.measureobjectintensity.ALL_LOCATION_MEASUREMENTS
            for m in measurements
        ]
    )

    assert module.get_measurement_images(
        None,
        "MyObjects1",
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.MAX_INTENSITY,
    ) == ["MyImage"]


def test_get_measurement_columns(module):
    """test the get_measurement_columns method"""
    module.images_list.value = "MyImage"

    module.objects_list.value = "MyObjects1, MyObjects2"

    columns = module.get_measurement_columns(None)

    assert len(columns) == 2 * (
        len(cellprofiler.modules.measureobjectintensity.ALL_MEASUREMENTS)
        + len(cellprofiler.modules.measureobjectintensity.ALL_LOCATION_MEASUREMENTS)
    )

    for column in columns:
        assert column[0] in ("MyObjects1", "MyObjects2")

        assert column[2], COLTYPE_FLOAT

        category = column[1].split("_")[0]

        assert category in (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            C_LOCATION,
        )

        if category == cellprofiler.modules.measureobjectintensity.INTENSITY:
            assert column[1][column[1].find("_") + 1 :] in [
                m + "_MyImage"
                for m in cellprofiler.modules.measureobjectintensity.ALL_MEASUREMENTS
            ]
        else:
            assert column[1][column[1].find("_") + 1 :] in [
                m + "_MyImage"
                for m in cellprofiler.modules.measureobjectintensity.ALL_LOCATION_MEASUREMENTS
            ]


def test_zero(image, measurements, module, objects, workspace):
    """Make sure we can process a blank image"""
    image.pixel_data = numpy.zeros((10, 10))

    objects.segmented = numpy.zeros((10, 10))

    module.run(workspace)

    for category, features in (
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.ALL_MEASUREMENTS,
        ),
        (
            C_LOCATION,
            cellprofiler.modules.measureobjectintensity.ALL_LOCATION_MEASUREMENTS,
        ),
    ):
        for meas_name in features:
            feature_name = "%s_%s_%s" % (category, meas_name, "MyImage")

            data = measurements.get_current_measurement("MyObjects", feature_name)

            assert numpy.product(data.shape) == 0, (
                "Got data for feature %s" % feature_name
            )

        assert_features_and_columns_match(measurements, module)


def test_masked(image, measurements, module, objects, workspace):
    """Make sure we can process a completely masked image

    Regression test of IMG-971
    """
    image.pixel_data = numpy.zeros((10, 10))

    image.mask = numpy.zeros((10, 10), bool)

    objects.segmented = numpy.ones((10, 10), int)

    module.run(workspace)

    for meas_name in cellprofiler.modules.measureobjectintensity.ALL_MEASUREMENTS:
        feature_name = "%s_%s_%s" % (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            meas_name,
            "MyImage",
        )

        data = measurements.get_current_measurement("MyObjects", feature_name)

        assert numpy.product(data.shape) == 1

        assert numpy.all(numpy.isnan(data) | (data == 0))

    assert_features_and_columns_match(measurements, module)


def test_one(image, measurements, module, objects, workspace):
    """Check measurements on a 3x3 square of 1's"""
    data = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    image.pixel_data = data.astype(float)

    objects.segmented = data.astype(int)

    module.run(workspace)

    for category, meas_name, value in (
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.INTEGRATED_INTENSITY,
            9,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MEAN_INTENSITY,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.STD_INTENSITY,
            0,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MIN_INTENSITY,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MAX_INTENSITY,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.INTEGRATED_INTENSITY_EDGE,
            8,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MEAN_INTENSITY_EDGE,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.STD_INTENSITY_EDGE,
            0,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MIN_INTENSITY_EDGE,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MAX_INTENSITY_EDGE,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MASS_DISPLACEMENT,
            0,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.LOWER_QUARTILE_INTENSITY,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.MEDIAN_INTENSITY,
            1,
        ),
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.UPPER_QUARTILE_INTENSITY,
            1,
        ),
        (
            C_LOCATION,
            cellprofiler.modules.measureobjectintensity.LOC_CMI_X,
            3,
        ),
        (
            C_LOCATION,
            cellprofiler.modules.measureobjectintensity.LOC_CMI_Y,
            2,
        ),
    ):
        feature_name = "%s_%s_%s" % (category, meas_name, "MyImage")

        data = measurements.get_current_measurement("MyObjects", feature_name)

        assert numpy.product(data.shape) == 1

        assert data[0] == value, "%s expected %f != actual %f" % (
            meas_name,
            value,
            data[0],
        )


def test_one_masked(image, measurements, module, objects, workspace):
    """Check measurements on a 3x3 square of 1's"""
    img = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    mask = img > 0

    image.pixel_data = img.astype(float)

    image.mask = mask

    objects.segmented = img.astype(int)

    module.run(workspace)

    for meas_name, value in (
        (cellprofiler.modules.measureobjectintensity.INTEGRATED_INTENSITY, 9),
        (cellprofiler.modules.measureobjectintensity.MEAN_INTENSITY, 1),
        (cellprofiler.modules.measureobjectintensity.STD_INTENSITY, 0),
        (cellprofiler.modules.measureobjectintensity.MIN_INTENSITY, 1),
        (cellprofiler.modules.measureobjectintensity.MAX_INTENSITY, 1),
        (cellprofiler.modules.measureobjectintensity.INTEGRATED_INTENSITY_EDGE, 8),
        (cellprofiler.modules.measureobjectintensity.MEAN_INTENSITY_EDGE, 1),
        (cellprofiler.modules.measureobjectintensity.STD_INTENSITY_EDGE, 0),
        (cellprofiler.modules.measureobjectintensity.MIN_INTENSITY_EDGE, 1),
        (cellprofiler.modules.measureobjectintensity.MAX_INTENSITY_EDGE, 1),
        (cellprofiler.modules.measureobjectintensity.MASS_DISPLACEMENT, 0),
        (cellprofiler.modules.measureobjectintensity.LOWER_QUARTILE_INTENSITY, 1),
        (cellprofiler.modules.measureobjectintensity.MEDIAN_INTENSITY, 1),
        (cellprofiler.modules.measureobjectintensity.MAD_INTENSITY, 0),
        (cellprofiler.modules.measureobjectintensity.UPPER_QUARTILE_INTENSITY, 1),
    ):
        feature_name = "%s_%s_%s" % (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            meas_name,
            "MyImage",
        )

        data = measurements.get_current_measurement("MyObjects", feature_name)

        assert numpy.product(data.shape) == 1

        assert data[0] == value, "%s expected %f != actual %f" % (
            meas_name,
            value,
            data[0],
        )


def test_intensity_location(image, measurements, module, objects, workspace):
    data = (
        numpy.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 2, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ).astype(float)
        / 2.0
    )

    image.pixel_data = data

    labels = (data != 0).astype(int)

    objects.segmented = labels

    module.run(workspace)

    for feature, value in (
        (cellprofiler.modules.measureobjectintensity.LOC_MAX_X, 5),
        (cellprofiler.modules.measureobjectintensity.LOC_MAX_Y, 2),
    ):
        feature_name = "%s_%s_%s" % (
            C_LOCATION,
            feature,
            "MyImage",
        )

        values = measurements.get_current_measurement(OBJECT_NAME, feature_name)

        assert len(values) == 1

        assert values[0] == value


def test_mass_displacement(image, measurements, module, objects, workspace):
    """Check the mass displacement of three squares"""

    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    data = numpy.zeros(labels.shape, dtype=float)
    #
    # image # 1 has a single value in one of the corners
    # whose distance is sqrt(8) from the center
    #
    data[1, 1] = 1

    # image # 2 has a single value on the top edge
    # and should have distance 2
    #
    data[7, 3] = 1

    # image # 3 has a single value on the left edge
    # and should have distance 2
    data[15, 1] = 1

    image.pixel_data = data

    objects.segmented = labels

    module.run(workspace)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.MASS_DISPLACEMENT,
        "MyImage",
    )

    mass_displacement = measurements.get_current_measurement("MyObjects", feature_name)

    assert numpy.product(mass_displacement.shape) == 3

    numpy.testing.assert_almost_equal(mass_displacement[0], math.sqrt(8.0))

    numpy.testing.assert_almost_equal(mass_displacement[1], 2.0)

    numpy.testing.assert_almost_equal(mass_displacement[2], 2.0)


def test_mass_displacement_masked(image, measurements, module, objects, workspace):
    """Regression test IMG-766 - mass displacement of a masked image"""
    labels = numpy.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    data = numpy.zeros(labels.shape, dtype=float)

    #
    # image # 1 has a single value in one of the corners
    # whose distance is sqrt(8) from the center
    #
    data[1, 1] = 1

    # image # 2 has a single value on the top edge
    # and should have distance 2
    #
    data[7, 3] = 1

    # image # 3 has a single value on the left edge
    # and should have distance 2
    data[15, 1] = 1

    mask = numpy.zeros(data.shape, bool)

    mask[labels > 0] = True

    image.pixel_data = data

    image.mask = mask

    objects.segmented = labels

    module.run(workspace)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.MASS_DISPLACEMENT,
        "MyImage",
    )

    mass_displacement = measurements.get_current_measurement("MyObjects", feature_name)

    assert numpy.product(mass_displacement.shape) == 3

    numpy.testing.assert_almost_equal(mass_displacement[0], math.sqrt(8.0))

    numpy.testing.assert_almost_equal(mass_displacement[1], 2.0)

    numpy.testing.assert_almost_equal(mass_displacement[2], 2.0)


def test_quartiles_uniform(image, measurements, module, objects, workspace):
    """test quartile values on a 250x250 square filled with uniform values"""
    labels = numpy.ones((250, 250), int)

    numpy.random.seed(0)

    data = numpy.random.uniform(size=(250, 250))

    image.pixel_data = data

    objects.segmented = labels

    module.run(workspace)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.LOWER_QUARTILE_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 0.25, 2)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.MEDIAN_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 0.50, 2)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.MAD_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 0.25, 2)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.UPPER_QUARTILE_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 0.75, 2)


def test_quartiles_one_pixel(image, module, objects, workspace):
    """Regression test a bug that occurs in an image with one pixel"""
    labels = numpy.zeros((10, 20))

    labels[2:7, 3:8] = 1

    labels[5, 15] = 2

    numpy.random.seed(0)

    data = numpy.random.uniform(size=(10, 20))

    image.pixel_data = data

    objects.segmented = labels

    # Crashes when pipeline runs in measureobjectintensity.py revision 7146
    module.run(workspace)


def test_quartiles_four_objects(image, measurements, module, objects, workspace):
    """test quartile values on a 250x250 square with 4 objects"""
    labels = numpy.ones((250, 250), int)

    labels[125:, :] += 1

    labels[:, 125:] += 2

    numpy.random.seed(0)

    data = numpy.random.uniform(size=(250, 250))

    #
    # Make the distributions center around .5, .25, 1/6 and .125
    #
    data /= labels.astype(float)

    image.pixel_data = data

    objects.segmented = labels

    module.run(workspace)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.LOWER_QUARTILE_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 1.0 / 4.0, 2)

    numpy.testing.assert_almost_equal(data[1], 1.0 / 8.0, 2)

    numpy.testing.assert_almost_equal(data[2], 1.0 / 12.0, 2)

    numpy.testing.assert_almost_equal(data[3], 1.0 / 16.0, 2)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.MEDIAN_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 1.0 / 2.0, 2)

    numpy.testing.assert_almost_equal(data[1], 1.0 / 4.0, 2)

    numpy.testing.assert_almost_equal(data[2], 1.0 / 6.0, 2)

    numpy.testing.assert_almost_equal(data[3], 1.0 / 8.0, 2)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.UPPER_QUARTILE_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 3.0 / 4.0, 2)

    numpy.testing.assert_almost_equal(data[1], 3.0 / 8.0, 2)

    numpy.testing.assert_almost_equal(data[2], 3.0 / 12.0, 2)

    numpy.testing.assert_almost_equal(data[3], 3.0 / 16.0, 2)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.MAD_INTENSITY,
        "MyImage",
    )

    data = measurements.get_current_measurement("MyObjects", feature_name)

    numpy.testing.assert_almost_equal(data[0], 1.0 / 4.0, 2)

    numpy.testing.assert_almost_equal(data[1], 1.0 / 8.0, 2)

    numpy.testing.assert_almost_equal(data[2], 1.0 / 12.0, 2)

    numpy.testing.assert_almost_equal(data[3], 1.0 / 16.0, 2)


def test_median_intensity_masked(image, measurements, module, objects, workspace):
    numpy.random.seed(37)

    labels = numpy.ones((10, 10), int)

    mask = numpy.ones((10, 10), bool)

    mask[:, :5] = False

    pixel_data = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)

    pixel_data[~mask] = 1

    image.pixel_data = pixel_data

    image.mask = mask

    objects.segmented = labels

    expected = numpy.sort(pixel_data[mask])[numpy.sum(mask) // 2]

    assert not expected == numpy.median(pixel_data)

    module.run(workspace)

    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)

    values = measurements.get_current_measurement(
        OBJECT_NAME,
        "_".join(
            (
                cellprofiler.modules.measureobjectintensity.INTENSITY,
                cellprofiler.modules.measureobjectintensity.MEDIAN_INTENSITY,
                IMAGE_NAME,
            )
        ),
    )

    assert len(values) == 1

    assert expected == values[0]


def test_std_intensity(image, measurements, module, objects, workspace):
    numpy.random.seed(38)

    labels = numpy.ones((40, 30), int)

    labels[:, 15:] = 3

    labels[20:, :] += 1

    data = numpy.random.uniform(size=(40, 30)).astype(numpy.float32)

    image.pixel_data = data

    objects.segmented = labels

    module.run(workspace)

    values = measurements.get_current_measurement(
        OBJECT_NAME,
        "_".join(
            (
                cellprofiler.modules.measureobjectintensity.INTENSITY,
                cellprofiler.modules.measureobjectintensity.STD_INTENSITY,
                IMAGE_NAME,
            )
        ),
    )

    assert len(values) == 4

    for i in range(1, 5):
        numpy.testing.assert_almost_equal(values[i - 1], numpy.std(data[labels == i]))


def test_std_intensity_edge(image, measurements, module, objects, workspace):
    numpy.random.seed(39)

    labels = numpy.ones((40, 30), int)

    labels[:, 15:] = 3

    labels[20:, :] += 1

    edge_mask = skimage.segmentation.find_boundaries(labels, mode="inner")

    elabels = labels * edge_mask

    pixel_data = numpy.random.uniform(size=(40, 30)).astype(numpy.float32)

    image.pixel_data = pixel_data

    objects.segmented = labels

    module.run(workspace)

    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)

    values = measurements.get_current_measurement(
        OBJECT_NAME,
        "_".join(
            (
                cellprofiler.modules.measureobjectintensity.INTENSITY,
                cellprofiler.modules.measureobjectintensity.STD_INTENSITY_EDGE,
                IMAGE_NAME,
            )
        ),
    )

    assert len(values) == 4

    for i in range(1, 5):
        numpy.testing.assert_almost_equal(
            values[i - 1], numpy.std(pixel_data[elabels == i])
        )


def test_ijv(image, module, objects, workspace):
    #
    # Test the module on overlapping objects
    #
    numpy.random.seed(310)

    i, j = numpy.mgrid[0:30, 0:35]

    o1 = numpy.argwhere((i - 10) ** 2 + (j - 10) ** 2 < 49)  # circle radius 7 at 10,10

    o2 = numpy.argwhere((i - 12) ** 2 + (j - 15) ** 2 < 25)  # circle radius 5 at 12,15

    o3 = numpy.argwhere((i - 15) ** 2 + (j - 25) ** 2 < 49)  # circle radius 7 at 15, 25

    labels = numpy.vstack(
        [
            numpy.column_stack([x, n * numpy.ones(x.shape[0], int)])
            for n, x in ((1, o1), (2, o2), (3, o3))
        ]
    )

    pixel_data = numpy.random.uniform(size=i.shape)

    image.pixel_data = pixel_data

    objects.ijv = labels

    module.run(workspace)

    measurements0 = workspace.measurements

    expected = {}

    for cname, fnames in (
        (
            cellprofiler.modules.measureobjectintensity.INTENSITY,
            cellprofiler.modules.measureobjectintensity.ALL_MEASUREMENTS,
        ),
        (
            C_LOCATION,
            cellprofiler.modules.measureobjectintensity.ALL_LOCATION_MEASUREMENTS,
        ),
    ):
        for fname in fnames:
            mname = "_".join([cname, fname, IMAGE_NAME])

            m0 = measurements0.get_measurement(OBJECT_NAME, mname)

            expected[mname] = m0

    for i, ijv in enumerate(
        [
            numpy.column_stack([o1, numpy.ones(o1.shape[0], int)]),
            numpy.column_stack([o2, numpy.ones(o2.shape[0], int)]),
            numpy.column_stack([o3, numpy.ones(o3.shape[0], int)]),
        ]
    ):
        objects.ijv = ijv

        module.run(workspace)

        measurements1 = workspace.measurements

        for mname, m0 in list(expected.items()):
            m1 = measurements1.get_measurement(OBJECT_NAME, mname)

            assert len(m0) == 3, "{} feature expected 3 measurements, got {}".format(
                mname, len(m0)
            )

            assert len(m1) == 1, "{} feature expected 1 measurement, got {}".format(
                mname, len(m1)
            )

            assert m0[i] == m1[0], "{} feature expected {}, got {}".format(
                mname, m0[i], m1[0]
            )


def test_wrong_image_size(image, measurements, module, objects, workspace):
    """Regression test of IMG-961 - object and image size differ"""
    numpy.random.seed(41)

    labels = numpy.ones((20, 50), int)

    pixel_data = numpy.random.uniform(size=(30, 40)).astype(numpy.float32)

    image.pixel_data = pixel_data

    objects.segmented = labels

    module.run(workspace)

    feature_name = "%s_%s_%s" % (
        cellprofiler.modules.measureobjectintensity.INTENSITY,
        cellprofiler.modules.measureobjectintensity.INTEGRATED_INTENSITY,
        "MyImage",
    )

    m = measurements.get_current_measurement("MyObjects", feature_name)

    assert len(m) == 1

    numpy.testing.assert_almost_equal(m[0], numpy.sum(pixel_data[:20, :40]), 4)


def test_masked_edge(image, module, objects, workspace):
    # Regression test of issue #1115
    labels = numpy.zeros((20, 50), int)

    labels[15:25, 15:25] = 1

    pixel_data = numpy.random.uniform(size=labels.shape).astype(numpy.float32)

    #
    # Mask the edge of the object
    #
    mask = ~centrosome.outline.outline(labels).astype(bool)

    image.pixel_data = pixel_data

    image.mask = mask

    objects.segmented = labels

    module.run(workspace)


def test_image(measurements, module, objects, image, workspace):
    data = numpy.zeros((11, 20))

    i, j = numpy.mgrid[-5:6, -5:15]
    data[i ** 2 + j ** 2 <= 25] = 0.5
    data[i ** 2 + j ** 2 <= 16] = 0.25
    data[i ** 2 + j ** 2 == 0] = 1

    i, j = numpy.mgrid[-5:6, -15:5]
    data[i ** 2 + j ** 2 <= 9] = 0.5
    data[i ** 2 + j ** 2 <= 4] = 0.75
    data[i ** 2 + j ** 2 == 0] = 1

    image.pixel_data = data

    image.dimensions = 2

    labels = skimage.measure.label(data > 0)

    objects.segmented = labels

    module.run(workspace)

    expected = {
        "Intensity_IntegratedIntensity_MyImage": [29.0, 18.0],
        "Intensity_MeanIntensity_MyImage": [0.35802469135802467, 0.62068965517241381],
        "Intensity_StdIntensity_MyImage": [0.14130275484271107, 0.14112677192883538],
        "Intensity_MinIntensity_MyImage": [0.25, 0.5],
        "Intensity_MaxIntensity_MyImage": [1.0, 1.0],
        "Intensity_IntegratedIntensityEdge_MyImage": [14.0, 8.0],
        "Intensity_MeanIntensityEdge_MyImage": [0.5, 0.5],
        "Intensity_StdIntensityEdge_MyImage": [0.0, 0.0],
        "Intensity_MinIntensityEdge_MyImage": [0.5, 0.5],
        "Intensity_MaxIntensityEdge_MyImage": [0.5, 0.5],
        "Intensity_MassDisplacement_MyImage": [0.0, 0.0],
        "Intensity_LowerQuartileIntensity_MyImage": [0.25, 0.5],
        "Intensity_MedianIntensity_MyImage": [0.25, 0.5],
        "Intensity_MADIntensity_MyImage": [0.0, 0.0],
        "Intensity_UpperQuartileIntensity_MyImage": [0.5, 0.75],
        "Location_CenterMassIntensity_X_MyImage": [5.0, 15.0],
        "Location_CenterMassIntensity_Y_MyImage": [5.0, 5.0],
        "Location_CenterMassIntensity_Z_MyImage": [0.0, 0.0],
        "Location_MaxIntensity_X_MyImage": [5.0, 15.0],
        "Location_MaxIntensity_Y_MyImage": [5.0, 5.0],
        "Location_MaxIntensity_Z_MyImage": [0.0, 0.0],
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_measurement(OBJECT_NAME, feature)

        numpy.testing.assert_array_almost_equal(
            value,
            actual,
            err_msg="{} expected {}, got {}".format(feature, value, actual),
        )


def test_image_masked(measurements, module, objects, image, workspace):
    data = numpy.zeros((11, 20))

    i, j = numpy.mgrid[-5:6, -5:15]
    data[i ** 2 + j ** 2 <= 25] = 0.5
    data[i ** 2 + j ** 2 <= 16] = 0.25
    data[i ** 2 + j ** 2 == 0] = 1

    i, j = numpy.mgrid[-5:6, -15:5]
    data[i ** 2 + j ** 2 <= 9] = 0.5
    data[i ** 2 + j ** 2 <= 4] = 0.75
    data[i ** 2 + j ** 2 == 0] = 1

    mask = numpy.ones_like(data)
    mask[:, 11:] = 0

    image.pixel_data = data
    image.dimensions = 2
    image.mask = mask > 0

    labels = skimage.measure.label(data > 0)

    objects.segmented = labels

    module.run(workspace)

    expected = {
        "Intensity_IntegratedIntensity_MyImage": [29.0, 0.0],
        "Intensity_MeanIntensity_MyImage": [0.35802469135802467, numpy.nan],
        "Intensity_StdIntensity_MyImage": [0.14130275484271107, numpy.nan],
        "Intensity_MinIntensity_MyImage": [0.25, 0.0],
        "Intensity_MaxIntensity_MyImage": [1.0, 0.0],
        "Intensity_IntegratedIntensityEdge_MyImage": [14.0, 0.0],
        "Intensity_MeanIntensityEdge_MyImage": [0.5, numpy.nan],
        "Intensity_StdIntensityEdge_MyImage": [0.0, numpy.nan],
        "Intensity_MinIntensityEdge_MyImage": [0.5, 0.0],
        "Intensity_MaxIntensityEdge_MyImage": [0.5, 0.0],
        "Intensity_MassDisplacement_MyImage": [0.0, numpy.nan],
        "Intensity_LowerQuartileIntensity_MyImage": [0.25, 0.0],
        "Intensity_MedianIntensity_MyImage": [0.25, 0.0],
        "Intensity_MADIntensity_MyImage": [0.0, 0.0],
        "Intensity_UpperQuartileIntensity_MyImage": [0.5, 0.0],
        "Location_CenterMassIntensity_X_MyImage": [5.0, numpy.nan],
        "Location_CenterMassIntensity_Y_MyImage": [5.0, numpy.nan],
        "Location_CenterMassIntensity_Z_MyImage": [0.0, numpy.nan],
        "Location_MaxIntensity_X_MyImage": [5.0, 5.0],
        "Location_MaxIntensity_Y_MyImage": [5.0, 0.0],
        "Location_MaxIntensity_Z_MyImage": [0.0, 0.0],
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_measurement(OBJECT_NAME, feature)

        numpy.testing.assert_array_almost_equal(
            value,
            actual,
            err_msg="{} expected {}, got {}".format(feature, value, actual),
        )


def test_volume(measurements, module, objects, volume, workspace):
    labels = skimage.measure.label(volume.pixel_data > 0)

    objects.segmented = labels

    module.run(workspace)

    expected = {
        "Intensity_IntegratedIntensity_MyImage": [194.0, 70.0],
        "Intensity_MeanIntensity_MyImage": [0.37669902912621361, 0.56910569105691056],
        "Intensity_StdIntensity_MyImage": [0.12786816898600722, 0.11626300525264302],
        "Intensity_MinIntensity_MyImage": [0.25, 0.5],
        "Intensity_MaxIntensity_MyImage": [1.0, 1.0],
        "Intensity_IntegratedIntensityEdge_MyImage": [111.0, 45.0],
        "Intensity_MeanIntensityEdge_MyImage": [0.5, 0.5],
        "Intensity_StdIntensityEdge_MyImage": [0.0, 0.0],
        "Intensity_MinIntensityEdge_MyImage": [0.5, 0.5],
        "Intensity_MaxIntensityEdge_MyImage": [0.5, 0.5],
        "Intensity_MassDisplacement_MyImage": [0.0, 0.0],
        "Intensity_LowerQuartileIntensity_MyImage": [0.25, 0.5],
        "Intensity_MedianIntensity_MyImage": [0.5, 0.5],
        "Intensity_MADIntensity_MyImage": [0.0, 0.0],
        "Intensity_UpperQuartileIntensity_MyImage": [0.5, 0.75],
        "Location_CenterMassIntensity_X_MyImage": [5.0, 15.0],
        "Location_CenterMassIntensity_Y_MyImage": [5.0, 5.0],
        "Location_CenterMassIntensity_Z_MyImage": [5.0, 5.0],
        "Location_MaxIntensity_X_MyImage": [5.0, 15.0],
        "Location_MaxIntensity_Y_MyImage": [5.0, 5.0],
        "Location_MaxIntensity_Z_MyImage": [5.0, 5.0],
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_measurement(OBJECT_NAME, feature)

        numpy.testing.assert_array_almost_equal(
            value,
            actual,
            err_msg="{} expected {}, got {}".format(feature, value, actual),
        )


def test_volume_masked(measurements, module, objects, volume, workspace):
    mask = numpy.ones_like(volume.pixel_data)

    mask[:, :, 11:] = 0

    volume.mask = mask > 0

    labels = skimage.measure.label(volume.pixel_data > 0)

    objects.segmented = labels

    module.run(workspace)

    expected = {
        "Intensity_IntegratedIntensity_MyImage": [194.0, 0.0],
        "Intensity_MeanIntensity_MyImage": [0.37669902912621361, numpy.nan],
        "Intensity_StdIntensity_MyImage": [0.12786816898600722, numpy.nan],
        "Intensity_MinIntensity_MyImage": [0.25, 0.0],
        "Intensity_MaxIntensity_MyImage": [1.0, 0.0],
        "Intensity_IntegratedIntensityEdge_MyImage": [111.0, 0.0],
        "Intensity_MeanIntensityEdge_MyImage": [0.5, numpy.nan],
        "Intensity_StdIntensityEdge_MyImage": [0.0, numpy.nan],
        "Intensity_MinIntensityEdge_MyImage": [0.5, 0.0],
        "Intensity_MaxIntensityEdge_MyImage": [0.5, 0.0],
        "Intensity_MassDisplacement_MyImage": [0.0, numpy.nan],
        "Intensity_LowerQuartileIntensity_MyImage": [0.25, 0.0],
        "Intensity_MedianIntensity_MyImage": [0.5, 0.0],
        "Intensity_MADIntensity_MyImage": [0.0, 0.0],
        "Intensity_UpperQuartileIntensity_MyImage": [0.5, 0.0],
        "Location_CenterMassIntensity_X_MyImage": [5.0, numpy.nan],
        "Location_CenterMassIntensity_Y_MyImage": [5.0, numpy.nan],
        "Location_CenterMassIntensity_Z_MyImage": [5.0, numpy.nan],
        "Location_MaxIntensity_X_MyImage": [5.0, 5.0],
        "Location_MaxIntensity_Y_MyImage": [5.0, 5.0],
        "Location_MaxIntensity_Z_MyImage": [5.0, 0.0],
    }

    for feature, value in list(expected.items()):
        actual = measurements.get_measurement(OBJECT_NAME, feature)

        numpy.testing.assert_array_almost_equal(
            value,
            actual,
            err_msg="{} expected {}, got {}".format(feature, value, actual),
        )
