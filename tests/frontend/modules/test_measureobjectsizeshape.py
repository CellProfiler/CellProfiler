import io
import os
import numpy
import skimage.io

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.modules.injectimage
import cellprofiler.modules.measureobjectsizeshape
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT

import tests.frontend
import tests.frontend.modules

OBJECTS_NAME = "myobjects"


def make_workspace(labels):
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, OBJECTS_NAME)
    m = cellprofiler_core.measurement.Measurements()
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    module.set_module_num(1)
    module.objects_list.value = OBJECTS_NAME
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, image_set, object_set, m, image_set_list
    )
    return workspace, module


def test_01_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("measureobjectsizeshape/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.measureobjectsizeshape.MeasureObjectSizeShape
    )
    assert len(module.objects_list.value) == 2
    for object_name in module.objects_list.value:
        assert object_name in ("Nuclei", "Cells")
    assert module.calculate_zernikes


def test_zeros():
    """Run on an empty labels matrix"""
    object_set = cellprofiler_core.object.ObjectSet()
    labels = numpy.zeros((10, 20), int)
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "SomeObjects")
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    settings = ["SomeObjects", "Yes"]
    module.set_settings_from_values(settings, 1, module.module_class())
    module.set_module_num(1)
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        measurements,
        image_set_list,
    )
    module.run(workspace)

    for f in (
        cellprofiler.modules.measureobjectsizeshape.F_AREA,
        cellprofiler.modules.measureobjectsizeshape.F_CENTER_X,
        cellprofiler.modules.measureobjectsizeshape.F_CENTER_Y,
        cellprofiler.modules.measureobjectsizeshape.F_ECCENTRICITY,
        cellprofiler.modules.measureobjectsizeshape.F_EULER_NUMBER,
        cellprofiler.modules.measureobjectsizeshape.F_EXTENT,
        cellprofiler.modules.measureobjectsizeshape.F_FORM_FACTOR,
        cellprofiler.modules.measureobjectsizeshape.F_MAJOR_AXIS_LENGTH,
        cellprofiler.modules.measureobjectsizeshape.F_MINOR_AXIS_LENGTH,
        cellprofiler.modules.measureobjectsizeshape.F_ORIENTATION,
        cellprofiler.modules.measureobjectsizeshape.F_PERIMETER,
        cellprofiler.modules.measureobjectsizeshape.F_SOLIDITY,
        cellprofiler.modules.measureobjectsizeshape.F_COMPACTNESS,
        cellprofiler.modules.measureobjectsizeshape.F_MAXIMUM_RADIUS,
        cellprofiler.modules.measureobjectsizeshape.F_MEAN_RADIUS,
        cellprofiler.modules.measureobjectsizeshape.F_MEDIAN_RADIUS,
        cellprofiler.modules.measureobjectsizeshape.F_MIN_FERET_DIAMETER,
        cellprofiler.modules.measureobjectsizeshape.F_MAX_FERET_DIAMETER,
    ):
        m = cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE + "_" + f
        a = measurements.get_current_measurement("SomeObjects", m)
        assert len(a) == 0


def test_run():
    """Run with a rectangle, cross and circle"""
    object_set = cellprofiler_core.object.ObjectSet()
    labels = numpy.zeros((10, 20), int)
    labels[1:9, 1:5] = 1
    labels[1:9, 11] = 2
    labels[4, 6:19] = 2
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "SomeObjects")
    labels = numpy.zeros((115, 115), int)
    x, y = numpy.mgrid[-50:51, -50:51]
    labels[:101, :101][x ** 2 + y ** 2 <= 2500] = 1
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "OtherObjects")
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    settings = ["SomeObjects", "OtherObjects", "Yes"]
    module.set_settings_from_values(settings, 1, module.module_class())
    module.set_module_num(1)
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        measurements,
        image_set_list,
    )
    module.run(workspace)
    features_and_columns_match(measurements, module, pipeline)

    a = measurements.get_current_measurement("SomeObjects", "AreaShape_Area")
    assert len(a) == 2
    assert a[0] == 32
    assert a[1] == 20
    #
    # Mini-test of the form factor of a circle
    #
    ff = measurements.get_current_measurement("OtherObjects", "AreaShape_FormFactor")
    assert len(ff) == 1
    perim = measurements.get_current_measurement("OtherObjects", "AreaShape_Perimeter")
    area = measurements.get_current_measurement("OtherObjects", "AreaShape_Area")
    # The perimeter is obtained geometrically and is overestimated.
    expected = 100 * numpy.pi
    diff = abs((perim[0] - expected) / (perim[0] + expected))
    assert diff < 0.05, "perimeter off by %f" % diff
    wrongness = (perim[0] / expected) ** 2

    # It's an approximate circle...
    expected = numpy.pi * 50.0 ** 2
    diff = abs((area[0] - expected) / (area[0] + expected))
    assert diff < 0.05, "area off by %f" % diff
    wrongness *= expected / area[0]

    assert round(abs(ff[0] * wrongness - 1.0), 7) == 0
    for object_name, object_count in (("SomeObjects", 2), ("OtherObjects", 1)):
        for measurement in module.get_measurements(pipeline, object_name, "AreaShape"):
            feature_name = "AreaShape_%s" % measurement
            m = measurements.get_current_measurement(object_name, feature_name)
            assert len(m) == object_count


def test_categories():
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    settings = ["SomeObjects", "OtherObjects", "Yes"]
    module.set_settings_from_values(settings, 1, module.module_class())
    for object_name in settings[:-1]:
        categories = module.get_categories(None, object_name)
        assert len(categories) == 1
        assert categories[0] == "AreaShape"
    assert len(module.get_categories(None, "Bogus")) == 0


def test_measurements_zernike():
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    settings = ["SomeObjects", "OtherObjects", "Yes"]
    module.set_settings_from_values(settings, 1, module.module_class())
    pipeline = cellprofiler_core.pipeline.Pipeline()
    for object_name in settings[:-1]:
        measurements = module.get_measurements(pipeline, object_name, "AreaShape")
        for measurement in (
            cellprofiler.modules.measureobjectsizeshape.F_STANDARD
            + cellprofiler.modules.measureobjectsizeshape.F_STD_2D
        ):
            assert measurement in measurements
        assert "Zernike_3_1" in measurements


def test_measurements_no_zernike():
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    settings = ["SomeObjects", "OtherObjects", "No"]
    module.set_settings_from_values(settings, 1, module.module_class())
    pipeline = cellprofiler_core.pipeline.Pipeline()
    for object_name in settings[:-1]:
        measurements = module.get_measurements(pipeline, object_name, "AreaShape")
        for measurement in (
            cellprofiler.modules.measureobjectsizeshape.F_STANDARD
            + cellprofiler.modules.measureobjectsizeshape.F_STD_2D
        ):
            assert measurement in measurements
        assert not ("Zernike_3_1" in measurements)


def test_non_contiguous():
    """make sure MeasureObjectAreaShape doesn't crash if fed non-contiguous objects"""
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    module.objects_list.value.append("SomeObjects")
    module.calculate_zernikes.value = True
    object_set = cellprofiler_core.object.ObjectSet()
    labels = numpy.zeros((10, 20), int)
    labels[1:9, 1:5] = 1
    labels[4:6, 6:19] = 1
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "SomeObjects")
    module.set_module_num(1)
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        measurements,
        image_set_list,
    )
    module.run(workspace)
    values = measurements.get_current_measurement("SomeObjects", "AreaShape_Perimeter")
    assert len(values) == 1
    assert values[0] == 46


def test_zernikes_are_different():
    """Regression test of IMG-773"""

    numpy.random.seed(32)
    labels = numpy.zeros((40, 20), int)
    #
    # Make two "objects" composed of random foreground/background
    #
    labels[1:19, 1:19] = (numpy.random.uniform(size=(18, 18)) > 0.5).astype(int)
    labels[21:39, 1:19] = (numpy.random.uniform(size=(18, 18)) > 0.5).astype(int) * 2
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set = cellprofiler_core.object.ObjectSet()
    object_set.add_objects(objects, "SomeObjects")
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    module.objects_list.value.append("SomeObjects")
    module.calculate_zernikes.value = True
    module.set_module_num(1)
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        measurements,
        image_set_list,
    )
    module.run(workspace)
    features = [
        x[1]
        for x in module.get_measurement_columns(pipeline)
        if x[0] == "SomeObjects" and x[1].startswith("AreaShape_Zernike")
    ]
    for feature in features:
        values = measurements.get_current_measurement("SomeObjects", feature)
        assert len(values) == 2
        assert values[0] != values[1]


def test_extent():
    module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
    module.objects_list.value.append("SomeObjects")
    module.calculate_zernikes.value = True
    object_set = cellprofiler_core.object.ObjectSet()
    labels = numpy.zeros((10, 20), int)
    # 3/4 of a square is covered
    labels[5:7, 5:10] = 1
    labels[7:9, 5:15] = 1
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels
    object_set.add_objects(objects, "SomeObjects")
    module.set_module_num(1)
    image_set_list = cellprofiler_core.image.ImageSetList()
    measurements = cellprofiler_core.measurement.Measurements()
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        object_set,
        measurements,
        image_set_list,
    )
    module.run(workspace)
    values = measurements.get_current_measurement(
        "SomeObjects",
        "_".join(
            (
                cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE,
                cellprofiler.modules.measureobjectsizeshape.F_EXTENT,
            )
        ),
    )
    assert len(values) == 1
    assert round(abs(values[0] - 0.75), 7) == 0


def test_overlapping():
    """Test object measurement with two overlapping objects in ijv format"""

    i, j = numpy.mgrid[0:10, 0:20]
    m = (i > 1) & (i < 9) & (j > 1) & (j < 19)
    m1 = m & (i < j)
    m2 = m & (i < 9 - j)
    mlist = []
    olist = []
    for m in (m1, m2):
        objects = cellprofiler_core.object.Objects()
        objects.segmented = m.astype(int)
        olist.append(objects)
    ijv = numpy.column_stack(
        (
            numpy.hstack([numpy.argwhere(m)[:, 0] for m in (m1, m2)]),
            numpy.hstack([numpy.argwhere(m)[:, 1] for m in (m1, m2)]),
            numpy.array([1] * numpy.sum(m1) + [2] * numpy.sum(m2)),
        )
    )
    objects = cellprofiler_core.object.Objects()
    objects.ijv = ijv
    olist.append(objects)
    for objects in olist:
        module = cellprofiler.modules.measureobjectsizeshape.MeasureObjectAreaShape()
        module.objects_list.value.append("SomeObjects")
        module.calculate_zernikes.value = True
        object_set = cellprofiler_core.object.ObjectSet()
        object_set.add_objects(objects, "SomeObjects")
        module.set_module_num(1)
        image_set_list = cellprofiler_core.image.ImageSetList()
        measurements = cellprofiler_core.measurement.Measurements()
        mlist.append(measurements)
        pipeline = cellprofiler_core.pipeline.Pipeline()
        pipeline.add_module(module)

        def callback(caller, event):
            assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)
            pipeline.add_listener(callback)

        workspace = cellprofiler_core.workspace.Workspace(
            pipeline,
            module,
            image_set_list.get_image_set(0),
            object_set,
            measurements,
            image_set_list,
        )
        module.run(workspace)

    pipeline = cellprofiler_core.pipeline.Pipeline()
    for c in module.get_measurement_columns(pipeline):
        oname, feature = c[:2]
        if oname != "SomeObjects":
            continue
        measurements = mlist[0]
        assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
        v1 = measurements.get_current_measurement(oname, feature)
        assert len(v1) == 1
        v1 = v1[0]
        measurements = mlist[1]
        v2 = measurements.get_current_measurement(oname, feature)
        assert len(v2) == 1
        v2 = v2[0]
        expected = (v1, v2)
        v = mlist[2].get_current_measurement(oname, feature)
        if numpy.all(numpy.isnan(v)):
            assert numpy.all(numpy.isnan(v))
        else:
            assert tuple(v) == expected


def test_max_radius():
    labels = numpy.zeros((20, 10), int)
    labels[3:8, 3:6] = 1
    labels[11:19, 2:7] = 2
    workspace, module = make_workspace(labels)
    module.run(workspace)
    m = workspace.measurements
    max_radius = m.get_current_measurement(
        OBJECTS_NAME,
        cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE
        + "_"
        + cellprofiler.modules.measureobjectsizeshape.F_MAXIMUM_RADIUS,
    )
    assert len(max_radius) == 2
    assert max_radius[0] == 2
    assert max_radius[1] == 3


def features_and_columns_match(measurements, module, pipeline):
    assert len(measurements.get_object_names()) == 3
    assert "SomeObjects" in measurements.get_object_names()
    assert "OtherObjects" in measurements.get_object_names()
    features = measurements.get_feature_names("SomeObjects")
    features += measurements.get_feature_names("OtherObjects")
    columns = module.get_measurement_columns(pipeline)
    assert len(features) == len(columns)
    for column in columns:
        assert column[0] in ["SomeObjects", "OtherObjects"]
        assert column[1] in features
        assert column[2] == COLTYPE_FLOAT


def test_run_volume():
    labels = numpy.zeros((10, 20, 40), dtype=numpy.uint8)
    labels[:, 5:15, 25:35] = 1

    workspace, module = make_workspace(labels)
    workspace.pipeline.set_volumetric(True)
    module.run(workspace)

    for feature in [
        cellprofiler.modules.measureobjectsizeshape.F_VOLUME,
        cellprofiler.modules.measureobjectsizeshape.F_EXTENT,
        cellprofiler.modules.measureobjectsizeshape.F_CENTER_X,
        cellprofiler.modules.measureobjectsizeshape.F_CENTER_Y,
        cellprofiler.modules.measureobjectsizeshape.F_CENTER_Z,
        cellprofiler.modules.measureobjectsizeshape.F_SURFACE_AREA,
    ]:
        assert workspace.measurements.has_current_measurements(
            OBJECTS_NAME,
            cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE + "_" + feature,
        )

        assert (
            len(
                workspace.measurements.get_current_measurement(
                    OBJECTS_NAME,
                    cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE
                    + "_"
                    + feature,
                )
            )
            == 1
        )

    # Assert AreaShape_Center_X and AreaShape_Center_Y aren't flipped. See:
    # https://github.com/CellProfiler/CellProfiler/issues/3352
    center_x = workspace.measurements.get_current_measurement(
        OBJECTS_NAME,
        "_".join(
            [
                cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE,
                cellprofiler.modules.measureobjectsizeshape.F_CENTER_X,
            ]
        ),
    )[0]

    assert center_x == 29.5

    center_y = workspace.measurements.get_current_measurement(
        OBJECTS_NAME,
        "_".join(
            [
                cellprofiler.modules.measureobjectsizeshape.AREA_SHAPE,
                cellprofiler.modules.measureobjectsizeshape.F_CENTER_Y,
            ]
        ),
    )[0]

    assert center_y == 9.5


# https://github.com/CellProfiler/CellProfiler/issues/2813
def test_run_without_zernikes():
    cells_resource = os.path.realpath(
        os.path.join(os.path.dirname(tests.frontend.__file__), "resources/cells.tiff")
    )

    workspace, module = make_workspace(skimage.io.imread(cells_resource))

    module.calculate_zernikes.value = False

    module.run(workspace)

    measurements = workspace.measurements

    for feature in measurements.get_feature_names(OBJECTS_NAME):
        assert "Zernike_" not in feature


def test_run_with_zernikes():
    cells_resource = os.path.realpath(
        os.path.join(os.path.dirname(tests.frontend.__file__), "resources/cells.tiff")
    )

    workspace, module = make_workspace(skimage.io.imread(cells_resource))

    module.calculate_zernikes.value = True

    module.run(workspace)

    measurements = workspace.measurements

    zernikes = [
        feature
        for feature in measurements.get_feature_names(OBJECTS_NAME)
        if "Zernike_" in feature
    ]

    assert len(zernikes) > 0


def test_run_without_advanced():
    cells_resource = os.path.realpath(
        os.path.join(os.path.dirname(tests.frontend.__file__), "resources/cells.tiff")
    )

    workspace, module = make_workspace(skimage.io.imread(cells_resource))

    module.calculate_advanced.value = False
    module.calculate_zernikes.value = False

    module.run(workspace)

    measurements = workspace.measurements

    standard = [
        f"AreaShape_{name}"
        for name in cellprofiler.modules.measureobjectsizeshape.F_STANDARD
        + cellprofiler.modules.measureobjectsizeshape.F_STD_2D
    ]
    advanced = [
        f"AreaShape_{name}"
        for name in cellprofiler.modules.measureobjectsizeshape.F_ADV_2D
    ]
    measures = measurements.get_feature_names(OBJECTS_NAME)
    for feature in standard:
        assert feature in measures
    for feature in advanced:
        assert feature not in measures


def test_run_with_advanced():
    cells_resource = os.path.realpath(
        os.path.join(os.path.dirname(tests.frontend.__file__), "resources/cells.tiff")
    )

    workspace, module = make_workspace(skimage.io.imread(cells_resource))

    module.calculate_advanced.value = True
    module.calculate_zernikes.value = False

    module.run(workspace)

    measurements = workspace.measurements

    allfeatures = [
        f"AreaShape_{name}"
        for name in cellprofiler.modules.measureobjectsizeshape.F_STANDARD
        + cellprofiler.modules.measureobjectsizeshape.F_STD_2D
        + cellprofiler.modules.measureobjectsizeshape.F_ADV_2D
    ]
    measures = measurements.get_feature_names(OBJECTS_NAME)
    for feature in allfeatures:
        assert feature in measures
