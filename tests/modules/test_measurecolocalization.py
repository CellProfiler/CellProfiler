import numpy
import six.moves

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.measurecolocalization
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace

IMAGE1_NAME = "image1"
IMAGE2_NAME = "image2"
OBJECTS_NAME = "objects"


def make_workspace(image1, image2, objects=None):
    """Make a workspace for testing Threshold"""
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    for image_group, name, image in zip(
        module.image_groups, (IMAGE1_NAME, IMAGE2_NAME), (image1, image2)
    ):
        image_group.image_name.value = name
        image_set.add(name, image)
    object_set = cellprofiler.object.ObjectSet()
    if objects is None:
        module.images_or_objects.value = (
            cellprofiler.modules.measurecolocalization.M_IMAGES
        )
    else:
        module.images_or_objects.value = (
            cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
        )
        module.object_groups[0].object_name.value = OBJECTS_NAME
        object_set.add_objects(objects, OBJECTS_NAME)
    pipeline = cellprofiler.pipeline.Pipeline()
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_load_v2():
    with open("./tests/resources/modules/measurecolocalization/v2.pipeline", "r") as fd:
        data = fd.read()

    fd = six.moves.StringIO(data)
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(fd)
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert (
        module.images_or_objects.value
        == cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    assert module.image_count.value == 2
    assert module.thr == 15.0
    for name in [x.image_name.value for x in module.image_groups]:
        assert name in ["DNA", "Cytoplasm"]

    assert module.object_count.value == 2
    for name in [x.object_name.value for x in module.object_groups]:
        assert name in ["Nuclei", "Cells"]


def test_load_v3():
    with open("./tests/resources/modules/measurecolocalization/v3.pipeline", "r") as fd:
        data = fd.read()

    fd = six.moves.StringIO(data)
    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(fd)
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert (
        module.images_or_objects.value
        == cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    assert module.image_count.value == 2
    assert module.thr == 25.0
    for name in [x.image_name.value for x in module.image_groups]:
        assert name in ["DNA", "Cytoplasm"]

    assert module.object_count.value == 2
    for name in [x.object_name.value for x in module.object_groups]:
        assert name in ["Nuclei", "Cells"]


all_object_measurement_formats = [
    cellprofiler.modules.measurecolocalization.F_CORRELATION_FORMAT,
    cellprofiler.modules.measurecolocalization.F_COSTES_FORMAT,
    cellprofiler.modules.measurecolocalization.F_K_FORMAT,
    cellprofiler.modules.measurecolocalization.F_MANDERS_FORMAT,
    cellprofiler.modules.measurecolocalization.F_OVERLAP_FORMAT,
    cellprofiler.modules.measurecolocalization.F_RWC_FORMAT,
]
all_image_measurement_formats = all_object_measurement_formats + [
    cellprofiler.modules.measurecolocalization.F_SLOPE_FORMAT
]
asymmetrical_measurement_formats = [
    cellprofiler.modules.measurecolocalization.F_COSTES_FORMAT,
    cellprofiler.modules.measurecolocalization.F_K_FORMAT,
    cellprofiler.modules.measurecolocalization.F_MANDERS_FORMAT,
    cellprofiler.modules.measurecolocalization.F_RWC_FORMAT,
]


def test_get_categories():
    """Test the get_categories function for some different cases"""
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    module.image_groups[0].image_name.value = IMAGE1_NAME
    module.image_groups[1].image_name.value = IMAGE2_NAME
    module.object_groups[0].object_name.value = OBJECTS_NAME
    module.images_or_objects.value = cellprofiler.modules.measurecolocalization.M_IMAGES

    def cat(name):
        return module.get_categories(None, name) == ["Correlation"]

    assert cat("Image")
    assert not cat(OBJECTS_NAME)
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_OBJECTS
    )
    assert not cat("Image")
    assert cat(OBJECTS_NAME)
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    assert cat("Image")
    assert cat(OBJECTS_NAME)


def test_get_measurements():
    """Test the get_measurements function for some different cases"""
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    module.image_groups[0].image_name.value = IMAGE1_NAME
    module.image_groups[1].image_name.value = IMAGE2_NAME
    module.object_groups[0].object_name.value = OBJECTS_NAME
    module.images_or_objects.value = cellprofiler.modules.measurecolocalization.M_IMAGES

    def meas(name):
        ans = list(module.get_measurements(None, name, "Correlation"))
        ans.sort()
        if name == "Image":
            mf = all_image_measurement_formats
        else:
            mf = all_object_measurement_formats
        expected = sorted([_.split("_")[1] for _ in mf])
        return ans == expected

    assert meas("Image")
    assert not meas(OBJECTS_NAME)
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_OBJECTS
    )
    assert not meas("Image")
    assert meas(OBJECTS_NAME)
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    assert meas("Image")
    assert meas(OBJECTS_NAME)


def test_get_measurement_images():
    """Test the get_measurment_images function for some different cases"""
    for iocase, names in (
        (
            cellprofiler.modules.measurecolocalization.M_IMAGES,
            [cellprofiler.measurement.IMAGE],
        ),
        (cellprofiler.modules.measurecolocalization.M_OBJECTS, [OBJECTS_NAME]),
        (
            cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS,
            [cellprofiler.measurement.IMAGE, OBJECTS_NAME],
        ),
    ):
        module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
        module.image_groups[0].image_name.value = IMAGE1_NAME
        module.image_groups[1].image_name.value = IMAGE2_NAME
        module.object_groups[0].object_name.value = OBJECTS_NAME
        module.images_or_objects.value = iocase
        for name, mfs in (
            (cellprofiler.measurement.IMAGE, all_image_measurement_formats),
            (OBJECTS_NAME, all_object_measurement_formats),
        ):
            if name not in names:
                continue
            for mf in mfs:
                ftr = mf.split("_")[1]
                ans = module.get_measurement_images(None, name, "Correlation", ftr)
                expected = [
                    "%s_%s" % (i1, i2)
                    for i1, i2 in (
                        (IMAGE1_NAME, IMAGE2_NAME),
                        (IMAGE2_NAME, IMAGE1_NAME),
                    )
                ]
                if mf in asymmetrical_measurement_formats:
                    assert all([e in ans for e in expected])
                else:
                    assert any([e in ans for e in expected])


def test_01_get_measurement_columns_images():
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    module.image_groups[0].image_name.value = IMAGE1_NAME
    module.image_groups[1].image_name.value = IMAGE2_NAME
    module.object_groups[0].object_name.value = OBJECTS_NAME
    module.images_or_objects.value = cellprofiler.modules.measurecolocalization.M_IMAGES
    columns = module.get_measurement_columns(None)
    expected = [
        (
            cellprofiler.measurement.IMAGE,
            ftr % (IMAGE1_NAME, IMAGE2_NAME),
            cellprofiler.measurement.COLTYPE_FLOAT,
        )
        for ftr in all_image_measurement_formats
    ] + [
        (
            cellprofiler.measurement.IMAGE,
            ftr % (IMAGE2_NAME, IMAGE1_NAME),
            cellprofiler.measurement.COLTYPE_FLOAT,
        )
        for ftr in asymmetrical_measurement_formats
    ]
    assert len(columns) == len(expected)
    for column in columns:
        assert any([all([cf == ef for cf, ef in zip(column, ex)]) for ex in expected])


def test_02_get_measurement_columns_objects():
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    module.image_groups[0].image_name.value = IMAGE1_NAME
    module.image_groups[1].image_name.value = IMAGE2_NAME
    module.object_groups[0].object_name.value = OBJECTS_NAME
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_OBJECTS
    )
    columns = module.get_measurement_columns(None)
    expected = [
        (
            OBJECTS_NAME,
            ftr % (IMAGE1_NAME, IMAGE2_NAME),
            cellprofiler.measurement.COLTYPE_FLOAT,
        )
        for ftr in all_object_measurement_formats
    ] + [
        (
            OBJECTS_NAME,
            ftr % (IMAGE2_NAME, IMAGE1_NAME),
            cellprofiler.measurement.COLTYPE_FLOAT,
        )
        for ftr in asymmetrical_measurement_formats
    ]
    assert len(columns) == len(expected)
    for column in columns:
        assert any([all([cf == ef for cf, ef in zip(column, ex)]) for ex in expected])


def test_03_get_measurement_columns_both():
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    module.image_groups[0].image_name.value = IMAGE1_NAME
    module.image_groups[1].image_name.value = IMAGE2_NAME
    module.object_groups[0].object_name.value = OBJECTS_NAME
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    columns = module.get_measurement_columns(None)
    expected = (
        [
            (
                cellprofiler.measurement.IMAGE,
                ftr % (IMAGE1_NAME, IMAGE2_NAME),
                cellprofiler.measurement.COLTYPE_FLOAT,
            )
            for ftr in all_image_measurement_formats
        ]
        + [
            (
                cellprofiler.measurement.IMAGE,
                ftr % (IMAGE2_NAME, IMAGE1_NAME),
                cellprofiler.measurement.COLTYPE_FLOAT,
            )
            for ftr in asymmetrical_measurement_formats
        ]
        + [
            (
                OBJECTS_NAME,
                ftr % (IMAGE1_NAME, IMAGE2_NAME),
                cellprofiler.measurement.COLTYPE_FLOAT,
            )
            for ftr in all_object_measurement_formats
        ]
        + [
            (
                OBJECTS_NAME,
                ftr % (IMAGE2_NAME, IMAGE1_NAME),
                cellprofiler.measurement.COLTYPE_FLOAT,
            )
            for ftr in asymmetrical_measurement_formats
        ]
    )

    assert len(columns) == len(expected)
    for column in columns:
        assert any([all([cf == ef for cf, ef in zip(column, ex)]) for ex in expected])


def test_correlated():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(10, 10))
    i1 = cellprofiler.image.Image(image)
    i2 = cellprofiler.image.Image(image)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, cellprofiler.measurement.IMAGE, "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        cellprofiler.measurement.IMAGE, "Correlation_Correlation_%s" % mi[0]
    )
    assert round(abs(corr - 1), 7) == 0

    assert len(m.get_object_names()) == 1
    assert m.get_object_names()[0] == cellprofiler.measurement.IMAGE
    columns = module.get_measurement_columns(None)
    features = m.get_feature_names(cellprofiler.measurement.IMAGE)
    assert len(columns) == len(features)
    for column in columns:
        assert column[1] in features


def test_anticorrelated():
    """Test two anticorrelated images"""
    #
    # Make a checkerboard pattern and reverse it for one image
    #
    i, j = numpy.mgrid[0:10, 0:10]
    image1 = ((i + j) % 2).astype(float)
    image2 = 1 - image1
    i1 = cellprofiler.image.Image(image1)
    i2 = cellprofiler.image.Image(image2)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, cellprofiler.measurement.IMAGE, "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        cellprofiler.measurement.IMAGE, "Correlation_Correlation_%s" % mi[0]
    )
    assert round(abs(corr - -1), 7) == 0


def test_slope():
    """Test the slope measurement"""
    numpy.random.seed(0)
    image1 = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
    image2 = image1 * 0.5
    i1 = cellprofiler.image.Image(image1)
    i2 = cellprofiler.image.Image(image2)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, cellprofiler.measurement.IMAGE, "Correlation", "Slope"
    )
    slope = m.get_current_measurement(
        cellprofiler.measurement.IMAGE, "Correlation_Slope_%s" % mi[0]
    )
    if mi[0] == "%s_%s" % (IMAGE1_NAME, IMAGE2_NAME):
        assert round(abs(slope - 0.5), 5) == 0
    else:
        assert round(abs(slope - 2), 7) == 0


def test_crop():
    """Test similarly cropping one image to another"""
    numpy.random.seed(0)
    image1 = numpy.random.uniform(size=(20, 20))
    i1 = cellprofiler.image.Image(image1)
    crop_mask = numpy.zeros((20, 20), bool)
    crop_mask[5:16, 5:16] = True
    i2 = cellprofiler.image.Image(image1[5:16, 5:16], crop_mask=crop_mask)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, cellprofiler.measurement.IMAGE, "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        cellprofiler.measurement.IMAGE, "Correlation_Correlation_%s" % mi[0]
    )
    assert round(abs(corr - 1), 7) == 0


def test_mask():
    """Test images with two different masks"""
    numpy.random.seed(0)
    image1 = numpy.random.uniform(size=(20, 20))
    mask1 = numpy.ones((20, 20), bool)
    mask1[5:8, 8:12] = False
    mask2 = numpy.ones((20, 20), bool)
    mask2[14:18, 2:5] = False
    mask = mask1 & mask2
    image2 = image1.copy()
    #
    # Try to confound the module by making masked points anti-correlated
    #
    image2[~mask] = 1 - image1[~mask]
    i1 = cellprofiler.image.Image(image1, mask=mask1)
    i2 = cellprofiler.image.Image(image2, mask=mask2)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, cellprofiler.measurement.IMAGE, "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        cellprofiler.measurement.IMAGE, "Correlation_Correlation_%s" % mi[0]
    )
    assert round(abs(corr - 1), 7) == 0


def test_objects():
    """Test images with two objects"""
    labels = numpy.zeros((10, 10), int)
    labels[:4, :4] = 1
    labels[6:, 6:] = 2
    i, j = numpy.mgrid[0:10, 0:10]
    image1 = ((i + j) % 2).astype(float)
    image2 = image1.copy()
    #
    # Anti-correlate the second object
    #
    image2[labels == 2] = 1 - image1[labels == 2]
    i1 = cellprofiler.image.Image(image1)
    i2 = cellprofiler.image.Image(image2)
    o = cellprofiler.object.Objects()
    o.segmented = labels
    workspace, module = make_workspace(i1, i2, o)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
    corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
    assert len(corr) == 2
    assert round(abs(corr[0] - 1), 7) == 0
    assert round(abs(corr[1] - -1), 7) == 0

    assert len(m.get_object_names()) == 2
    assert OBJECTS_NAME in m.get_object_names()
    columns = module.get_measurement_columns(None)
    image_features = m.get_feature_names(cellprofiler.measurement.IMAGE)
    object_features = m.get_feature_names(OBJECTS_NAME)
    assert len(columns) == len(image_features) + len(object_features)
    for column in columns:
        if column[0] == cellprofiler.measurement.IMAGE:
            assert column[1] in image_features
        else:
            assert column[0] == OBJECTS_NAME
            assert column[1] in object_features


def test_cropped_objects():
    """Test images and objects with a cropping mask"""
    numpy.random.seed(0)
    image1 = numpy.random.uniform(size=(20, 20))
    i1 = cellprofiler.image.Image(image1)
    crop_mask = numpy.zeros((20, 20), bool)
    crop_mask[5:15, 5:15] = True
    i2 = cellprofiler.image.Image(image1[5:15, 5:15], crop_mask=crop_mask)
    labels = numpy.zeros((10, 10), int)
    labels[:4, :4] = 1
    labels[6:, 6:] = 2
    o = cellprofiler.object.Objects()
    o.segmented = labels
    #
    # Make the objects have the cropped image as a parent
    #
    o.parent_image = i2
    workspace, module = make_workspace(i1, i2, o)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
    corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
    assert round(abs(corr[0] - 1), 7) == 0
    assert round(abs(corr[1] - 1), 7) == 0


def test_no_objects():
    """Test images with no objects"""
    labels = numpy.zeros((10, 10), int)
    i, j = numpy.mgrid[0:10, 0:10]
    image1 = ((i + j) % 2).astype(float)
    image2 = image1.copy()
    i1 = cellprofiler.image.Image(image1)
    i2 = cellprofiler.image.Image(image2)
    o = cellprofiler.object.Objects()
    o.segmented = labels
    workspace, module = make_workspace(i1, i2, o)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
    corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
    assert len(corr) == 0
    assert len(m.get_object_names()) == 2
    assert OBJECTS_NAME in m.get_object_names()
    columns = module.get_measurement_columns(None)
    image_features = m.get_feature_names(cellprofiler.measurement.IMAGE)
    object_features = m.get_feature_names(OBJECTS_NAME)
    assert len(columns) == len(image_features) + len(object_features)
    for column in columns:
        if column[0] == cellprofiler.measurement.IMAGE:
            assert column[1] in image_features
        else:
            assert column[0] == OBJECTS_NAME
            assert column[1] in object_features


def test_wrong_size():
    """Regression test of IMG-961 - objects and images of different sizes"""
    numpy.random.seed(0)
    image1 = numpy.random.uniform(size=(20, 20))
    i1 = cellprofiler.image.Image(image1)
    labels = numpy.zeros((10, 30), int)
    labels[:4, :4] = 1
    labels[6:, 6:] = 2
    o = cellprofiler.object.Objects()
    o.segmented = labels
    workspace, module = make_workspace(i1, i1, o)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(None, OBJECTS_NAME, "Correlation", "Correlation")
    corr = m.get_current_measurement(OBJECTS_NAME, "Correlation_Correlation_%s" % mi[0])
    assert round(abs(corr[0] - 1), 7) == 0
    assert round(abs(corr[1] - 1), 7) == 0


def test_last_object_masked():
    # Regression test of issue #1553
    # MeasureColocalization was truncating the measurements
    # if the last had no pixels or all pixels masked.
    #
    r = numpy.random.RandomState()
    r.seed(65)
    image1 = r.uniform(size=(20, 20))
    image2 = r.uniform(size=(20, 20))
    labels = numpy.zeros((20, 20), int)
    labels[3:8, 3:8] = 1
    labels[13:18, 13:18] = 2
    mask = labels != 2
    objects = cellprofiler.object.Objects()
    objects.segmented = labels

    for mask1, mask2 in ((mask, None), (None, mask), (mask, mask)):
        workspace, module = make_workspace(
            cellprofiler.image.Image(image1, mask=mask1),
            cellprofiler.image.Image(image2, mask=mask2),
            objects,
        )
        module.run(workspace)
        m = workspace.measurements
        feature = cellprofiler.modules.measurecolocalization.F_CORRELATION_FORMAT % (
            IMAGE1_NAME,
            IMAGE2_NAME,
        )
        values = m[OBJECTS_NAME, feature]
        assert len(values) == 2
        assert numpy.isnan(values[1])


def test_zero_valued_intensity():
    # https://github.com/CellProfiler/CellProfiler/issues/2680
    image1 = numpy.zeros((10, 10), dtype=numpy.float32)

    image2 = numpy.random.rand(10, 10).astype(numpy.float32)

    labels = numpy.zeros_like(image1, dtype=numpy.uint8)

    labels[5, 5] = 1

    objects = cellprofiler.object.Objects()

    objects.segmented = labels

    workspace, module = make_workspace(
        cellprofiler.image.Image(image1), cellprofiler.image.Image(image2), objects
    )

    module.run(workspace)

    m = workspace.measurements

    feature = cellprofiler.modules.measurecolocalization.F_CORRELATION_FORMAT % (
        IMAGE1_NAME,
        IMAGE2_NAME,
    )

    values = m[OBJECTS_NAME, feature]

    assert len(values) == 1

    assert numpy.isnan(values[0])


def test_non_overlapping_object_intensity():
    # https://github.com/CellProfiler/CellProfiler/issues/2764
    image1 = numpy.random.rand(10, 10)
    image1[:5, :] = 0

    image2 = numpy.random.rand(10, 10)
    image2[5:, :] = 0

    objects = cellprofiler.object.Objects()
    objects.segmented = numpy.ones_like(image1, dtype=numpy.uint8)

    workspace, module = make_workspace(
        cellprofiler.image.Image(image1), cellprofiler.image.Image(image2), objects
    )

    module.run(workspace)

    m = workspace.measurements

    feature = cellprofiler.modules.measurecolocalization.F_OVERLAP_FORMAT % (
        IMAGE1_NAME,
        IMAGE2_NAME,
    )

    values = m[OBJECTS_NAME, feature]

    assert len(values) == 1

    assert values[0] == 0.0
