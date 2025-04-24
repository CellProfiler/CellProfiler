import numpy
import pytest
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT


import cellprofiler.modules.measurecolocalization
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

IMAGE1_NAME = "image1"
IMAGE2_NAME = "image2"
OBJECTS_NAME = "objects"

def make_workspace(image1, image2, objects=None, thresholds=None):
    """Make a workspace for testing Threshold"""
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    module.images_list.value = ", ".join((IMAGE1_NAME, IMAGE2_NAME))
    for image_name, name, image in zip(
        module.images_list.value, (IMAGE1_NAME, IMAGE2_NAME), (image1, image2)
    ):
        image_set.add(name, image)
    object_set = cellprofiler_core.object.ObjectSet()
    if objects is None:
        module.images_or_objects.value = (
            cellprofiler.modules.measurecolocalization.M_IMAGES
        )
    else:
        module.images_or_objects.value = (
            cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
        )
        module.objects_list.value = OBJECTS_NAME
        object_set.add_objects(objects, OBJECTS_NAME)
    if thresholds is not None:
        module.wants_channel_thresholds.set_value(True)
        for idx, threshold in enumerate(thresholds):
            if threshold is None:
                continue
            module.add_threshold()
            module.thresholds_list[-1].image_name.value = f'image{idx+1}'
            module.thresholds_list[-1].threshold_for_channel.value = (
                threshold
            )
    pipeline = cellprofiler_core.pipeline.Pipeline()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def get_x_measurement(workspace, module, x, image_or_object="Image"):
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, image_or_object, "Correlation", x
    )
    x = m.get_current_measurement(
        image_or_object, f"Correlation_{x}_{mi[0]}"
    )
    return x


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("measurecolocalization/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    fd = six.moves.StringIO(data)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(fd)
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert (
        module.images_or_objects.value
        == cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    assert len(module.images_list.value) == 2
    assert module.thr == 15.0
    for name in module.images_list.value:
        assert name in ["DNA", "Cytoplasm"]

    assert len(module.objects_list.value) == 2
    for name in module.objects_list.value:
        assert name in ["Nuclei", "Cells"]


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("measurecolocalization/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    fd = six.moves.StringIO(data)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(fd)
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[-1]
    assert (
        module.images_or_objects.value
        == cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    assert len(module.images_list.value) == 2
    assert module.thr == 25.0
    for name in module.images_list.value:
        assert name in ["DNA", "Cytoplasm"]

    assert len(module.objects_list.value) == 2
    for name in module.objects_list.value:
        assert name in ["Nuclei", "Cells"]


def test_load_v5():
    file = tests.frontend.modules.get_test_resources_directory("measurecolocalization/v5.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    fd = six.moves.StringIO(data)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(fd)
    assert len(pipeline.modules()) == 5
    module = pipeline.modules()[-1]
    assert (
        module.images_or_objects.value
        == cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    assert len(module.images_list.value) == 2
    assert module.thr == 15.0
    for name in module.images_list.value:
        assert name in ["OrigStain1", "OrigStain2"]
    assert module.wants_channel_thresholds.value is False
    assert module.wants_threshold_visualization.value is False

    assert len(module.objects_list.value) == 0


def test_load_v6():
    file = tests.frontend.modules.get_test_resources_directory("measurecolocalization/v6.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    fd = six.moves.StringIO(data)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(fd)
    assert len(pipeline.modules()) == 5
    module = pipeline.modules()[-1]
    assert (
        module.images_or_objects.value
        == cellprofiler.modules.measurecolocalization.M_IMAGES
    )
    assert len(module.images_list.value) == 2
    assert module.thr == 15.0
    for name in module.images_list.value:
        assert name in ["OrigStain1", "OrigStain2"]

    assert module.get_image_threshold_value("OrigStain1") == 42.5
    assert module.get_image_threshold_value("OrigStain2") == 15.0
    assert module.wants_channel_thresholds.value is True
    assert module.wants_threshold_visualization.value is True
    assert module.threshold_visualization_list.value[0] == "OrigStain2"
    assert len(module.threshold_visualization_list.value) == 1
    assert module.wants_masks_saved.value is True
    assert module.save_mask_list[0].save_image_name.value == "ColocalizationMask12"
    assert module.save_mask_list[0].image_name.value == "OrigStain1"
    assert module.save_mask_list[0].save_mask_wants_objects.value is False
    assert module.save_mask_list[1].save_image_name.value == "ColocalizationMask34"
    assert module.save_mask_list[1].image_name.value == "OrigStain2"
    assert module.save_mask_list[1].save_mask_wants_objects.value is False

    


    assert len(module.objects_list.value) == 0


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
    module.images_list.value = ", ".join((IMAGE1_NAME, IMAGE2_NAME))
    module.objects_list.value = OBJECTS_NAME
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
    module.images_list.value = ", ".join((IMAGE1_NAME, IMAGE2_NAME))
    module.objects_list.value = OBJECTS_NAME
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
            ["Image"],
        ),
        (cellprofiler.modules.measurecolocalization.M_OBJECTS, [OBJECTS_NAME]),
        (
            cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS,
            ["Image", OBJECTS_NAME],
        ),
    ):
        module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
        module.images_list.value = ", ".join((IMAGE1_NAME, IMAGE2_NAME))
        module.objects_list.value = OBJECTS_NAME
        module.images_or_objects.value = iocase
        for name, mfs in (
            ("Image", all_image_measurement_formats),
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
    module.images_list.value = ", ".join((IMAGE1_NAME, IMAGE2_NAME))
    module.objects_list.value = OBJECTS_NAME
    module.images_or_objects.value = cellprofiler.modules.measurecolocalization.M_IMAGES
    columns = module.get_measurement_columns(None)
    expected = [
        (
            "Image",
            ftr % (IMAGE1_NAME, IMAGE2_NAME),
            COLTYPE_FLOAT,
        )
        for ftr in all_image_measurement_formats
    ] + [
        (
            "Image",
            ftr % (IMAGE2_NAME, IMAGE1_NAME),
            COLTYPE_FLOAT,
        )
        for ftr in asymmetrical_measurement_formats
    ]
    assert len(columns) == len(expected)
    for column in columns:
        assert any([all([cf == ef for cf, ef in zip(column, ex)]) for ex in expected])


def test_02_get_measurement_columns_objects():
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    module.images_list.value = ", ".join((IMAGE1_NAME, IMAGE2_NAME))
    module.objects_list.value = OBJECTS_NAME
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_OBJECTS
    )
    columns = module.get_measurement_columns(None)
    expected = [
        (
            OBJECTS_NAME,
            ftr % (IMAGE1_NAME, IMAGE2_NAME),
            COLTYPE_FLOAT,
        )
        for ftr in all_object_measurement_formats
    ] + [
        (
            OBJECTS_NAME,
            ftr % (IMAGE2_NAME, IMAGE1_NAME),
            COLTYPE_FLOAT,
        )
        for ftr in asymmetrical_measurement_formats
    ]
    assert len(columns) == len(expected)
    for column in columns:
        assert any([all([cf == ef for cf, ef in zip(column, ex)]) for ex in expected])


def test_03_get_measurement_columns_both():
    module = cellprofiler.modules.measurecolocalization.MeasureColocalization()
    module.images_list.value = ", ".join((IMAGE1_NAME, IMAGE2_NAME))
    module.objects_list.value = OBJECTS_NAME
    module.images_or_objects.value = (
        cellprofiler.modules.measurecolocalization.M_IMAGES_AND_OBJECTS
    )
    columns = module.get_measurement_columns(None)
    expected = (
        [
            (
                "Image",
                ftr % (IMAGE1_NAME, IMAGE2_NAME),
                COLTYPE_FLOAT,
            )
            for ftr in all_image_measurement_formats
        ]
        + [
            (
                "Image",
                ftr % (IMAGE2_NAME, IMAGE1_NAME),
                COLTYPE_FLOAT,
            )
            for ftr in asymmetrical_measurement_formats
        ]
        + [
            (
                OBJECTS_NAME,
                ftr % (IMAGE1_NAME, IMAGE2_NAME),
                COLTYPE_FLOAT,
            )
            for ftr in all_object_measurement_formats
        ]
        + [
            (
                OBJECTS_NAME,
                ftr % (IMAGE2_NAME, IMAGE1_NAME),
                COLTYPE_FLOAT,
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
    i1 = cellprofiler_core.image.Image(image)
    i2 = cellprofiler_core.image.Image(image)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, "Image", "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        "Image", "Correlation_Correlation_%s" % mi[0]
    )
    assert round(abs(corr - 1), 7) == 0

    assert len(m.get_object_names()) == 1
    assert m.get_object_names()[0] == "Image"
    columns = module.get_measurement_columns(None)
    features = m.get_feature_names("Image")
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
    i1 = cellprofiler_core.image.Image(image1)
    i2 = cellprofiler_core.image.Image(image2)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, "Image", "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        "Image", "Correlation_Correlation_%s" % mi[0]
    )
    assert round(abs(corr - -1), 7) == 0


def test_slope(uniform_random_image_20):
    """Test the slope measurement"""
    image1, i1 = uniform_random_image_20
    image2 = image1 * 0.5
    i2 = cellprofiler_core.image.Image(image2)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, "Image", "Correlation", "Slope"
    )
    slope = m.get_current_measurement(
        "Image", "Correlation_Slope_%s" % mi[0]
    )
    if mi[0] == "%s_%s" % (IMAGE1_NAME, IMAGE2_NAME):
        assert round(abs(slope - 0.5), 5) == 0
    else:
        assert round(abs(slope - 2), 7) == 0


@pytest.fixture(scope="function")
def uniform_random_image_20():
    numpy.random.seed(0)
    image1 = numpy.random.uniform(size=(20, 20))
    i1 = cellprofiler_core.image.Image(image1)
    return image1, i1

def test_crop(uniform_random_image_20):
    """Test similarly cropping one image to another"""
    image1, i1 = uniform_random_image_20
    crop_mask = numpy.zeros((20, 20), bool)
    crop_mask[5:16, 5:16] = True
    i2 = cellprofiler_core.image.Image(image1[5:16, 5:16], crop_mask=crop_mask)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, "Image", "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        "Image", "Correlation_Correlation_%s" % mi[0]
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
    i1 = cellprofiler_core.image.Image(image1, mask=mask1)
    i2 = cellprofiler_core.image.Image(image2, mask=mask2)
    workspace, module = make_workspace(i1, i2)
    module.run(workspace)
    m = workspace.measurements
    mi = module.get_measurement_images(
        None, "Image", "Correlation", "Correlation"
    )
    corr = m.get_current_measurement(
        "Image", "Correlation_Correlation_%s" % mi[0]
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
    i1 = cellprofiler_core.image.Image(image1)
    i2 = cellprofiler_core.image.Image(image2)
    o = cellprofiler_core.object.Objects()
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
    image_features = m.get_feature_names("Image")
    object_features = m.get_feature_names(OBJECTS_NAME)
    assert len(columns) == len(image_features) + len(object_features)
    for column in columns:
        if column[0] == "Image":
            assert column[1] in image_features
        else:
            assert column[0] == OBJECTS_NAME
            assert column[1] in object_features


def test_cropped_objects(uniform_random_image_20):
    """Test images and objects with a cropping mask"""
    image1, i1 = uniform_random_image_20
    crop_mask = numpy.zeros((20, 20), bool)
    crop_mask[5:15, 5:15] = True
    i2 = cellprofiler_core.image.Image(image1[5:15, 5:15], crop_mask=crop_mask)
    labels = numpy.zeros((10, 10), int)
    labels[:4, :4] = 1
    labels[6:, 6:] = 2
    o = cellprofiler_core.object.Objects()
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
    i1 = cellprofiler_core.image.Image(image1)
    i2 = cellprofiler_core.image.Image(image2)
    o = cellprofiler_core.object.Objects()
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
    image_features = m.get_feature_names("Image")
    object_features = m.get_feature_names(OBJECTS_NAME)
    assert len(columns) == len(image_features) + len(object_features)
    for column in columns:
        if column[0] == "Image":
            assert column[1] in image_features
        else:
            assert column[0] == OBJECTS_NAME
            assert column[1] in object_features


def test_wrong_size(uniform_random_image_20):
    """Regression test of IMG-961 - objects and images of different sizes"""
    image1, i1 = uniform_random_image_20
    labels = numpy.zeros((10, 30), int)
    labels[:4, :4] = 1
    labels[6:, 6:] = 2
    o = cellprofiler_core.object.Objects()
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
    objects = cellprofiler_core.object.Objects()
    objects.segmented = labels

    for mask1, mask2 in ((mask, None), (None, mask), (mask, mask)):
        workspace, module = make_workspace(
            cellprofiler_core.image.Image(image1, mask=mask1),
            cellprofiler_core.image.Image(image2, mask=mask2),
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

    objects = cellprofiler_core.object.Objects()

    objects.segmented = labels

    workspace, module = make_workspace(
        cellprofiler_core.image.Image(image1),
        cellprofiler_core.image.Image(image2),
        objects,
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

    objects = cellprofiler_core.object.Objects()
    objects.segmented = numpy.ones_like(image1, dtype=numpy.uint8)

    workspace, module = make_workspace(
        cellprofiler_core.image.Image(image1),
        cellprofiler_core.image.Image(image2),
        objects,
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


@pytest.fixture(scope="function")
def two_images_with_50_percent_overlap():
    numpy.random.seed(0)
    image1 = numpy.random.rand(10, 10)
    image1[:2, :] = 0
    image1[6:, :] = 0
    image1[-1, -1] = 1
    image2 = numpy.random.rand(10, 10)
    image2[:4, :] = 0
    image2[8:, :] = 0
    image2[-1, -1] = 1
    image1 = cellprofiler_core.image.Image(image1)
    image1.image_name = IMAGE1_NAME
    image2 = cellprofiler_core.image.Image(image2)
    image2.image_name = IMAGE2_NAME
    image1.mask = numpy.ones_like(image1.pixel_data, dtype=bool)
    image2.mask = numpy.ones_like(image2.pixel_data, dtype=bool)
    return image1, image2


@pytest.mark.parametrize('measure',['Manders','RWC'])
@pytest.mark.parametrize('method',['Image','objects'])
@pytest.mark.parametrize(
    "inp, expected",
    [
        (((90, 90), (10, 90)), False),
        (((90, 90), (90, 10)), False),
        (((90, 90), (10, 10)), False),
        (((90, 90), (90, 90)), True),
        (((20, 20), (50, 50)), False),
        (((20, 20), (20, 50)), False),
        (((20, 20), (50, 20)), False),
        (((20, 20), (20, 20)), True),
        (((1, 1), (1, 10)), False)

    ]
    )


def test_channel_specific_threshold_changes_manders(inp, expected, measure, method, two_images_with_50_percent_overlap):
    image1, image2 = two_images_with_50_percent_overlap

    thr1a, thr1b = inp[0]
    thr2a, thr2b = inp[1]
    objects = cellprofiler_core.object.Objects()
    objects.segmented = numpy.ones_like(image1.pixel_data, dtype=numpy.uint8)
    workspace1, module1 = make_workspace(
        image1,
        image2,
        objects=objects,
        thresholds=[thr1a, thr1b]
    )
    module1.run(workspace1)

    workspace2, module2 = make_workspace(
        image1,
        image2,
        objects=objects,
        thresholds=[thr2a, thr2b]
    )
    module2.run(workspace2)

    measure1 = get_x_measurement(workspace1, module1, measure, method)
    measure2 = get_x_measurement(workspace2, module2, measure, method)
    assert numpy.isnan(measure1) == False
    assert numpy.isnan(measure2) == False
    assert (measure1 == measure2) == expected
