import io

import numpy
import numpy.random
import numpy.testing
import scipy.ndimage

import cellprofiler_library.opts.measureimageoverlap

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT

import cellprofiler.modules.measureimageoverlap

import tests.frontend.modules

GROUND_TRUTH_IMAGE_NAME = "groundtruth"
TEST_IMAGE_NAME = "test"
O_IMG = "Foreground/background segmentation"
O_OBJ = "Segmented objects"
GROUND_TRUTH_OBJ_IMAGE_NAME = "DNA"
ID_OBJ_IMAGE_NAME = "Protein"
GROUND_TRUTH_OBJ = "Nuclei"
ID_OBJ = "Protein"


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("measureimageoverlap/v1.pipeline")
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
        module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap
    )
    assert module.ground_truth == "GroundTruth"
    assert module.test_img == "Segmentation"


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("measureimageoverlap/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap
    )
    assert module.ground_truth == "Foo"
    assert module.test_img == "Bar"
    # assertEqual(module.object_name_GT, "Cell2_0")
    # assertEqual(module.object_name_ID, "Cell2_1")
    assert not module.wants_emd


def test_load_v4():
    data = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20141015195823
GitHash:051040e
ModuleCount:1
HasImagePlaneDetails:False

MeasureImageOverlap:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
Compare segmented objects, or foreground/background?:Foreground/background segmentation
Select the image to be used as the ground truth basis for calculating the amount of overlap:Foo
Select the image to be used to test for overlap:Bar
Select the objects to be used as the ground truth basis for calculating the amount of overlap:Cell2_0
Select the objects to be tested for overlap against the ground truth:Cell2_1
Calculate earth mover\'s distance?:Yes
Maximum # of points:101
Point selection method:Skeleton
Maximum distance:102
Penalize missing pixels:Yes
"""
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))

    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap
    )
    assert module.ground_truth == "Foo"
    assert module.test_img == "Bar"
    # assertEqual(module.object_name_GT, "Cell2_0")
    # assertEqual(module.object_name_ID, "Cell2_1")
    # assertTrue(module.wants_emd)
    # assertEqual(module.decimation_method, cellprofiler.modules.measureimageoverlap.DM_SKEL)
    # assertEqual(module.max_distance, 102)
    # assertEqual(module.max_points, 101)
    # assertTrue(module.penalize_missing)


def make_workspace(ground_truth, test, dimensions=2):
    """Make a workspace with a ground-truth image and a test image

    ground_truth and test are dictionaries with the following keys:
    image     - the pixel data
    mask      - (optional) the mask data
    crop_mask - (optional) a cropping mask

    returns a workspace and module
    """
    module = cellprofiler.modules.measureimageoverlap.MeasureImageOverlap()
    module.set_module_num(1)
    module.ground_truth.value = GROUND_TRUTH_IMAGE_NAME
    module.test_img.value = TEST_IMAGE_NAME
    module.wants_emd.value = True

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    pipeline.add_module(module)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)

    for name, d in ((GROUND_TRUTH_IMAGE_NAME, ground_truth), (TEST_IMAGE_NAME, test)):
        image = cellprofiler_core.image.Image(
            d["image"],
            mask=d.get("mask"),
            crop_mask=d.get("crop_mask"),
            dimensions=dimensions,
        )
        image_set.add(name, image)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def test_zeros():
    """Test ground-truth of zeros and image of zeros"""

    workspace, module = make_workspace(
        dict(image=numpy.ones((20, 10), bool)), dict(image=numpy.ones((20, 10), bool))
    )

    assert isinstance(
        module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap
    )
    module.run(workspace)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    assert measurements.get_current_image_measurement("Overlap_FalseNegRate_test") == 0
    features = measurements.get_feature_names("Image")
    for feature in cellprofiler.modules.measureimageoverlap.FTR_ALL + [
        cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
    ]:
        field = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        assert field in features, "Missing feature: %s" % feature
    ftr_emd = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
    )
    assert measurements["Image", ftr_emd] == 0


def test_ones():
    """Test ground-truth of ones and image of ones"""

    workspace, module = make_workspace(
        dict(image=numpy.zeros((20, 10), bool)), dict(image=numpy.zeros((20, 10), bool))
    )

    assert isinstance(
        module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap
    )
    module.run(workspace)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert expected == value
    mname = "_".join(
        (
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
            TEST_IMAGE_NAME,
        )
    )
    assert numpy.isnan(measurements.get_current_image_measurement(mname))


def test_masked():
    """Test ground-truth of a masked image"""

    workspace, module = make_workspace(
        dict(image=numpy.zeros((20, 10), bool)),
        dict(image=numpy.zeros((20, 10), bool), mask=numpy.zeros((20, 10), bool)),
    )

    assert isinstance(
        module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap
    )
    module.run(workspace)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert expected == value
    for feature in (
        cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX,
        cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert numpy.isnan(value)


def test_all_right():
    numpy.random.seed(34)
    image = numpy.random.uniform(size=(10, 20)) > 0.5
    workspace, module = make_workspace(dict(image=image), dict(image=image))
    module.run(workspace)
    measurements = workspace.measurements
    assert isinstance(measurements,cellprofiler_core.measurement.Measurements)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert expected == value


def test_one_false_positive():
    i, j = numpy.mgrid[0:10, 0:20]
    ground_truth = ((i + j) % 2) == 0
    test = ground_truth.copy()
    test[0, 1] = True
    workspace, module = make_workspace(dict(image=ground_truth), dict(image=test))
    module.run(workspace)
    measurements = workspace.measurements
    precision = 100.0 / 101.0
    f_factor = 2 * precision / (1 + precision)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0.01),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 0.99),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, precision),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor),
        (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert round(abs(expected - value), 7) == 0, "%s is wrong" % feature


def test_one_false_negative():
    i, j = numpy.mgrid[0:10, 0:20]
    ground_truth = ((i + j) % 2) == 0
    test = ground_truth.copy()
    test[0, 0] = False
    workspace, module = make_workspace(dict(image=ground_truth), dict(image=test))
    module.run(workspace)
    measurements = workspace.measurements
    recall = 0.99
    f_factor = 2 * recall / (1 + recall)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0.01),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 0.99),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, recall),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor),
        (cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE, 0),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert round(abs(expected - value), 7) == 0, "%s is wrong" % feature


def test_one_false_positive_and_mask():
    i, j = numpy.mgrid[0:10, 0:20]
    ground_truth = ((i + j) % 2) == 0
    test = ground_truth.copy()
    test[0, 1] = True
    mask = j < 10
    workspace, module = make_workspace(
        dict(image=ground_truth), dict(image=test, mask=mask)
    )
    module.run(workspace)
    measurements = workspace.measurements
    precision = 50.0 / 51.0
    f_factor = 2 * precision / (1 + precision)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0.02),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 0.98),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, precision),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert round(abs(expected - value), 7) == 0, "%s is wrong" % feature


def test_one_false_negative_and_mask():
    i, j = numpy.mgrid[0:10, 0:20]
    ground_truth = ((i + j) % 2) == 0
    test = ground_truth.copy()
    test[0, 0] = False
    mask = j < 10
    workspace, module = make_workspace(
        dict(image=ground_truth), dict(image=test, mask=mask)
    )
    module.run(workspace)
    measurements = workspace.measurements
    recall = 0.98
    f_factor = 2 * recall / (1 + recall)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0.02),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 0.98),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, recall),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert round(abs(expected - value), 7) == 0, "%s is wrong" % feature


def test_masked_errors():
    numpy.random.seed(38)
    ground_truth = numpy.random.uniform(size=(20, 10)) > 0.5
    test = ground_truth.copy()
    mask = numpy.random.uniform(size=(20, 10)) > 0.5
    test[~mask] = numpy.random.uniform(size=numpy.sum(~mask)) > 0.5
    workspace, module = make_workspace(
        dict(image=ground_truth), dict(image=test, mask=mask)
    )
    module.run(workspace)
    measurements = workspace.measurements
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, 1),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert round(abs(expected - value), 7) == 0, "%s is wrong" % feature


def test_cropped():
    numpy.random.seed(39)
    i, j = numpy.mgrid[0:10, 0:20]
    ground_truth = ((i + j) % 2) == 0
    test = ground_truth.copy()
    test[0, 1] = True
    cropping = numpy.zeros((20, 40), bool)
    cropping[10:20, 10:30] = True
    big_ground_truth = numpy.random.uniform(size=(20, 40)) > 0.5
    big_ground_truth[10:20, 10:30] = ground_truth
    workspace, module = make_workspace(
        dict(image=big_ground_truth), dict(image=test, crop_mask=cropping)
    )
    module.run(workspace)
    measurements = workspace.measurements
    precision = 100.0 / 101.0
    f_factor = 2 * precision / (1 + precision)
    for feature, expected in (
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE, 0.01),
        (cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE, 0),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE, 0.99),
        (cellprofiler.modules.measureimageoverlap.FTR_RECALL, 1),
        (cellprofiler.modules.measureimageoverlap.FTR_PRECISION, precision),
        (cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR, f_factor),
    ):
        mname = "_".join(
            (
                cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
                feature,
                TEST_IMAGE_NAME,
            )
        )
        value = measurements.get_current_image_measurement(mname)
        assert (
            round(abs(expected - value), 7) == 0
        ), "%s is wrong. Expected %f, got %f" % (feature, expected, value)


def test_rand_index():
    numpy.random.seed(310)
    i, j = numpy.mgrid[0:10, 0:20]
    #
    # Create a labeling with two objects 0:10, 0:5 and 0:10, 15:20.
    # The background class is 0:10, 5:15
    #
    ground_truth = (j < 5) | (j >= 15)
    #
    # Add a 3x4 square in the middle
    #
    test = ground_truth.copy()
    test[4:7, 8:12] = True
    #
    # I used R to generate the rand index and adjusted rand index
    # of the two segmentations: a 10 x 5 rectangle, a 10x10 background
    # and a 10x5 rectangle with 12 pixels that disagree in the middle
    #
    # The rand index is from rand.index in the fossil package and
    # the adjusted rand index is from cluster.stats in the fpc package.
    # There's an adjusted rand index in the fossil package but it gives
    # the wrong numbers (!!!!)
    #
    expected_rand_index = 0.9469347
    expected_adj_rand_index = 0.8830027
    workspace, module = make_workspace(dict(image=ground_truth), dict(image=test))
    module.run(workspace)
    measurements = workspace.measurements
    mname = "_".join(
        (
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX,
            TEST_IMAGE_NAME,
        )
    )
    assert (
        round(
            abs(
                measurements.get_current_image_measurement(mname) - expected_rand_index
            ),
            6,
        )
        == 0
    )
    mname = "_".join(
        (
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
            TEST_IMAGE_NAME,
        )
    )
    assert (
        round(
            abs(
                measurements.get_current_image_measurement(mname)
                - expected_adj_rand_index
            ),
            6,
        )
        == 0
    )


def test_masked_rand_index():
    numpy.random.seed(310)
    i, j = numpy.mgrid[0:10, 0:20]
    #
    # Create a labeling with two objects 0:10, 0:5 and 0:10, 15:20.
    # The background class is 0:10, 5:15
    #
    ground_truth = (j < 5) | (j >= 15)
    #
    # Add a 3x4 square in the middle
    #
    test = ground_truth.copy()
    test[4:7, 8:12] = True
    #
    # Remove both one correct and one incorect pixel
    #
    mask = numpy.ones(ground_truth.shape, bool)
    mask[4, 4] = False
    mask[5, 9] = False
    #
    # See notes from 03_10
    #
    expected_rand_index = 0.9503666
    expected_adj_rand_index = 0.8907784
    workspace, module = make_workspace(
        dict(image=ground_truth, mask=mask), dict(image=test, mask=mask)
    )
    module.run(workspace)
    measurements = workspace.measurements

    mname = "_".join(
        (
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX,
            TEST_IMAGE_NAME,
        )
    )
    assert (
        round(
            abs(
                measurements.get_current_image_measurement(mname) - expected_rand_index
            ),
            6,
        )
        == 0
    )
    mname = "_".join(
        (
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
            TEST_IMAGE_NAME,
        )
    )
    assert (
        round(
            abs(
                measurements.get_current_image_measurement(mname)
                - expected_adj_rand_index
            ),
            6,
        )
        == 0
    )


def test_get_measurement_columns():
    workspace, module = make_workspace(
        dict(image=numpy.zeros((20, 10), bool)), dict(image=numpy.zeros((20, 10), bool))
    )

    assert isinstance(
        module, cellprofiler.modules.measureimageoverlap.MeasureImageOverlap
    )
    name = TEST_IMAGE_NAME

    columns = module.get_measurement_columns(workspace.pipeline)
    # All columns should be unique
    assert len(columns) == len(set([x[1] for x in columns]))
    # All columns should be floats and done on images
    x = columns[-1]
    assert all([x[0] == "Image"])
    assert all([x[2] == COLTYPE_FLOAT])
    for feature in cellprofiler.modules.measureimageoverlap.FTR_ALL:
        field = "_".join(
            (cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP, feature, name)
        )
        assert field in [x[1] for x in columns]


def test_get_categories():
    workspace, module = make_workspace(
        dict(image=numpy.zeros((20, 10), bool)), dict(image=numpy.zeros((20, 10), bool))
    )
    categories = module.get_categories(workspace.pipeline, "Foo")
    assert len(categories) == 0
    categories = module.get_categories(
        workspace.pipeline, "Image"
    )
    assert len(categories) == 1
    assert categories[0] == cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP


def test_get_measurements():
    workspace, module = make_workspace(
        dict(image=numpy.zeros((20, 10), bool)), dict(image=numpy.zeros((20, 10), bool))
    )
    for wants_emd, features in (
        (
            True,
            list(cellprofiler.modules.measureimageoverlap.FTR_ALL)
            + [cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE],
        ),
        (False, cellprofiler.modules.measureimageoverlap.FTR_ALL),
    ):
        module.wants_emd.value = wants_emd
        mnames = module.get_measurements(
            workspace.pipeline,
            "Image",
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
        )
        assert len(mnames) == len(features)
        assert all(n in features for n in mnames)
        assert all(f in mnames for f in features)
        mnames = module.get_measurements(
            workspace.pipeline,
            "Foo",
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
        )
        assert len(mnames) == 0
        mnames = module.get_measurements(
            workspace.pipeline, "Image", "Foo"
        )
        assert len(mnames) == 0


def test_get_measurement_images():
    workspace, module = make_workspace(
        dict(image=numpy.zeros((20, 10), bool)), dict(image=numpy.zeros((20, 10), bool))
    )

    for feature in cellprofiler.modules.measureimageoverlap.FTR_ALL:
        imnames = module.get_measurement_images(
            workspace.pipeline,
            "Image",
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            feature,
        )
        assert len(imnames) == 1
        assert imnames[0] == TEST_IMAGE_NAME
    imnames = module.get_measurement_images(
        workspace.pipeline,
        "Image",
        cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
        "Foo",
    )
    assert len(imnames) == 0
    imnames = module.get_measurement_images(
        workspace.pipeline,
        "Image",
        "Foo",
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE,
    )
    assert len(imnames) == 0
    imnames = module.get_measurement_images(
        workspace.pipeline,
        "Foo",
        cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE,
    )
    assert len(imnames) == 0


def test_get_measurement_scales():
    workspace, module = make_workspace(
        dict(image=numpy.zeros((20, 10), bool)), dict(image=numpy.zeros((20, 10), bool))
    )
    scales = module.get_measurement_scales(
        workspace.pipeline,
        "Image",
        cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
        cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX,
        None,
    )
    assert len(scales) == 0


def test_test_objects_rand_index():
    r = numpy.random.RandomState()
    r.seed(52)
    base = numpy.zeros((100, 100), bool)
    base[r.randint(0, 100, size=10), r.randint(0, 100, size=10)] = True
    gt = base.copy()
    gt[r.randint(0, 100, size=5), r.randint(0, 100, size=5)] = True
    test = base.copy()
    test[r.randint(0, 100, size=5), r.randint(0, 100, size=5)] = True
    gt = scipy.ndimage.binary_dilation(gt, numpy.ones((5, 5), bool))
    test = scipy.ndimage.binary_dilation(test, numpy.ones((5, 5), bool))
    workspace, module = make_workspace(dict(image=gt), dict(image=test))
    module.wants_emd.value = False
    module.run(workspace)

    measurements = workspace.measurements
    mname = "_".join(
        (
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX,
            TEST_IMAGE_NAME,
        )
    )
    expected_rand_index = measurements.get_current_image_measurement(mname)
    mname = "_".join(
        (
            cellprofiler.modules.measureimageoverlap.C_IMAGE_OVERLAP,
            cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX,
            TEST_IMAGE_NAME,
        )
    )
    expected_adjusted_rand_index = measurements.get_current_image_measurement(mname)


def test_no_emd():
    workspace, module = make_workspace(
        dict(image=numpy.ones((20, 10), bool)), dict(image=numpy.ones((20, 10), bool))
    )
    module.wants_emd.value = False
    module.run(workspace)
    assert not workspace.measurements.has_feature(
        "Image",
        module.measurement_name(
            cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
        ),
    )


def test_one_pixel():
    #
    # The earth movers distance should be sqrt((8-5)**2 + (7 - 3) ** 2) = 5
    #
    src = numpy.zeros((20, 10), bool)
    dest = numpy.zeros((20, 10), bool)
    src[5, 3] = True
    dest[8, 7] = True
    workspace, module = make_workspace(dict(image=src), dict(image=dest))
    module.run(workspace)
    assert (
        workspace.measurements[
            "Image",
            module.measurement_name(
                cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
            ),
        ]
        == 5
    )


def test_missing_penalty():
    #
    # Test that the missing penalty works
    #
    src = numpy.zeros((20, 10), bool)
    dest = numpy.zeros((20, 10), bool)
    src[2, 2] = True
    dest[2, 2] = True
    dest[8, 7] = True
    dest[2, 6] = True
    workspace, module = make_workspace(dict(image=src), dict(image=dest))
    module.penalize_missing.value = True
    module.max_distance.value = 8
    module.run(workspace)
    assert (
        workspace.measurements[
            "Image",
            module.measurement_name(
                cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
            ),
        ]
        == 16
    )


def test_max_distance():
    src = numpy.zeros((20, 10), bool)
    dest = numpy.zeros((20, 10), bool)
    src[5, 3] = True
    dest[8, 7] = True
    src[19, 9] = True
    dest[11, 9] = True
    workspace, module = make_workspace(dict(image=src), dict(image=dest))
    module.max_distance.value = 6
    module.run(workspace)
    assert (
        workspace.measurements[
            "Image",
            module.measurement_name(
                cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
            ),
        ]
        == 11
    )


def test_decimate_k_means():
    r = numpy.random.RandomState()
    r.seed(64)
    img = r.uniform(size=(10, 10)) > 0.5
    workspace, module = make_workspace(dict(image=img), dict(image=img.transpose()))
    #
    # Pick a single point for decimation - the emd should be zero
    #
    module.max_points._Number__minval = 1
    module.max_points.value = 1
    module.run(workspace)
    assert (
        workspace.measurements[
            "Image",
            module.measurement_name(
                cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
            ),
        ]
        == 0
    )
    #
    # Pick a large number of points to get the real EMD
    #
    workspace, module = make_workspace(dict(image=img), dict(image=img.transpose()))
    module.max_points._Number__minval = 1
    module.max_points.value = 100
    module.run(workspace)
    emd = workspace.measurements[
        "Image",
        module.measurement_name(
            cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
        ),
    ]
    #
    # The EMD after decimation is going to be randomly different,
    # but not by much.
    #
    workspace, module = make_workspace(dict(image=img), dict(image=img.transpose()))
    module.max_points._Number__minval = 1
    module.max_points.value = numpy.sum(img | img.transpose()) / 2
    module.run(workspace)
    decimated_emd = workspace.measurements[
        "Image",
        module.measurement_name(
            cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
        ),
    ]
    assert decimated_emd < emd * 2
    assert decimated_emd > emd / 2


def test_decimate_skel():
    #
    # Mostly, this is to check that the skeleton method doesn't crash
    #
    i, j = numpy.mgrid[0:10, 0:20]
    image1 = ((i - 4) ** 2) * 4 + (j - 8) ** 2 < 32
    image2 = ((i - 6) ** 2) * 4 + (j - 12) ** 2 < 32
    workspace, module = make_workspace(dict(image=image1), dict(image=image2))
    module.max_points._Number__minval = 1
    module.max_points.value = 5
    module.decimation_method.value = cellprofiler_library.opts.measureimageoverlap.DM.SKELETON.value
    module.run(workspace)
    emd = workspace.measurements[
        "Image",
        module.measurement_name(
            cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
        ),
    ]
    assert emd > numpy.sum(image1) * 3
    assert emd < numpy.sum(image1) * 6


def test_3D_perfect_overlap():
    image_data = numpy.random.rand(10, 100, 100) >= 0.5

    workspace, module = make_workspace(
        ground_truth={"image": image_data}, test={"image": image_data}, dimensions=3
    )

    module.run(workspace)

    measurements = workspace.measurements

    ftr_precision = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_PRECISION
    )
    ftr_precision_measurement = measurements.get_current_image_measurement(
        ftr_precision
    )
    assert ftr_precision_measurement == 1.0

    ftr_true_pos_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE
    )
    ftr_true_pos_rate_measurement = measurements.get_current_image_measurement(
        ftr_true_pos_rate
    )
    assert ftr_true_pos_rate_measurement == 1.0

    ftr_false_pos_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE
    )
    ftr_false_pos_rate_measurement = measurements.get_current_image_measurement(
        ftr_false_pos_rate
    )
    assert ftr_false_pos_rate_measurement == 0

    ftr_true_neg_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE
    )
    ftr_true_neg_rate_measurement = measurements.get_current_image_measurement(
        ftr_true_neg_rate
    )
    assert ftr_true_neg_rate_measurement == 1.0

    ftr_false_neg_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE
    )
    ftr_false_neg_rate_measurement = measurements.get_current_image_measurement(
        ftr_false_neg_rate
    )
    assert ftr_false_neg_rate_measurement == 0

    ftr_precision = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_PRECISION
    )
    ftr_precision_measurement = measurements.get_current_image_measurement(
        ftr_precision
    )
    assert round(abs(ftr_precision_measurement - 1), 7) == 0

    ftr_F_factor = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR
    )
    ftr_F_factor_measurement = measurements.get_current_image_measurement(ftr_F_factor)
    assert round(abs(ftr_F_factor_measurement - 1), 7) == 0

    ftr_Earth_Movers_Distance = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
    )
    ftr_Earth_Movers_Distance_measurement = measurements.get_current_image_measurement(
        ftr_Earth_Movers_Distance
    )
    assert round(abs(ftr_Earth_Movers_Distance_measurement - 0), 7) == 0

    ftr_rand_index = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX
    )
    ftr_rand_index_measurement = measurements.get_current_image_measurement(
        ftr_rand_index
    )
    assert round(abs(ftr_rand_index_measurement - 1), 7) == 0

    ftr_adj_rand_index = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX
    )
    ftr_adj_rand_index_measurement = measurements.get_current_image_measurement(
        ftr_adj_rand_index
    )
    assert round(abs(ftr_adj_rand_index_measurement - 1), 7) == 0


def test_3D_no_overlap():
    ground_truth_image_data = numpy.zeros((10, 100, 100), dtype=numpy.bool_)
    ground_truth_image_data[4:6, 30:40, 30:40] = True
    ground_truth_image_data[7:10, 50:60, 50:60] = True

    test_image_data = numpy.zeros((10, 100, 100), dtype=numpy.bool_)
    test_image_data[7:9, 30:40, 30:40] = True
    test_image_data[7:10, 70:80, 70:80] = True

    workspace, module = make_workspace(
        ground_truth={"image": ground_truth_image_data},
        test={"image": test_image_data},
        dimensions=3,
    )

    module.run(workspace)

    measurements = workspace.measurements

    ftr_true_pos_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE
    )
    ftr_true_pos_rate_measurement = measurements.get_current_image_measurement(
        ftr_true_pos_rate
    )
    assert round(abs(ftr_true_pos_rate_measurement - 0), 7) == 0

    ftr_false_pos_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE
    )
    ftr_false_pos_rate_measurement = measurements.get_current_image_measurement(
        ftr_false_pos_rate
    )
    assert round(abs(ftr_false_pos_rate_measurement - 0.005025125628141), 7) == 0

    ftr_true_neg_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE
    )
    ftr_true_neg_rate_measurement = measurements.get_current_image_measurement(
        ftr_true_neg_rate
    )
    assert round(abs(ftr_true_neg_rate_measurement - 0.994974874371859), 7) == 0

    ftr_false_neg_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE
    )
    ftr_false_neg_rate_measurement = measurements.get_current_image_measurement(
        ftr_false_neg_rate
    )
    assert round(abs(ftr_false_neg_rate_measurement - 1), 7) == 0

    ftr_precision = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_PRECISION
    )
    ftr_precision_measurement = measurements.get_current_image_measurement(
        ftr_precision
    )
    assert round(abs(ftr_precision_measurement - 0), 7) == 0

    ftr_recall = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_RECALL
    )
    ftr_recall_measurement = measurements.get_current_image_measurement(ftr_recall)
    assert round(abs(ftr_recall_measurement - 0), 7) == 0

    ftr_F_factor = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR
    )
    ftr_F_factor_measurement = measurements.get_current_image_measurement(ftr_F_factor)
    assert round(abs(ftr_F_factor_measurement - 0), 7) == 0

    ftr_Earth_Movers_Distance = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
    )
    ftr_Earth_Movers_Distance_measurement = measurements.get_current_image_measurement(
        ftr_Earth_Movers_Distance
    )
    # test within 0.1% because sklearn keeps changing things so this kinda breaks
    assert abs(ftr_Earth_Movers_Distance_measurement - 52812) / 52812 <= 0.001

    ftr_rand_index = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX
    )
    ftr_rand_index_measurement = measurements.get_current_image_measurement(
        ftr_rand_index
    )
    assert round(abs(ftr_rand_index_measurement - 0.980199801998020), 2) == 0

    ftr_adj_rand_index = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX
    )
    ftr_adj_rand_index_measurement = measurements.get_current_image_measurement(
        ftr_adj_rand_index
    )
    numpy.testing.assert_almost_equal(ftr_adj_rand_index_measurement, 0, 2)


def test_3D_half_overlap():
    ground_truth_image_data = numpy.zeros((10, 100, 100), dtype=numpy.bool_)
    ground_truth_image_data[2:7, 30:40, 30:40] = True
    ground_truth_image_data[8, 10:20, 10:20] = True
    ground_truth_image_data[1:5, 50:60, 50:60] = True

    test_image_data = numpy.zeros((10, 100, 100), dtype=numpy.bool_)
    test_image_data[2:7, 35:45, 30:40] = True
    test_image_data[8, 10:20, 15:25] = True
    test_image_data[3:7, 50:60, 50:60] = True

    workspace, module = make_workspace(
        ground_truth={"image": ground_truth_image_data},
        test={"image": test_image_data},
        dimensions=3,
    )

    module.run(workspace)

    measurements = workspace.measurements

    ftr_true_pos_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_TRUE_POS_RATE
    )
    ftr_true_pos_rate_measurement = measurements.get_current_image_measurement(
        ftr_true_pos_rate
    )
    assert round(abs(ftr_true_pos_rate_measurement - 0.5), 7) == 0

    ftr_false_pos_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_POS_RATE
    )
    ftr_false_pos_rate_measurement = measurements.get_current_image_measurement(
        ftr_false_pos_rate
    )
    assert round(abs(ftr_false_pos_rate_measurement - 0.005050505), 7) == 0

    ftr_true_neg_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_TRUE_NEG_RATE
    )
    ftr_true_neg_rate_measurement = measurements.get_current_image_measurement(
        ftr_true_neg_rate
    )
    assert round(abs(ftr_true_neg_rate_measurement - 0.99494949), 7) == 0

    ftr_false_neg_rate = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_FALSE_NEG_RATE
    )
    ftr_false_neg_rate_measurement = measurements.get_current_image_measurement(
        ftr_false_neg_rate
    )
    assert round(abs(ftr_false_neg_rate_measurement - 0.5), 7) == 0

    ftr_precision = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_PRECISION
    )
    ftr_precision_measurement = measurements.get_current_image_measurement(
        ftr_precision
    )
    assert round(abs(ftr_precision_measurement - 0.5), 7) == 0

    ftr_recall = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_RECALL
    )
    ftr_recall_measurement = measurements.get_current_image_measurement(ftr_recall)
    assert round(abs(ftr_recall_measurement - 0.5), 7) == 0

    ftr_F_factor = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_F_FACTOR
    )
    ftr_F_factor_measurement = measurements.get_current_image_measurement(ftr_F_factor)
    assert round(abs(ftr_F_factor_measurement - 0.5), 7) == 0

    ftr_Earth_Movers_Distance = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_EARTH_MOVERS_DISTANCE
    )
    ftr_Earth_Movers_Distance_measurement = measurements.get_current_image_measurement(
        ftr_Earth_Movers_Distance
    )
    # test within 0.1% because sklearn keeps changing things so this kinda breaks
    assert abs(ftr_Earth_Movers_Distance_measurement - 52921) / 52921 <= 0.001

    ftr_rand_index = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_RAND_INDEX
    )
    ftr_rand_index_measurement = measurements.get_current_image_measurement(
        ftr_rand_index
    )
    assert round(abs(ftr_rand_index_measurement - 0.980199801998020), 2) == 0

    ftr_adj_rand_index = module.measurement_name(
        cellprofiler.modules.measureimageoverlap.FTR_ADJUSTED_RAND_INDEX
    )
    ftr_adj_rand_index_measurement = measurements.get_current_image_measurement(
        ftr_adj_rand_index
    )
    numpy.testing.assert_almost_equal(ftr_adj_rand_index_measurement, 0.5, 2)
