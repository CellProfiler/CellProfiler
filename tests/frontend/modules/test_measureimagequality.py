import io

import centrosome.threshold
import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT, EXPERIMENT
from cellprofiler_core.constants.module._identify import O_FOREGROUND, O_THREE_CLASS, O_WEIGHTED_VARIANCE, O_TWO_CLASS, \
    O_ENTROPY, O_BACKGROUND


import cellprofiler.modules.measureimagequality
import cellprofiler_core.modules.namesandtypes
import cellprofiler.modules.smooth
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

IMAGES_NAME = "my_image"
OBJECTS_NAME = "my_objects"


def make_workspace(pixel_data, mask=None, objects=None, dimensions=2):
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    object_set = cellprofiler_core.object.ObjectSet()
    image = cellprofiler_core.image.Image(pixel_data, dimensions=dimensions)
    if not mask is None:
        image.mask = mask
    image_set.add(IMAGES_NAME, image)
    if not objects is None:
        o = cellprofiler_core.object.Objects()
        o.segmented = objects
        object_set.add_objects(o, OBJECTS_NAME)
    module = cellprofiler.modules.measureimagequality.MeasureImageQuality()
    module.images_choice.value = cellprofiler.modules.measureimagequality.O_SELECT
    module.image_groups[0].include_image_scalings.value = False
    module.image_groups[0].image_names.value = IMAGES_NAME
    module.image_groups[0].use_all_threshold_methods.value = False
    module.set_module_num(1)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace


def test_zeros():
    workspace = make_workspace(numpy.zeros((100, 100)))
    q = workspace.module
    q.image_groups[0].check_blur.value = True
    q.image_groups[0].check_saturation.value = True
    q.image_groups[0].check_intensity.value = True
    q.image_groups[0].calculate_threshold.value = False
    q.image_groups[0].scale_groups[0].scale.value = 20
    q.run(workspace)
    m = workspace.measurements
    for feature_name, value in (
        ("ImageQuality_FocusScore_my_image", 0),
        ("ImageQuality_LocalFocusScore_my_image_20", 0),
        ("ImageQuality_Correlation_my_image_20", 0),
        ("ImageQuality_PercentMaximal_my_image", 100),
        ("ImageQuality_PercentMinimal_my_image", 100),
        ("ImageQuality_PowerLogLogSlope_my_image", 0),
        ("ImageQuality_TotalIntensity_my_image", 0),
        ("ImageQuality_MeanIntensity_my_image", 0),
        ("ImageQuality_MedianIntensity_my_image", 0),
        ("ImageQuality_StdIntensity_my_image", 0),
        ("ImageQuality_MADIntensity_my_image", 0),
        ("ImageQuality_MaxIntensity_my_image", 0),
        ("ImageQuality_MinIntensity_my_image", 0),
    ):
        assert m.has_current_measurements(
            "Image", feature_name
        ), ("Missing feature %s" % feature_name)
        m_value = m.get_current_measurement(
            "Image", feature_name
        )
        if not value is None:
            assert m_value == value, "Measured value, %f, for feature %s was not %f" % (
                m_value,
                feature_name,
                value,
            )
    features_and_columns_match(m, q, pipeline=workspace.pipeline)


def features_and_columns_match(
    measurements, module, object_name="Image", pipeline=None
):
    assert object_name in measurements.get_object_names()
    features = measurements.get_feature_names(object_name)
    columns = list(
        filter(
            (lambda x: x[0] == object_name), module.get_measurement_columns(pipeline)
        )
    )
    assert len(features) == len(columns)
    for column in columns:
        assert column[1] in features, "features_and_columns_match, %s not in %s" % (
            column[1],
            features,
        )
        assert column[2] == COLTYPE_FLOAT, (
            "features_and_columns_match, %s type not %s"
            % (column[2], COLTYPE_FLOAT)
        )


def test_zeros_and_mask():
    workspace = make_workspace(numpy.zeros((100, 100)), numpy.zeros((100, 100), bool))
    q = workspace.module
    q.image_groups[0].check_blur.value = True
    q.image_groups[0].check_saturation.value = True
    q.image_groups[0].check_intensity.value = True
    q.image_groups[0].calculate_threshold.value = False
    q.image_groups[0].scale_groups[0].scale.value = 20
    q.run(workspace)
    m = workspace.measurements
    for feature_name, value in (
        ("ImageQuality_FocusScore_my_image", 0),
        ("ImageQuality_LocalFocusScore_my_image_20", 0),
        ("ImageQuality_Correlation_my_image_20", 0),
        ("ImageQuality_PercentMaximal_my_image", 0),
        ("ImageQuality_PercentMinimal_my_image", 0),
        ("ImageQuality_PowerLogLogSlope_my_image", 0),
        ("ImageQuality_TotalIntensity_my_image", 0),
        ("ImageQuality_MeanIntensity_my_image", 0),
        ("ImageQuality_MedianIntensity_my_image", 0),
        ("ImageQuality_StdIntensity_my_image", 0),
        ("ImageQuality_MADIntensity_my_image", 0),
        ("ImageQuality_MaxIntensity_my_image", 0),
        ("ImageQuality_MinIntensity_my_image", 0),
    ):
        assert m.has_current_measurements(
            "Image", feature_name
        ), ("Missing feature %s" % feature_name)
        m_value = m.get_current_measurement(
            "Image", feature_name
        )
        assert m_value == value, "Measured value, %f, for feature %s was not %f" % (
            m_value,
            feature_name,
            value,
        )


def test_image_blur():
    """Test the focus scores of a random image

    The expected variance of a uniform distribution is 1/12 of the
    difference of the extents (=(0,1)). We divide this by the mean
    and the focus_score should be 1/6

    The local focus score is the variance among the 25 focus scores
    divided by the median focus score. This should be low.
    """
    numpy.random.seed(0)
    workspace = make_workspace(numpy.random.uniform(size=(100, 100)))
    q = workspace.module
    q.image_groups[0].check_blur.value = True
    q.image_groups[0].check_saturation.value = False
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False
    q.image_groups[0].scale_groups[0].scale.value = 20
    q.run(workspace)
    m = workspace.measurements
    for feature_name, value in (
        ("ImageQuality_FocusScore_my_image", 1.0 / 6.0),
        ("ImageQuality_LocalFocusScore_my_image_20", 0),
        ("ImageQuality_PercentSaturation_my_image", None),
        ("ImageQuality_PercentMaximal_my_image", None),
    ):
        if value is None:
            assert not m.has_current_measurements(
                "Image", feature_name
            ), ("Feature %s should not be present" % feature_name)
        else:
            assert m.has_current_measurements(
                "Image", feature_name
            ), ("Missing feature %s" % feature_name)

            m_value = m.get_current_measurement(
                "Image", feature_name
            )
            assert round(abs(m_value - value), 2) == 0, (
                "Measured value, %f, for feature %s was not %f"
                % (m_value, feature_name, value)
            )
    features_and_columns_match(m, q)


def test_local_focus_score():
    """Test the local focus score by creating one deviant grid block

    Create one grid block out of four that has a uniform value. That one
    should have a focus score of zero. The others have a focus score of
    1/6, so the local focus score should be the variance of (1/6,1/6,1/6,0)
    divided by the median local norm variance (=1/6)
    """
    expected_value = numpy.var([1.0 / 6.0] * 3 + [0]) * 6.0
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(1000, 1000))
    image[:500, :500] = 0.5
    workspace = make_workspace(image)
    q = workspace.module
    q.image_groups[0].check_blur.value = True
    q.image_groups[0].check_saturation.value = False
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False
    q.image_groups[0].scale_groups[0].scale.value = 500
    q.run(workspace)
    m = workspace.measurements
    value = m.get_current_measurement(
        "Image", "ImageQuality_LocalFocusScore_my_image_500"
    )
    assert round(abs(value - expected_value), 3) == 0


def test_focus_score_with_mask():
    """Test focus score with a mask to block out an aberrant part of the image"""
    numpy.random.seed(0)
    expected_value = 1.0 / 6.0
    image = numpy.random.uniform(size=(1000, 1000))
    mask = numpy.ones(image.shape, bool)
    mask[400:600, 400:600] = False
    image[mask == False] = 0.5
    workspace = make_workspace(image, mask)
    q = workspace.module
    q.image_groups[0].check_blur.value = True
    q.image_groups[0].check_saturation.value = False
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False
    q.image_groups[0].scale_groups[0].scale.value = 500
    q.run(workspace)
    m = workspace.measurements
    value = m.get_current_measurement(
        "Image", "ImageQuality_FocusScore_my_image"
    )
    assert round(abs(value - expected_value), 3) == 0


def test_local_focus_score_with_mask():
    """Test local focus score and mask"""
    numpy.random.seed(0)
    expected_value = numpy.var([1.0 / 6.0] * 3 + [0]) * 6.0
    image = numpy.random.uniform(size=(1000, 1000))
    image[:500, :500] = 0.5
    mask = numpy.ones(image.shape, bool)
    mask[400:600, 400:600] = False
    image[mask == False] = 0.5
    workspace = make_workspace(image, mask)
    q = workspace.module
    q.image_groups[0].check_blur.value = True
    q.image_groups[0].check_saturation.value = False
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False
    q.image_groups[0].scale_groups[0].scale.value = 500
    q.run(workspace)
    m = workspace.measurements
    value = m.get_current_measurement(
        "Image", "ImageQuality_LocalFocusScore_my_image_500"
    )
    assert round(abs(value - expected_value), 3) == 0


def test_saturation():
    """Test percent saturation"""
    image = numpy.zeros((10, 10))
    image[:5, :5] = 1
    workspace = make_workspace(image)
    q = workspace.module
    q.image_groups[0].check_blur.value = False
    q.image_groups[0].check_saturation.value = True
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False
    q.run(workspace)
    m = workspace.measurements
    for feature_name in (
        "ImageQuality_ThresholdOtsu_my_image",
        "ImageQuality_FocusScore_my_image",
        "ImageQuality_LocalFocusScore_my_image_20",
    ):
        assert not m.has_current_measurements(
            "Image", feature_name
        ), ("%s should not be present" % feature_name)
    for (feature_name, expected_value) in (
        ("ImageQuality_PercentMaximal_my_image", 25),
        ("ImageQuality_PercentMinimal_my_image", 75),
    ):
        assert m.has_current_measurements(
            "Image", feature_name
        )
        assert (
            round(
                abs(
                    m.get_current_measurement(
                        "Image", feature_name
                    )
                    - expected_value
                ),
                7,
            )
            == 0
        )
    features_and_columns_match(m, q)


def test_maximal():
    """Test percent maximal"""
    image = numpy.zeros((10, 10))
    image[:5, :5] = 0.5
    expected_value = 100.0 / 4.0
    workspace = make_workspace(image)
    q = workspace.module
    q.image_groups[0].check_blur.value = False
    q.image_groups[0].check_saturation.value = True
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False
    q.run(workspace)
    m = workspace.measurements
    assert (
        round(
            abs(
                expected_value
                - m.get_current_measurement(
                    "Image",
                    "ImageQuality_PercentMaximal_my_image",
                )
            ),
            7,
        )
        == 0
    )


def test_saturation_mask():
    """Test percent saturation with mask"""
    image = numpy.zeros((10, 10))
    # 1/2 of image is saturated
    # 1/4 of image is saturated but masked
    image[:5, :] = 1
    mask = numpy.ones((10, 10), bool)
    mask[:5, 5:] = False
    workspace = make_workspace(image, mask)
    q = workspace.module
    q.image_groups[0].check_blur.value = False
    q.image_groups[0].check_saturation.value = True
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False

    q.run(workspace)
    m = workspace.measurements
    for feature_name in (
        "ImageQuality_ThresholdOtsu_my_image",
        "ImageQuality_FocusScore_my_image",
        "ImageQuality_LocalFocusScore_my_image_20",
    ):
        assert not m.has_current_measurements(
            "Image", feature_name
        ), ("%s should not be present" % feature_name)
    for (feature_name, expected_value) in (
        ("ImageQuality_PercentMaximal_my_image", 100.0 / 3),
        ("ImageQuality_PercentMinimal_my_image", 200.0 / 3),
    ):
        assert m.has_current_measurements(
            "Image", feature_name
        )
        print(
            (
                feature_name,
                expected_value,
                m.get_current_measurement(
                    "Image", feature_name
                ),
            )
        )
        assert (
            round(
                abs(
                    m.get_current_measurement(
                        "Image", feature_name
                    )
                    - expected_value
                ),
                7,
            )
            == 0
        )


def test_maximal_mask():
    """Test percent maximal with mask"""
    image = numpy.zeros((10, 10))
    image[:5, :5] = 0.5
    mask = numpy.ones((10, 10), bool)
    mask[:5, 5:] = False
    expected_value = 100.0 / 3.0
    workspace = make_workspace(image, mask)
    q = workspace.module
    q.image_groups[0].check_blur.value = False
    q.image_groups[0].check_saturation.value = True
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False
    q.run(workspace)
    m = workspace.measurements
    assert (
        round(
            abs(
                expected_value
                - m.get_current_measurement(
                    "Image",
                    "ImageQuality_PercentMaximal_my_image",
                )
            ),
            7,
        )
        == 0
    )


def test_threshold():
    """Test all thresholding methods

    Use an image that has 1/5 of "foreground" pixels to make MOG
    happy and set the object fraction to 1/5 to test this.
    """
    numpy.random.seed(0)
    image = numpy.random.beta(2, 5, size=(100, 100))
    object_fraction = 0.2
    mask = numpy.random.binomial(1, object_fraction, size=(100, 100))
    count = numpy.sum(mask)
    image[mask == 1] = 1.0 - numpy.random.beta(2, 20, size=count)
    #
    # Kapur needs to be quantized
    #
    image = numpy.around(image, 2)

    workspace = make_workspace(image)
    q = workspace.module

    for tm, idx in zip(
        centrosome.threshold.TM_GLOBAL_METHODS,
        list(range(len(centrosome.threshold.TM_GLOBAL_METHODS))),
    ):
        if idx != 0:
            q.add_image_group()
        q.image_groups[idx].image_names.value = "my_image"
        q.image_groups[idx].include_image_scalings.value = False
        q.image_groups[idx].check_blur.value = False
        q.image_groups[idx].check_saturation.value = False
        q.image_groups[idx].check_intensity.value = False
        q.image_groups[idx].calculate_threshold.value = True
        q.image_groups[idx].use_all_threshold_methods.value = False
        t = q.image_groups[idx].threshold_groups[0]
        t.threshold_method.value = tm
        t.object_fraction.value = object_fraction
        t.two_class_otsu.value = O_THREE_CLASS
        t.assign_middle_to_foreground.value = (
            O_FOREGROUND
        )
        t.use_weighted_variance.value = (
            O_WEIGHTED_VARIANCE
        )
    q.run(workspace)
    m = workspace.measurements
    for feature_name in (
        "ImageQuality_FocusScore_my_image",
        "ImageQuality_LocalFocusScore_my_image_20",
        "ImageQuality_PercentSaturation_my_image",
        "ImageQuality_PercentMaximal_my_image",
    ):
        assert not m.has_current_measurements(
            "Image", feature_name
        )
    for tm, idx in zip(
        centrosome.threshold.TM_GLOBAL_METHODS,
        list(range(len(centrosome.threshold.TM_GLOBAL_METHODS))),
    ):
        if tm == centrosome.threshold.TM_OTSU_GLOBAL:
            feature_name = "ImageQuality_ThresholdOtsu_my_image_3FW"
        elif tm == centrosome.threshold.TM_MOG_GLOBAL:
            feature_name = "ImageQuality_ThresholdMoG_my_image_20"
        else:
            feature_name = "ImageQuality_Threshold%s_my_image" % tm.split(" ")[0]
        assert m.has_current_measurements(
            "Image", feature_name
        )
    features_and_columns_match(m, q)


def test_experiment_threshold():
    """Test experiment-wide thresholds"""
    numpy.random.seed(32)
    workspace = make_workspace(numpy.zeros((10, 10)))
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module = workspace.module
    assert isinstance(
        module, cellprofiler.modules.measureimagequality.MeasureImageQuality
    )
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    image_name = module.image_groups[0].image_names.value[0]
    feature = (
        module.image_groups[0].threshold_groups[0].threshold_feature_name(image_name)
    )
    data = numpy.random.uniform(size=100)
    m.add_all_measurements("Image", feature, data.tolist())
    module.post_run(workspace)

    # Check threshold algorithms
    threshold_group = module.image_groups[0].threshold_groups[0]
    threshold_algorithm = threshold_group.threshold_algorithm
    f_mean, f_median, f_std = [
        threshold_group.threshold_feature_name(image_name, agg)
        for agg in (
            cellprofiler.modules.measureimagequality.AGG_MEAN,
            cellprofiler.modules.measureimagequality.AGG_MEDIAN,
            cellprofiler.modules.measureimagequality.AGG_STD,
        )
    ]

    expected = (
        (f_mean, numpy.mean(data)),
        (f_median, numpy.median(data)),
        (f_std, numpy.std(data)),
    )
    for feature, expected_value in expected:
        value = m.get_experiment_measurement(feature)
        assert round(abs(value - expected_value), 7) == 0


def test_experiment_threshold_cycle_skipping():
    """Regression test of IMG-970: can you handle nulls in measurements?"""

    numpy.random.seed(33)
    workspace = make_workspace(numpy.zeros((10, 10)))
    assert isinstance(workspace, cellprofiler_core.workspace.Workspace)
    module = workspace.module
    assert isinstance(
        module, cellprofiler.modules.measureimagequality.MeasureImageQuality
    )
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    image_name = module.image_groups[0].image_names.value[0]
    feature = (
        module.image_groups[0].threshold_groups[0].threshold_feature_name(image_name)
    )
    data = numpy.random.uniform(size=100)
    dlist = data.tolist()
    #
    # Erase 10 randomly
    #
    eraser = numpy.lexsort([numpy.random.uniform(size=100)])[:10]
    mask = numpy.ones(data.shape, bool)
    mask[eraser] = False
    for e in eraser:
        dlist[e] = None

    m.add_all_measurements("Image", feature, dlist)
    module.post_run(workspace)
    features_and_columns_match(
        m, module, EXPERIMENT, pipeline=workspace.pipeline
    )

    # Check threshold algorithms
    threshold_group = module.image_groups[0].threshold_groups[0]
    threshold_algorithm = threshold_group.threshold_algorithm
    image_name = module.image_groups[0].image_names.value_text
    f_mean, f_median, f_std = [
        threshold_group.threshold_feature_name(image_name, agg)
        for agg in (
            cellprofiler.modules.measureimagequality.AGG_MEAN,
            cellprofiler.modules.measureimagequality.AGG_MEDIAN,
            cellprofiler.modules.measureimagequality.AGG_STD,
        )
    ]

    expected = (
        (f_mean, numpy.mean(data[mask])),
        (f_median, numpy.median(data[mask])),
        (f_std, numpy.std(data[mask])),
    )
    for feature, expected_value in expected:
        value = m.get_experiment_measurement(feature)
        assert round(abs(value - expected_value), 7) == 0


def test_use_all_thresholding_methods():
    workspace = make_workspace(numpy.zeros((100, 100)))
    q = workspace.module
    q.image_groups[0].check_blur.value = False
    q.image_groups[0].check_saturation.value = False
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = True
    q.image_groups[0].use_all_threshold_methods.value = True
    q.run(workspace)
    m = workspace.measurements
    for feature_name in [
        "ImageQuality_ThresholdOtsu_my_image_2S",
        "ImageQuality_ThresholdOtsu_my_image_2W",
        "ImageQuality_ThresholdOtsu_my_image_3BW",
        "ImageQuality_ThresholdOtsu_my_image_3BS",
        "ImageQuality_ThresholdOtsu_my_image_3FS",
        "ImageQuality_ThresholdOtsu_my_image_3FW",
        "ImageQuality_ThresholdMoG_my_image_5",
        "ImageQuality_ThresholdMoG_my_image_75",
        "ImageQuality_ThresholdMoG_my_image_95",
        "ImageQuality_ThresholdMoG_my_image_25",
        "ImageQuality_ThresholdBackground_my_image",
        "ImageQuality_ThresholdRobustBackground_my_image",
        "ImageQuality_ThresholdKapur_my_image",
        "ImageQuality_ThresholdRidlerCalvard_my_image",
    ]:
        assert m.has_current_measurements(
            "Image", feature_name
        )
    features_and_columns_match(m, q)


def check_error(caller, event):
    assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("measureimagequality/v3.pipeline")
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
        module, cellprofiler.modules.measureimagequality.MeasureImageQuality
    )
    assert len(module.image_groups) == 2

    group = module.image_groups[0]
    thr = group.threshold_groups[0]
    assert group.image_names.value_text == "Alpha"
    assert group.check_blur
    assert group.scale_groups[0].scale == 25
    assert group.check_saturation
    assert group.calculate_threshold
    assert thr.threshold_method == centrosome.threshold.TM_OTSU
    assert round(abs(thr.object_fraction.value - 0.2), 7) == 0
    assert thr.two_class_otsu == O_THREE_CLASS
    assert (
        thr.use_weighted_variance
        == O_WEIGHTED_VARIANCE
    )
    assert (
        thr.assign_middle_to_foreground
        == O_FOREGROUND
    )

    group = module.image_groups[1]
    thr = group.threshold_groups[0]
    assert group.image_names.value_text == "Beta"
    assert not group.check_blur
    assert group.scale_groups[0].scale == 15
    assert not group.check_saturation
    assert not group.calculate_threshold
    assert thr.threshold_method == centrosome.threshold.TM_MOG
    assert round(abs(thr.object_fraction.value - 0.3), 7) == 0
    assert thr.two_class_otsu == O_TWO_CLASS
    assert thr.use_weighted_variance == O_ENTROPY
    assert (
        thr.assign_middle_to_foreground
        == O_BACKGROUND
    )


def test_load_v4():
    file = tests.frontend.modules.get_test_resources_directory("measureimagequality/v4.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 5
    for module in pipeline.modules():
        assert isinstance(
            module, cellprofiler.modules.measureimagequality.MeasureImageQuality
        )

    module = pipeline.modules()[0]
    assert len(module.image_groups) == 1
    group = module.image_groups[0]
    assert group.threshold_groups == []
    assert module.images_choice == cellprofiler.modules.measureimagequality.O_ALL_LOADED
    assert group.check_blur
    assert group.scale_groups[0].scale == 20
    assert group.check_saturation
    assert group.check_intensity
    assert group.calculate_threshold
    assert group.use_all_threshold_methods

    module = pipeline.modules()[1]
    assert len(module.image_groups) == 1
    group = module.image_groups[0]
    assert module.images_choice == cellprofiler.modules.measureimagequality.O_SELECT
    assert group.image_names.value_text == "Alpha"

    module = pipeline.modules()[2]
    assert len(module.image_groups) == 1
    group = module.image_groups[0]
    assert module.images_choice == cellprofiler.modules.measureimagequality.O_SELECT
    assert "Delta" in group.image_names.value
    assert "Beta" in group.image_names.value
    assert len(group.image_names.value) == 2

    module = pipeline.modules()[3]
    assert len(module.image_groups) == 2
    group = module.image_groups[0]
    assert module.images_choice == cellprofiler.modules.measureimagequality.O_SELECT
    assert group.image_names.value_text == "Delta"
    assert group.check_intensity
    assert not group.use_all_threshold_methods
    thr = group.threshold_groups[0]
    assert thr.threshold_method == centrosome.threshold.TM_OTSU
    assert (
        thr.use_weighted_variance
        == O_WEIGHTED_VARIANCE
    )
    assert thr.two_class_otsu == O_TWO_CLASS
    group = module.image_groups[1]
    assert group.image_names.value_text == "Epsilon"

    module = pipeline.modules()[4]
    assert len(module.image_groups) == 1
    group = module.image_groups[0]
    assert module.images_choice == cellprofiler.modules.measureimagequality.O_SELECT
    assert group.image_names.value_text == "Zeta"
    assert not group.use_all_threshold_methods
    thr = group.threshold_groups[0]
    assert thr.threshold_method == centrosome.threshold.TM_OTSU
    assert (
        thr.use_weighted_variance
        == O_WEIGHTED_VARIANCE
    )
    assert thr.two_class_otsu == O_TWO_CLASS
    thr = group.threshold_groups[1]
    assert thr.threshold_method == centrosome.threshold.TM_OTSU
    assert (
        thr.use_weighted_variance
        == O_WEIGHTED_VARIANCE
    )
    assert thr.two_class_otsu == O_THREE_CLASS
    assert (
        thr.assign_middle_to_foreground
        == O_FOREGROUND
    )


def test_intensity_image():
    """Test operation on a single unmasked image"""
    numpy.random.seed(0)
    pixels = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * 0.99
    pixels[0:2, 0:2] = 1
    workspace = make_workspace(pixels, None)
    q = workspace.module
    q.image_groups[0].check_blur.value = False
    q.image_groups[0].check_saturation.value = False
    q.image_groups[0].check_intensity.value = True
    q.image_groups[0].calculate_threshold.value = False
    q.run(workspace)
    m = workspace.measurements
    assert m.get_current_measurement(
        "Image", "ImageQuality_TotalIntensity_my_image"
    ) == numpy.sum(pixels)
    assert (
        m.get_current_measurement(
            "Image", "ImageQuality_MeanIntensity_my_image"
        )
        == numpy.sum(pixels) / 100.0
    )
    assert m.get_current_image_measurement(
        "ImageQuality_MinIntensity_my_image"
    ) == numpy.min(pixels)
    assert m.get_current_image_measurement(
        "ImageQuality_MaxIntensity_my_image"
    ) == numpy.max(pixels)


def test_check_image_groups():
    workspace = make_workspace(numpy.zeros((100, 100)))
    image_set_list = workspace.image_set_list
    image_set = image_set_list.get_image_set(0)
    for i in range(1, 5):
        image_set.add(
            "my_image%s" % i, cellprofiler_core.image.Image(numpy.zeros((100, 100)))
        )

    q = workspace.module
    # Set my_image1 and my_image2 settings: Saturation only
    q.image_groups[0].image_names.value = "my_image1, my_image2"
    q.image_groups[0].include_image_scalings.value = False
    q.image_groups[0].check_blur.value = False
    q.image_groups[0].check_saturation.value = True
    q.image_groups[0].check_intensity.value = False
    q.image_groups[0].calculate_threshold.value = False

    # Set my_image3 and my_image4's settings: Blur only
    q.add_image_group()
    q.image_groups[1].image_names.value = "my_image3, my_image4"
    q.image_groups[1].include_image_scalings.value = False
    q.image_groups[1].check_blur.value = True
    q.image_groups[1].check_saturation.value = False
    q.image_groups[1].check_intensity.value = False
    q.image_groups[1].calculate_threshold.value = False
    q.run(workspace)
    m = workspace.measurements

    # Make sure each group of settings has (and *doesn't* have) the correct measures
    for i in [1, 2]:
        for feature_name in (
            ("ImageQuality_PercentMaximal_my_image%s" % i),
            ("ImageQuality_PercentMinimal_my_image%s" % i),
        ):
            assert m.has_current_measurements(
                "Image", feature_name
            ), ("Missing feature %s" % feature_name)

        for feature_name in (
            ("ImageQuality_FocusScore_my_image%s" % i),
            ("ImageQuality_LocalFocusScore_my_image%s_20" % i),
            ("ImageQuality_PowerLogLogSlope_my_image%s" % i),
        ):
            assert not m.has_current_measurements(
                "Image", feature_name
            ), ("Erroneously present feature %s" % feature_name)
    for i in [3, 4]:
        for feature_name in (
            ("ImageQuality_FocusScore_my_image%s" % i),
            ("ImageQuality_LocalFocusScore_my_image%s_20" % i),
            ("ImageQuality_PowerLogLogSlope_my_image%s" % i),
        ):
            assert m.has_current_measurements(
                "Image", feature_name
            ), ("Missing feature %s" % feature_name)
        for feature_name in (
            ("ImageQuality_PercentMaximal_my_image%s" % i),
            ("ImageQuality_PercentMinimal_my_image%s" % i),
        ):
            assert not m.has_current_measurements(
                "Image", feature_name
            ), ("Erroneously present feature %s" % feature_name)


def test_images_to_process():
    #
    # Test MeasureImageQuality.images_to_process on a pipeline with a
    # variety of image providers.
    #
    expected_names = ["foo", "bar"]
    pipeline = cellprofiler_core.pipeline.Pipeline()
    module1 = cellprofiler_core.modules.namesandtypes.NamesAndTypes()
    module1.set_module_num(1)
    module1.assignment_method.value = (
        cellprofiler_core.modules.namesandtypes.ASSIGN_RULES
    )
    module1.add_assignment()
    module1.add_assignment()
    module1.assignments[0].image_name.value = expected_names[0]
    module1.assignments[
        0
    ].load_as_choice.value = (
        cellprofiler_core.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
    )
    #
    # TO_DO: issue #652
    #    This test should fail at some later date when we can detect
    #    that an illumination function should not be QA measured
    #
    module1.assignments[1].image_name.value = expected_names[1]
    module1.assignments[
        1
    ].load_as_choice.value = (
        cellprofiler_core.modules.namesandtypes.LOAD_AS_ILLUMINATION_FUNCTION
    )
    module1.assignments[
        2
    ].load_as_choice.value = cellprofiler_core.modules.namesandtypes.LOAD_AS_OBJECTS
    pipeline.add_module(module1)

    module2 = cellprofiler.modules.smooth.Smooth()
    module2.set_module_num(2)
    module2.image_name.value = expected_names[0]
    module2.filtered_image_name.value = "henry"
    pipeline.add_module(module2)

    miq_module = cellprofiler.modules.measureimagequality.MeasureImageQuality()
    miq_module.set_module_num(3)
    miq_module.images_choice.value = (
        cellprofiler.modules.measureimagequality.O_ALL_LOADED
    )
    image_names = miq_module.images_to_process(
        miq_module.image_groups[0], None, pipeline
    )
    assert len(image_names) == len(expected_names)
    for image_name in image_names:
        assert image_name in expected_names


def test_volumetric_measurements():
    # Test that a volumetric pipeline returns volumetric measurements
    labels = numpy.zeros((10, 20, 40), dtype=numpy.uint8)
    labels[:, 5:15, 25:35] = 1
    labels[:, 7, 27] = 2

    workspace = make_workspace(labels, dimensions=3)
    workspace.pipeline.set_volumetric(True)
    module = workspace.module
    module.run(workspace)

    # Names and values will be associated directly
    names = [
        "_".join(
            [
                cellprofiler.modules.measureimagequality.C_IMAGE_QUALITY,
                feature,
                IMAGES_NAME,
            ]
        )
        for feature in [
            cellprofiler.modules.measureimagequality.F_TOTAL_VOLUME,
            cellprofiler.modules.measureimagequality.F_TOTAL_INTENSITY,
            cellprofiler.modules.measureimagequality.F_MEAN_INTENSITY,
            cellprofiler.modules.measureimagequality.F_MEDIAN_INTENSITY,
            cellprofiler.modules.measureimagequality.F_STD_INTENSITY,
            cellprofiler.modules.measureimagequality.F_MAD_INTENSITY,
            cellprofiler.modules.measureimagequality.F_MAX_INTENSITY,
            cellprofiler.modules.measureimagequality.F_MIN_INTENSITY,
        ]
    ]
    values = [
        8000,
        3.9607843137254903,
        0.0004950980392156863,
        0.0,
        0.0013171505688094403,
        0.0,
        0.00784313725490196,
        0.0,
    ]
    expected = dict(list(zip(names, values)))

    for feature, value in list(expected.items()):
        assert workspace.measurements.has_current_measurements(
            "Image", feature
        )

        actual = workspace.measurements.get_current_measurement(
            "Image", feature
        )

        numpy.testing.assert_almost_equal(actual, value, decimal=5)
        print(("{} expected {}, got {}".format(feature, value, actual)))
