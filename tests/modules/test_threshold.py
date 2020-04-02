import io

import centrosome.filter
import centrosome.otsu
import centrosome.threshold
import numpy
import numpy.testing
import skimage.filters
import skimage.filters.rank
import skimage.morphology

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler_core.module
import cellprofiler.modules.threshold
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"


def make_workspace(image, mask=None, dimensions=2):
    """Make a workspace for testing Threshold"""
    module = cellprofiler.modules.threshold.Threshold()
    module.x_name.value = INPUT_IMAGE_NAME
    module.y_name.value = OUTPUT_IMAGE_NAME
    pipeline = cellprofiler.pipeline.Pipeline()
    object_set = cellprofiler.object.ObjectSet()
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cellprofiler.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler.measurement.Measurements(),
        image_set_list,
    )
    image_set.add(
        INPUT_IMAGE_NAME,
        cellprofiler.image.Image(image, dimensions=dimensions)
        if mask is None
        else cellprofiler.image.Image(image, mask, dimensions=dimensions),
    )
    return workspace, module


def test_write_a_test_for_the_new_variable_revision_please():
    assert cellprofiler.modules.threshold.Threshold.variable_revision_number == 10


def test_load_v7():
    with open("./tests/resources/modules/threshold/v7.pipeline", "r") as fd:
        data = fd.read()

    fd = io.StringIO(data)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.loadtxt(fd)
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.threshold.Threshold)
    assert module.x_name.value == "RainbowPony"
    assert module.y_name.value == "GrayscalePony"
    assert module.threshold_scope.value == cellprofiler.modules.threshold.TS_GLOBAL
    assert module.global_operation.value == cellprofiler.modules.threshold.TM_LI
    assert module.threshold_smoothing_scale.value == 1.3488
    assert module.threshold_correction_factor.value == 1.1
    assert module.threshold_range.min == 0.07
    assert module.threshold_range.max == 0.99
    assert module.manual_threshold.value == 0.1
    assert module.thresholding_measurement.value == "Pony_Perimeter"
    assert module.two_class_otsu.value == cellprofiler.modules.threshold.O_TWO_CLASS
    assert (
        module.assign_middle_to_foreground.value
        == cellprofiler.modules.threshold.O_FOREGROUND
    )
    assert module.adaptive_window_size.value == 13


def test_load_v8():
    with open("./tests/resources/modules/threshold/v8.pipeline", "r") as fd:
        data = fd.read()

    fd = io.StringIO(data)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.loadtxt(fd)
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.threshold.Threshold)
    assert module.x_name == "DNA"
    assert module.y_name == "ThreshBlue"
    assert module.threshold_scope == cellprofiler.modules.threshold.TS_GLOBAL
    assert module.global_operation.value == cellprofiler.modules.threshold.TM_LI
    assert module.threshold_smoothing_scale == 0
    assert module.threshold_correction_factor == 1.0
    assert module.threshold_range.min == 0.0
    assert module.threshold_range.max == 1.0
    assert module.manual_threshold == 0.0
    assert module.thresholding_measurement == "None"
    assert module.two_class_otsu == cellprofiler.modules.threshold.O_TWO_CLASS
    assert (
        module.assign_middle_to_foreground
        == cellprofiler.modules.threshold.O_FOREGROUND
    )
    assert module.adaptive_window_size == 50


def test_load_v9():
    with open("./tests/resources/modules/threshold/v9.pipeline", "r") as fd:
        data = fd.read()

    fd = io.StringIO(data)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.loadtxt(fd)
    module = pipeline.modules()[4]
    assert isinstance(module, cellprofiler.modules.threshold.Threshold)
    assert module.x_name == "DNA"
    assert module.y_name == "Threshold"
    assert module.threshold_scope == cellprofiler.modules.threshold.TS_GLOBAL
    assert module.global_operation.value == cellprofiler.modules.threshold.TM_LI
    assert module.threshold_smoothing_scale == 0
    assert module.threshold_correction_factor == 1.0
    assert module.threshold_range.min == 0.0
    assert module.threshold_range.max == 1.0
    assert module.manual_threshold == 0.0
    assert module.thresholding_measurement == "None"
    assert module.two_class_otsu == cellprofiler.modules.threshold.O_TWO_CLASS
    assert (
        module.assign_middle_to_foreground
        == cellprofiler.modules.threshold.O_FOREGROUND
    )
    assert module.adaptive_window_size == 50

    module = pipeline.modules()[5]
    assert module.threshold_scope.value == cellprofiler.modules.threshold.TS_GLOBAL
    assert module.global_operation.value == cellprofiler.modules.threshold.TM_MANUAL

    module = pipeline.modules()[6]
    assert module.threshold_scope.value == cellprofiler.modules.threshold.TS_GLOBAL
    assert module.global_operation.value == centrosome.threshold.TM_ROBUST_BACKGROUND

    module = pipeline.modules()[7]
    assert module.threshold_scope.value == cellprofiler.modules.threshold.TS_GLOBAL
    assert module.global_operation.value == cellprofiler.modules.threshold.TM_LI


def test_load_v10():
    with open("./tests/resources/modules/threshold/v10.pipeline", "r") as fd:
        data = fd.read()

    fd = io.StringIO(data)
    pipeline = cellprofiler.pipeline.Pipeline()
    pipeline.loadtxt(fd)
    module = pipeline.modules()[-1]
    assert isinstance(module, cellprofiler.modules.threshold.Threshold)
    assert module.x_name == "DNA"
    assert module.y_name == "Threshold"
    assert module.threshold_scope == cellprofiler.modules.threshold.TS_GLOBAL
    assert module.global_operation.value == cellprofiler.modules.threshold.TM_LI
    assert module.threshold_smoothing_scale == 0
    assert module.threshold_correction_factor == 1.0
    assert module.threshold_range.min == 0.0
    assert module.threshold_range.max == 1.0
    assert module.manual_threshold == 0.0
    assert module.thresholding_measurement == "None"
    assert module.two_class_otsu == cellprofiler.modules.threshold.O_TWO_CLASS
    assert (
        module.assign_middle_to_foreground
        == cellprofiler.modules.threshold.O_FOREGROUND
    )
    assert module.adaptive_window_size == 50
    assert module.local_operation.value == centrosome.threshold.TM_OTSU


def test_binary_manual():
    """Test a binary threshold with manual threshold value"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20))
    expected = image > 0.5
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.manual_threshold.value = 0.5
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == expected)


def test_binary_global():
    """Test a binary threshold with Otsu global method"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20))
    threshold = skimage.filters.threshold_otsu(image)
    expected = image > threshold
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = centrosome.threshold.TM_OTSU
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == expected)


def test_binary_correction():
    """Test a binary threshold with a correction factor"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20))
    threshold = skimage.filters.threshold_otsu(image) * 0.5
    expected = image > threshold
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = centrosome.threshold.TM_OTSU
    module.threshold_correction_factor.value = 0.5
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == expected)


def test_low_bounds():
    """Test a binary threshold with a low bound"""

    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20))
    image[(image > 0.4) & (image < 0.6)] = 0.5
    expected = image > 0.7
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = centrosome.threshold.TM_OTSU
    module.threshold_range.min = 0.7
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == expected)


def test_high_bounds():
    """Test a binary threshold with a high bound"""

    numpy.random.seed(0)
    image = numpy.random.uniform(size=(40, 40))
    expected = image > 0.1
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = centrosome.threshold.TM_OTSU
    module.threshold_range.max = 0.1
    module.run(workspace)
    output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output.pixel_data == expected)


def test_threshold_from_measurement():
    """Test a binary threshold from previous measurements"""
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(20, 20))
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    module.manual_threshold.value = 0.5
    module.run(workspace)

    module2 = cellprofiler.modules.threshold.Threshold()
    module2.x_name.value = OUTPUT_IMAGE_NAME
    module2.y_name.value = OUTPUT_IMAGE_NAME + "new"
    module2.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module2.global_operation.value = cellprofiler.modules.threshold.TM_MEASUREMENT
    module2.thresholding_measurement.value = (
        "Threshold_FinalThreshold_" + OUTPUT_IMAGE_NAME
    )
    module2.run(workspace)


def test_otsu3_low():
    """Test the three-class otsu, weighted variance middle = background"""
    numpy.random.seed(0)
    image = numpy.hstack(
        (
            numpy.random.exponential(1.5, size=300),
            numpy.random.poisson(15, size=300),
            numpy.random.poisson(30, size=300),
        )
    ).astype(numpy.float32)
    image.shape = (30, 30)
    image = centrosome.filter.stretch(image)
    limage, d = centrosome.threshold.log_transform(image)
    t1, t2 = centrosome.otsu.otsu3(limage)
    threshold = centrosome.threshold.inverse_log_transform(t2, d)
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = centrosome.threshold.TM_OTSU
    module.two_class_otsu.value = cellprofiler.modules.threshold.O_THREE_CLASS
    module.assign_middle_to_foreground.value = (
        cellprofiler.modules.threshold.O_BACKGROUND
    )
    module.run(workspace)
    m = workspace.measurements
    m_threshold = m[
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.FF_ORIG_THRESHOLD % module.y_name.value,
    ]
    assert round(abs(m_threshold - threshold), 7) == 0


def test_otsu3_high():
    """Test the three-class otsu, weighted variance middle = foreground"""
    numpy.random.seed(0)
    image = numpy.hstack(
        (
            numpy.random.exponential(1.5, size=300),
            numpy.random.poisson(15, size=300),
            numpy.random.poisson(30, size=300),
        )
    )
    image.shape = (30, 30)
    image = centrosome.filter.stretch(image)
    limage, d = centrosome.threshold.log_transform(image)
    t1, t2 = centrosome.otsu.otsu3(limage)
    threshold = centrosome.threshold.inverse_log_transform(t1, d)
    workspace, module = make_workspace(image)
    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    module.global_operation.value = centrosome.threshold.TM_OTSU
    module.two_class_otsu.value = cellprofiler.modules.threshold.O_THREE_CLASS
    module.assign_middle_to_foreground.value = (
        cellprofiler.modules.threshold.O_FOREGROUND
    )
    module.run(workspace)
    m = workspace.measurements
    m_threshold = m[
        cellprofiler.measurement.IMAGE,
        cellprofiler.measurement.FF_ORIG_THRESHOLD % module.y_name.value,
    ]
    assert round(abs(m_threshold - threshold), 7) == 0


def test_adaptive_otsu_small():
    """Test the function, get_threshold, using Otsu adaptive / small

    Use a small image (125 x 125) to break the image into four
    pieces, check that the threshold is different in each block
    and that there are four blocks broken at the 75 boundary
    """
    numpy.random.seed(0)
    image = numpy.zeros((120, 110))
    for i0, i1 in ((0, 60), (60, 120)):
        for j0, j1 in ((0, 55), (55, 110)):
            dmin = float(i0 * 2 + j0) / 500.0
            dmult = 1.0 - dmin
            # use the sine here to get a bimodal distribution of values
            r = numpy.random.uniform(0, numpy.pi * 2, (60, 55))
            rsin = (numpy.sin(r) + 1) / 2
            image[i0:i1, j0:j1] = dmin + rsin * dmult
    workspace, x = make_workspace(image)
    x.threshold_scope.value = centrosome.threshold.TM_ADAPTIVE
    x.global_operation.value = centrosome.threshold.TM_OTSU
    threshold, global_threshold = x.get_threshold(
        cellprofiler.image.Image(image, mask=numpy.ones_like(image, bool)), workspace
    )
    assert threshold[0, 0] != threshold[0, 109]
    assert threshold[0, 0] != threshold[119, 0]
    assert threshold[0, 0] != threshold[119, 109]


def test_adaptive_otsu_small():
    """Test the function, get_threshold, using Otsu adaptive / small

    Use a small image (125 x 125) to break the image into four
    pieces, check that the threshold is different in each block
    and that there are four blocks broken at the 75 boundary
    """
    numpy.random.seed(0)
    image = numpy.zeros((120, 110))
    for i0, i1 in ((0, 60), (60, 120)):
        for j0, j1 in ((0, 55), (55, 110)):
            dmin = float(i0 * 2 + j0) / 500.0
            dmult = 1.0 - dmin
            # use the sine here to get a bimodal distribution of values
            r = numpy.random.uniform(0, numpy.pi * 2, (60, 55))
            rsin = (numpy.sin(r) + 1) / 2
            image[i0:i1, j0:j1] = dmin + rsin * dmult
    workspace, x = make_workspace(image)
    x.threshold_scope.value = centrosome.threshold.TM_ADAPTIVE
    x.global_operation.value = centrosome.threshold.TM_OTSU
    threshold, global_threshold = x.get_threshold(
        cellprofiler.image.Image(image, mask=numpy.ones_like(image, bool)), workspace
    )
    assert threshold[0, 0] != threshold[0, 109]
    assert threshold[0, 0] != threshold[119, 0]
    assert threshold[0, 0] != threshold[119, 109]


def test_small_images():
    """Test mixture of gaussians thresholding with few pixels

    Run MOG to see if it blows up, given 0-10 pixels"""
    r = numpy.random.RandomState()
    r.seed(91)
    image = r.uniform(size=(9, 11))
    ii, jj = numpy.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    ii, jj = ii.flatten(), jj.flatten()

    for threshold_method in (
        cellprofiler.modules.threshold.TM_LI,
        centrosome.threshold.TM_OTSU,
        centrosome.threshold.TM_ROBUST_BACKGROUND,
    ):
        for i in range(11):
            mask = numpy.zeros(image.shape, bool)
            if i:
                p = r.permutation(numpy.prod(image.shape))[:i]
                mask[ii[p], jj[p]] = True
            workspace, x = make_workspace(image, mask)
            x.global_operation.value = threshold_method
            x.threshold_scope.value = cellprofiler.modules.identify.TS_GLOBAL
            l, g = x.get_threshold(
                cellprofiler.image.Image(image, mask=mask), workspace
            )
            v = image[mask]
            image = r.uniform(size=(9, 11))
            image[mask] = v
            l1, g1 = x.get_threshold(
                cellprofiler.image.Image(image, mask=mask), workspace
            )
            assert round(abs(l1 - l), 7) == 0


def test_test_manual_background():
    """Test manual background"""
    workspace, x = make_workspace(numpy.zeros((10, 10)))
    x = cellprofiler.modules.threshold.Threshold()
    x.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL
    x.global_operation.value = cellprofiler.modules.threshold.TM_MANUAL
    x.manual_threshold.value = 0.5
    local_threshold, threshold = x.get_threshold(
        cellprofiler.image.Image(
            numpy.zeros((10, 10)), mask=numpy.ones((10, 10), bool)
        ),
        workspace,
    )
    assert threshold == 0.5
    assert threshold == 0.5


def test_threshold_li_uniform_image():
    workspace, module = make_workspace(0.1 * numpy.ones((10, 10)))

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = cellprofiler.modules.threshold.TM_LI

    t_local, t_global = module.get_threshold(image, workspace)

    numpy.testing.assert_almost_equal(t_local, 0.1)

    numpy.testing.assert_almost_equal(t_global, 0.1)


def test_threshold_li_uniform_partial_mask():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    mask[1:3, 1:3] = True

    data[mask] = 0.0

    workspace, module = make_workspace(data, mask)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = cellprofiler.modules.threshold.TM_LI

    t_local, t_global = module.get_threshold(image, workspace)

    numpy.testing.assert_almost_equal(t_local, 0.0)

    numpy.testing.assert_almost_equal(t_global, 0.0)


def test_threshold_li_full_mask():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    workspace, module = make_workspace(data, mask)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = cellprofiler.modules.threshold.TM_LI

    t_local, t_global = module.get_threshold(image, workspace)

    numpy.testing.assert_almost_equal(t_local, 0.0)

    numpy.testing.assert_almost_equal(t_global, 0.0)


def test_threshold_li_image():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    workspace, module = make_workspace(data)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = cellprofiler.modules.threshold.TM_LI

    t_local, t_global = module.get_threshold(image, workspace)

    expected = skimage.filters.threshold_li(data)

    numpy.testing.assert_almost_equal(t_local, expected)

    numpy.testing.assert_almost_equal(t_global, expected)


def test_threshold_li_image_automatic():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    workspace, module = make_workspace(data)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = cellprofiler.modules.threshold.TM_LI

    module.threshold_range.maximum = 0.0  # expected to be ignored

    t_local, t_global = module.get_threshold(image, workspace, automatic=True)

    expected = skimage.filters.threshold_li(data)

    assert t_local != 0.0

    assert t_global != 0.0

    numpy.testing.assert_almost_equal(t_local, expected)

    numpy.testing.assert_almost_equal(t_global, expected)


def test_threshold_li_volume():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    workspace, module = make_workspace(data, dimensions=3)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = cellprofiler.modules.threshold.TM_LI

    t_local, t_global = module.get_threshold(image, workspace)

    expected = skimage.filters.threshold_li(data)

    numpy.testing.assert_almost_equal(t_local, expected)

    numpy.testing.assert_almost_equal(t_global, expected)


def test_threshold_robust_background_mean_sd_volume():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    workspace, module = make_workspace(data, dimensions=3)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = centrosome.threshold.TM_ROBUST_BACKGROUND

    module.averaging_method.value = cellprofiler.modules.threshold.RB_MEAN

    module.variance_method.value = cellprofiler.modules.threshold.RB_SD

    t_local, t_global = module.get_threshold(image, workspace)

    t_local_expected, t_global_expected = centrosome.threshold.get_threshold(
        centrosome.threshold.TM_ROBUST_BACKGROUND,
        cellprofiler.modules.threshold.TS_GLOBAL,
        data,
        threshold_range_min=0.0,
        threshold_range_max=1.0,
        threshold_correction_factor=1.0,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        deviations_above_average=2,
        average_fn=numpy.mean,
        variance_fn=numpy.std,
    )

    numpy.testing.assert_almost_equal(t_local, t_local_expected)

    numpy.testing.assert_almost_equal(t_global, t_global_expected)


def test_threshold_robust_background_median_sd_volume():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    workspace, module = make_workspace(data, dimensions=3)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = centrosome.threshold.TM_ROBUST_BACKGROUND

    module.averaging_method.value = cellprofiler.modules.threshold.RB_MEDIAN

    module.variance_method.value = cellprofiler.modules.threshold.RB_SD

    t_local, t_global = module.get_threshold(image, workspace)

    t_local_expected, t_global_expected = centrosome.threshold.get_threshold(
        centrosome.threshold.TM_ROBUST_BACKGROUND,
        cellprofiler.modules.threshold.TS_GLOBAL,
        data,
        threshold_range_min=0.0,
        threshold_range_max=1.0,
        threshold_correction_factor=1.0,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        deviations_above_average=2,
        average_fn=numpy.median,
        variance_fn=numpy.std,
    )

    numpy.testing.assert_almost_equal(t_local, t_local_expected)

    numpy.testing.assert_almost_equal(t_global, t_global_expected)


def test_threshold_robust_background_mode_sd_volume():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    workspace, module = make_workspace(data, dimensions=3)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = centrosome.threshold.TM_ROBUST_BACKGROUND

    module.averaging_method.value = cellprofiler.modules.threshold.RB_MODE

    module.variance_method.value = cellprofiler.modules.threshold.RB_SD

    t_local, t_global = module.get_threshold(image, workspace)

    t_local_expected, t_global_expected = centrosome.threshold.get_threshold(
        centrosome.threshold.TM_ROBUST_BACKGROUND,
        cellprofiler.modules.threshold.TS_GLOBAL,
        data,
        threshold_range_min=0.0,
        threshold_range_max=1.0,
        threshold_correction_factor=1.0,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        deviations_above_average=2,
        average_fn=centrosome.threshold.binned_mode,
        variance_fn=numpy.std,
    )

    numpy.testing.assert_almost_equal(t_local, t_local_expected)

    numpy.testing.assert_almost_equal(t_global, t_global_expected)


def test_threshold_robust_background_mean_mad_volume():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    workspace, module = make_workspace(data, dimensions=3)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_GLOBAL

    module.global_operation.value = centrosome.threshold.TM_ROBUST_BACKGROUND

    module.averaging_method.value = cellprofiler.modules.threshold.RB_MEAN

    module.variance_method.value = cellprofiler.modules.threshold.RB_MAD

    t_local, t_global = module.get_threshold(image, workspace)

    t_local_expected, t_global_expected = centrosome.threshold.get_threshold(
        centrosome.threshold.TM_ROBUST_BACKGROUND,
        cellprofiler.modules.threshold.TS_GLOBAL,
        data,
        threshold_range_min=0.0,
        threshold_range_max=1.0,
        threshold_correction_factor=1.0,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        deviations_above_average=2,
        average_fn=numpy.mean,
        variance_fn=centrosome.threshold.mad,
    )

    numpy.testing.assert_almost_equal(t_local, t_local_expected)

    numpy.testing.assert_almost_equal(t_global, t_global_expected)


def test_threshold_otsu_full_mask():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    workspace, module = make_workspace(data, mask=mask)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_TWO_CLASS

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_local_expected = numpy.zeros_like(data)

    t_global_expected = 0.0

    numpy.testing.assert_array_equal(t_local, t_local_expected)

    assert t_global == t_global_expected


def test_threshold_otsu_partial_mask_uniform_data():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    mask[2:5, 2:5] = True

    data[mask] = 0.2

    workspace, module = make_workspace(data, mask=mask)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_TWO_CLASS

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_global_expected = 0.2

    t_local_expected = 0.7 * t_global_expected * numpy.ones_like(data)

    numpy.testing.assert_array_almost_equal(t_local, t_local_expected)

    numpy.testing.assert_almost_equal(t_global, t_global_expected)


def test_threshold_otsu_uniform_data():
    data = numpy.ones((10, 10), dtype=numpy.float32)

    data *= 0.2

    workspace, module = make_workspace(data)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_TWO_CLASS

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_global_expected = 0.2

    t_local_expected = 0.7 * t_global_expected * numpy.ones_like(data)

    numpy.testing.assert_array_almost_equal(t_local, t_local_expected)

    numpy.testing.assert_almost_equal(t_global, t_global_expected)


def test_threshold_otsu_image():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    mask[1:-1, 1:-1] = True

    workspace, module = make_workspace(data, mask=mask)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_TWO_CLASS

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_global_expected = skimage.filters.threshold_otsu(data[mask])

    t_local_expected = skimage.filters.rank.otsu(
        skimage.img_as_ubyte(data), skimage.morphology.square(3), mask=mask
    )

    t_local_expected = skimage.img_as_float(t_local_expected)

    t_min = 0.7 * t_global_expected

    t_max = min(1.0, 1.5 * t_global_expected)

    t_local_expected[t_local_expected < t_min] = t_min

    t_local_expected[t_local_expected > t_max] = t_max

    numpy.testing.assert_almost_equal(t_global, t_global_expected)

    numpy.testing.assert_array_almost_equal(t_local, t_local_expected)


def test_threshold_otsu_volume():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(data, mask=mask, dimensions=3)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_TWO_CLASS

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_global_expected = skimage.filters.threshold_otsu(data[mask])

    data = skimage.img_as_ubyte(data)

    t_local_expected = numpy.zeros_like(data)

    for index, plane in enumerate(data):
        t_local_expected[index] = skimage.filters.rank.otsu(
            plane, skimage.morphology.square(3), mask=mask[index]
        )

    t_local_expected = skimage.img_as_float(t_local_expected)

    t_min = 0.7 * t_global_expected

    t_max = min(1.0, 1.5 * t_global_expected)

    t_local_expected[t_local_expected < t_min] = t_min

    t_local_expected[t_local_expected > t_max] = t_max

    numpy.testing.assert_almost_equal(t_global, t_global_expected)

    numpy.testing.assert_array_almost_equal(t_local, t_local_expected)


def test_threshold_otsu3_full_mask():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    workspace, module = make_workspace(data, mask=mask)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_THREE_CLASS

    module.assign_middle_to_foreground.value = (
        cellprofiler.modules.threshold.O_FOREGROUND
    )

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_local_expected, t_global_expected = centrosome.threshold.get_threshold(
        centrosome.threshold.TM_OTSU,
        cellprofiler.modules.threshold.TS_ADAPTIVE,
        data,
        mask=mask,
        adaptive_window_size=3,
        threshold_range_min=0.0,
        threshold_range_max=1.0,
        threshold_correction_factor=1.0,
        two_class_otsu=False,
        assign_middle_to_foreground=True,
    )

    numpy.testing.assert_almost_equal(t_global, t_global_expected)

    numpy.testing.assert_array_almost_equal(t_local, t_local_expected)


def test_threshold_otsu3_image():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    mask[1:-1, 1:-1] = True

    workspace, module = make_workspace(data, mask=mask)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_THREE_CLASS

    module.assign_middle_to_foreground.value = (
        cellprofiler.modules.threshold.O_FOREGROUND
    )

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_local_expected, t_global_expected = centrosome.threshold.get_threshold(
        centrosome.threshold.TM_OTSU,
        cellprofiler.modules.threshold.TS_ADAPTIVE,
        data,
        mask=mask,
        adaptive_window_size=3,
        threshold_range_min=0.0,
        threshold_range_max=1.0,
        threshold_correction_factor=1.0,
        two_class_otsu=False,
        assign_middle_to_foreground=True,
    )

    numpy.testing.assert_almost_equal(t_global, t_global_expected)

    numpy.testing.assert_array_almost_equal(t_local, t_local_expected)


def test_threshold_otsu3_volume():
    numpy.random.seed(73)

    data = numpy.random.rand(10, 10, 10)

    mask = numpy.zeros_like(data, dtype=numpy.bool)

    mask[1:-1, 1:-1, 1:-1] = True

    workspace, module = make_workspace(data, mask=mask, dimensions=3)

    image = workspace.image_set.get_image(INPUT_IMAGE_NAME)

    module.threshold_scope.value = cellprofiler.modules.threshold.TS_ADAPTIVE

    module.global_operation.value = centrosome.threshold.TM_OTSU

    module.two_class_otsu.value = cellprofiler.modules.threshold.O_THREE_CLASS

    module.assign_middle_to_foreground.value = (
        cellprofiler.modules.threshold.O_FOREGROUND
    )

    module.adaptive_window_size.value = 3

    t_local, t_global = module.get_threshold(image, workspace)

    t_global_expected = centrosome.threshold.get_otsu_threshold(
        image.pixel_data,
        image.mask,
        two_class_otsu=False,
        assign_middle_to_foreground=True,
    )

    t_local_expected = numpy.zeros_like(data)

    for index, plane in enumerate(data):
        t_local_expected[index] = centrosome.threshold.get_adaptive_threshold(
            centrosome.threshold.TM_OTSU,
            plane,
            t_global_expected,
            mask=mask[index],
            adaptive_window_size=3,
            two_class_otsu=False,
            assign_middle_to_foreground=True,
        )

    t_min = t_global_expected * 0.7

    t_max = min(1.0, t_global_expected * 1.5)

    t_local_expected[t_local_expected < t_min] = t_min

    t_local_expected[t_local_expected > t_max] = t_max

    numpy.testing.assert_almost_equal(t_global, t_global_expected)

    assert t_local.ndim == 3

    numpy.testing.assert_array_almost_equal(t_local, t_local_expected)
