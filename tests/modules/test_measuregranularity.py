import sys

import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.plugins.measuregranularity
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

print((sys.path))

IMAGE_NAME = "myimage"
OBJECTS_NAME = "myobjects"


def test_load_v3():
    with open("./tests/resources/modules/measuregranularity/v3.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    assert len(module.images) == 2
    for image_setting, image_name, subsample_size, bsize, elsize, glen, objs in (
        (module.images[0], "DNA", 0.25, 0.25, 10, 16, ("Nuclei", "Cells")),
        (
            module.images[1],
            "Actin",
            0.33,
            0.50,
            12,
            20,
            ("Nuclei", "Cells", "Cytoplasm"),
        ),
    ):
        # assertTrue(isinstance(image_setting, M.MeasureGranularity))
        assert image_setting.image_name == image_name
        assert round(abs(image_setting.subsample_size.value - subsample_size), 7) == 0
        assert round(abs(image_setting.image_sample_size.value - bsize), 7) == 0
        assert image_setting.element_size.value == elsize
        assert image_setting.granular_spectrum_length.value == glen
        assert len(image_setting.objects) == len(objs)
        assert image_setting.object_count.value == len(objs)
        assert all([ob.objects_name.value in objs for ob in image_setting.objects])


def make_pipeline(
    image,
    mask,
    subsample_size,
    image_sample_size,
    element_size,
    granular_spectrum_length,
    labels=None,
):
    """Make a pipeline with a MeasureGranularity module

    image - measure granularity on this image
    mask - exclude / include pixels from measurement. None = no mask
    subsample_size, etc. - values for corresponding settings in the module
    returns tuple of module & workspace
    """
    module = cellprofiler.modules.plugins.measuregranularity.MeasureGranularity()
    module.set_module_num(1)
    image_setting = module.images[0]
    # assert isinstance(image_setting, M.MeasureGranularity)
    image_setting.image_name.value = IMAGE_NAME
    image_setting.subsample_size.value = subsample_size
    image_setting.image_sample_size.value = image_sample_size
    image_setting.element_size.value = element_size
    image_setting.granular_spectrum_length.value = granular_spectrum_length
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    img = cellprofiler_core.image.Image(image, mask)
    image_set.add(IMAGE_NAME, img)
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def error_callback(event, caller):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(error_callback)
    pipeline.add_module(module)
    object_set = cellprofiler_core.object.ObjectSet()
    if labels is not None:
        objects = cellprofiler_core.object.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)
        image_setting.add_objects()
        image_setting.objects[0].objects_name.value = OBJECTS_NAME
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        object_set,
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return module, workspace


def test_all_masked():
    """Run on a totally masked image"""
    module, workspace = make_pipeline(
        numpy.zeros((40, 40)), numpy.zeros((40, 40), bool), 0.25, 0.25, 10, 16
    )
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert numpy.isnan(value)


def test_zeros():
    """Run on an image of all zeros"""
    module, workspace = make_pipeline(numpy.zeros((40, 40)), None, 0.25, 0.25, 10, 16)
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - 0), 7) == 0


def test_no_scaling():
    """Run on an image without subsampling or background scaling"""
    #
    # Make an image with granularity at scale 1
    #
    i, j = numpy.mgrid[0:10, 0:10]
    image = (i % 2 == j % 2).astype(float)
    expected = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    module, workspace = make_pipeline(image, None, 1, 1, 10, 16)
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - expected[i - 1]), 7) == 0


def test_subsampling():
    """Run on an image with subsampling"""
    #
    # Make an image with granularity at scale 2
    #
    i, j = numpy.mgrid[0:80, 0:80]
    image = ((i / 8).astype(int) % 2 == (j / 8).astype(int) % 2).astype(float)
    #
    # The 4x4 blocks need two erosions before disappearing. The corners
    # need an additional two erosions before disappearing
    #
    expected = [0, 96, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    module, workspace = make_pipeline(image, None, 0.5, 1, 10, 16)
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - expected[i - 1]), 7) == 0


def test_background_sampling():
    """Run on an image with background subsampling"""
    #
    # Make an image with granularity at scale 2
    #
    i, j = numpy.mgrid[0:80, 0:80]
    image = ((i / 4).astype(int) % 2 == (j / 4).astype(int) % 2).astype(float)
    #
    # Add in a background offset
    #
    image = image * 0.5 + 0.5
    #
    # The 4x4 blocks need two erosions before disappearing. The corners
    # need an additional two erosions before disappearing
    #
    expected = [0, 99, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    module, workspace = make_pipeline(image, None, 1, 0.5, 10, 16)
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - expected[i - 1]), 7) == 0


def test_filter_background():
    """Run on an image, filtering out the background

    This test makes sure that the grey_closing happens correctly
    over the user-specified radius.
    """
    #
    # Make an image with granularity at scale 2
    #
    i, j = numpy.mgrid[0:80, 0:80]
    image = ((i / 4).astype(int) % 2 == (j / 4).astype(int) % 2).astype(float)
    #
    # Scale the pixels down so we have some dynamic range and offset
    # so the background is .2
    #
    image = image * 0.5 + 0.2
    #
    # Paint all background pixels on the edge and 1 in to be 0
    #
    image[:, :2][image[:, :2] < 0.5] = 0
    #
    # Paint all of the foreground pixels on the edge to be .5
    #
    image[:, 0][image[:, 0] > 0.5] = 0.5
    #
    # The pixel at 0,0 doesn't get a background of zero
    #
    image[0, 0] = 0.7
    #
    # The 4x4 blocks need two erosions before disappearing. The corners
    # need an additional two erosions before disappearing
    #
    expected = [0, 99, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    module, workspace = make_pipeline(image, None, 1, 1, 5, 16)
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - expected[i - 1]), 7) == 0


def test_all_masked():
    """Run on objects and a totally masked image"""
    labels = numpy.ones((40, 40), int)
    labels[20:, :] = 2
    module, workspace = make_pipeline(
        numpy.zeros((40, 40)), numpy.zeros((40, 40), bool), 0.25, 0.25, 10, 16, labels
    )
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert numpy.isnan(value)
        values = m.get_current_measurement(OBJECTS_NAME, feature)
        assert len(values) == 2
        assert numpy.all(numpy.isnan(values)) or numpy.all(values == 0)


def test_no_objects():
    """Run on a labels matrix with no objects"""
    module, workspace = make_pipeline(
        numpy.zeros((40, 40)), None, 0.25, 0.25, 10, 16, numpy.zeros((40, 40), int)
    )
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - 0), 7) == 0
        values = m.get_current_measurement(OBJECTS_NAME, feature)
        assert len(values) == 0


def test_zeros():
    """Run on an image of all zeros"""
    labels = numpy.ones((40, 40), int)
    labels[20:, :] = 2
    module, workspace = make_pipeline(
        numpy.zeros((40, 40)), None, 0.25, 0.25, 10, 16, labels
    )
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - 0), 7) == 0
        values = m.get_current_measurement(OBJECTS_NAME, feature)
        assert len(values) == 2
        numpy.testing.assert_almost_equal(values, 0)


def test_no_scaling():
    """Run on an image without subsampling or background scaling"""
    #
    # Make an image with granularity at scale 1
    #
    i, j = numpy.mgrid[0:40, 0:30]
    image = (i % 2 == j % 2).astype(float)
    expected = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = numpy.ones((40, 30), int)
    labels[20:, :] = 2
    module, workspace = make_pipeline(image, None, 1, 1, 10, 16, labels)
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - expected[i - 1]), 7) == 0
        values = m.get_current_measurement(OBJECTS_NAME, feature)
        assert len(values) == 2
        numpy.testing.assert_almost_equal(values, expected[i - 1])


def test_subsampling():
    """Run on an image with subsampling"""
    #
    # Make an image with granularity at scale 2
    #
    i, j = numpy.mgrid[0:80, 0:80]
    image = ((i / 8).astype(int) % 2 == (j / 8).astype(int) % 2).astype(float)
    #
    # The 4x4 blocks need two erosions before disappearing. The corners
    # need an additional two erosions before disappearing
    #
    expected = [0, 96, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = numpy.ones((80, 80), int)
    labels[40:, :] = 2
    module, workspace = make_pipeline(image, None, 0.5, 1, 10, 16, labels)
    assert isinstance(
        module, cellprofiler.modules.plugins.measuregranularity.MeasureGranularity
    )
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(1, 16):
        feature = module.images[0].granularity_feature(i)
        assert feature in m.get_feature_names(cellprofiler_core.measurement.IMAGE)
        value = m.get_current_image_measurement(feature)
        assert round(abs(value - expected[i - 1]), 7) == 0
        values = m.get_current_measurement(OBJECTS_NAME, feature)
        assert len(values) == 2
        #
        # We rescale the downscaled image to the size of the labels
        # and this throws the images off during interpolation
        #
        numpy.testing.assert_almost_equal(values, expected[i - 1], 0)
