import centrosome.cpmorphology
import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.modules
from cellprofiler_core.constants.measurement import C_COUNT, M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y, \
    M_NUMBER_OBJECT_NUMBER, C_LOCATION, C_NUMBER, FTR_CENTER_X, FTR_CENTER_Y, FTR_OBJECT_NUMBER, COLTYPE_FLOAT

import cellprofiler.modules.straightenworms
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.setting
import cellprofiler_core.workspace
import cellprofiler_core.preferences
import tests.frontend.modules

OBJECTS_NAME = "worms"
STRAIGHTENED_OBJECTS_NAME = "straightenedworms"
IMAGE_NAME = "wormimage"
STRAIGHTENED_IMAGE_NAME = "straightenedimage"
AUX_IMAGE_NAME = "auximage"
AUX_STRAIGHTENED_IMAGE_NAME = "auxstraightenedimage"


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("straightenworms/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    assert module.objects_name == "OverlappingWorms"
    assert module.straightened_objects_name == "StraightenedWorms"
    assert module.width == 20
    assert (
        module.training_set_directory.dir_choice
        == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
    )
    assert module.training_set_file_name == "TrainingSet.xml"
    assert module.image_count.value == 2
    for group, input_name, output_name in (
        (module.images[0], "Brightfield", "StraightenedBrightfield"),
        (module.images[1], "Fluorescence", "StraightenedFluorescence"),
    ):
        assert group.image_name == input_name
        assert group.straightened_image_name == output_name


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("straightenworms/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 3
    for alignment, module in zip(
        (
            cellprofiler.modules.straightenworms.FLIP_TOP,
            cellprofiler.modules.straightenworms.FLIP_BOTTOM,
            cellprofiler.modules.straightenworms.FLIP_NONE,
        ),
        pipeline.modules(),
    ):
        assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
        assert module.objects_name == "OverlappingWorms"
        assert module.straightened_objects_name == "StraightenedWorms"
        assert module.width == 20
        assert (
            module.training_set_directory.dir_choice
            == cellprofiler_core.preferences.DEFAULT_INPUT_FOLDER_NAME
        )
        assert module.training_set_file_name == "TrainingSet.mat"
        assert len(module.images) == 1
        assert module.wants_measurements
        assert module.number_of_segments == 4
        assert module.number_of_stripes == 1
        assert module.flip_worms == alignment
        assert module.flip_image == "Brightfield"
        assert module.images[0].image_name == "Brightfield"
        assert module.images[0].straightened_image_name == "StraightenedImage"


def make_workspace(control_points, lengths, radii, image, mask=None, auximage=None):
    """Create a workspace containing the control point measurements

    control_points - an n x 2 x m array where n is the # of control points,
                        and m is the number of objects.
    lengths - the length of each object
    radii - the radii_from_training defining the radius at each control pt
    image - the image to be straightened
    mask - the mask associated with the image (default = no mask)
    auximage - a second image to be straightnened (default = no second image)
    """
    module = cellprofiler.modules.straightenworms.StraightenWorms()
    module.objects_name.value = OBJECTS_NAME
    module.straightened_objects_name.value = STRAIGHTENED_OBJECTS_NAME
    module.images[0].image_name.value = IMAGE_NAME
    module.images[0].straightened_image_name.value = STRAIGHTENED_IMAGE_NAME
    module.flip_image.value = IMAGE_NAME
    module.set_module_num(1)

    # Trick the module into thinking it's read the data file

    class P:
        def __init__(self):
            self.radii_from_training = radii

    module.training_set_directory.dir_choice = (
        cellprofiler_core.preferences.URL_FOLDER_NAME
    )
    module.training_set_directory.custom_path = "http://www.cellprofiler.org"
    module.training_set_file_name.value = "TrainingSet.xml"
    module.training_params = {"TrainingSet.xml": (P(), "URL")}

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)

    m = cellprofiler_core.measurement.Measurements()
    for i, (y, x) in enumerate(control_points):
        for v, f in (
            (x, cellprofiler.modules.straightenworms.F_CONTROL_POINT_X),
            (y, cellprofiler.modules.straightenworms.F_CONTROL_POINT_Y),
        ):
            feature = "_".join(
                (cellprofiler.modules.straightenworms.C_WORM, f, str(i + 1))
            )
            m.add_measurement(OBJECTS_NAME, feature, v)
    feature = "_".join(
        (
            cellprofiler.modules.straightenworms.C_WORM,
            cellprofiler.modules.straightenworms.F_LENGTH,
        )
    )
    m.add_measurement(OBJECTS_NAME, feature, lengths)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(IMAGE_NAME, cellprofiler_core.image.Image(image, mask))

    if auximage is not None:
        image_set.add(AUX_IMAGE_NAME, cellprofiler_core.image.Image(auximage))
        module.add_image()
        module.images[1].image_name.value = AUX_IMAGE_NAME
        module.images[1].straightened_image_name.value = AUX_STRAIGHTENED_IMAGE_NAME

    object_set = cellprofiler_core.object.ObjectSet()
    objects = cellprofiler_core.object.Objects()
    labels = numpy.zeros(image.shape, int)
    for i in range(control_points.shape[2]):
        if lengths[i] == 0:
            continue
        rebuild_worm_from_control_points_approx(
            control_points[:, :, i], radii, labels, i + 1
        )
    objects.segmented = labels

    object_set.add_objects(objects, OBJECTS_NAME)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, image_set, object_set, m, image_set_list
    )
    return workspace, module


def rebuild_worm_from_control_points_approx(control_coords, worm_radii, labels, idx):
    """Rebuild a worm from its control coordinates

    Given a worm specified by some control points along its spline,
    reconstructs an approximate binary image representing the worm.

    Specifically, this function generates an image where successive control
    points have been joined by line segments, and then dilates that by a
    certain (specified) radius.

    Inputs:

    control_coords: A N x 2 double array, where each column contains the x
    and y coordinates for a control point.

    worm_radius: Scalar double. Approximate radius of a typical worm; the
    radius by which the reconstructed worm spline is dilated to form the
    final worm.

    Outputs:
    The coordinates of all pixels in the worm in an N x 2 array"""
    index, count, i, j = centrosome.cpmorphology.get_line_pts(
        control_coords[:-1, 0],
        control_coords[:-1, 1],
        control_coords[1:, 0],
        control_coords[1:, 1],
    )
    #
    # Get rid of the last point for the middle elements - these are
    # duplicated by the first point in the next line
    #
    i = numpy.delete(i, index[1:])
    j = numpy.delete(j, index[1:])
    index = index - numpy.arange(len(index))
    count -= 1
    #
    # Find the control point and within-control-point index of each point
    #
    label = numpy.zeros(len(i), int)
    label[index[1:]] = 1
    label = numpy.cumsum(label)
    order = numpy.arange(len(i)) - index[label]
    frac = order.astype(float) / count[label].astype(float)
    radius = worm_radii[label] * (1 - frac) + worm_radii[label + 1] * frac
    iworm_radius = int(numpy.max(numpy.ceil(radius)))
    #
    # Get dilation coordinates
    #
    ii, jj = numpy.mgrid[
        -iworm_radius : iworm_radius + 1, -iworm_radius : iworm_radius + 1
    ]
    dd = numpy.sqrt((ii * ii + jj * jj).astype(float))
    mask = ii * ii + jj * jj <= iworm_radius * iworm_radius
    ii = ii[mask]
    jj = jj[mask]
    dd = dd[mask]
    #
    # All points (with repeats)
    #
    i = (i[:, numpy.newaxis] + ii[numpy.newaxis, :]).flatten()
    j = (j[:, numpy.newaxis] + jj[numpy.newaxis, :]).flatten()
    #
    # We further mask out any dilation coordinates outside of
    # the radius at our point in question
    #
    m = (radius[:, numpy.newaxis] >= dd[numpy.newaxis, :]).flatten()
    i = i[m]
    j = j[m]
    #
    # Find repeats by sorting and comparing against next
    #
    order = numpy.lexsort((i, j))
    i = i[order]
    j = j[order]
    mask = numpy.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
    i = i[mask]
    j = j[mask]
    mask = (i >= 0) & (j >= 0) & (i < labels.shape[0]) & (j < labels.shape[1])
    labels[i[mask], j[mask]] = idx


def test_straighten_nothing():
    workspace, module = make_workspace(
        numpy.zeros((5, 2, 0)), numpy.zeros(0), numpy.zeros(5), numpy.zeros((20, 10))
    )
    module.run(workspace)

    image = workspace.image_set.get_image(STRAIGHTENED_IMAGE_NAME)
    # TODO: This test is failing due to a chance in ndimage.map_coordinates as of scipy v1.1
    # TODO: Previously the map_coordinates call on line 542 of straightenworms would return
    # TODO: an all False array - as of v1.1 it is returning an all True array. I don't know
    # TODO: enough about map_coordinates to know what it should be doing here or why, so I'll
    # TODO: need to come back to this and see what needs to be changed for this to be resolved
    # assertFalse(np.any(image.mask))
    objectset = workspace.object_set
    assert isinstance(objectset, cellprofiler_core.object.ObjectSet)
    labels = objectset.get_objects(STRAIGHTENED_OBJECTS_NAME).segmented
    assert numpy.all(labels == 0)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    assert (
        m.get_current_image_measurement(
            "_".join((C_COUNT, STRAIGHTENED_OBJECTS_NAME))
        )
        == 0
    )
    assert (
        len(
            m.get_current_measurement(
                STRAIGHTENED_OBJECTS_NAME,
                M_LOCATION_CENTER_X,
            )
        )
        == 0
    )
    assert (
        len(
            m.get_current_measurement(
                STRAIGHTENED_OBJECTS_NAME,
                M_LOCATION_CENTER_Y,
            )
        )
        == 0
    )


def test_straighten_straight_worm():
    """Do a "straightening" that is a 1-1 mapping"""
    r = numpy.random.RandomState()
    r.seed(0)
    image = r.uniform(size=(60, 30))
    control_points = numpy.array(
        [[[21], [15]], [[23], [15]], [[25], [15]], [[27], [15]], [[29], [15]]]
    )
    lengths = numpy.array([8])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.wants_measurements.value = False
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 11
    assert pixels.shape[0] == 19
    numpy.testing.assert_almost_equal(pixels, image[16:35, 10:21])

    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    assert (
        m.get_current_image_measurement(
            "_".join((C_COUNT, STRAIGHTENED_OBJECTS_NAME))
        )
        == 1
    )
    v = m.get_current_measurement(
        STRAIGHTENED_OBJECTS_NAME, M_LOCATION_CENTER_X
    )
    assert len(v) == 1
    assert round(abs(v[0] - 5), 7) == 0
    v = m.get_current_measurement(
        STRAIGHTENED_OBJECTS_NAME, M_LOCATION_CENTER_Y
    )
    assert len(v) == 1
    assert round(abs(v[0] - 9), 7) == 0
    object_set = workspace.object_set
    objects = object_set.get_objects(STRAIGHTENED_OBJECTS_NAME)
    orig_objects = object_set.get_objects(OBJECTS_NAME)
    assert numpy.all(objects.segmented == orig_objects.segmented[16:35, 10:21])


def test_straighten_diagonal_worm():
    """Do a straightening on a worm on the 3x4x5 diagonal"""
    r = numpy.random.RandomState()
    r.seed(23)
    image = r.uniform(size=(60, 30))
    control_points = numpy.array(
        [[[10], [10]], [[13], [14]], [[16], [18]], [[19], [22]], [[22], [26]]]
    )
    lengths = numpy.array([20])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 11
    assert pixels.shape[0] == 31
    expected = image[control_points[:, 0, 0], control_points[:, 1, 0]]
    samples = pixels[5:26:5, 5]
    numpy.testing.assert_almost_equal(expected, samples)


def test_straighten_two_worms():
    """Straighten the worms from tests 02_02 and 02_03 together"""
    r = numpy.random.RandomState()
    r.seed(0)
    image = r.uniform(size=(60, 30))
    control_points = numpy.array(
        [
            [[21, 10], [15, 10]],
            [[23, 13], [15, 14]],
            [[25, 16], [15, 18]],
            [[27, 19], [15, 22]],
            [[29, 22], [15, 26]],
        ]
    )
    lengths = numpy.array([8, 20])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 22
    assert pixels.shape[0] == 31
    numpy.testing.assert_almost_equal(pixels[0:19, 0:11], image[16:35, 10:21])
    expected = image[control_points[:, 0, 1], control_points[:, 1, 1]]
    samples = pixels[5:26:5, 16]
    numpy.testing.assert_almost_equal(expected, samples)


def test_straighten_missing_worm():
    r = numpy.random.RandomState()
    r.seed(0)
    image = r.uniform(size=(60, 30))
    control_points = numpy.array(
        [
            [[21, numpy.nan, 10], [15, numpy.nan, 10]],
            [[23, numpy.nan, 13], [15, numpy.nan, 14]],
            [[25, numpy.nan, 16], [15, numpy.nan, 18]],
            [[27, numpy.nan, 19], [15, numpy.nan, 22]],
            [[29, numpy.nan, 22], [15, numpy.nan, 26]],
        ]
    )
    lengths = numpy.array([8, 0, 20])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 33
    assert pixels.shape[0] == 31
    numpy.testing.assert_almost_equal(pixels[0:19, 0:11], image[16:35, 10:21])
    expected = image[
        control_points[:, 0, 2].astype(int), control_points[:, 1, 2].astype(int)
    ]
    samples = pixels[5:26:5, 27]
    numpy.testing.assert_almost_equal(expected, samples)


def test_get_measurement_columns():
    workspace, module = make_workspace(
        numpy.zeros((5, 2, 0)), numpy.zeros(0), numpy.zeros(5), numpy.zeros((20, 10))
    )
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.wants_measurements.value = False
    columns = module.get_measurement_columns(workspace.pipeline)
    for object_name, feature_name in (
        (
            "Image",
            "_".join(
                (C_COUNT, STRAIGHTENED_OBJECTS_NAME)
            ),
        ),
        (STRAIGHTENED_OBJECTS_NAME, M_LOCATION_CENTER_X),
        (STRAIGHTENED_OBJECTS_NAME, M_LOCATION_CENTER_Y),
        (
            STRAIGHTENED_OBJECTS_NAME,
            M_NUMBER_OBJECT_NUMBER,
        ),
    ):
        assert any([(o == object_name) and (f == feature_name) for o, f, t in columns])

    categories = module.get_categories(
        workspace.pipeline, "Image"
    )
    assert len(categories) == 1
    assert categories[0] == C_COUNT

    categories = module.get_categories(workspace.pipeline, STRAIGHTENED_OBJECTS_NAME)
    assert len(categories) == 2
    assert C_LOCATION in categories
    assert C_NUMBER in categories

    f = module.get_measurements(
        workspace.pipeline,
        "Image",
        C_COUNT,
    )
    assert len(f) == 1
    assert f[0] == STRAIGHTENED_OBJECTS_NAME

    f = module.get_measurements(
        workspace.pipeline,
        STRAIGHTENED_OBJECTS_NAME,
        C_NUMBER,
    )
    assert len(f) == 1
    assert f[0] == FTR_OBJECT_NUMBER

    f = module.get_measurements(
        workspace.pipeline,
        STRAIGHTENED_OBJECTS_NAME,
        C_LOCATION,
    )
    assert len(f) == 2
    assert FTR_CENTER_X in f
    assert FTR_CENTER_Y in f


def test_get_measurement_columns_wants_images_vertical():
    workspace, module = make_workspace(
        numpy.zeros((5, 2, 0)),
        numpy.zeros(0),
        numpy.zeros(5),
        numpy.zeros((20, 10)),
        auximage=numpy.zeros((20, 10)),
    )
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.wants_measurements.value = True
    module.number_of_segments.value = 5
    module.number_of_stripes.value = 1

    expected_columns = [
        (
            OBJECTS_NAME,
            "_".join(
                (
                    cellprofiler.modules.straightenworms.C_WORM,
                    ftr,
                    image,
                    module.get_scale_name(None, segno),
                )
            ),
            COLTYPE_FLOAT,
        )
        for ftr, image, segno in zip(
            [cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY] * 10
            + [cellprofiler.modules.straightenworms.FTR_STD_INTENSITY] * 10,
            ([STRAIGHTENED_IMAGE_NAME] * 5 + [AUX_STRAIGHTENED_IMAGE_NAME] * 5) * 2,
            list(range(5)) * 4,
        )
    ]

    columns = module.get_measurement_columns(workspace.pipeline)
    columns = [column for column in columns if column[0] == OBJECTS_NAME]
    for expected_column in expected_columns:
        assert any(
            [
                all([x == y for x, y in zip(column, expected_column)])
                for column in columns
            ]
        )
    for column in columns:
        assert any(
            [
                all([x == y for x, y in zip(column, expected_column)])
                for expected_column in expected_columns
            ]
        )

    categories = module.get_categories(workspace.pipeline, OBJECTS_NAME)
    assert cellprofiler.modules.straightenworms.C_WORM in categories

    features = module.get_measurements(
        workspace.pipeline, OBJECTS_NAME, cellprofiler.modules.straightenworms.C_WORM
    )
    assert len(features) == 2
    assert cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY in features
    assert cellprofiler.modules.straightenworms.FTR_STD_INTENSITY in features

    for ftr in (
        cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY,
        cellprofiler.modules.straightenworms.FTR_STD_INTENSITY,
    ):
        images = module.get_measurement_images(
            workspace.pipeline,
            OBJECTS_NAME,
            cellprofiler.modules.straightenworms.C_WORM,
            ftr,
        )
        assert len(images) == 2
        assert STRAIGHTENED_IMAGE_NAME in images
        assert AUX_STRAIGHTENED_IMAGE_NAME in images

    for ftr, image in zip(
        [
            cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY,
            cellprofiler.modules.straightenworms.FTR_STD_INTENSITY,
        ]
        * 2,
        [STRAIGHTENED_IMAGE_NAME] * 2 + [AUX_STRAIGHTENED_IMAGE_NAME] * 2,
    ):
        scales = module.get_measurement_scales(
            workspace.pipeline,
            OBJECTS_NAME,
            cellprofiler.modules.straightenworms.C_WORM,
            ftr,
            image,
        )
        assert len(scales) == 5
        for expected_scale in range(5):
            assert module.get_scale_name(None, expected_scale) in scales


def test_get_measurement_columns_horizontal():
    workspace, module = make_workspace(
        numpy.zeros((5, 2, 0)),
        numpy.zeros(0),
        numpy.zeros(5),
        numpy.zeros((20, 10)),
        auximage=numpy.zeros((20, 10)),
    )
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.wants_measurements.value = True
    module.number_of_segments.value = 1
    module.number_of_stripes.value = 5
    expected_columns = [
        (
            OBJECTS_NAME,
            "_".join(
                (
                    cellprofiler.modules.straightenworms.C_WORM,
                    ftr,
                    image,
                    module.get_scale_name(segno, None),
                )
            ),
            COLTYPE_FLOAT,
        )
        for ftr, image, segno in zip(
            [cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY] * 10
            + [cellprofiler.modules.straightenworms.FTR_STD_INTENSITY] * 10,
            ([STRAIGHTENED_IMAGE_NAME] * 5 + [AUX_STRAIGHTENED_IMAGE_NAME] * 5) * 2,
            list(range(5)) * 4,
        )
    ]

    columns = module.get_measurement_columns(workspace.pipeline)
    columns = [column for column in columns if column[0] == OBJECTS_NAME]
    for expected_column in expected_columns:
        assert any(
            [
                all([x == y for x, y in zip(column, expected_column)])
                for column in columns
            ]
        )
    for column in columns:
        assert any(
            [
                all([x == y for x, y in zip(column, expected_column)])
                for expected_column in expected_columns
            ]
        )

    categories = module.get_categories(workspace.pipeline, OBJECTS_NAME)
    assert cellprofiler.modules.straightenworms.C_WORM in categories

    features = module.get_measurements(
        workspace.pipeline, OBJECTS_NAME, cellprofiler.modules.straightenworms.C_WORM
    )
    assert len(features) == 2
    assert cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY in features
    assert cellprofiler.modules.straightenworms.FTR_STD_INTENSITY in features

    for ftr in (
        cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY,
        cellprofiler.modules.straightenworms.FTR_STD_INTENSITY,
    ):
        images = module.get_measurement_images(
            workspace.pipeline,
            OBJECTS_NAME,
            cellprofiler.modules.straightenworms.C_WORM,
            ftr,
        )
        assert len(images) == 2
        assert STRAIGHTENED_IMAGE_NAME in images
        assert AUX_STRAIGHTENED_IMAGE_NAME in images

    for ftr, image in zip(
        [
            cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY,
            cellprofiler.modules.straightenworms.FTR_STD_INTENSITY,
        ]
        * 2,
        [STRAIGHTENED_IMAGE_NAME] * 2 + [AUX_STRAIGHTENED_IMAGE_NAME] * 2,
    ):
        scales = module.get_measurement_scales(
            workspace.pipeline,
            OBJECTS_NAME,
            cellprofiler.modules.straightenworms.C_WORM,
            ftr,
            image,
        )
        assert len(scales) == 5
        for expected_scale in range(5):
            assert module.get_scale_name(expected_scale, None) in scales


def test_get_measurement_columns_both():
    workspace, module = make_workspace(
        numpy.zeros((5, 2, 0)),
        numpy.zeros(0),
        numpy.zeros(5),
        numpy.zeros((20, 10)),
        auximage=numpy.zeros((20, 10)),
    )
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.wants_measurements.value = True
    module.number_of_segments.value = 2
    module.number_of_stripes.value = 3
    expected_columns = []
    vscales = (None, 0, 1)
    hscales = (None, 0, 1, 2)
    for image in (STRAIGHTENED_IMAGE_NAME, AUX_STRAIGHTENED_IMAGE_NAME):
        for ftr in (
            cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY,
            cellprofiler.modules.straightenworms.FTR_STD_INTENSITY,
        ):
            for vscale in vscales:
                for hscale in hscales:
                    if vscale is None and hscale is None:
                        continue
                    meas = "_".join(
                        (
                            cellprofiler.modules.straightenworms.C_WORM,
                            ftr,
                            image,
                            module.get_scale_name(hscale, vscale),
                        )
                    )
                    expected_columns.append(
                        (
                            OBJECTS_NAME,
                            meas,
                            COLTYPE_FLOAT,
                        )
                    )
    columns = module.get_measurement_columns(workspace.pipeline)
    columns = [column for column in columns if column[0] == OBJECTS_NAME]
    for expected_column in expected_columns:
        assert any(
            [
                all([x == y for x, y in zip(column, expected_column)])
                for column in columns
            ]
        )
    for column in columns:
        assert any(
            [
                all([x == y for x, y in zip(column, expected_column)])
                for expected_column in expected_columns
            ]
        )
    for ftr in (
        cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY,
        cellprofiler.modules.straightenworms.FTR_STD_INTENSITY,
    ):
        for image in (STRAIGHTENED_IMAGE_NAME, AUX_STRAIGHTENED_IMAGE_NAME):
            scales = module.get_measurement_scales(
                workspace.pipeline,
                OBJECTS_NAME,
                cellprofiler.modules.straightenworms.C_WORM,
                ftr,
                image,
            )
            assert len(scales) == 11
            for vscale in vscales:
                for hscale in hscales:
                    if vscale is None and hscale is None:
                        continue
                    assert module.get_scale_name(hscale, vscale) in scales


def test_measure_no_worms():
    workspace, module = make_workspace(
        numpy.zeros((5, 2, 0)), numpy.zeros(0), numpy.zeros(5), numpy.zeros((20, 10))
    )
    module.wants_measurements.value = True
    module.number_of_segments.value = 5
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for i in range(5):
        for ftr, function in (
            (cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY, numpy.mean),
            (cellprofiler.modules.straightenworms.FTR_STD_INTENSITY, numpy.std),
        ):
            mname = "_".join(
                (
                    cellprofiler.modules.straightenworms.C_WORM,
                    ftr,
                    STRAIGHTENED_IMAGE_NAME,
                    module.get_scale_name(None, i),
                )
            )
            v = m.get_current_measurement(OBJECTS_NAME, mname)
            assert len(v) == 0


def test_measure_one_worm():
    r = numpy.random.RandomState()
    r.seed(0)
    image = r.uniform(size=(60, 30))
    control_points = numpy.array(
        [[[21], [15]], [[24], [15]], [[27], [15]], [[30], [15]], [[33], [15]]]
    )
    lengths = numpy.array([12])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.wants_measurements.value = True
    module.number_of_segments.value = 4
    module.number_of_stripes.value = 3
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    oo = workspace.object_set.get_objects(OBJECTS_NAME)
    #
    # The worm goes from 20 to 34. Each segment is 15 / 4 = 3 3/4 long
    #
    # 20, 21, and 22 are in 1 as is 3/4 of 23
    # 1/4 of 23 is in 2 as is 24, 25 and 26 and 1/2 of 27
    # 1/2 of 27 is in 3 as is 28, 29, 30 and 1/4 of 31
    # 3/4 of 31 is in 4 as is 32, 33, and 34
    #
    segments = [
        [(20, 1.0), (21, 1.0), (22, 1.0), (23, 0.75)],
        [(23, 0.25), (24, 1.0), (25, 1.0), (26, 1.0), (27, 0.5)],
        [(27, 0.50), (28, 1.0), (29, 1.0), (30, 1.0), (31, 0.25)],
        [(31, 0.75), (32, 1.0), (33, 1.0), (34, 1.0)],
    ]

    def weighted_mean(img, segments, mask):
        accumulator = 0.0
        weight_accumulator = 0.0
        for i, w in segments:
            piece = img[i, mask[i, :]]
            accumulator += numpy.sum(piece) * w
            weight_accumulator += w * numpy.sum(mask[i, :])
        return accumulator / weight_accumulator

    def weighted_std(img, segments, mask):
        mean = weighted_mean(img, segments, mask)
        accumulator = 0.0
        weight_accumulator = 0.0
        pixel_count = 0.0
        for i, w in segments:
            piece = img[i, mask[i, :]]
            accumulator += numpy.sum((piece - mean) ** 2) * w
            weight_accumulator += w * numpy.sum(mask[i, :])
            pixel_count += numpy.sum(mask[i, :])
        return numpy.sqrt(
            accumulator / weight_accumulator / (pixel_count - 1) * pixel_count
        )

    for i, segment in enumerate(segments):
        for ftr, function in (
            (cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY, weighted_mean),
            (cellprofiler.modules.straightenworms.FTR_STD_INTENSITY, weighted_std),
        ):
            mname = "_".join(
                (
                    cellprofiler.modules.straightenworms.C_WORM,
                    ftr,
                    STRAIGHTENED_IMAGE_NAME,
                    module.get_scale_name(None, i),
                )
            )
            v = m.get_current_measurement(OBJECTS_NAME, mname)
            expected = function(image, segment, oo.segmented == 1)
            assert len(v) == 1
            assert round(abs(v[0] - expected), 7) == 0


def test_measure_checkerboarded_worm():
    r = numpy.random.RandomState()
    r.seed(42)
    image = r.uniform(size=(60, 30))
    control_points = numpy.array(
        [[[21], [15]], [[24], [15]], [[27], [15]], [[30], [15]], [[33], [15]]]
    )
    lengths = numpy.array([12])
    radii = numpy.array([1, 4, 7, 4, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 21
    module.wants_measurements.value = True
    module.number_of_segments.value = 4
    module.number_of_stripes.value = 3
    module.run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    oo = workspace.object_set.get_objects(OBJECTS_NAME)
    image = workspace.image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    f1 = 1.0 / 3.0
    f2 = 2.0 / 3.0
    stripes = [
        (9, ((10, 11, 2, 1),)),
        (
            10,
            (
                (8, 9, 1, 1),
                (9, 10, 1, f2),
                (9, 10, 2, f1),
                (10, 11, 2, 1),
                (11, 12, 2, f1),
                (11, 12, 3, f2),
                (12, 13, 3, 1),
            ),
        ),
        (
            11,
            (
                (7, 9, 1, 1),
                (9, 10, 1, f1),
                (9, 10, 2, f2),
                (10, 11, 2, 1),
                (11, 12, 2, f2),
                (11, 12, 3, f1),
                (12, 14, 3, 1),
            ),
        ),
        (12, ((6, 9, 1, 1), (9, 12, 2, 1), (12, 15, 3, 1))),
        (
            13,
            (
                (5, 8, 1, 1),
                (8, 9, 1, f2),
                (8, 9, 2, f1),
                (9, 12, 2, 1),
                (12, 13, 2, f1),
                (12, 13, 3, f2),
                (13, 16, 3, 1),
            ),
        ),
        (
            14,
            (
                (4, 8, 1, 1),
                (8, 9, 1, f1),
                (8, 9, 2, f2),
                (9, 12, 2, 1),
                (12, 13, 2, f2),
                (12, 13, 3, f1),
                (13, 17, 3, 1),
            ),
        ),
        (15, ((4, 8, 1, 1), (8, 13, 2, 1), (13, 17, 3, 1))),
        (16, ((3, 8, 1, 1), (8, 13, 2, 1), (13, 18, 3, 1))),
        (17, ((4, 8, 1, 1), (8, 13, 2, 1), (13, 17, 3, 1))),
        (
            18,
            (
                (4, 8, 1, 1),
                (8, 9, 1, f1),
                (8, 9, 2, f2),
                (9, 12, 2, 1),
                (12, 13, 2, f2),
                (12, 13, 3, f1),
                (13, 17, 3, 1),
            ),
        ),
        (
            19,
            (
                (5, 8, 1, 1),
                (8, 9, 1, f2),
                (8, 9, 2, f1),
                (9, 12, 2, 1),
                (12, 13, 2, f1),
                (12, 13, 3, f2),
                (13, 16, 3, 1),
            ),
        ),
        (20, ((6, 9, 1, 1), (9, 12, 2, 1), (12, 15, 3, 1))),
        (
            21,
            (
                (7, 9, 1, 1),
                (9, 10, 1, f1),
                (9, 10, 2, f2),
                (10, 11, 2, 1),
                (11, 12, 2, f2),
                (11, 12, 3, f1),
                (12, 14, 3, 1),
            ),
        ),
        (
            22,
            (
                (8, 9, 1, 1),
                (9, 10, 1, f2),
                (9, 10, 2, f1),
                (10, 11, 2, 1),
                (11, 12, 2, f1),
                (11, 12, 3, f2),
                (12, 13, 3, 1),
            ),
        ),
        (23, ((10, 11, 2, 1),)),
    ]
    segments = [
        [(9, 1.0), (10, 1.0), (11, 1.0), (12, 0.75)],
        [(12, 0.25), (13, 1.0), (14, 1.0), (15, 1.0), (16, 0.5)],
        [(16, 0.50), (17, 1.0), (18, 1.0), (19, 1.0), (20, 0.25)],
        [(20, 0.75), (21, 1.0), (22, 1.0), (23, 1.0)],
    ]

    i_w = numpy.zeros((image.shape[0], image.shape[1], 4))
    j_w = numpy.zeros((image.shape[0], image.shape[1], 3))
    mask = numpy.zeros(image.shape, bool)
    for i, sstripes in stripes:
        for jstart, jend, idx, w in sstripes:
            for j in range(jstart, jend):
                j_w[i, j, idx - 1] = w
                mask[i, j] = True

    for idx, segment in enumerate(segments):
        for i, w in segment:
            i_w[i, mask[i, :], idx] = w

    s2 = lambda x: numpy.sum(numpy.sum(x, 0), 0)
    weights = s2(i_w)
    expected_means = s2(image[:, :, numpy.newaxis] * i_w) / weights
    counts = s2(i_w > 0)
    expected_sds = numpy.sqrt(
        s2(
            i_w
            * (
                image[:, :, numpy.newaxis]
                - expected_means[numpy.newaxis, numpy.newaxis, :]
            )
            ** 2
        )
        / weights
        * counts
        / (counts - 1)
    )
    for i in range(4):
        for ftr, expected in (
            (cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY, expected_means),
            (cellprofiler.modules.straightenworms.FTR_STD_INTENSITY, expected_sds),
        ):
            value = m.get_current_measurement(
                OBJECTS_NAME,
                "_".join(
                    (
                        cellprofiler.modules.straightenworms.C_WORM,
                        ftr,
                        STRAIGHTENED_IMAGE_NAME,
                        module.get_scale_name(None, i),
                    )
                ),
            )
            assert len(value) == 1
            assert round(abs(value[0] - expected[i]), 7) == 0
    weights = s2(j_w)
    expected_means = s2(image[:, :, numpy.newaxis] * j_w) / weights
    counts = s2(j_w > 0)
    expected_sds = numpy.sqrt(
        s2(
            j_w
            * (
                image[:, :, numpy.newaxis]
                - expected_means[numpy.newaxis, numpy.newaxis, :]
            )
            ** 2
        )
        / weights
        * counts
        / (counts - 1)
    )
    for i in range(3):
        for ftr, expected in (
            (cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY, expected_means),
            (cellprofiler.modules.straightenworms.FTR_STD_INTENSITY, expected_sds),
        ):
            value = m.get_current_measurement(
                OBJECTS_NAME,
                "_".join(
                    (
                        cellprofiler.modules.straightenworms.C_WORM,
                        ftr,
                        STRAIGHTENED_IMAGE_NAME,
                        module.get_scale_name(i, None),
                    )
                ),
            )
            assert len(value) == 1
            assert round(abs(value[0] - expected[i]), 7) == 0

    ww = i_w[:, :, :, numpy.newaxis] * j_w[:, :, numpy.newaxis, :]
    weights = s2(ww)
    expected_means = s2(image[:, :, numpy.newaxis, numpy.newaxis] * ww) / weights
    counts = s2(ww > 0)
    expected_sds = numpy.sqrt(
        s2(
            ww
            * (
                image[:, :, numpy.newaxis, numpy.newaxis]
                - expected_means[numpy.newaxis, numpy.newaxis, :, :]
            )
            ** 2
        )
        / weights
        * counts
        / (counts - 1)
    )
    for stripe in range(3):
        for segment in range(4):
            for ftr, expected in (
                (
                    cellprofiler.modules.straightenworms.FTR_MEAN_INTENSITY,
                    expected_means,
                ),
                (cellprofiler.modules.straightenworms.FTR_STD_INTENSITY, expected_sds),
            ):
                mname = "_".join(
                    (
                        cellprofiler.modules.straightenworms.C_WORM,
                        ftr,
                        STRAIGHTENED_IMAGE_NAME,
                        module.get_scale_name(stripe, segment),
                    )
                )
                value = m.get_current_measurement(OBJECTS_NAME, mname)
                assert len(value) == 1
                assert round(abs(value[0] - expected[segment, stripe]), 7) == 0


def test_flip_no_worms():
    workspace, module = make_workspace(
        numpy.zeros((5, 2, 0)), numpy.zeros(0), numpy.zeros(5), numpy.zeros((20, 10))
    )
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.wants_measurements.value = True
    module.number_of_segments.value = 5
    module.flip_worms.value = cellprofiler.modules.straightenworms.FLIP_TOP
    module.run(workspace)


def test_flip_dont_flip_top():
    r = numpy.random.RandomState()
    r.seed(0)
    image = r.uniform(size=(60, 30))
    image[25:] /= 5.0
    control_points = numpy.array(
        [[[21], [15]], [[23], [15]], [[25], [15]], [[27], [15]], [[29], [15]]]
    )
    lengths = numpy.array([8])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.wants_measurements.value = True
    module.number_of_segments.value = 3
    module.flip_worms.value = cellprofiler.modules.straightenworms.FLIP_TOP
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 11
    assert pixels.shape[0] == 19
    numpy.testing.assert_almost_equal(pixels, image[16:35, 10:21])


def test_flip_top():
    r = numpy.random.RandomState()
    r.seed(0)
    image = r.uniform(size=(60, 30))
    image[:25] /= 5.0
    control_points = numpy.array(
        [[[21], [15]], [[23], [15]], [[25], [15]], [[27], [15]], [[29], [15]]]
    )
    lengths = numpy.array([8])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.wants_measurements.value = True
    module.number_of_segments.value = 3
    module.flip_worms.value = cellprofiler.modules.straightenworms.FLIP_TOP
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 11
    assert pixels.shape[0] == 19
    i, j = numpy.mgrid[34:15:-1, 20:9:-1]
    numpy.testing.assert_almost_equal(pixels, image[i, j])


def test_flip_dont_flip_bottom():
    r = numpy.random.RandomState()
    r.seed(53)
    image = r.uniform(size=(60, 30))
    image[:25] /= 5.0
    control_points = numpy.array(
        [[[21], [15]], [[23], [15]], [[25], [15]], [[27], [15]], [[29], [15]]]
    )
    lengths = numpy.array([8])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.wants_measurements.value = True
    module.number_of_segments.value = 3
    module.flip_worms.value = cellprofiler.modules.straightenworms.FLIP_BOTTOM
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 11
    assert pixels.shape[0] == 19
    numpy.testing.assert_almost_equal(pixels, image[16:35, 10:21])


def test_flip_bottom():
    r = numpy.random.RandomState()
    r.seed(54)
    image = r.uniform(size=(60, 30))
    image[25:] /= 5.0
    control_points = numpy.array(
        [[[21], [15]], [[23], [15]], [[25], [15]], [[27], [15]], [[29], [15]]]
    )
    lengths = numpy.array([8])
    radii = numpy.array([1, 3, 5, 3, 1])
    workspace, module = make_workspace(control_points, lengths, radii, image)
    assert isinstance(module, cellprofiler.modules.straightenworms.StraightenWorms)
    module.width.value = 11
    module.wants_measurements.value = True
    module.number_of_segments.value = 3
    module.flip_worms.value = cellprofiler.modules.straightenworms.FLIP_BOTTOM
    module.run(workspace)
    image_set = workspace.image_set
    assert isinstance(image_set, cellprofiler_core.image.ImageSet)
    pixels = image_set.get_image(STRAIGHTENED_IMAGE_NAME).pixel_data
    assert pixels.shape[1] == 11
    assert pixels.shape[0] == 19
    i, j = numpy.mgrid[34:15:-1, 20:9:-1]
    numpy.testing.assert_almost_equal(pixels, image[i, j])
