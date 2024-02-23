import io

import numpy
import numpy.testing
import skimage.color
import skimage.segmentation

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.module
import cellprofiler.modules.overlayoutlines
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"
OUTLINE_NAME = "outlineimage"
OBJECTS_NAME = "objectsname"


def make_workspace(image, labels=None, dimensions=2):
    """Make a workspace for testing Threshold"""
    m = cellprofiler_core.measurement.Measurements()
    object_set = cellprofiler_core.object.ObjectSet()
    module = cellprofiler.modules.overlayoutlines.OverlayOutlines()
    module.blank_image.value = False
    module.image_name.value = INPUT_IMAGE_NAME
    module.output_image_name.value = OUTPUT_IMAGE_NAME

    objects = cellprofiler_core.object.Objects()
    if len(labels) > 1:
        ijv = numpy.vstack(
            [numpy.column_stack(list(numpy.where(l > 0)) + [l[l > 0]]) for l in labels]
        )
        objects.set_ijv(ijv, shape=labels[0].shape)
    else:
        objects.segmented = labels[0]
    object_set.add_objects(objects, OBJECTS_NAME)
    module.outlines[0].objects_name.value = OBJECTS_NAME

    pipeline = cellprofiler_core.pipeline.Pipeline()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline, module, m, object_set, m, None
    )
    m.add(INPUT_IMAGE_NAME, cellprofiler_core.image.Image(image, dimensions=dimensions))
    return workspace, module


def test_load_v2():
    file = tests.frontend.modules.get_test_resources_directory("overlayoutlines/v2.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.overlayoutlines.OverlayOutlines)
    assert not module.blank_image
    assert module.image_name == "DNA"
    assert module.output_image_name == "PrimaryOverlay"
    assert module.wants_color == "Color"
    assert module.max_type == cellprofiler.modules.overlayoutlines.MAX_IMAGE
    assert module.line_mode.value == "Inner"
    assert len(module.outlines) == 2
    for outline, name, color in zip(
        module.outlines, ("PrimaryOutlines", "SecondaryOutlines"), ("Red", "Green")
    ):
        assert outline.objects_name.value == "None"
        assert outline.color == color


def test_load_v3():
    file = tests.frontend.modules.get_test_resources_directory("overlayoutlines/v3.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.load(io.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.overlayoutlines.OverlayOutlines)
    assert not module.blank_image
    assert module.image_name == "DNA"
    assert module.output_image_name == "PrimaryOverlay"
    assert module.wants_color == "Color"
    assert module.max_type == cellprofiler.modules.overlayoutlines.MAX_IMAGE
    assert module.line_mode.value == "Inner"
    assert len(module.outlines) == 2
    for outline, name, color, choice, objects_name in (
        (
            module.outlines[0],
            "PrimaryOutlines",
            "Red",
            cellprofiler.modules.overlayoutlines.FROM_IMAGES,
            "Nuclei",
        ),
        (
            module.outlines[1],
            "SecondaryOutlines",
            "Green",
            cellprofiler.modules.overlayoutlines.FROM_OBJECTS,
            "Cells",
        ),
    ):
        assert outline.color == color
        assert outline.objects_name == objects_name


def test_gray_to_color_outlines():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32)
    image[0, 0] = 1
    outline = numpy.zeros((50, 50), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = numpy.dstack((image, image, image))
    expected[:, :, 0][outline.astype(bool)] = 1
    expected[:, :, 1][outline.astype(bool)] = 0
    expected[:, :, 2][outline.astype(bool)] = 0
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
    module.outlines[0].color.value = "Red"
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_color_to_color_outlines():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(50, 50, 3)).astype(numpy.float32)
    image[0, 0] = 1
    outline = numpy.zeros((50, 50), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = image.copy()
    expected[:, :, 0][outline.astype(bool)] = 1
    expected[:, :, 1][outline.astype(bool)] = 0
    expected[:, :, 2][outline.astype(bool)] = 0
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
    module.outlines[0].color.value = "Red"
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_blank_to_color_outlines():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(50, 50, 3))
    image[0, 0] = 1
    outline = numpy.zeros((50, 50), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = numpy.zeros((50, 50, 3))
    expected[:, :, 0][outline.astype(bool)] = 1
    expected[:, :, 1][outline.astype(bool)] = 0
    expected[:, :, 2][outline.astype(bool)] = 0
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.blank_image.value = True
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
    module.outlines[0].color.value = "Red"
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_wrong_size_gray_to_color():
    """Regression test of img-961"""
    numpy.random.seed(24)
    image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32)
    image[0, 0] = 1
    outline = numpy.zeros((60, 40), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = numpy.dstack((image, image, image))
    sub_expected = expected[:50, :40]
    sub_expected[:, :, 0][outline[:50, :40].astype(bool)] = 1
    sub_expected[:, :, 1][outline[:50, :40].astype(bool)] = 0
    sub_expected[:, :, 2][outline[:50, :40].astype(bool)] = 0
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
    module.outlines[0].color.value = "Red"
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_wrong_size_color_to_color():
    numpy.random.seed(25)
    image = numpy.random.uniform(size=(50, 50, 3)).astype(numpy.float32)
    image[0, 0] = 1
    outline = numpy.zeros((60, 40), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = image.copy()
    sub_expected = expected[:50, :40]
    sub_expected[:, :, 0][outline[:50, :40].astype(bool)] = 1
    sub_expected[:, :, 1][outline[:50, :40].astype(bool)] = 0
    sub_expected[:, :, 2][outline[:50, :40].astype(bool)] = 0
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
    module.outlines[0].color.value = "Red"
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    assert numpy.all(output_image.pixel_data == expected)


def test_blank_to_gray():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(50, 50))
    outline = numpy.zeros((50, 50), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = numpy.zeros_like(image)
    expected[outline.astype(bool)] = 1
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.blank_image.value = True
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_gray_max_image():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32) * 0.5
    outline = numpy.zeros((50, 50), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = image.copy()
    expected[outline.astype(bool)] = numpy.max(image)
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.blank_image.value = False
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
    module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_IMAGE
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_almost_equal(output_image.pixel_data, expected)


def test_gray_max_possible():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32) * 0.5
    outline = numpy.zeros((50, 50), bool)
    outline[20:31, 20:31] = 1
    outline[21:30, 21:30] = 0
    expected = image.copy()
    expected[outline.astype(bool)] = 1
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.blank_image.value = False
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
    module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_POSSIBLE
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_almost_equal(output_image.pixel_data, expected)


def test_wrong_size_gray():
    """Regression test of IMG-961 - image and outline size differ"""
    numpy.random.seed(41)
    image = numpy.random.uniform(size=(50, 50)).astype(numpy.float32) * 0.5
    outline = numpy.zeros((60, 40), bool)
    outline[20:31, 20:31] = True
    outline[21:30, 21:30] = False
    expected = image.copy()
    expected[:50, :40][outline[:50, :40]] = 1
    workspace, module = make_workspace(image, labels=[outline.astype(int)])
    module.blank_image.value = False
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE
    module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_POSSIBLE
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_almost_equal(output_image.pixel_data, expected)


def test_ijv():
    numpy.random.seed(0)
    image = numpy.random.uniform(size=(50, 50, 3)).astype(numpy.float32)
    image[0, 0] = 1
    labels0 = numpy.zeros(image.shape[:2], int)
    labels0[20:30, 20:30] = 1
    labels1 = numpy.zeros(image.shape[:2], int)
    labels1[25:35, 25:35] = 2
    labels = [labels0, labels1]
    expected = image.copy()
    mask = numpy.zeros(image.shape[:2], bool)
    mask[20:30, 20] = True
    mask[20:30, 29] = True
    mask[20, 20:30] = True
    mask[29, 20:30] = True
    mask[25:35, 25] = True
    mask[25:35, 34] = True
    mask[25, 25:35] = True
    mask[34, 25:35] = True
    expected[mask, 0] = 1
    expected[mask, 1:] = 0
    workspace, module = make_workspace(image, labels=labels)
    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR
    module.outlines[0].color.value = "Red"
    module.line_mode.value = "Inner"
    module.run(workspace)
    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_color_outlines_on_blank_volume():
    image = numpy.zeros((9, 9, 9))

    labels = numpy.zeros_like(image)

    k, i, j = numpy.mgrid[-4:5, -4:5, -4:5]

    labels[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    workspace, module = make_workspace(image, labels=[labels.astype(int)], dimensions=3)

    module.blank_image.value = True

    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR

    module.outlines[0].color.value = "Red"

    module.run(workspace)

    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected = numpy.zeros(labels.shape + (3,))

    for index, plane in enumerate(labels):
        expected[index] = skimage.segmentation.mark_boundaries(
            image[index], plane, color=(1, 0, 0), mode="inner"
        )

    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_color_outlines_on_gray_volume():
    numpy.random.seed(0)

    image = numpy.random.uniform(size=(9, 9, 9)).astype(numpy.float32)

    labels = numpy.zeros_like(image)

    k, i, j = numpy.mgrid[-4:5, -4:5, -4:5]

    labels[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    workspace, module = make_workspace(image, labels=[labels.astype(int)], dimensions=3)

    module.blank_image.value = False

    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR

    module.outlines[0].color.value = "Red"

    module.run(workspace)

    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected = numpy.zeros(labels.shape + (3,))

    for index, plane in enumerate(labels):
        expected[index] = skimage.segmentation.mark_boundaries(
            image[index], plane, color=(1, 0, 0), mode="inner"
        )

    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_color_outlines_on_color_volume():
    numpy.random.seed(0)

    image = numpy.random.uniform(size=(9, 9, 9, 3)).astype(numpy.float32)

    labels = numpy.zeros((9, 9, 9))

    k, i, j = numpy.mgrid[-4:5, -4:5, -4:5]

    labels[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    workspace, module = make_workspace(image, labels=[labels.astype(int)], dimensions=3)

    module.blank_image.value = False

    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_COLOR

    module.outlines[0].color.value = "Red"

    module.run(workspace)

    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected = numpy.zeros_like(image)

    for index, plane in enumerate(labels):
        expected[index] = skimage.segmentation.mark_boundaries(
            image[index], plane, color=(1, 0, 0), mode="inner"
        )

    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_gray_outlines_on_blank_volume():
    image = numpy.zeros((9, 9, 9))

    labels = numpy.zeros_like(image)

    k, i, j = numpy.mgrid[-4:5, -4:5, -4:5]

    labels[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    workspace, module = make_workspace(image, labels=[labels.astype(int)], dimensions=3)

    module.blank_image.value = True

    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE

    module.run(workspace)

    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected = numpy.zeros(labels.shape + (3,))

    for index, plane in enumerate(labels):
        expected[index] = skimage.segmentation.mark_boundaries(
            image[index], plane, color=1.0, mode="inner"
        )

    expected = skimage.color.rgb2gray(expected)

    numpy.testing.assert_array_equal(output_image.pixel_data, expected)


def test_gray_outlines_max_possible_on_volume():
    numpy.random.seed(0)

    image = numpy.random.uniform(size=(9, 9, 9)).astype(numpy.float32)

    labels = numpy.zeros_like(image)

    k, i, j = numpy.mgrid[-4:5, -4:5, -4:5]

    labels[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    workspace, module = make_workspace(image, labels=[labels.astype(int)], dimensions=3)

    module.blank_image.value = False

    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE

    module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_POSSIBLE

    module.run(workspace)

    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected = numpy.zeros(labels.shape + (3,))

    for index, plane in enumerate(labels):
        expected[index] = skimage.segmentation.mark_boundaries(
            image[index], plane, color=1.0, mode="inner"
        )

    expected = skimage.color.rgb2gray(expected)

    numpy.testing.assert_array_almost_equal(output_image.pixel_data, expected)


def test_gray_outlines_image_max_on_volume():
    numpy.random.seed(0)

    image = numpy.random.uniform(size=(9, 9, 9)).astype(numpy.float32)

    image_max = numpy.max(image)

    labels = numpy.zeros_like(image)

    k, i, j = numpy.mgrid[-4:5, -4:5, -4:5]

    labels[k ** 2 + i ** 2 + j ** 2 <= 9] = 1

    workspace, module = make_workspace(image, labels=[labels.astype(int)], dimensions=3)

    module.blank_image.value = False

    module.wants_color.value = cellprofiler.modules.overlayoutlines.WANTS_GRAYSCALE

    module.max_type.value = cellprofiler.modules.overlayoutlines.MAX_IMAGE

    module.run(workspace)

    output_image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)

    expected = numpy.zeros(labels.shape + (3,))

    for index, plane in enumerate(labels):
        expected[index] = skimage.segmentation.mark_boundaries(
            image[index], plane, color=image_max, mode="inner"
        )

    expected = skimage.color.rgb2gray(expected)

    numpy.testing.assert_array_almost_equal(output_image.pixel_data, expected)
