import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler.modules.unmixcolors
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace
import tests.frontend.modules

INPUT_IMAGE = "inputimage"


def output_image_name(idx):
    return "outputimage%d" % idx


def test_load_v1():
    file = tests.frontend.modules.get_test_resources_directory("unmixcolors/v1.pipeline")
    with open(file, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(module, cellprofiler.modules.unmixcolors.UnmixColors)
    assert module.input_image_name == "Color"
    assert module.stain_count.value == 13
    assert module.outputs[0].image_name == "Hematoxylin"
    assert module.outputs[-1].image_name == "RedWine"
    for i, stain in enumerate(
        (
            cellprofiler.modules.unmixcolors.CHOICE_HEMATOXYLIN,
            cellprofiler.modules.unmixcolors.CHOICE_EOSIN,
            cellprofiler.modules.unmixcolors.CHOICE_DAB,
            cellprofiler.modules.unmixcolors.CHOICE_FAST_RED,
            cellprofiler.modules.unmixcolors.CHOICE_FAST_BLUE,
            cellprofiler.modules.unmixcolors.CHOICE_METHYL_GREEN,
            cellprofiler.modules.unmixcolors.CHOICE_AEC,
            cellprofiler.modules.unmixcolors.CHOICE_ANILINE_BLUE,
            cellprofiler.modules.unmixcolors.CHOICE_AZOCARMINE,
            cellprofiler.modules.unmixcolors.CHOICE_ALCIAN_BLUE,
            cellprofiler.modules.unmixcolors.CHOICE_PAS,
        )
    ):
        assert module.outputs[i].stain_choice == stain
    assert round(abs(module.outputs[-1].red_absorbance.value - 0.1), 7) == 0
    assert round(abs(module.outputs[-1].green_absorbance.value - 0.2), 7) == 0
    assert round(abs(module.outputs[-1].blue_absorbance.value - 0.3), 7) == 0


def make_workspace(pixels, choices):
    """Make a workspace for running UnmixColors

    pixels - input image
    choices - a list of choice strings for the images desired
    """
    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)

    module = cellprofiler.modules.unmixcolors.UnmixColors()
    module.input_image_name.value = INPUT_IMAGE
    module.outputs[0].image_name.value = output_image_name(0)
    module.outputs[0].stain_choice.value = choices[0]
    for i, choice in enumerate(choices[1:]):
        module.add_image()
        module.outputs[i + 1].image_name.value = output_image_name(i + 1)
        module.outputs[i + 1].stain_choice.value = choice

    module.set_module_num(1)
    pipeline.add_module(module)

    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image = cellprofiler_core.image.Image(pixels)
    image_set.add(INPUT_IMAGE, image)

    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    return workspace, module


def make_image(expected, absorbances):
    eps = 1.0 / 256.0 / 2.0
    absorbance = 1 - expected
    log_absorbance = numpy.log(absorbance + eps)
    absorbances = numpy.array(absorbances)
    absorbances = absorbances / numpy.sqrt(numpy.sum(absorbances ** 2))
    log_absorbance = (
        log_absorbance[:, :, numpy.newaxis]
        * absorbances[numpy.newaxis, numpy.newaxis, :]
    )
    image = numpy.exp(log_absorbance) - eps
    return image


def test_zeros():
    """Test on an image of all zeros"""
    workspace, module = make_workspace(
        numpy.zeros((10, 20, 3)), [cellprofiler.modules.unmixcolors.CHOICE_HEMATOXYLIN]
    )
    module.run(workspace)
    image = workspace.image_set.get_image(output_image_name(0))
    #
    # All zeros in brightfield should be all 1 in stain
    #
    numpy.testing.assert_almost_equal(image.pixel_data, 1, 2)


def test_ones():
    """Test on an image of all ones"""
    workspace, module = make_workspace(
        numpy.ones((10, 20, 3)), [cellprofiler.modules.unmixcolors.CHOICE_HEMATOXYLIN]
    )
    module.run(workspace)
    image = workspace.image_set.get_image(output_image_name(0))
    #
    # All ones in brightfield should be no stain
    #
    numpy.testing.assert_almost_equal(image.pixel_data, 0, 2)


def test_one_stain():
    """Test on a single stain"""

    numpy.random.seed(23)
    expected = numpy.random.uniform(size=(10, 20))
    image = make_image(expected, cellprofiler.modules.unmixcolors.ST_HEMATOXYLIN)
    workspace, module = make_workspace(
        image, [cellprofiler.modules.unmixcolors.CHOICE_HEMATOXYLIN]
    )
    module.run(workspace)
    image = workspace.image_set.get_image(output_image_name(0))
    numpy.testing.assert_almost_equal(image.pixel_data, expected, 2)


def test_two_stains():
    """Test on two stains mixed together"""
    numpy.random.seed(24)
    expected_1 = numpy.random.uniform(size=(10, 20)) * 0.5
    expected_2 = numpy.random.uniform(size=(10, 20)) * 0.5
    #
    # The absorbances should add in log space and multiply in
    # the image space
    #
    image = make_image(expected_1, cellprofiler.modules.unmixcolors.ST_HEMATOXYLIN)
    image *= make_image(expected_2, cellprofiler.modules.unmixcolors.ST_EOSIN)
    workspace, module = make_workspace(
        image,
        [
            cellprofiler.modules.unmixcolors.CHOICE_HEMATOXYLIN,
            cellprofiler.modules.unmixcolors.CHOICE_EOSIN,
        ],
    )
    module.run(workspace)
    image_1 = workspace.image_set.get_image(output_image_name(0))
    numpy.testing.assert_almost_equal(image_1.pixel_data, expected_1, 2)
    image_2 = workspace.image_set.get_image(output_image_name(1))
    numpy.testing.assert_almost_equal(image_2.pixel_data, expected_2, 2)


def test_custom_stain():
    """Test on a custom value for the stains"""
    numpy.random.seed(25)
    absorbance = numpy.random.uniform(size=3)
    expected = numpy.random.uniform(size=(10, 20))
    image = make_image(expected, absorbance)
    workspace, module = make_workspace(
        image, [cellprofiler.modules.unmixcolors.CHOICE_CUSTOM]
    )
    (
        module.outputs[0].red_absorbance.value,
        module.outputs[0].green_absorbance.value,
        module.outputs[0].blue_absorbance.value,
    ) = absorbance
    module.run(workspace)
    image = workspace.image_set.get_image(output_image_name(0))
    numpy.testing.assert_almost_equal(image.pixel_data, expected, 2)
