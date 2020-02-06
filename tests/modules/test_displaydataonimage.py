import centrosome.cpmorphology
import numpy
import six.moves

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.modules.displaydataonimage
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.workspace

INPUT_IMAGE_NAME = "inputimage"
OUTPUT_IMAGE_NAME = "outputimage"
OBJECTS_NAME = "objects"
MEASUREMENT_NAME = "measurement"


def test_load_v4():
    with open("./tests/resources/modules/displaydataonimage/v4.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.displaydataonimage.DisplayDataOnImage
    )
    assert module.objects_or_image == cellprofiler.modules.displaydataonimage.OI_OBJECTS
    assert module.measurement == "AreaShape_Zernike_0_0"
    assert module.image_name == "DNA"
    assert module.text_color == "green"
    assert module.objects_name == "Nuclei"
    assert module.display_image == "Zernike"
    assert module.font_size == 10
    assert module.decimals == 2
    assert module.saved_image_contents == cellprofiler.modules.displaydataonimage.E_AXES
    assert module.offset == 5
    assert module.color_or_text == cellprofiler.modules.displaydataonimage.CT_COLOR
    assert module.colormap == "jet"
    assert module.wants_image


def test_load_v5():
    with open("./tests/resources/modules/displaydataonimage/v5.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.displaydataonimage.DisplayDataOnImage
    )
    assert module.objects_or_image == cellprofiler.modules.displaydataonimage.OI_OBJECTS
    assert module.measurement == "AreaShape_Zernike_0_0"
    assert module.image_name == "DNA"
    assert module.text_color == "green"
    assert module.objects_name == "Nuclei"
    assert module.display_image == "Zernike"
    assert module.font_size == 10
    assert module.decimals == 2
    assert module.saved_image_contents == cellprofiler.modules.displaydataonimage.E_AXES
    assert module.offset == 5
    assert module.color_or_text == cellprofiler.modules.displaydataonimage.CT_COLOR
    assert module.colormap == "jet"
    assert not module.wants_image
    assert (
        module.color_map_scale_choice
        == cellprofiler.modules.displaydataonimage.CMS_USE_MEASUREMENT_RANGE
    )
    assert module.color_map_scale.min == 0
    assert module.color_map_scale.max == 1


def test_load_v6():
    with open("./tests/resources/modules/displaydataonimage/v6.pipeline", "r") as fd:
        data = fd.read()

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.LoadExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 2
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.displaydataonimage.DisplayDataOnImage
    )
    assert module.objects_or_image == cellprofiler.modules.displaydataonimage.OI_OBJECTS
    assert module.objects_name == "Nuclei"
    assert module.measurement == "Texture_AngularSecondMoment_CropBlue_3_0"
    assert module.image_name == "RGBImage"
    assert module.display_image == "Whatever"
    assert module.font_size == 11
    assert module.decimals == 3
    assert (
        module.saved_image_contents == cellprofiler.modules.displaydataonimage.E_IMAGE
    )
    assert module.offset == 1
    assert module.color_or_text == cellprofiler.modules.displaydataonimage.CT_COLOR
    assert module.colormap == "jet"
    assert module.wants_image
    assert (
        module.color_map_scale_choice
        == cellprofiler.modules.displaydataonimage.CMS_MANUAL
    )
    assert module.color_map_scale.min == 0.05
    assert module.color_map_scale.max == 1.5
    module = pipeline.modules()[1]
    assert (
        module.color_map_scale_choice
        == cellprofiler.modules.displaydataonimage.CMS_USE_MEASUREMENT_RANGE
    )


def make_workspace(measurement, labels=None, image=None):
    object_set = cellprofiler.object.ObjectSet()
    module = cellprofiler.modules.displaydataonimage.DisplayDataOnImage()
    module.set_module_num(1)
    module.image_name.value = INPUT_IMAGE_NAME
    module.display_image.value = OUTPUT_IMAGE_NAME
    module.objects_name.value = OBJECTS_NAME
    m = cellprofiler.measurement.Measurements()

    if labels is None:
        module.objects_or_image.value = cellprofiler.modules.displaydataonimage.OI_IMAGE
        m.add_image_measurement(MEASUREMENT_NAME, measurement)
        if image is None:
            image = numpy.zeros((50, 120))
    else:
        module.objects_or_image.value = (
            cellprofiler.modules.displaydataonimage.OI_OBJECTS
        )
        o = cellprofiler.object.Objects()
        o.segmented = labels
        object_set.add_objects(o, OBJECTS_NAME)
        m.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME, numpy.array(measurement))
        y, x = centrosome.cpmorphology.centers_of_labels(labels)
        m.add_measurement(OBJECTS_NAME, "Location_Center_X", x)
        m.add_measurement(OBJECTS_NAME, "Location_Center_Y", y)
        if image is None:
            image = numpy.zeros(labels.shape)
    module.measurement.value = MEASUREMENT_NAME

    pipeline = cellprofiler.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler.pipeline.RunExceptionEvent)

    pipeline.add_listener(callback)
    pipeline.add_module(module)
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    image_set.add(INPUT_IMAGE_NAME, cellprofiler.image.Image(image))

    workspace = cellprofiler.workspace.Workspace(
        pipeline, module, image_set, object_set, m, image_set_list
    )
    return workspace, module


def test_display_image():
    for display in (
        cellprofiler.modules.displaydataonimage.E_AXES,
        cellprofiler.modules.displaydataonimage.E_FIGURE,
        cellprofiler.modules.displaydataonimage.E_IMAGE,
    ):
        workspace, module = make_workspace(0)
        module.saved_image_contents.value = display
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_objects():
    labels = numpy.zeros((50, 120), int)
    labels[10:20, 20:27] = 1
    labels[30:35, 35:50] = 2
    labels[5:18, 44:100] = 3
    for display in (
        cellprofiler.modules.displaydataonimage.E_AXES,
        cellprofiler.modules.displaydataonimage.E_FIGURE,
        cellprofiler.modules.displaydataonimage.E_IMAGE,
    ):
        workspace, module = make_workspace([0, 1, 2], labels)
        module.saved_image_contents.value = display
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_no_objects():
    workspace, module = make_workspace([], numpy.zeros((50, 120)))
    module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_nan_objects():
    labels = numpy.zeros((50, 120), int)
    labels[10:20, 20:27] = 1
    labels[30:35, 35:50] = 2
    labels[5:18, 44:100] = 3
    for measurements in (
        numpy.array([1.0, numpy.nan, 5.0]),
        numpy.array([numpy.nan] * 3),
    ):
        workspace, module = make_workspace(measurements, labels)
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_objects_wrong_size():
    labels = numpy.zeros((50, 120), int)
    labels[10:20, 20:27] = 1
    labels[30:35, 35:50] = 2
    labels[5:18, 44:100] = 3
    input_image = numpy.random.uniform(size=(60, 110))
    for display in (
        cellprofiler.modules.displaydataonimage.E_AXES,
        cellprofiler.modules.displaydataonimage.E_FIGURE,
        cellprofiler.modules.displaydataonimage.E_IMAGE,
    ):
        workspace, module = make_workspace([0, 1, 2], labels, input_image)
        module.saved_image_contents.value = display
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_colors():
    labels = numpy.zeros((50, 120), int)
    labels[10:20, 20:27] = 1
    labels[30:35, 35:50] = 2
    labels[5:18, 44:100] = 3
    workspace, module = make_workspace([1.1, 2.2, 3.3], labels)
    assert isinstance(
        module, cellprofiler.modules.displaydataonimage.DisplayDataOnImage
    )
    module.color_or_text.value = cellprofiler.modules.displaydataonimage.CT_COLOR
    module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_colors_missing_measurement():
    #
    # Regression test of issue 1084
    #
    labels = numpy.zeros((50, 120), int)
    labels[10:20, 20:27] = 1
    labels[30:35, 35:50] = 2
    labels[5:18, 44:100] = 3
    workspace, module = make_workspace([1.1, 2.2], labels)
    assert isinstance(
        module, cellprofiler.modules.displaydataonimage.DisplayDataOnImage
    )
    module.color_or_text.value = cellprofiler.modules.displaydataonimage.CT_COLOR
    module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_colors_nan_measurement():
    #
    # Regression test of issue 1084
    #
    labels = numpy.zeros((50, 120), int)
    labels[10:20, 20:27] = 1
    labels[30:35, 35:50] = 2
    labels[5:18, 44:100] = 3
    workspace, module = make_workspace([1.1, numpy.nan, 2.2], labels)
    assert isinstance(
        module, cellprofiler.modules.displaydataonimage.DisplayDataOnImage
    )
    module.color_or_text.value = cellprofiler.modules.displaydataonimage.CT_COLOR
    module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)


def test_display_colors_manual():
    #
    # Just run the code path for manual color map scale
    #
    labels = numpy.zeros((50, 120), int)
    labels[10:20, 20:27] = 1
    labels[30:35, 35:50] = 2
    labels[5:18, 44:100] = 3
    workspace, module = make_workspace([1.1, 2.2, 3.3], labels)
    assert isinstance(
        module, cellprofiler.modules.displaydataonimage.DisplayDataOnImage
    )
    module.color_or_text.value = cellprofiler.modules.displaydataonimage.CT_COLOR
    module.color_map_scale_choice.value = (
        cellprofiler.modules.displaydataonimage.CMS_MANUAL
    )
    module.color_map_scale.min = 2.0
    module.color_map_scale.max = 3.0
    module.run(workspace)
    image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
