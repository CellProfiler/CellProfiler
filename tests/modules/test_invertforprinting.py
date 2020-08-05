import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement


import cellprofiler.modules.invertforprinting
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

I_RED_IN = "RedInput"
I_GREEN_IN = "GreenInput"
I_BLUE_IN = "BlueInput"
I_COLOR_IN = "ColorInput"
I_RED_OUT = "RedOutput"
I_GREEN_OUT = "GreenOutput"
I_BLUE_OUT = "BlueOutput"
I_COLOR_OUT = "ColorOutput"


def run_module(
    color_image=None, red_image=None, green_image=None, blue_image=None, fn=None
):
    """Run the InvertForPrinting module

    Call this with Numpy arrays for the images and optionally
    specify a function (fn) whose argument is an InvertForPrinting module.
    You can specialize the module inside this function.

    Returns a dictionary of the pixel data of the images in the image set
    """
    image_set_list = cellprofiler_core.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    module = cellprofiler.modules.invertforprinting.InvertForPrinting()
    module.set_module_num(1)
    for image, name, setting, check in (
        (color_image, I_COLOR_IN, module.color_input_image, None),
        (red_image, I_RED_IN, module.red_input_image, module.wants_red_input),
        (green_image, I_GREEN_IN, module.green_input_image, module.wants_green_input),
        (blue_image, I_BLUE_IN, module.blue_input_image, module.wants_blue_input),
    ):
        if image is not None:
            img = cellprofiler_core.image.Image(image)
            image_set.add(name, img)
            setting.value = name
            if check is not None:
                check.value = True
        elif check is not None:
            check.value = False
    for name, setting in (
        (I_COLOR_OUT, module.color_output_image),
        (I_RED_OUT, module.red_output_image),
        (I_GREEN_OUT, module.green_output_image),
        (I_BLUE_OUT, module.blue_output_image),
    ):
        setting.value = name
    if fn is not None:
        fn(module)
    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.RunException)

    pipeline.add_listener(callback)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        cellprofiler_core.measurement.Measurements(),
        image_set_list,
    )
    module.run(workspace)
    result = {}
    for provider in image_set.providers:
        result[provider.get_name()] = provider.provide_image(image_set).pixel_data
    return result


def test_color_to_color():
    numpy.random.seed(0)
    color_image = numpy.random.uniform(size=(10, 20, 3)).astype(numpy.float32)

    def fn(module):
        assert isinstance(
            module, cellprofiler.modules.invertforprinting.InvertForPrinting
        )
        module.input_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_COLOR
        )
        module.output_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_COLOR
        )

    d = run_module(color_image=color_image, fn=fn)
    assert len(d) == 2
    assert I_COLOR_OUT in list(d.keys())
    result = d[I_COLOR_OUT]
    for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        diff = result[:, :, o] - (
            (1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2])
        )
        assert numpy.all(numpy.abs(diff) <= numpy.finfo(float).eps)


def test_color_to_bw():
    numpy.random.seed(0)
    color_image = numpy.random.uniform(size=(10, 20, 3)).astype(numpy.float32)

    def fn(module):
        assert isinstance(
            module, cellprofiler.modules.invertforprinting.InvertForPrinting
        )
        module.input_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_COLOR
        )
        module.output_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_GRAYSCALE
        )

    d = run_module(color_image=color_image, fn=fn)
    assert len(d) == 4
    assert all(
        [color in list(d.keys()) for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)]
    )
    result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
    for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        diff = result[o] - ((1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2]))
        assert numpy.all(numpy.abs(diff) <= numpy.finfo(float).eps)


def test_bw_to_color():
    numpy.random.seed(0)
    color_image = numpy.random.uniform(size=(10, 20, 3)).astype(numpy.float32)

    def fn(module):
        assert isinstance(
            module, cellprofiler.modules.invertforprinting.InvertForPrinting
        )
        module.input_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_GRAYSCALE
        )
        module.output_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_COLOR
        )

    d = run_module(
        red_image=color_image[:, :, 0],
        green_image=color_image[:, :, 1],
        blue_image=color_image[:, :, 2],
        fn=fn,
    )
    assert len(d) == 4
    assert I_COLOR_OUT in list(d.keys())
    result = d[I_COLOR_OUT]
    for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        numpy.testing.assert_almost_equal(
            result[:, :, o], ((1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2]))
        )


def test_bw_to_bw():
    numpy.random.seed(0)
    color_image = numpy.random.uniform(size=(10, 20, 3)).astype(numpy.float32)

    def fn(module):
        assert isinstance(
            module, cellprofiler.modules.invertforprinting.InvertForPrinting
        )
        module.input_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_GRAYSCALE
        )
        module.output_color_choice.value = (
            cellprofiler.modules.invertforprinting.CC_GRAYSCALE
        )

    d = run_module(
        red_image=color_image[:, :, 0],
        green_image=color_image[:, :, 1],
        blue_image=color_image[:, :, 2],
        fn=fn,
    )
    assert len(d) == 6
    assert all(
        [color in list(d.keys()) for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)]
    )
    result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
    for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        numpy.testing.assert_almost_equal(
            result[o], ((1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2]))
        )


def test_missing_image():
    numpy.random.seed(0)
    color_image = numpy.random.uniform(size=(10, 20, 3)).astype(numpy.float32)
    for present in (
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
    ):

        def fn(module):
            assert isinstance(
                module, cellprofiler.modules.invertforprinting.InvertForPrinting
            )
            module.input_color_choice.value = (
                cellprofiler.modules.invertforprinting.CC_GRAYSCALE
            )
            module.output_color_choice.value = (
                cellprofiler.modules.invertforprinting.CC_GRAYSCALE
            )

        d = run_module(
            red_image=color_image[:, :, 0] if present[0] else None,
            green_image=color_image[:, :, 1] if present[1] else None,
            blue_image=color_image[:, :, 2] if present[2] else None,
            fn=fn,
        )
        assert len(d) == 3 + numpy.sum(present)
        assert all(
            [color in list(d.keys()) for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)]
        )
        result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            numpy.testing.assert_almost_equal(
                result[o],
                (
                    (1 - color_image[:, :, i1] if present[i1] else 1)
                    * (1 - color_image[:, :, i2] if present[i2] else 1)
                ),
            )
