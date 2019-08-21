"""test_invertforprinting - Test the InvertForPrinting module
"""

import base64
import unittest
import zlib
from six.moves import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.object as cpo
import cellprofiler.image as cpi
import cellprofiler.measurement as cpm
import cellprofiler.modules.invertforprinting as I

I_RED_IN = "RedInput"
I_GREEN_IN = "GreenInput"
I_BLUE_IN = "BlueInput"
I_COLOR_IN = "ColorInput"
I_RED_OUT = "RedOutput"
I_GREEN_OUT = "GreenOutput"
I_BLUE_OUT = "BlueOutput"
I_COLOR_OUT = "ColorOutput"


class TestInvertForPrinting:
    def run_module(
        self,
        color_image=None,
        red_image=None,
        green_image=None,
        blue_image=None,
        fn=None,
    ):
        """Run the InvertForPrinting module

        Call this with Numpy arrays for the images and optionally
        specify a function (fn) whose argument is an InvertForPrinting module.
        You can specialize the module inside this function.

        Returns a dictionary of the pixel data of the images in the image set
        """
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = I.InvertForPrinting()
        module.set_module_num(1)
        for image, name, setting, check in (
            (color_image, I_COLOR_IN, module.color_input_image, None),
            (red_image, I_RED_IN, module.red_input_image, module.wants_red_input),
            (
                green_image,
                I_GREEN_IN,
                module.green_input_image,
                module.wants_green_input,
            ),
            (blue_image, I_BLUE_IN, module.blue_input_image, module.wants_blue_input),
        ):
            if image is not None:
                img = cpi.Image(image)
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
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)

        def callback(caller, event):
            assert not isinstance(event, cpp.RunExceptionEvent)

        pipeline.add_listener(callback)
        workspace = cpw.Workspace(
            pipeline,
            module,
            image_set,
            cpo.ObjectSet(),
            cpm.Measurements(),
            image_set_list,
        )
        module.run(workspace)
        result = {}
        for provider in image_set.providers:
            result[provider.get_name()] = provider.provide_image(image_set).pixel_data
        return result

    def test_color_to_color(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            assert isinstance(module, I.InvertForPrinting)
            module.input_color_choice.value = I.CC_COLOR
            module.output_color_choice.value = I.CC_COLOR

        d = self.run_module(color_image=color_image, fn=fn)
        assert len(d) == 2
        assert I_COLOR_OUT in list(d.keys())
        result = d[I_COLOR_OUT]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            diff = result[:, :, o] - (
                (1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2])
            )
            assert np.all(np.abs(diff) <= np.finfo(float).eps)

    def test_color_to_bw(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            assert isinstance(module, I.InvertForPrinting)
            module.input_color_choice.value = I.CC_COLOR
            module.output_color_choice.value = I.CC_GRAYSCALE

        d = self.run_module(color_image=color_image, fn=fn)
        assert len(d) == 4
        assert all(
            [color in list(d.keys()) for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)]
        )
        result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            diff = result[o] - (
                (1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2])
            )
            assert np.all(np.abs(diff) <= np.finfo(float).eps)

    def test_bw_to_color(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            assert isinstance(module, I.InvertForPrinting)
            module.input_color_choice.value = I.CC_GRAYSCALE
            module.output_color_choice.value = I.CC_COLOR

        d = self.run_module(
            red_image=color_image[:, :, 0],
            green_image=color_image[:, :, 1],
            blue_image=color_image[:, :, 2],
            fn=fn,
        )
        assert len(d) == 4
        assert I_COLOR_OUT in list(d.keys())
        result = d[I_COLOR_OUT]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            np.testing.assert_almost_equal(
                result[:, :, o],
                ((1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2])),
            )

    def test_bw_to_bw(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            assert isinstance(module, I.InvertForPrinting)
            module.input_color_choice.value = I.CC_GRAYSCALE
            module.output_color_choice.value = I.CC_GRAYSCALE

        d = self.run_module(
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
            np.testing.assert_almost_equal(
                result[o], ((1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2]))
            )

    def test_missing_image(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)
        for present in (
            (True, True, False),
            (True, False, True),
            (True, False, False),
            (False, True, True),
            (False, True, False),
            (False, False, True),
        ):

            def fn(module):
                assert isinstance(module, I.InvertForPrinting)
                module.input_color_choice.value = I.CC_GRAYSCALE
                module.output_color_choice.value = I.CC_GRAYSCALE

            d = self.run_module(
                red_image=color_image[:, :, 0] if present[0] else None,
                green_image=color_image[:, :, 1] if present[1] else None,
                blue_image=color_image[:, :, 2] if present[2] else None,
                fn=fn,
            )
            assert len(d) == 3 + np.sum(present)
            assert all(
                [
                    color in list(d.keys())
                    for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)
                ]
            )
            result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
            for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                np.testing.assert_almost_equal(
                    result[o],
                    (
                        (1 - color_image[:, :, i1] if present[i1] else 1)
                        * (1 - color_image[:, :, i2] if present[i2] else 1)
                    ),
                )
