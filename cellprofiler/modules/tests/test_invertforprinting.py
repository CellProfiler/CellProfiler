'''test_invertforprinting - Test the InvertForPrinting module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

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


class TestInvertForPrinting(unittest.TestCase):
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggH0sGpBQpGxgoGFlamllbGlgpGBgaWCiQDBkZPX34GBoZiJgaG'
                'ijlXI3zzbxlIlCkEWKlcuShntnJy0Z3U+Fupx/OWvp5g1hUVkunufOdMkNDi'
                'GoUbmftX3So8/nfO35abD1iOq5RKBS2YpvPa9P3vPed/HymTiWJoiM9leFWS'
                'cyGOsTvs8Ta1krT4hQoH1j293sA37+NP+y3xzvyHF/auWyirJH9Ed9b7XeYp'
                'BtLH7WuV0zIWy5W4pN+qWmjsMq++UJj10/zfFV1vtP8IN6Z039G7yPJ7xg6b'
                'eY9Pby6ME/KojN7y/278qwWeViGZ+dyfz/xn0fn7VeQuH8t/5nvsGVJ7rGYf'
                'CkywMJD4e6J5s/5Dg9+HDl/T3Nip/uHnggM3Jxd8szinavT8Y77oxyqlOPa5'
                '/X/zLu5ZfkQySfzpJcPV/4r5FcNX7NeRDbUKEbYIfbCvceeX/IaUf2zn/3fm'
                'zbT5M9/j45l3AodVGwV/ZZzcr9FfzH8u4o+4a010V3rZ9bkK+g5p7L+Tpc/d'
                'mNT1c0ro9Fc7MzhlePc8LlxV1bfkj/vqr0e83xpeKdZ4ah+ztl1354zj217f'
                'NOlf8zfarL/0/ONFu/adj3S/V1x6edeKcyc53Lfyz/OyOHNsXaf6DZ6FuQns'
                'UR91F0w1WlWyXuWD9gL3rocnzr8vZZsvzOp36/OWFzPf9S/oDfrw6dPk76fL'
                'V90y+xr09n7Gn+X3a+dnc31+ve7zz6RNNkJ9p4/fFNu/5fdroeMLojl75ZvC'
                'b/4p432+8Mv90LspdZGP132u09nBq/v7zuTz5xf9/qbdmBZe0fjsYCbQGfPq'
                'q3k5fn/pKOzac138++3zR15X30j9H8DxtfP95F1O/Vxbb539FfX95u9nqzNv'
                'ndrrIbq8eP5t8fP96ce79+w/+Hfe79PWfxZv0/z/39JNhxUAgHNw/Q==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, I.InvertForPrinting))
        self.assertEqual(module.input_color_choice.value, I.CC_GRAYSCALE)
        self.assertTrue(module.wants_red_input.value)
        self.assertEqual(module.red_input_image.value, "OrigRed")
        self.assertTrue(module.wants_green_input.value)
        self.assertEqual(module.green_input_image.value, "OrigGreen")
        self.assertTrue(module.wants_blue_input.value)
        self.assertEqual(module.blue_input_image.value, "OrigBlue")
        self.assertEqual(module.output_color_choice.value, I.CC_GRAYSCALE)
        self.assertTrue(module.wants_red_output.value)
        self.assertEqual(module.red_output_image.value, "InvertedDisplayRed")
        self.assertTrue(module.wants_green_output.value)
        self.assertEqual(module.green_output_image.value, "InvertedDisplayGreen")
        self.assertTrue(module.wants_blue_output.value)
        self.assertEqual(module.blue_output_image.value, "InvertedDisplayBlue")

    def test_01_02_load_v1(self):
        data = ('eJztWF1PGkEUHRCNVm00adI+zqO0QhBto6RBUPpBKpQI0Rij7QiDTLI7Q4Zd'
                'i21M+tif1cf+nP6EzsAsu0yRhUXTalyzWe/de+6dc2ZnuTuFbGUvuwNfxhOw'
                'kK3E6sTAsGQgq864mYLUWoW7HCML1yCjKXgormXchMl1mNhMbWylNl7BZCKx'
                'BYIdoXzhsbi0nwAwI66z4gyrW9PKDnlOaZexZRF63poGEfBM+X+K8wBxgs4M'
                'fIAMG7fcEo4/T+usctns3Sqwmm3gIjK9weIo2uYZ5q2PdQeobpdIGxtl8hVr'
                'FJywfXxBWoRRhVf5dW+vLrO0ulKHX7OuDiFNB6nLsscv498DNz4yQDdv/JKy'
                'Ca2RC1KzkQGJic57o5D5Nn3yTWv5pL3LDMYVPuODX9Lw8qzgthV700ZVC5rI'
                'qjZknoRPnnBfnjAosm59P1yoDxcC62A0XKQPFxH1KB6F7yONr7RzDFJmQbuF'
                'Xd1vi29Q3HV8/Z63pxpfaedwHdmGBfPyYYM5wnHVYvzyxsY/o+Gcw8HNgdF1'
                'vo63H26qDzcFjsSqCoLLFbMT6TLp+r2t52kUfcaZx4xPvXmNp7Tz9AJz8TO2'
                'j2u9PEH0qo6h100/F368F7XxLnp4v+MY0xHHrb8n126Zb9D364LGd8HDd0e0'
                'AUCN+7tPng9aHmmfrmyXXstGCKfjL6KfpHWIDWOffUkfZ2Olk6jjEQvINmn6'
                'OBHbOvm2tpq86gaXiUB2nFHgjGOS+fOu04ZPnk0tj7TlmI4w4mqgG1fRmHQV'
                'GLUaypdUvhy6dD2TrNOVMfuZm3iv30dcBgyf70F9Rqe5O+fMbj7wfeD7f9e/'
                '77gMGK7voO8iV18oPtlw8y6P+67y/9e63dX6QXHSeV2/oH9/yPjPPjyfazyl'
                'XRUtW5Mzuc/F42ZnM6YVNxiqdXdD4nvi37xnYyRovzWwDuk0c3XGm5xQuY0V'
                '77Z3bxkvKY+u39yAel4dwuJveWm47rre7jz83g5SLxL6u968Dy6ilJO4H2C8'
                'eV4ZEu9wCxr/B5P4OHs=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, I.InvertForPrinting))
        self.assertEqual(module.input_color_choice.value, I.CC_COLOR)
        self.assertEqual(module.color_input_image.value, "DNA")
        self.assertEqual(module.output_color_choice.value, I.CC_COLOR)
        self.assertEqual(module.color_output_image.value, "InvertedColor")

    def run_module(self, color_image=None,
                   red_image=None, green_image=None, blue_image=None,
                   fn=None):
        '''Run the InvertForPrinting module

        Call this with Numpy arrays for the images and optionally
        specify a function (fn) whose argument is an InvertForPrinting module.
        You can specialize the module inside this function.

        Returns a dictionary of the pixel data of the images in the image set
        '''
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = I.InvertForPrinting()
        module.module_num = 1
        for image, name, setting, check in \
                ((color_image, I_COLOR_IN, module.color_input_image, None),
                 (red_image, I_RED_IN, module.red_input_image, module.wants_red_input),
                 (green_image, I_GREEN_IN, module.green_input_image, module.wants_green_input),
                 (blue_image, I_BLUE_IN, module.blue_input_image, module.wants_blue_input)):
            if image is not None:
                img = cpi.Image(image)
                image_set.add(name, img)
                setting.value = name
                if check is not None:
                    check.value = True
            elif check is not None:
                check.value = False
        for name, setting in ((I_COLOR_OUT, module.color_output_image),
                              (I_RED_OUT, module.red_output_image),
                              (I_GREEN_OUT, module.green_output_image),
                              (I_BLUE_OUT, module.blue_output_image)):
            setting.value = name
        if fn is not None:
            fn(module)
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(),
                                  cpm.Measurements(),
                                  image_set_list)
        module.run(workspace)
        result = {}
        for provider in image_set.providers:
            result[provider.get_name()] = provider.provide_image(image_set).pixel_data
        return result

    def test_02_01_color_to_color(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            self.assertTrue(isinstance(module, I.InvertForPrinting))
            module.input_color_choice.value = I.CC_COLOR
            module.output_color_choice.value = I.CC_COLOR

        d = self.run_module(color_image=color_image, fn=fn)
        self.assertEqual(len(d), 2)
        self.assertTrue(I_COLOR_OUT in d.keys())
        result = d[I_COLOR_OUT]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            diff = (result[:, :, o] - ((1 - color_image[:, :, i1]) *
                                       (1 - color_image[:, :, i2])))
            self.assertTrue(np.all(np.abs(diff) <= np.finfo(float).eps))

    def test_02_02_color_to_bw(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            self.assertTrue(isinstance(module, I.InvertForPrinting))
            module.input_color_choice.value = I.CC_COLOR
            module.output_color_choice.value = I.CC_GRAYSCALE

        d = self.run_module(color_image=color_image, fn=fn)
        self.assertEqual(len(d), 4)
        self.assertTrue(all([color in d.keys()
                             for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)]))
        result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            diff = (result[o] - ((1 - color_image[:, :, i1]) *
                                 (1 - color_image[:, :, i2])))
            self.assertTrue(np.all(np.abs(diff) <= np.finfo(float).eps))

    def test_02_03_bw_to_color(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            self.assertTrue(isinstance(module, I.InvertForPrinting))
            module.input_color_choice.value = I.CC_GRAYSCALE
            module.output_color_choice.value = I.CC_COLOR

        d = self.run_module(red_image=color_image[:, :, 0],
                            green_image=color_image[:, :, 1],
                            blue_image=color_image[:, :, 2],
                            fn=fn)
        self.assertEqual(len(d), 4)
        self.assertTrue(I_COLOR_OUT in d.keys())
        result = d[I_COLOR_OUT]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            np.testing.assert_almost_equal(
                    result[:, :, o],
                    ((1 - color_image[:, :, i1]) * (1 - color_image[:, :, i2])))

    def test_02_04_bw_to_bw(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)

        def fn(module):
            self.assertTrue(isinstance(module, I.InvertForPrinting))
            module.input_color_choice.value = I.CC_GRAYSCALE
            module.output_color_choice.value = I.CC_GRAYSCALE

        d = self.run_module(red_image=color_image[:, :, 0],
                            green_image=color_image[:, :, 1],
                            blue_image=color_image[:, :, 2],
                            fn=fn)
        self.assertEqual(len(d), 6)
        self.assertTrue(all([color in d.keys()
                             for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)]))
        result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
        for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            np.testing.assert_almost_equal(
                    result[o],
                    ((1 - color_image[:, :, i1]) *
                     (1 - color_image[:, :, i2])))

    def test_03_01_missing_image(self):
        np.random.seed(0)
        color_image = np.random.uniform(size=(10, 20, 3)).astype(np.float32)
        for present in ((True, True, False),
                        (True, False, True),
                        (True, False, False),
                        (False, True, True),
                        (False, True, False),
                        (False, False, True)):
            def fn(module):
                self.assertTrue(isinstance(module, I.InvertForPrinting))
                module.input_color_choice.value = I.CC_GRAYSCALE
                module.output_color_choice.value = I.CC_GRAYSCALE

            d = self.run_module(red_image=color_image[:, :, 0] if present[0] else None,
                                green_image=color_image[:, :, 1] if present[1] else None,
                                blue_image=color_image[:, :, 2] if present[2] else None,
                                fn=fn)
            self.assertEqual(len(d), 3 + np.sum(present))
            self.assertTrue(all([color in d.keys()
                                 for color in (I_RED_OUT, I_GREEN_OUT, I_BLUE_OUT)]))
            result = [d[I_RED_OUT], d[I_GREEN_OUT], d[I_BLUE_OUT]]
            for o, i1, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                np.testing.assert_almost_equal(
                        result[o],
                        ((1 - color_image[:, :, i1] if present[i1] else 1) *
                         (1 - color_image[:, :, i2] if present[i2] else 1)))
