"""test_correctilluminationapply.py
"""

import base64
import os
import sys
import tempfile
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.modules.correctilluminationapply as cpmcia
import cellprofiler.modules.injectimage as inj
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.object as cpo
import cellprofiler.measurement as cpm


class TestCorrectIlluminationApply(unittest.TestCase):
    def error_callback(self, calller, event):
        if isinstance(event, cpp.RunExceptionEvent):
            self.fail(event.error.message)

    def test_00_00_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwTy5RMLBUMDSwMjG3MjRQMDIA8kgGDIyevvwMDAweTAwM'
                'FXPeTj/vf9tA5PhcM9cAgZI3TOwBfbsv8L7d9HjZ2w1ea9ZqODnxKG4vNc72'
                'k7nJl2Xu+0je6ZHJ2x3/G+o5d6uoGmr+3Liuh9M77Muc5/3732+t3BbemPOa'
                'QefE8UVvVAsuazXmuuzaZO3B8ebWHiafDYI9/6f9FTDi2c0i6/RImU9NVyZ/'
                'B3OVwSe19XYX9/kHCFsFpt+cte+ya892todHHpRK8wu+fJDPcST6xbb+cwbH'
                'r/A+Y6+pLdmkJ3pr1+pN8Xp3nx288Clg2t/KlJif9Szb/6U53TNT+rf1gJ2D'
                '88bN/TnzHdIWLjwuEM6089Fer4dTk/cYFcQf2ti3XfPzNBPZU3mGOdv3O8g9'
                'M+ZOj3yssfvuqyv99t/3v/9nMbPg0iODJBnmK8br5oscrz4Vcc8ibWq/il+z'
                'wP2V3DYnDq7MmpN1opND69RPB93pXKf/B/tNmmeuOce113idfF24wQ1hfll/'
                'wXNdxwLLJkd9En7nqNT9aM/XlKc2L3/tL3m6dv36/x//nGk6fTT89MJ0izvX'
                'Ju+o/H5dmVP70Mxt+5/2GmfWb5SUWVnULFQR+XVq0p1JaxL5T88sWzdTbP4/'
                'G/P39tttu3vOv/rfJpuhPDvwvva5tZl3onqyE7f374ks+he25/O16dO/L/le'
                '+LpW5/wsw718L0vXvSjf4Z64f2HR0c3F10zORR14ecJ49Z6Z2rdmXPZ7mJMl'
                'k7r6t/irzo9fGXYHnNj95qX1eeXFHnMu/to/YdL0SVuurZ7q9PtN3sGvWmk/'
                'z9Wq5c6T/PHnzq4b+27tivwufuL77/07jtvnfP60n+3dXInPAHjXVAU=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cpmcia.CorrectIlluminationApply))
        self.assertEqual(len(module.images), 1)
        image = module.images[0]
        self.assertEqual(image.image_name, "OrigPhase")
        self.assertEqual(image.illum_correct_function_image_name, "IllumPhase")
        self.assertEqual(image.corrected_image_name, "CorrPhase")
        self.assertEqual(image.divide_or_subtract, "Subtract")

    def test_00_01_load_v1(self):
        data = ('eJztWd1O2zAUdkph/EwTu2LalS/pRqu2YxpUE9C1Q1SjpaIVE0KMmdallty4'
                'chJoNyHtco/IY+wRZpeEpCaQ/kARUlJF6Tn2d77j43Pc2C1mq7vZL/BjIgmL'
                '2Wq8QSiGZYrMBuOtDGwzg3RWYI5jZOI6ZHoGbnMC92omhOswlcx8+JRJrcJ0'
                'MrkORri0QvGVeOy8BWBGPGfFHbGbpm1Z89xSrmDTJPqZMQ2i4I2tvxL3AeIE'
                'nVJ8gKiFDZfC0Rf0Bqt22zdNRVa3KC6hlrezuEpW6xRzY6/hAO3mMulgWiG/'
                'sDIEp9s+PicGYbqNt+2r2hteZiq8lSa72ObCHcW+jM/VghsfTYmPjNeSR9+L'
                'J3D7R33i+drTf9GWiV4n56RuIQpJC53deCftJQPsTfXZmwL5UraH2wrALSp+'
                'yLuKO2b8aweJDGshs9aUdtYC7LxQ7Eg5xzh3/AjyX+vDa+CDPe4g3lnQzyvl'
                'AqVWSxAPMv55BS/lPIM6M6FlYDf+QX7MKHaknJeziQebv0gfPgJKDAw0fy8V'
                'XimXGOTYqCEqitTxPygflxQ7Us7jBrKoCQsyGWGecFwzGe/eGs+MYs+5HHtz'
                'wI3jU+IGqZtDUXWPyXfXPD9HvkngRqm7Xr6mJps3fuvcQ/Kp62PKB/eQfNE+'
                'vqjIGx2Pw/cngO8b6J9HKf9Y3ix/li9CeCPxPnYipe+Y0n12sXGUjZePY44m'
                'x8SCr28cJePrx79TK+nL684VIpA9Zcx33MP43wzwf03xX8rSh0OMuO3Y6mUs'
                'LlVFpptNW5e2dXnUdTUTrxf5ezlmvYz7Oz2pukn74MK6Cevm7/xw7/ej8mwF'
                'xMPvfbS3GTjjzGo/Pr/ffsDlh2KLgtvPefwhLsSFuOeH2/LgnmLdCnEhLszX'
                'EDcp3KJ29/uoeq4j+/8E9+fbO9Cfb1KuiS1AmzP5fwNPtHqH4kaCMlS/Pn1O'
                '7IqvBc9BtOTpBPDsKDw7d/HUGJfniUTuAYmOTML0E9Ru024id91S8LZkZYsa'
                'zzkffm9cIuKzNH3/PKjxd+fl3+YofFHtNt9CAC5qR7K3DwHDzfvyPf2dsY3T'
                'f9jxa0L4DxGFUhU=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cpmcia.CorrectIlluminationApply))
        self.assertEqual(len(module.images), 1)
        image = module.images[0]
        self.assertEqual(image.image_name, "DNA")
        self.assertEqual(image.illum_correct_function_image_name, "IllumDNA")
        self.assertEqual(image.corrected_image_name, "CorrDNA")

    def test_00_02_load_v2(self):
        data = ('eJztWu9OGkEQv0M0aptWkyb2435qpBWClCZqGgWhKqkgEWvTGGtXWGSTZZcs'
                'h0Ibkz5CH6eP0o99hD5Cd+FOjpV6xx8RmrvkArO7v/nNzuzM3e1dOn60H98G'
                'b0JhkI4fBYuYIJAl0CgyXt4A1FgBCY6ggQqA0Q2wwzE4yBsgvA5WwxvR6EYk'
                'DCLh8LrW36Gn0k/k7wtNmxE/s+L0mV3TpqzbTinnkGFgelGd1vzac7P9lziP'
                'IcfwnKBjSGqo2qaw2lO0yI4alZuuNCvUCMrAsn2wODK18jni1YOiBTS7s7iO'
                'SA5/RcoUrGGH6BJXMaMm3tSvtt7wMkPhzZXY1Q4X5ij6pX9+LLb9oyv+mRLn'
                'kq1djt/T2uP9Xfy5aBu/YMqYFvAlLtQgAbgML26sc6PvmaJPyklUhDVitJSB'
                'IiMFxC19aw76ZhR9Uubwancn28LHHPALCl6eR6huBN/VoVi8ZWjkS1JP2EGP'
                '3qFH10KaO/45hV/KCcb5LkeI2vzaK/9rl/HoFt9ECVKKyGpqf/9DOiRcYLPD'
                'KR6zij4ppwiplUVE3PhjXsFLOckAZQaoVdHw7OhnXSXlqkfu4uHrwPu0DBs4'
                'HpFu8XDy52NFn5QzDHBUzUMiiqOlx8muJUWPlK28TTXzNok5yhuMNwaLTzIT'
                'HyDvBfp2fGYUvHVY+DmbP/vF9TNfmefb4hLkZj1NdeCntE+i6o7aXnt8xs3e'
                '+8rXfu2NOfC5zctRr+P7mm/X65x5nQk+oN2D3l88ZHzc3AesDpFvmHb6O+z0'
                'i7VPb9XBUdj53cHO91pn/KX8eXkr+1Y++KDN0KvAmZQ+IkIO2dXmSTyYPQ1Y'
                'LQkmCibdPAkH10+/ra5ErluDc1ggm42Bge3vF1dymPeaMm8pS9s/IcjNCUWv'
                'A0HZlGbUKJltEbMtCRvtlgmrR5EJrEcPdt/Taz2KDJHPq0dePZqUevT7aW/7'
                'MuNWL7s9lzf3ay44q1XG1+5u+zttuwGmBVQZR7vv299O/OPqt0nBxbTh+NdJ'
                'j5eX/1deejgPNw7rfFzz08N5OA83OC5mw3n57+E83JD2b/Q2Tn3Ol7L9vZgc'
                '/0W7Ow9fap15KOU8IqTCmfz+hofKzY9EqiHCYKH1NUZoX/xN2T7McLO/ElV4'
                'onfxVDG9IKjJ1iTLNeUmpcVXd+DbU/j2/sWXZ1y+V8by3SOm0MCMnsFKhTRC'
                'iVZPyt4Tlz1q/Oa68Nvj4BPS0sz8nXFX491eB3+2+uHz675b+72PHHB+m02W'
                'n39qva235TvGW3Mc5fhe/abruvYXbnrGEg==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cpmcia.CorrectIlluminationApply))
        self.assertEqual(len(module.images), 2)
        image = module.images[0]
        self.assertEqual(image.image_name, "rawGFP")
        self.assertEqual(image.illum_correct_function_image_name, "IllumGFP")
        self.assertEqual(image.corrected_image_name, "CorrGreen")
        image = module.images[1]
        self.assertEqual(image.image_name, "rawDNA")
        self.assertEqual(image.illum_correct_function_image_name, "IllumDNA")
        self.assertEqual(image.corrected_image_name, "CorrBlue")

    def test_01_01_divide(self):
        """Test correction by division"""
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = image / illum
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.module_num = 3
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_DIVIDE
        image.rescale_option = cpmcia.RE_NONE
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline, None, None, None,
                                  measurements, image_set_list)
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))

    def test_01_02_subtract(self):
        """Test correction by subtraction"""
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = image - illum
        expected[expected < 0] = 0
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.module_num = 3
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_SUBTRACT
        image.rescale_option = cpmcia.RE_NONE
        measurements = cpm.Measurements()
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline, None, None, None,
                                  measurements, image_set_list)
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))

    def test_02_01_color_by_bw(self):
        '''Correct a color image with a black & white illumination fn'''
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10, 3)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = image - illum[:, :, np.newaxis]
        expected[expected < 0] = 0
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.module_num = 3
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_SUBTRACT
        image.rescale_option = cpmcia.RE_NONE
        measurements = cpm.Measurements()
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline, None, None, None,
                                  measurements, image_set_list)
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))

    def test_02_02_color_by_color(self):
        '''Correct a color image with a black & white illumination fn'''
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10, 3)).astype(np.float32)
        illum = np.random.uniform(size=(10, 10, 3)).astype(np.float32)
        expected = image - illum
        expected[expected < 0] = 0
        pipeline = cpp.Pipeline()
        pipeline.add_listener(self.error_callback)
        input_module = inj.InjectImage("InputImage", image)
        input_module.module_num = 1
        pipeline.add_module(input_module)
        illum_module = inj.InjectImage("IllumImage", illum)
        illum_module.module_num = 2
        pipeline.add_module(illum_module)
        module = cpmcia.CorrectIlluminationApply()
        module.module_num = 3
        pipeline.add_module(module)
        image = module.images[0]
        image.image_name.value = "InputImage"
        image.illum_correct_function_image_name.value = "IllumImage"
        image.corrected_image_name.value = "OutputImage"
        image.divide_or_subtract.value = cpmcia.DOS_SUBTRACT
        image.rescale_option = cpmcia.RE_NONE
        measurements = cpm.Measurements()
        image_set_list = cpi.ImageSetList()
        measurements = cpm.Measurements()
        workspace = cpw.Workspace(pipeline, None, None, None,
                                  measurements, image_set_list)
        pipeline.prepare_run(workspace)
        input_module.prepare_group(workspace, {}, [1])
        illum_module.prepare_group(workspace, {}, [1])
        module.prepare_group(workspace, {}, [1])
        image_set = image_set_list.get_image_set(0)
        object_set = cpo.ObjectSet()
        workspace = cpw.Workspace(pipeline,
                                  input_module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        input_module.run(workspace)
        illum_module.run(workspace)
        module.run(workspace)
        output_image = workspace.image_set.get_image("OutputImage")
        self.assertTrue(np.all(output_image.pixel_data == expected))
