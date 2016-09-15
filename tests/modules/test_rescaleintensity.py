'''test_rescaleintensity.py - test the RescaleIntensity module
'''

import StringIO
import base64
import unittest
import zlib

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.rescaleintensity as R
from cellprofiler.modules.injectimage import InjectImage

INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
REFERENCE_NAME = 'reference'
MEASUREMENT_NAME = 'measurement'


class TestRescaleIntensity(unittest.TestCase):
    def test_01_0000_load_matlab_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:1234
FromMatlab:True

RescaleIntensity:[module_num:1|svn_version:\'8913\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    What did you call the image to be rescaled?:MyImage
    What do you want to call the rescaled image?:MyRescaledImage
    Rescaling method. (S) Stretch the image (0 to 1). (E) Enter the minimum and maximum values in the boxes below. (G) rescale so all pixels are equal to or Greater than one. (M) Match the maximum of one image to the maximum of another. (C) Convert to 8 bit. See the help for details.:Stretch 0 to 1
    Enter the intensity from the original image that should be set to the lowest value in the rescaled image, or type AA to calculate the lowest intensity automatically from all of the images to be analyzed and AE to calculate the lowest intensity from each image independently.:0.1
    Enter the intensity from the original image that should be set to the highest value in the rescaled image, or type AA to calculate the highest intensity automatically from all of the images to be analyzed and AE to calculate the highest intensity from each image independently.:AA
    What should the lowest intensity of the rescaled image be (range 0,1)?:0.2
    What should the highest intensity of the rescaled image be (range 0,1)?:0.9
    What did you call the image whose maximum you want the rescaled image to match?:MyOtherImage

RescaleIntensity:[module_num:2|svn_version:\'8913\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    What did you call the image to be rescaled?:MyImage
    What do you want to call the rescaled image?:MyRescaledImage
    Rescaling method. (S) Stretch the image (0 to 1). (E) Enter the minimum and maximum values in the boxes below. (G) rescale so all pixels are equal to or Greater than one. (M) Match the maximum of one image to the maximum of another. (C) Convert to 8 bit. See the help for details.:Enter min/max below
    Enter the intensity from the original image that should be set to the lowest value in the rescaled image, or type AA to calculate the lowest intensity automatically from all of the images to be analyzed and AE to calculate the lowest intensity from each image independently.:0.1
    Enter the intensity from the original image that should be set to the highest value in the rescaled image, or type AA to calculate the highest intensity automatically from all of the images to be analyzed and AE to calculate the highest intensity from each image independently.:AA
    What should the lowest intensity of the rescaled image be (range 0,1)?:0.2
    What should the highest intensity of the rescaled image be (range 0,1)?:0.9
    What did you call the image whose maximum you want the rescaled image to match?:MyOtherImage
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.image_name, "MyImage")
        self.assertEqual(module.rescaled_image_name, "MyRescaledImage")
        self.assertEqual(module.rescale_method, R.M_STRETCH)

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.image_name, "MyImage")
        self.assertEqual(module.rescaled_image_name, "MyRescaledImage")
        self.assertEqual(module.rescale_method, R.M_MANUAL_IO_RANGE)
        self.assertEqual(module.wants_automatic_low, R.CUSTOM_VALUE)
        self.assertEqual(module.wants_automatic_high, R.HIGH_ALL_IMAGES)
        self.assertAlmostEqual(module.dest_scale.min, 0.2)
        self.assertAlmostEqual(module.dest_scale.max, 0.9)

    def test_01_000_load_matlab_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:1234
FromMatlab:True

RescaleIntensity:[module_num:1|svn_version:\'8913\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    What did you call the image to be rescaled?:MyImage
    What do you want to call the rescaled image?:MyRescaledImage
    Rescaling method. (S) Stretch the image (0 to 1). (E) Enter the minimum and maximum values in the boxes below. (G) rescale so all pixels are equal to or Greater than one. (M) Match the maximum of one image to the maximum of another. (C) Convert to 8 bit. See the help for details.:Enter min/max below
    Enter the intensity from the original image that should be set to the lowest value in the rescaled image, or type AA to calculate the lowest intensity automatically from all of the images to be analyzed and AE to calculate the lowest intensity from each image independently.:0.1
    Enter the intensity from the original image that should be set to the highest value in the rescaled image, or type AA to calculate the highest intensity automatically from all of the images to be analyzed and AE to calculate the highest intensity from each image independently.:AA
    What should the lowest intensity of the rescaled image be (range 0,1)?:0.2
    What should the highest intensity of the rescaled image be (range 0,1)?:0.9
    What value should pixels *below* the low end of the original intensity range be mapped to (range 0,1)?:0.01
    What value should pixels *above* the high end of the original intensity range be mapped to (range 0,1)?:0.99
    What did you call the image whose maximum you want the rescaled image to match?:MyOtherImage
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.image_name, "MyImage")
        self.assertEqual(module.rescaled_image_name, "MyRescaledImage")
        self.assertEqual(module.rescale_method, R.M_MANUAL_IO_RANGE)
        self.assertEqual(module.wants_automatic_low, R.CUSTOM_VALUE)
        self.assertEqual(module.wants_automatic_high, R.HIGH_ALL_IMAGES)
        self.assertAlmostEqual(module.dest_scale.min, 0.2)
        self.assertAlmostEqual(module.dest_scale.max, 0.9)
        self.assertAlmostEqual(module.custom_low_truncation.value, 0.01)
        self.assertAlmostEqual(module.custom_high_truncation.value, 0.99)

    def test_01_01_load_matlab_stretch(self):
        '''Load a pipeline with RescaleIntensity set up to stretch'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDQAIitjUysTEwUjAwNLBZIBA6OnLz8DA0MwIwND'
                'xZy9ISf9DhsIzHXJjOxaJOP4aILy8W/8GtvlNFY4ioZvn8nmfXOKRUfRo8s/'
                'TB4d85Nd/UroVp9JUdqfe7MrbDbxMkwXZ4jdNy3PKVVWULrr8w3nYxkyT2Q3'
                'PAh7Uu89f1LPpMiOYCGxGc65RvZd4idvG63dPk94P0ehj5JozK0dHWkznupz'
                'e4rcXv+F27m/TtnE95DaD8/iWUea9L98zYk1T4yKnT3/cMz3gzcutM+b9PyH'
                'e8Tjtbw/pFnfd1Q0/ZzT8sJ+aq7V9r61Lz/cSJbc9LP9dnnkI75VMfXtfTcv'
                'GCZa2Oiez/jQI7qPN//CpPJTVZeuVDfWf6rreXzu49aw9uMmSoxskUFLfr+2'
                'mNV3+UJh5pvkLO3/pS+q1hXHZFrZfONU7jrjpd769vuCc963/3WWJ4gY9xSf'
                'mvbrTkn5FdPT6+Vk8vd9X5m+9UGk6ud3+x7tlb9xvV23ItPoP3veqZxrAHZa'
                'yfc=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.image_name.value, "RescaledBlue")
        self.assertEqual(module.rescaled_image_name.value, "RescaledBlue")
        self.assertEqual(module.rescale_method, R.M_STRETCH)

    def test_01_02_load_matlab_enter_auto_low(self):
        '''Load a pipeline with automatic low manual scaling'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDQAIisTAytTcwUjAwNLBZIBA6OnLz8DA0MLIwND'
                'xZy9U7z9DzkIHF/ue5C9zeTjos52OVvFCQIWXZ2dMi7MbRWxb5zMXmrXalR8'
                'WvzT8I9FXuL2HHPeWo85KZfOvfnz7/K5vNssDI85FvitP34l9EZU79GuB4U2'
                'RxM95qQGMPPqRfyb/YvnJ8tZXRvnsBZmj57n7ucO1UZWn7y0ONtNTE45WmeZ'
                'Wd8by2/Ny1nZb34yjkzM9rIzXNI73feDNWfHa4N5L0/fXLJe2qt29pX6K3dl'
                'pvRYmrTUxNRz767SXfzPZ8X1SJuDB6U3bah0P16SMbddZm6xp3rf+7kCv9t8'
                '50hLr+2x3Han/8nfW5ZVWV7+rQ83SF1/Jfp4i8hyx91X34Y+/iWdN+Hw1SWV'
                'Rwt3X3357UbNqo/Nds+/25qbT2y3L4o/lXj++8or6n2dvdMqXu6eGB7jPr3k'
                'y5Q+Rxnhat+HO0/F/X9t66cy18IssINlZm8f++/vjo1FvyeY2n+rl7h4/jan'
                'x45p531tl6/3ei8V8Ga3iOXce6VGO5T+buGa43by9Grd71FF665Xv9r58Y++'
                '17uiSgA7HuOs')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.rescale_method, R.M_MANUAL_IO_RANGE)
        self.assertEqual(module.wants_automatic_low.value, R.LOW_ALL_IMAGES)
        self.assertEqual(module.wants_automatic_high.value, R.CUSTOM_VALUE)
        self.assertAlmostEqual(module.source_high.value, .5)
        self.assertAlmostEqual(module.dest_scale.min, .25)
        self.assertAlmostEqual(module.dest_scale.max, .75)
        self.assertAlmostEqual(module.custom_low_truncation.value, 0.125)
        self.assertAlmostEqual(module.custom_high_truncation.value, 0.875)
        self.assertEqual(module.low_truncation_choice.value, R.R_SET_TO_CUSTOM)
        self.assertEqual(module.high_truncation_choice.value, R.R_SET_TO_CUSTOM)

    def test_01_03_load_matlab_enter_auto_high(self):
        '''Load a pipeline with automatic high manual scaling'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDQAIitTQytTCwUjAwNLBZIBA6OnLz8DA0MLIwND'
                'xZy9U7z9DzkIHF/ue5C9zeTjos52OVvFCQIWXZ2dMi7MbRWxb5zMXmrXalR8'
                'WvzT8I9FXuL2HHPeWo85KZfOvfnz7/K5vNssDI85FvitP34l9EZU79GuB4U2'
                'RxM95qQGMPPqRfyb/YvnJ8tZXRvnsBZmj57n7ucO1UZWn7y0ONtNTE45WmeZ'
                'Wd8by2/Ny1nZb34yjkzM9rIzXNI73feDNWfHa4N5L0/fXLJe2qt29pX6K3dl'
                'pvRYmrTUxNRz767SXfzPZ8X1SJuDB6U3bah0P16SMbddZm6xp3rf+7kCv9t8'
                '50hLr+2x3Han/8nfW5ZVWV7+rWHrpcxlxB5vEVnuuPvq29DHv6TzJhy+uqTy'
                'aOHuqy+/3ahZ9bHZ7vl3W3Pzie32RfGnEs9/X3lFva+zd1rFy90Tw2Pcp5d8'
                'mdLnKCNc7ftw56m4/69t/VTmWpgFdrDM7O1j//3dsbHo9wRT+2/1EhfP3+b0'
                '2DHtvK/t8vVe76UC3uwWsZx7r9Roh9LfLVxz3E6eXq37Papo3fXqVzs//tF3'
                'USuqBAART+Do')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.wants_automatic_low.value, R.CUSTOM_VALUE)
        self.assertEqual(module.wants_automatic_high.value, R.HIGH_ALL_IMAGES)
        self.assertAlmostEqual(module.source_low.value, .5)

    def test_01_04_load_matlab_enter_manual(self):
        '''Load a pipeline with manual low and high scaling'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDQAIitTYytTAwUjAwNLBZIBA6OnLz8DA0MLIwND'
                'xZy9U856HXYQOK6uuFZ+qtCHRQ8U3f236k65NpujyTZi6dRHpdnbUhVvttSs'
                '+Ppk8U8Ju77ZZeZ3dRZ6JhcmeKSV1f33TDPPZGooZQhIv+78zGrN2mPJvp9U'
                '5j3QUCoRvHagwXje/zQ7V0ZHY3b/icKPBYKEfyrOrT7vbR9b9eHNjYPKT04d'
                'PeO20fqw4w8LcYX4OiX5GzMn+PNECqcm/3FzehR4rrjWdPa3r0pL8yf/7N83'
                's3macXvbFP9X+QnB+U+uzPf4qtZ/3KLGlW9b/s0Z39YWxy8oXvVC5aNsb0HC'
                'zpa4nS8Oz0o8Xu1/f33sguXpX0oySrWKrj+1UD4seSGvtPrqjn9d0iIFpqlx'
                'nnfySnfZ2s3P+rNA/0fueV2fZw/770zfdv9z7ZovnkmPFrHGscS9mX7dUFTW'
                'eHGyYFbyfk3b6+uu/6/qV3ZV78wVUfRMW/r48LfaCwnba0/Mlv9vz3Pzua+z'
                'kO7V4mD9F6s372ab8lsv9VxK+VyZdT72i5nSDT79STq7943F9dO6Xw+5/BPf'
                'nF4UBQBPz9+M')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.wants_automatic_low.value, R.CUSTOM_VALUE)
        self.assertEqual(module.wants_automatic_high.value, R.CUSTOM_VALUE)
        self.assertAlmostEqual(module.source_scale.min, .1)
        self.assertAlmostEqual(module.source_scale.max, .9)

    def test_01_05_load_matlab_greater_than_one(self):
        '''Load a pipeline, dividing by the minimum'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDQAIitTMysTSwUjAwNLBZIBA6OnLz8DA0M1IwND'
                'xZy9U876HXIQOL7c9yA/z8OOTva+eRwTBD6k7ZHgW8TovEev7IDvqW2Z291s'
                'fmbYNfU/0XtzaWnmAtfDKpPf+Tz/Pv9tyW0thiv+DKceVWndads6NSRHUu+x'
                'YZHfTRutRubSqf/F7BQVVYP51Y90Gzb1vvzTcPbP89QNO78+2L2x3XmTQXcr'
                '7+OjIodfFC5zLP3dWbfAfNl8jtvLTrv98FN+ZXxU+29J2eXrz1fuC31oX2bE'
                'rp3MJ6qs/zpfcWW50Sv9nl+zutwl9grx37U2XJb8+kqQ1aX3zioyh9nqldc8'
                'NJ5itdbW6tu0Pt6z33bVJ75Tk3+eU8i3Zdbcd4kx+Ue/fi9YemVOyp2sbdax'
                'ps8P/6jz/7z729qzTx7235n+df/zXzZX3Y+VOD6z4P9rcWL95FI54blsqXps'
                'ocnxruuPV/VtntLXE3uoU3HzJc6E79UbFtyoPjFr/i99Hgl3scB9PvNnf5q3'
                'b1tEqML98swi041Z6QWVvyzZC/oU9Ndfu+N9cGfc+xXXl33er5ZeqAwAZ7jf'
                'Pg==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.rescale_method.value,
                         R.M_DIVIDE_BY_IMAGE_MINIMUM)

    def test_01_06_load_matlab_match(self):
        '''Load a pipeline, matching to another image'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDRQMDS0MjCwMjRTMDIwsFQgGTAwevryMzAwTGFk'
                'YKiYs3eKt/8hB4Hjy30PsntVXTnC7O7P4sFRoXzPgoGN5+OuCwvzLoX/Mnhw'
                'sv/Z8Q+Pd2jvvpAndauoYJKv9eWff9OtN+cKM0SJM5SW7U3xbt97tWX9UrHv'
                '4hLHr2xeLCVRoP7ofo6up8fJUwdVC4V6HiQY2X0z+/daz8Jm1o+wRw+m7HiZ'
                'knb5SGyixFkzIZH++Bdic1n/upyY5TP7MXvviveccS9WX3SOtQ97ZXbz18X+'
                '2gtFvy54HK38J9D3082vLuq4yD+dJd+MswWfu7P+6T1UXN93cZZFRiHvzn6l'
                'udGu7b+FhdbfjxbetDTvEddcv8a+eRcvLrxrO7la67ik5LkGv93X7fulv5w+'
                's3rZl87fc8u2X28v+nzueTvzyX6uH3sszxY2K1ZlizDrV9VnZsoK3Jke9rZG'
                '/O33qJWHa18JSb+94N/0yWrZouS0kqgrmw0yln0L/for/n3h0hcF95UMPU/+'
                'eclSUMUnFTLbOe7/kfqkN0dfuJrcW/l8WVxAeHo5r8b3ucssJ98tLd5h9LXH'
                '2KJ54nf+6ndrk+Ll/Z94T/4YH9ldfBAAEkrtMg==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.image_name.value, "OrigBlue")
        self.assertEqual(module.rescaled_image_name.value, "RescaledBlue")
        self.assertEqual(module.rescale_method, R.M_SCALE_BY_IMAGE_MAXIMUM)
        self.assertEqual(module.matching_image_name, "ReferenceBlue")

    def test_01_07_load_matlab_convert(self):
        '''Load a matlab pipeline with convert to 8-bit'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDRQMDS0MjC2MjJQMDIwsFQgGTAwevryMzAw1DAy'
                'MFTM2TvlfPYhA5Hj8d6l9jKPLh7ks32notXxzmzb5PadC53LerOnOPh6zT51'
                'J7P+1r+Wv57eJhclurbl/Xhwcc/17z/3fTHNFWV4lc6QWmgr/u7QLS+tMp7p'
                'HzmK/V7oaDayl079L2bnZug4WaA/cMYjFrNln2utUtefzih+3cma2cGekSfG'
                'Lb3kM2ff5Ocr2a6+Ff/UeFjw55EN2btyOsRVFHu04l7evukbf2ztvtCH9u/K'
                '+PyOyWi5xV27f3jbvdMr9hu9ytaSe/REttWiQiDZaqvf2UsT/514wXT8zvMz'
                '738KH1rq97mhb/fix06Lt76Ln7H79PFK6znsZ912F8tcjy3cyxxzctbkntm9'
                'YbsXP6uRuf86q+7+vq0zbf5M/F3073tMfFRtis+zjLZ0Q+/Ff6winxeufiws'
                '9WBF1pvVL76GnC9MfHnCwkFG0kh7Lse3/ezsXXvb1/xc+d9pfpGP8KHp33Zq'
                'yr943RXK+ql2ulq48OowtunL55ezPOf89Cvt7F7ROa/fzqvKtfrPHSxacBwA'
                '5zLjFA==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.rescale_method, R.M_CONVERT_TO_8_BIT)

    def test_01_08_load_matlab_measurement(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylTwKs1RMDRQMDS0MjCzMjBUMDIwsFQgGTAwevryMzAwLGJk'
                'YKiYs3eKb/YhB4E2dbNb/N0VmlsuOL9YuXxBU0a11lJ224zuri9h19KmtM5o'
                'SRF9u9J+rf3a3WWLT5rNkPBiN5s988xnP6PJS5gY5iY3CN33/6F1oF906XrX'
                'aR/aT0iUpjkKluyY8lF/37Ejy1OSHs+4wc2TkJ2457XP3Yu3bse92nnt0DG7'
                'me6Td34V/8z56+BnE5anT8V3OVV72As8ElvGW+DJ/Mpq0vRfWSXR11O97nZZ'
                'yLz9tUD1y0KZzu4/7Deevub8+Xzin1Vf2CX2CEuvdb+3bP+1St+LOwNLWuwP'
                'ie42n/diUlSGz7W2t0Lcus0fN2wU2WxSfj7a+6L+kS/cCdeKi8pvWtgeexlg'
                'LH22Jib78NbgHYsu2T+8fqZqsebvyf35253q7te2iz11LJo7ZZc4l/ofOaMk'
                'Qbd7u4qNnA3/H/i69N7yb9vvH5wZFP8jZMrcn+8OnV4XHFqyylC22UqsauqJ'
                'Wtld/68Xz9GyOiS+ueNETonNDXt2Xr7ZzkH/n9QveW1QwnTq2qvqCK+V0Q/S'
                'U0982+u2U8t+tXpB5KuXlhZzJ4df3/9/cuSvqt3q9lsM/vH2PK4vBwDa3/W+')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.rescale_method, R.M_DIVIDE_BY_MEASUREMENT)
        self.assertEqual(module.divisor_measurement, "Metadata_ImageDivisor")

    def test_01_09_load_v1(self):
        data = ('eJztWl9P2zAQd2lhMKSNaRKbtBc/7AE2GqXlj6CaoB3dtG6UVYA2IcSYaV3w'
                '5MRV4rB2ExIfa498pH2ExW3SpKYlbSm0aIlkNXfx737ns++SOM1n9rYyb+Gy'
                'osJ8Zi9eJhTDAkW8zAwtBXW+ADcNjDguQaan4HuDwI8WhQkVJpKpxZWUugyT'
                'qroG+jsiufwj8fsSgAn7Z9JuY86lcUeO+JqQdzHnRD8xx0EMPHf0l3b7ggyC'
                'jin+gqiFTY/C1ef0MturVZqX8qxkUbyNNH9n+9i2tGNsmJ/LLtC5XCBVTHfJ'
                'LywNwe22g8+ISZju4B37srbJy7jEK+JwOePFISLFIWq3WZ9e9P8AvP6xNnF7'
                '4us/48hEL5EzUrIQhURDJ00vhD01wF60xV4UZLczN8KlA3Azkv+i7eEqj7+r'
                'oiKHGuLF0274x1rsjIFtBrrin5b4hbyDzSKiuCTG0G3cIi12ImCxS1ysBRez'
                '/daxwJ0G4JYkv4W8ecqYiaFZwUVSJkV4Vs8TyBk8xtDAJubiHMGiZXKmQQPp'
                'J9gdX1CcHkp8Qs4yqDMOLdNJGHm8E5Id93DtTIH2uF7mdxC4u/AzKI+fgdb4'
                'CjmLy8iiHOZEEsMsMXCRM6N2q/7LeayoiYHxDdJPOW9UZW2tn/Ht29XxJn6u'
                'BvA9AK3zKmRVSSzY/vY1H8PxN2n7uzqUfAvKm6eSv0LebRQ6p8zVq2DT3uiv'
                'Y3V5UPfpUYlzN/fLRBvcLdeLruLc6f7cr5/pAL4p0BpnIW/WOKtQZGo+OxcB'
                'dj5JdoT8bW6j8EY8+ON15fX8kZC+Ykp32M/1g0y8cDjvajYZtTR9/UCNrx3+'
                'Tiwkzxudd4mNrCvn28Z9kPMlr4+kgwt6LlqVxi1k4fs+RoYzoKXz+bhQ5ZnO'
                'Tx1d0tFlUc3T3GR8hQA/X0h+Cjmnc6ybhNeO8hjpntTpOfSu69cw1vmo1etO'
                '63LU/Bx03eoXdxHg532qU73g7kud6hc3+bi3fYzbqiPt3kvrmx4nBrMqg7cz'
                'anW03f6J5zckeglXbpO/3f4JO/5hv6l6gRt2fQxxIS7Ehbj/EZf24YZx/whx'
                'dzufo/ocE+JGA5cG16+fsB6EuBDXO64a6fw+LGT/fr3o/x1cn4evQGseCrmI'
                'Ka0YTPx/w1C0+p8MTIUyVGp85Ve27NOc74O/4KkE8KQlnnQnHg0j0zJwnYq4'
                'W5NKvqGtszY3LLvZ/1iReFc68RqNj/EepfN1vsl2dd6m2vD54z9mS7PRmWvn'
                'W55nb/7/bvTDF41Ernx/mw7AxXw+iUPg/4De1tncNf3dMd5V/38aZ6rY')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        #
        # image_name = DNA
        # rescaled_image_name = RescaledDNA
        # rescaling_method = M_MANUAL_IO_RANGE
        # manual intensities
        # source_low = .01
        # source_high = .99
        # src_scale = .1 to .9
        # dest_scale = .2 to .8
        # low_truncation_choice = R_SET_TO_CUSTOM
        # custom_low_truncation = .05
        # high_truncation_choice = R_SET_TO_CUSTOM
        # custom_high_truncation = .95
        # matching_image_name = Cytoplasm
        # divisor_value = 2
        # divisor_measurement = Intensity_MeanIntensity_DNA
        self.assertEqual(module.image_name.value, "DNA")
        self.assertEqual(module.rescaled_image_name.value, "RescaledDNA")
        self.assertEqual(module.rescale_method.value, R.M_MANUAL_IO_RANGE)
        self.assertEqual(module.wants_automatic_high.value, R.CUSTOM_VALUE)
        self.assertEqual(module.wants_automatic_low.value, R.CUSTOM_VALUE)
        self.assertAlmostEqual(module.source_low.value, .01)
        self.assertAlmostEqual(module.source_high.value, .99)
        self.assertAlmostEqual(module.source_scale.min, .1)
        self.assertAlmostEqual(module.source_scale.max, .9)
        self.assertEqual(module.low_truncation_choice.value, R.R_SET_TO_CUSTOM)
        self.assertEqual(module.high_truncation_choice.value, R.R_SET_TO_CUSTOM)
        self.assertAlmostEqual(module.custom_low_truncation.value, .05)
        self.assertAlmostEqual(module.custom_high_truncation.value, .95)
        self.assertEqual(module.matching_image_name.value, "Cytoplasm")
        self.assertAlmostEqual(module.divisor_value.value, 2)
        self.assertEqual(module.divisor_measurement.value, "Intensity_MeanIntensity_DNA")

    def test_01_10_img_787(self):
        data = ('eJyVk/s7E4oDxkeJU5xJhJPLcr91rEMuyxR6ODqZ+yS5Lmm5bC5z35Rrclml'
                'EqYdB5PIDuUyxhzRYpg7oWJhmNvcRmM7Pef7F3zfH97f3ud5n8/zvg5Wbtes'
                'rEEXfgWDHKzczt1GhgQagJxC/KNuoyJCISAnGw97mAHIJiLQPyrwFggVBgG5'
                'YQJBDv4RILA5CHwech4MAZuAfgOfB4P+bwFE7B2AAABgWRwAiCXMeXo5ToFl'
                'o0vAE2K13N+44Hnxd6WjxiaryGxMNfDdA1AZtLm7pB69+jH8eotvI7Ge3mW8'
                '1HU8QqLEvvF3OMaFuL6XyA7Q/mQlcbh09p2iICwNIeVQk/N7Ta+U5hRI0x+b'
                'dMR9kLMoWEffDLBC4zs0SSJ7d+K7eFMrD32zW5d8K6aldKI20LiIWEyyPzD1'
                'a+tDuVIa86PEKwecyA0qC8lG2yR2eJapznJiKqIpIytrOmeIty3cjFV80hGo'
                'tbZCoWLfgsep2KniItqZnBMK3fHBqcpIqdqoY8fZpSLszPTSuRoQFnXP9Q3C'
                'a/hr7rbjUUBLxiVDYyVNYsq63/GwSx09efnVOGypiDxwO7gHL+rlTNDA5Yrx'
                'tpuSTGeoQ+inSnRssxZXyLogGTsqRtWDHLuaNXDRaLF18AC+iZyI+RwGar08'
                'hl005TSm+YibtPfezxSSBPUls3qfdnUe63xtQ9ygk360gM04gZyFpovrwd/5'
                'MZtb8CdV3dTAjS23YY+IoiE/M4/FAr6YPKuNf8UZqx3sfetTUiiQEHw8bYSW'
                'KSxL3Os596Fcbg/8jZGaJ0s3yWtPVKPrQGGTOxbbSrdYkVirJOXaIwQ9IG/g'
                'L2/NRmuV1AsaK34P7OBAHWHBJL4sHtVGxt4pmzNapWjrq2rx7bkHycWcthmR'
                'yKDdtzMaO28ydjHDkGhjrOaUy+VYyfpfns/BjYhNX0xV+iZbbnKqQsy0LPVO'
                'mMmNGFSqkqgTP6BYmDO/hf+iMnZ6e3zAWd/L8lCCGOfRjYIKwpUDa4PFobod'
                'FQxjwaXh7WeJvRT0T++5CSlqLHgUlZGex9uvDREEruLxz/KbfTLIJFpGThQH'
                'V/PGq1s5LHNKVq1alXzxBKyIMpfeJEKKIowJqd/+C26RCXYqVblrk/isL/nP'
                'R01ZB/jV2/Eqry6cQcYxitb/+uZmQqijJelChG93WzrX5YeA1eryCU+m2x/u'
                'kem0NO2v5QEtjBph4BZxWykqK9I6DjisC9QiuK/+sDIe2k70b2i7z1HXq7SF'
                'rTXDH/cQCleufwpjgWWllJsN7/ItZVTvDZVldaobWYeyxWypfYjj8GW9Tp0/'
                'ygWhl8qj25Ovcye4opVXNDgRzv1cyvj4H9PRp5mWYXZN9Wh+7bs5SqdP4ikz'
                '+b3GjKl/Nlv1blalBsutM2YmBIm8hvYdsaY8mbBZoJO2rae24z+spGcCJBUq'
                'svZRl+d4QEP/rKaQciJolPaA/Wj2rCCFjQl/Wxhzmo1SiGnsNH+y1yqDYyqq'
                'QVuxIrCGtmvckRzEqwel3KzlzXuDDonMnQTvryHTeluIp+1oSZILpF/Bgpvx'
                'Kq9fusfHX8lFeWTM23BNjl0Zoqxi9D1znfKSHQNtbCR5hVusL8bWxf2569me'
                '0LchkyQ5ZC99f8lV9PKBwrzOYya5Mg42Lv/s4Ml3vXfjHw3JMDY5gh/UkorK'
                'O78b+gHbvwj1xRdlF9gKcPOKhWKKhOp5/GTzFifanFZYmKSSm+NlgvsCseJi'
                'RXBFfCnsGD9/1NvdLokl2anba2nQSY5oU9/Zvthz3155ZPK6/NLbPu6j+VnP'
                'UaoT9Y6wOWD7pOoH172fK+KkMTMGElbfVbEkVmO5axnKFXMX4kuKjYXHFI/1'
                '5cEI7nzyaniVl10Dy7jr/dsGTnJXy/OtwKtAZuEQ8mFGUCiR8Xo7zX7DOVEX'
                'uRjAZAg3fvVzxSOLX5u5KzdvucIdfdsRHIHLU1UviAN3o7rn8MM9Qm579O68'
                'qy2QaxdnW5OhDXqYzZn1fNy/btGGoDao+6IEQ3wcnEkaM/IRWkw9xYw99fn8'
                'fWyV3tDGJOrjemz2yYTNmOI16jlL0fdH57cWNo3dx3OD6Oot+NJ8U3A3/IrD'
                'nH+o3Lr8/wb4wtkxBSxNZxyL7LkZlaVl+WJvsRR3JL9rUSbHljdTVFmtRyot'
                '0+/eXS1PNx+7PHlWFqyODBNfGRd644cCRACT/kYd0dKe0ocLfcUdwG0r+xki'
                '5cvR/c8b06ETNwaFvYsCM7bTDJpIxZ7N6Lo8uBOW3Ppkwx0j8IB92WcWaWcN'
                'n47dhZDrlD43u3GsRjyMfT98PtWUEDwd1fr6oHnM0YdQF+v+hl59OIo5wIS7'
                'ERHe8cvwpB/gPGDOVGqoz/cGwl10w1a3cE62S5+xIJeRXZqg2oIkBHIYdwPj'
                '69byo+arQrDNW82VffowxSCPGXSSqGJ2mmt3+MfXLTpeUZYN4SGhxdUrfBSw'
                'evwkz779b06XfiXkMF+OSalfjul25Oe+3DnE2M3c6iTOFkC0kVgzjRcD5ykl'
                '6JJe+0LHlwO99xVyLNKpBYzTevuXZMymVR0e0ad5XeqYipR7j3Vv1bgOWkCv'
                'mlWIsrczbJBNIFV/pWvehpK5sWSbB8UAm3RZirYoTyPt4gaWCqhqMtS4GZkT'
                'nHxWs2B8ELJsFYPFiKnw1k8SxGRzBglptg7jte3mPuEquIZgmicj0/RPCcsk'
                'a9XKLorhRpPQq5T3CKu3oJ8ZCeX9FIiIgMZIwxTocFFtS18TNVrQ1OHLVhRm'
                'BUdrHbaczqmpGqGIsJol0fviys/9ow0QFciCYljpYMiFAEjdxtDP/QnIsF0f'
                'c/ElniF+fh/131K+Ifz8BgABJqAFQIDtMQm/tE1/gJbla5zbIdoFCLCjWXv/'
                'Cz7DUEw=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        self.assertEqual(module.image_name, "OrigGreen")
        self.assertEqual(module.rescaled_image_name, "Rescaledgreen")
        self.assertEqual(module.rescale_method, R.M_MANUAL_INPUT_RANGE)
        self.assertEqual(module.wants_automatic_low, R.LOW_EACH_IMAGE)
        self.assertEqual(module.wants_automatic_high, R.HIGH_EACH_IMAGE)

    def make_workspace(self, input_image, input_mask=None,
                       reference_image=None, reference_mask=None,
                       measurement=None):
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        measurements = cpmeas.Measurements()
        module_number = 1
        module = R.RescaleIntensity()
        module.image_name.value = INPUT_NAME
        if isinstance(input_image, (tuple, list)):
            first_input_image = input_image[0]
        else:
            first_input_image = input_image
        if isinstance(input_mask, (tuple, list)):
            first_input_mask = input_mask[0]
        else:
            first_input_mask = input_mask
        if first_input_mask is None:
            image = cpi.Image(first_input_image)
        else:
            image = cpi.Image(first_input_image, first_input_mask)
        ii = InjectImage(INPUT_NAME, input_image, input_mask)
        ii.module_num = module_number
        module_number += 1
        pipeline.add_module(ii)
        image_set.add(INPUT_NAME, image)
        module.rescaled_image_name.value = OUTPUT_NAME
        if reference_image is not None:
            module.matching_image_name.value = REFERENCE_NAME
            if reference_mask is None:
                image = cpi.Image(reference_image)
            else:
                image = cpi.Image(reference_image, mask=reference_mask)
            image_set.add(REFERENCE_NAME, image)
            ii = InjectImage(REFERENCE_NAME, reference_image, reference_mask)
            ii.module_num = module_number
            module_number += 1
            pipeline.add_module(ii)
        module.module_num = module_number
        pipeline.add_module(module)
        if measurement is not None:
            module.divisor_measurement.value = MEASUREMENT_NAME
            measurements.add_image_measurement(MEASUREMENT_NAME, measurement)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  measurements,
                                  image_set_list)
        return workspace, module

    def test_03_01_stretch(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected[0, 0] = 1
        expected[9, 9] = 0
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = R.M_STRETCH
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_03_02_stretch_mask(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected[0, 0] = 1
        expected[9, 9] = 0
        mask = np.ones(expected.shape, bool)
        mask[3:5, 4:7] = False
        expected[~ mask] = 1.5
        workspace, module = self.make_workspace(expected / 2 + .1, mask)
        module.rescale_method.value = R.M_STRETCH
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_04_01_manual_input_range(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10))
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.CUSTOM_VALUE
        module.wants_automatic_high.value = R.CUSTOM_VALUE
        module.source_scale.min = .1
        module.source_scale.max = .6
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_04_02_00_manual_input_range_auto_low(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected[0, 0] = 0
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.LOW_EACH_IMAGE
        module.wants_automatic_high.value = R.CUSTOM_VALUE
        module.source_high.value = .6
        self.assertFalse(module.is_aggregation_module())
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_04_02_01_manual_input_range_auto_low_all(self):
        np.random.seed(421)
        image1 = np.random.uniform(size=(10, 20)).astype(np.float32) * .5 + .5
        image2 = np.random.uniform(size=(10, 20)).astype(np.float32)
        expected = (image1 - np.min(image2)) / (1 - np.min(image2))
        workspace, module = self.make_workspace([image1, image2])
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.LOW_ALL_IMAGES
        module.wants_automatic_high.value = R.CUSTOM_VALUE
        module.source_high.value = 1
        self.assertTrue(module.is_aggregation_module())
        module.prepare_group(workspace, {}, [1, 2])
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_04_03_00_manual_input_range_auto_high(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected[0, 0] = 1
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.CUSTOM_VALUE
        module.wants_automatic_high.value = R.HIGH_EACH_IMAGE
        module.source_low.value = .1
        self.assertFalse(module.is_aggregation_module())
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_04_03_01_manual_input_range_auto_high_all(self):
        np.random.seed(421)
        image1 = np.random.uniform(size=(10, 20)).astype(np.float32) * .5
        image2 = np.random.uniform(size=(10, 20)).astype(np.float32)
        expected = image1 / np.max(image2)
        workspace, module = self.make_workspace([image1, image2])
        self.assertTrue(isinstance(module, R.RescaleIntensity))
        image_set_2 = workspace.image_set_list.get_image_set(1)
        image_set_2.add(INPUT_NAME, cpi.Image(image2))
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.CUSTOM_VALUE
        module.wants_automatic_high.value = R.HIGH_ALL_IMAGES
        module.source_low.value = 0
        self.assertTrue(module.is_aggregation_module())
        module.prepare_group(workspace, {}, [1, 2])
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_04_03_02_manual_input_range_auto_low_and_high(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = expected - expected.min()
        expected = expected / expected.max()
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.LOW_EACH_IMAGE
        module.wants_automatic_high.value = R.HIGH_EACH_IMAGE
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_04_04_manual_input_range_mask(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected[0, 0] = 1
        mask = np.ones(expected.shape, bool)
        mask[3:5, 4:7] = False
        expected[~ mask] = 1.5
        workspace, module = self.make_workspace(expected / 2 + .1, mask)
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.CUSTOM_VALUE
        module.wants_automatic_high.value = R.HIGH_EACH_IMAGE
        module.source_low.value = .1
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_04_05_manual_input_range_truncate(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected_low_mask = np.zeros(expected.shape, bool)
        expected_low_mask[2:4, 1:3] = True
        expected[expected_low_mask] = -.05
        expected_high_mask = np.zeros(expected.shape, bool)
        expected_high_mask[6:8, 5:7] = True
        expected[expected_high_mask] = 1.05
        mask = ~(expected_low_mask | expected_high_mask)
        for low_truncate_method in (R.R_MASK, R.R_SCALE, R.R_SET_TO_CUSTOM,
                                    R.R_SET_TO_ZERO):
            for high_truncate_method in (R.R_MASK, R.R_SCALE, R.R_SET_TO_CUSTOM,
                                         R.R_SET_TO_ONE):
                workspace, module = self.make_workspace(expected / 2 + .1)
                module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
                module.wants_automatic_low.value = R.CUSTOM_VALUE
                module.wants_automatic_high.value = R.CUSTOM_VALUE
                module.source_scale.min = .1
                module.source_scale.max = .6
                module.low_truncation_choice.value = low_truncate_method
                module.high_truncation_choice.value = high_truncate_method
                module.custom_low_truncation.value = -1
                module.custom_high_truncation.value = 2
                module.run(workspace)
                image = workspace.image_set.get_image(OUTPUT_NAME)
                pixels = image.pixel_data
                np.testing.assert_almost_equal(pixels[mask], expected[mask])
                if low_truncate_method == R.R_MASK:
                    self.assertTrue(image.has_mask)
                    self.assertTrue(np.all(image.mask[expected_low_mask] == False))
                    if high_truncate_method != R.R_MASK:
                        self.assertTrue(np.all(image.mask[expected_high_mask]))
                else:
                    if low_truncate_method == R.R_SCALE:
                        low_value = -.05
                    elif low_truncate_method == R.R_SET_TO_CUSTOM:
                        low_value = -1
                    elif low_truncate_method == R.R_SET_TO_ZERO:
                        low_value = 0
                    np.testing.assert_almost_equal(pixels[expected_low_mask], low_value)
                if high_truncate_method == R.R_MASK:
                    self.assertTrue(image.has_mask)
                    self.assertTrue(np.all(image.mask[expected_high_mask] == False))
                else:
                    if high_truncate_method == R.R_SCALE:
                        high_value = 1.05
                    elif high_truncate_method == R.R_SET_TO_CUSTOM:
                        high_value = 2
                    elif high_truncate_method == R.R_SET_TO_ONE:
                        high_value = 1
                    np.testing.assert_almost_equal(
                            pixels[expected_high_mask], high_value)

    def test_04_06_color_mask(self):
        '''Regression test - color image + truncate with mask

        The bug: color image yielded a 3-d mask
        '''
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10, 3)).astype(np.float32)
        expected_mask = (expected >= .2) & (expected <= .8)
        expected_mask = expected_mask[:, :, 0] & expected_mask[:, :, 1] & expected_mask[:, :, 2]
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = R.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = R.CUSTOM_VALUE
        module.wants_automatic_high.value = R.CUSTOM_VALUE
        module.source_scale.min = .2
        module.source_scale.max = .5
        module.low_truncation_choice.value = R.R_MASK
        module.high_truncation_choice.value = R.R_MASK
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_NAME)
        self.assertEqual(image.mask.ndim, 2)
        self.assertTrue(np.all(image.mask == expected_mask))

    def test_05_01_manual_io_range(self):
        np.random.seed(0)
        expected = np.random.uniform(size=(10, 10)).astype(np.float32)
        workspace, module = self.make_workspace(expected / 2 + .1)
        expected = expected * .75 + .05
        module.rescale_method.value = R.M_MANUAL_IO_RANGE
        module.wants_automatic_low.value = R.CUSTOM_VALUE
        module.wants_automatic_high.value = R.CUSTOM_VALUE
        module.source_scale.min = .1
        module.source_scale.max = .6
        module.dest_scale.min = .05
        module.dest_scale.max = .80
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_06_01_divide_by_image_minimum(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        image[0, 0] = 0
        image = image / 2 + .25
        expected = image * 4
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = R.M_DIVIDE_BY_IMAGE_MINIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_06_02_divide_by_image_minimum_masked(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        image[0, 0] = 0
        image = image / 2 + .25
        mask = np.ones(image.shape, bool)
        mask[3:6, 7:9] = False
        image[~mask] = .05
        expected = image * 4
        workspace, module = self.make_workspace(image, mask)
        module.rescale_method.value = R.M_DIVIDE_BY_IMAGE_MINIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_07_01_divide_by_image_maximum(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        image = image / 2 + .1
        image[0, 0] = .8
        expected = image / .8
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = R.M_DIVIDE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_07_02_divide_by_image_minimum_masked(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        image = image / 2 + .1
        image[0, 0] = .8
        mask = np.ones(image.shape, bool)
        mask[3:6, 7:9] = False
        image[~mask] = .9
        expected = image / .8
        workspace, module = self.make_workspace(image, mask)
        module.rescale_method.value = R.M_DIVIDE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_08_01_divide_by_value(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        image = image / 2 + .1
        value = .9
        expected = image / value
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = R.M_DIVIDE_BY_VALUE
        module.divisor_value.value = value
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_09_01_divide_by_measurement(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        image = image / 2 + .1
        value = .75
        expected = image / value
        workspace, module = self.make_workspace(image, measurement=value)
        module.rescale_method.value = R.M_DIVIDE_BY_MEASUREMENT
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_10_01_scale_by_image_maximum(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        image[0, 0] = 1
        image = image / 2 + .1
        reference = np.random.uniform(size=(10, 10)).astype(np.float32) * .75
        reference[0, 0] = .75
        expected = image * .75 / .60
        workspace, module = self.make_workspace(image,
                                                reference_image=reference)
        module.rescale_method.value = R.M_SCALE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)

    def test_10_02_scale_by_image_maximum_mask(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        image[0, 0] = 1
        image = image / 2 + .1
        mask = np.ones(image.shape, bool)
        mask[3:6, 4:8] = False
        image[~mask] = .9
        reference = np.random.uniform(size=(10, 10)) * .75
        reference[0, 0] = .75
        rmask = np.ones(reference.shape, bool)
        rmask[7:9, 1:3] = False
        reference[~rmask] = .91
        expected = image * .75 / .60
        workspace, module = self.make_workspace(image,
                                                input_mask=mask,
                                                reference_image=reference,
                                                reference_mask=rmask)
        module.rescale_method.value = R.M_SCALE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_11_01_convert_to_8_bit(self):
        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        expected = (image * 255).astype(np.uint8)
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = R.M_CONVERT_TO_8_BIT
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        np.testing.assert_almost_equal(pixels, expected)
