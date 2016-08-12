'''test_classifyobjects - test the ClassifyObjects module
'''

import base64
import unittest
import zlib
from StringIO import StringIO

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.grid as cpg
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.classifyobjects as C

OBJECTS_NAME = "myobjects"
MEASUREMENT_NAME_1 = "Measurement1"
MEASUREMENT_NAME_2 = "Measurement2"
IMAGE_NAME = "image"


class TestClassifyObjects(unittest.TestCase):
    def test_01_01_load_matlab_classify_objects(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylRwSU1WMDBRMDSzMja3MjRQMDIwsFQgGTAwevryMzAwsDIz'
                'MFTMuRtxN++ygUjZ7m23GXKTRTOEFjyd1G146ePiWL5jLFNvWG3MbPI2L7RW'
                'vr4uKT1q5zPPCiG9CgG9Sp43prlB68/8zL6RdrKr7/Hnn3PO26kfj2RieGPD'
                'tEBtPvem6Qc3hDLvFHnpZefklHDraWgAY/bBH/43/Re0J85eIOxRc+Tgjt5j'
                's7cc7mP/+3F17bNfLLIH4ko25bDXre+1ufeqR6T5evsPmYnKf3gbT5rN+sY5'
                'N7FPkvcLc01tyR494Vu7vDf9j+Or1Mi0CUlZ737s6h9ZgXPrfR7mqhrVq744'
                '/iLG5rZ1e9+GqgCL5Vzptxs2L3iosfPpw3fRnqWCv0x995dt9OdzrDjhPOtr'
                'ub/i0o4c0Y37POuPL9i/4qX2Tk/WRTLdjgZxR+xCTGtVzd/0TjfreH5c1jf+'
                '5PsZrjM31PQ52dtJFLyS8rN9sPi1Y+Mdz7RPW06bPTjj++BMo77Rsc8x9q//'
                'iR11rmRazsMXsfF3wKwDmo1sO2KfHxCRXy3VN+n7OQFV5u7vC5tjVvdq8n/8'
                'tOl0S3H44cAi74L7Vf4R0pMOvZ05LTw+98bs6E35zX9+elbnrfe+I360Wc7+'
                'm9tjtyN79/3rnlE1lXMv1wbzt25Se2tZXgpxvD17Ol/Y784vldX2y/aUx292'
                'Pjlj1w+e3XYqYu8Ovtvx6tzl9Xsq+k4H33r/2qT+3o+mk90MZf8W21fdyTm2'
                'RXmd/S9j3nVrdNnvNLK4Fly6xuj5c2ap7cQr8yKvl0f+XHg+cd3vyVPFZdNz'
                '/72qrjPpTz37XNxz/WvD/fE257ZU9W/dr+Te4hD5uZrl/KSdcl+FP0y9Yid8'
                'b8t/XcV858+THqhu1V+5t/Cjr0Xl3Om3T93P/Wr/bt/5U/8jH5/7VRn81u/z'
                '/8afTQldeqqvMn/M9/7/z+7V95Ytz78oN+cZFyl2PL7OzOt+0lc8sNG62Zn/'
                '+6Kt6TJ72qXmTXo6t/23XFZsS+aTTYLHz6fy+p/czVKk91aa6ZN3qc/ZA/mR'
                'Xqu//jyq8/39kteh76eKf/kvp7bXy+Bf1d/fv/rV98tp/7VXuGlZVXX0T+Du'
                'dZ//M3WZT9YGAF3qrqE=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice, C.BY_SINGLE_MEASUREMENT)
        self.assertEqual(len(module.single_measurements), 1)
        group = module.single_measurements[0]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement, "Intensity_IntegratedIntensity_OrigBlue_1")
        self.assertEqual(group.bin_choice, C.BC_EVEN)
        self.assertEqual(group.bin_count, 5)
        self.assertAlmostEqual(group.low_threshold.value, 0.2)
        self.assertFalse(group.wants_low_bin)
        self.assertAlmostEqual(group.high_threshold.value, 0.8)
        self.assertFalse(group.wants_high_bin)
        self.assertFalse(group.wants_custom_names)
        self.assertFalse(group.wants_images)

    def test_01_02_load_matlab_classify_by_two_measurements(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylRwSU1WMDBRMDS1MrawMjRTMDIwsFQgGTAwevryMzAw/GBi'
                'YKiYczfibt5lA5Gy3duk7Wef0hIQdz6tKp68pEFgeiqzo+q1sDWzHlzKvXst'
                'OCl8o3HUukqBI5PfFFbwvDHLDRZ+psKbae0dViJTP7/f3vr48ZvMDCZ8DBod'
                'z/tWRBpoVp4/92PVlvOCz2d8iZB7ciB5/d+oK9wvEm5susC6JebIgTU+t1cv'
                '8Z69+OCbq5Jvd4itcrx9P3LKrOztEw6/uqt5ofruAeeUuY/fc+hMy9x+/gWX'
                'sU9n+/cfV6/d3DFF+/G7zl+bLa4rLn4xQXjrk+zpf7UT7tmLL4i+eH3/i6tq'
                'Up/S37dvOMsr/aks/tOeix/eaNhcWme9o1xpiR9zfMCH1o9WW3Vb1K2OSP7f'
                'ws/peK3b5Nj1h9pSf2RZ9d9qPewwOOMeVy31ekL4h0t6s93X+wjkyZWL7tly'
                'P7A5IG8Hz9nnHwMPBwdVeM2Ts2UvuhIZoua534f3/DWm82I/tPZ9VDkeX1Z7'
                '+ZLi7gOTWU1nvbNKEaswWsCTd93uwseWXvHMn8X+DyeWzTW+/iPhotiWoO/p'
                '4Q8nfpNsOrL+otWb66HzT2eejf8bd2bxp6BdS1urJYWmRx942lCZM+3xgydy'
                'sWzhFx9c6rPPPB8/Pdwm4M80MzXZmlZrvS08OYfv6WTfP2Kx59VnW4/+7XrL'
                'Hyk13ZZe0y4l3ufcJx/G/1WyPfzJMeVrbWKfPh45mxxSHzaX3/Xr8/rny0+/'
                'cp532OV6emRsyDz92+b/l384flHjt5nH/cILby2vyC45Lzxl+6VrqlmfLSfW'
                '1xWeE5efr3hOyqRjQXjI/5Iax//dz3b53uk+JH51/4p/bU+vf1TZGp+3um57'
                '6b/XYvmLXm1/lR70+aZJbElhXfzH8FtXnoqpvVjvIH/k/vk/v2tC35sXfn+u'
                'zmUder9V7cweOS1gDJUJPNxhnWVx49vzdYk/5DRrLomVmz6zWmH370ztdNWT'
                '1y98kOSav8Vf98SMD/lOJ/zczuZfkT4sNs/YVF8r48s5ruz9W/bOvlLfZvD4'
                '6fv0za/+y8jqPd+63+5P4Y/XJ782HvnPdvHb1cvSv/xP7/r2n7ns6/KJAHF9'
                'smE=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice, C.BY_TWO_MEASUREMENTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.first_measurement, "Intensity_IntegratedIntensity_OrigBlue_1")
        self.assertEqual(module.second_measurement, "Intensity_MeanIntensity_OrigBlue_1")
        self.assertEqual(module.first_threshold_method, C.TM_MEAN)
        self.assertEqual(module.second_threshold_method, C.TM_MEAN)
        self.assertFalse(module.wants_custom_names)
        self.assertFalse(module.wants_image)

    def test_01_03_load_v1(self):
        data = ('eJztXNtu2zYYphwna1pgSLeLFRiG6SoYCkdQTluWmymHZjEQJ8EcdO1NM0Wi'
                'bQ4yaUhUEu9J9lh7lD7CRFu2ZEaObFmWLJdCFJs0v/8jP/48/JKg2tHNxdGx'
                'vK+ocu3oZquBLChfWzptELt9KGNakU9sqFNoygQfyjWC5VNoyOovsnpwuPvz'
                'obov76jqryDZIVVrX3sf/34CYM37fOGdJf+nVT8thU6WrkNKEW46q6AM3vj5'
                'n73zvW4j/c6C73XLhU5AMciv4ga56XaGP9WI6VrwUm+HC3vHpdu+g7Zz1RgA'
                '/Z+v0SO06ugfyDVhUOwPeI8cRLCP9+3zuUNeQjneeos8nNledTj7xzo1WnXq'
                '9cBofk+3SqCbxOnGPjdD+az8OQjKlyN0fh0qv+GnETbRPTJd3ZJRW28Oa83s'
                'HcTYe8HZY+krGzWPvS5KAx/Xnm85PEvXPefxXLwNdce1YRti6gzbo8XY2+Ds'
                'sfPmgYwYS2wHPtKtd4+6QeU26/JJ9Fnj7LD0pWtYEE3WnpccnqVPiYwJlV3H'
                '9zdmR42xI43YkcBOQtx2QtwumMwfV7n2svS2WtlTJ8TH6T0rPq7d5RF8GVwS'
                'DLPs55UROyvgozcb8Lg1Djc4Brh1EPB9iOELz18bfrqKKcQOot1b9q1ps6Up'
                'yBvMDrPpGYcrjeBKHm4yvklw0+gXN/99w+nH0u/uIba6stPRDW9Nv0PYyc/e'
                'd5w9lj6FDd21qFxli418imxoUGJ3c/HPaeeh/YS43YQ4BUw2z7/idGbpK+q4'
                '8u8WudOtyPamqRPfL6qyk6i9g3l6XuNzkvVovroczHXdTKrLk3pWtrPVJYIv'
                'TVxSXfj1Q1XU7Sx1iZrfslh3tBi+9RG+fnq4Ro9p7zT81zH8P3L8LH2GbIdW'
                '6tAg2KzctJBtVs6Ia9NW5Qw1aGum/UKSOOakpWMMrZ1F0DNrv5s17puWd1ud'
                'r79HxXMnlu44qIGg2d/hp2lnVn9LY18zjX5Fjauy3m8mmdeC2KemP0YEQkmv'
                '+1wQQ6eI4NsT6Bm1bz+E6jkvPSfBpRkfRV0fOnEdStpbJmwgHA5okvphDZpI'
                'x8Xyw3nFL0/3u/u5r1vz3c+rqY2/j2PqsUx6bft6pTnOsujfea2LUeu4slNR'
                '9ivKQR7jTYvhy/s6w6Jcl1s0vqRxy1dgtD9Z2iIPt965VPNQmnEK06eFmq0s'
                '6+sN64WIb5LoxbRiDpX1vMTu2xmsDTDY3+UV38TVPyr+7+nG/oXqMa99ctT+'
                '5E/ocbPHAu7ZDXBswBTqkef9mjT9Lep+2xmxYdMmLjaz16kGdZxH/LXo8/ys'
                '82ae95WyHldZ67II/TDJOCpK+5Z9HC2iLmnu6xcZl2T/c0EevL9itXNe+kTF'
                'Wefe9uoixThrkXFJ9PG0YRIVqp1ZXidi4pynGIcWBSfm5QCnged1mTSuT8r/'
                '39sAJ3E49rkZys86Hus9zMwCsk76doriH+cx7Y2K+8nd39CgvQbLCJuwk2P9'
                'i4LTwPM6R12fCum8sHaKor/ACZzACZzACZzACZzAPY/TQrhJ48YgDuqHBUWu'
                'd1HbvyzxSVF0KwpOA1+WPwucwAmcwC0aTgvhvoTrywIncMuI00I4sZ8aj9PA'
                '8zqJeEDg5oHTgBifAidwAidwAidwAidwAidwy4LTQrg89vetUoCTOJzkf5dC'
                '5f+Kqe9brr4sbUDL6tiEvY/ZVtq9lwY7ikV0s/8WXuXC+1oNvZCX8XRieDSO'
                'RxvHg0yIKWp0O7bH5lLS1ikylKqfe+3lHg1yGe9jDO8xx3s8jtd/kW7/GgAa'
                'vMxGqfWzr3rZ3Iuvev0Rw7/H8e+N4zf6Dz93+xVwFP9h6G6f2cmLL/z8/3oE'
                'X9jfSn769Q8rm9+D58cZAKP+Hfj959+S8pbLJUkCT8fpqxg80/EleHowO2+k'
                '6cbbT2B8+UGbl6l8kn6S2AFm1zfgKw/rNuBZhvL/A/z19oA=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[-2]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice.value, C.BY_SINGLE_MEASUREMENT)
        self.assertEqual(len(module.single_measurements), 2)

        group = module.single_measurements[0]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_IntegratedIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, C.BC_EVEN)
        self.assertEqual(group.bin_count, 3)
        self.assertAlmostEqual(group.low_threshold.value, 0.2)
        self.assertTrue(group.wants_low_bin)
        self.assertAlmostEqual(group.high_threshold.value, 0.8)
        self.assertTrue(group.wants_high_bin)
        self.assertTrue(group.wants_custom_names)
        for name, expected in zip(group.bin_names.value.split(','),
                                  ('First', 'Second', 'Third', 'Fourth', 'Fifth')):
            self.assertEqual(name, expected)
        self.assertTrue(group.wants_images)
        self.assertEqual(group.image_name, "ClassifiedNuclei")

        group = module.single_measurements[1]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_MaxIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, C.BC_CUSTOM)
        self.assertEqual(group.custom_thresholds, ".2,.5,.8")
        self.assertFalse(group.wants_custom_names)
        self.assertFalse(group.wants_images)

        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice, C.BY_TWO_MEASUREMENTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.first_measurement, "Location_Center_X")
        self.assertEqual(module.first_threshold_method, C.TM_MEDIAN)
        self.assertEqual(module.second_measurement, "Location_Center_Y")
        self.assertEqual(module.second_threshold_method, C.TM_MEDIAN)

    def test_01_04_load_v2(self):
        data = ('eJztne9z2jYYxwUhXdPe7dJtL/qmN7/a7XrEA5bs0rwZ+VEW7kLCDa5rX6WO'
                'LUA7I3G2nIT91fkTZoEdG8XExtgGU3EhIOHP80hfP5L1iBBax92L4xPpQK5I'
                'rePuXg/pUGrrCu0RY3gkYVqWTg2oUKhJBB9JLYKlM6hKtapUPTiq/XG0/0Gq'
                'VSofQLxbodn63n5o9wF4YT++tO9F56Vtp1zw3Vm5AylFuG9ugxJ469Q/2PdP'
                'ioGUGx1+UnQLmp4Lt76Je6Q7Hj2+1CKapcNLZeg/2L5dWsMbaJhXPRd0Xm6j'
                'e6h30H+Q64J72N/wFpmIYId37PO1j34J5fx2BuSuYdjN4eyfKFQddKh9Bmbr'
                'mW4PFU+3Aqcbe3zvq2fHnwPv+FKAzm98x+86ZYQ1dIs0S9ElNFT6j61m9g5D'
                '7L3k7LHylYH6J/YpYnwlhC/M8AVQjdiPHzi/rNyxg8YO7SFUTMuAQ4hpdF1+'
                '4uyxcltBhkR6foOma68eYm+Xs8fuXXhP9z7eKyqVhuyUR9H3BWeHlS8tVYdo'
                'Neen5vQ/7nldlPsdROvnNtdPVq5WyvuViPyyOofxYf0uzfAlcEkwjBJnrzi/'
                'rHxGJEyoZJnOfBLF/9aMnS3wxZ4FeO4Fx7k3l9sBnr/PIf5+4drNyk1MITYR'
                'HV+zZ32DXZK8Ojdql9MzjCvOcEWbi+YvCreIfnHmv4+3EOtjyRwpqn0tv0HY'
                'XJ29oPn0DPYUS6dSk11kpAbRNWi49rKOz7jzUFbzngyiXWdeczqz8hU1Lekv'
                'ndwoemB/k9SJPy8VuRarv+48ndb4jHJe0tQlKF6T5OLq8uT8lasZx8thquMq'
                'ri789aMiV6p5jZdF2lkP8bcz429afrxGz+lvqvETEK+LcHHWz6cDBWOo1/Ko'
                '17LzSdr5Bu+3WllOn3aIv5+59rJyAxkmLXegSrBW7g6QoZUbxDLooNxAPTpY'
                'av25bLwtOk8egHzFV1rr+qznyaD8/1RXTBP1ENSmmWGSdtLSLQq3iG55zcOz'
                'jq8485aXK7eU+4DEOe7+4AVRFYoIvj6FtlHj+rOvnaucV5LMp3/k+s3Kp5ZJ'
                'yXBPgz2E/Qlw3DhsQQ0pOF+6pbUuf5oHHGScd1RW4i+J8fdlTjuSvH6t276C'
                'G8dxxt10HKfb3qT3/xbVpxoQz9Hicn+j5qM016Npxos9/S3lL05eIdfK8kFZ'
                'PozCf8fxrKyTu2v7nvk81L0jksrWwNC7Hmc9ruPozfQaoP4gzXVM0PXjH2g7'
                'ZW/v37I3srEK5/Q/Sd2SzH+ZZizQ0oyzoPfRGsSAfYNYWEtfr3lxFtbuoP2l'
                'iV7s1xz/6zy/h/U3qXw46/k+6X2ZdciD4+qTNbfp/UtTlxZU8KrbuQ66JL1e'
                'yyMnxlEwJ8ZRMCfGUTCX9Xpknbk465gLcmf/5KufaekTlJef20v/iyXz8rxw'
                'cfSxtWES5aqfWe4LMXHOl9ynyCMn5mWPq4PndYmThy/U7t88rsBx7JH/3EOW'
                '+0KTD0mwjaHR8v7zwp2D53UK2nckN/9ClU6EkhDW4Mhnrx5iT+gef3/cp/va'
                '2smL/oITnOAEJzjBCU5wgnueq/u4qHmjt66fpgl56q/gvk2uDpKJ86TsrLof'
                'm5Lf5iX+BCc4wX07XN3Hif1SwQlOcIITnOAEt075XF64Onhep6TyTqG34AQn'
                'OMEJTnCCE5zgBCc4wW0mV/dxq8gHB0WPK3BcwXle8B3/NaS9/r+z33XKKtT1'
                'kUHY90gY8nDyZQemrBNFm357gHxhP236vkiA+RmF+Klzfurz/CANYop645Fh'
                'e7MoGSoUqXLTqW3btcduLfN7H+L3hPN7Ms+v8yUA0z0A5P5zNbk1rb6aVHP/'
                '2HNyPkL873P+9+f5V6cfrhhPG2DKzoctxlPP5qr8+T9ftBPgzx9vRaf85t3W'
                '+3fg+XEGwGx8e3H/8Gdcv6XSVrEIno7T1yE80/EVeHpjdt4WFhtvv4L5x7t9'
                '3qTj45ynAruB5fX1/JUe2+b62YTj/wfn4gog')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[-2]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice.value, C.BY_SINGLE_MEASUREMENT)
        self.assertEqual(len(module.single_measurements), 2)

        group = module.single_measurements[0]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_IntegratedIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, C.BC_EVEN)
        self.assertEqual(group.bin_count, 3)
        self.assertAlmostEqual(group.low_threshold.value, 0.2)
        self.assertTrue(group.wants_low_bin)
        self.assertAlmostEqual(group.high_threshold.value, 0.8)
        self.assertTrue(group.wants_high_bin)
        self.assertTrue(group.wants_custom_names)
        for name, expected in zip(group.bin_names.value.split(','),
                                  ('First', 'Second', 'Third', 'Fourth', 'Fifth')):
            self.assertEqual(name, expected)
        self.assertTrue(group.wants_images)
        self.assertEqual(group.image_name, "ClassifiedNuclei")

        group = module.single_measurements[1]
        self.assertEqual(group.object_name, "Nuclei")
        self.assertEqual(group.measurement.value, "Intensity_MaxIntensity_OrigBlue")
        self.assertEqual(group.bin_choice, C.BC_CUSTOM)
        self.assertEqual(group.custom_thresholds, ".2,.5,.8")
        self.assertFalse(group.wants_custom_names)
        self.assertFalse(group.wants_images)

        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        self.assertEqual(module.contrast_choice, C.BY_TWO_MEASUREMENTS)
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.first_measurement, "Location_Center_X")
        self.assertEqual(module.first_threshold_method, C.TM_MEDIAN)
        self.assertEqual(module.second_measurement, "Location_Center_Y")
        self.assertEqual(module.second_threshold_method, C.TM_CUSTOM)
        self.assertAlmostEqual(module.second_threshold.value, .4)

    def make_workspace(self, labels, contrast_choice,
                       measurement1=None, measurement2=None):
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, OBJECTS_NAME)

        measurements = cpmeas.Measurements()
        module = C.ClassifyObjects()
        m_names = []
        if measurement1 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_1,
                                         measurement1)
            m_names.append(MEASUREMENT_NAME_1)
        if measurement2 is not None:
            measurements.add_measurement(OBJECTS_NAME, MEASUREMENT_NAME_2,
                                         measurement2)
            module.add_single_measurement()
            m_names.append(MEASUREMENT_NAME_2)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)

        module.contrast_choice.value = contrast_choice
        if module.contrast_choice == C.BY_SINGLE_MEASUREMENT:
            for i, m in enumerate(m_names):
                group = module.single_measurements[i]
                group.object_name.value = OBJECTS_NAME
                group.measurement.value = m
                group.image_name.value = IMAGE_NAME
        else:
            module.object_name.value = OBJECTS_NAME
            module.image_name.value = IMAGE_NAME
            module.first_measurement.value = MEASUREMENT_NAME_1
            module.second_measurement.value = MEASUREMENT_NAME_2
        module.module_num = 1
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, measurements,
                                  image_set_list)
        return workspace, module

    def test_02_01_classify_single_none(self):
        '''Make sure the single measurement mode can handle no objects'''
        workspace, module = self.make_workspace(
                np.zeros((10, 10), int),
                C.BY_SINGLE_MEASUREMENT,
                np.zeros((0,), float))
        module.run(workspace)
        for m_name in ("Classify_Measurement1_Bin_1",
                       "Classify_Measurement1_Bin_2",
                       "Classify_Measurement1_Bin_3"):
            m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                               m_name)
            self.assertEqual(len(m), 0)

    def test_02_02_classify_single_even(self):
        m = np.array((.5, 0, 1, .1))
        labels = np.zeros((20, 10), int)
        labels[2:5, 3:7] = 1
        labels[12:15, 1:4] = 2
        labels[6:11, 5:9] = 3
        labels[16:19, 5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                C.BY_SINGLE_MEASUREMENT, m)
        module.single_measurements[0].bin_choice.value = C.BC_EVEN
        module.single_measurements[0].low_threshold.value = .2
        module.single_measurements[0].high_threshold.value = .7
        module.single_measurements[0].bin_count.value = 1
        module.single_measurements[0].wants_low_bin.value = True
        module.single_measurements[0].wants_high_bin.value = True
        module.single_measurements[0].wants_images.value = True

        expected_obj = dict(Classify_Measurement1_Bin_1=(0, 1, 0, 1),
                            Classify_Measurement1_Bin_2=(1, 0, 0, 0),
                            Classify_Measurement1_Bin_3=(0, 0, 1, 0))
        expected_img = dict(Classify_Measurement1_Bin_1_NumObjectsPerBin=2,
                            Classify_Measurement1_Bin_2_NumObjectsPerBin=1,
                            Classify_Measurement1_Bin_3_NumObjectsPerBin=1,
                            Classify_Measurement1_Bin_1_PctObjectsPerBin=50.0,
                            Classify_Measurement1_Bin_2_PctObjectsPerBin=25.0,
                            Classify_Measurement1_Bin_3_PctObjectsPerBin=25.0)
        module.run(workspace)
        for measurement, expected_values in expected_obj.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(np.all(values == np.array(expected_values)))
        for measurement, expected_values in expected_img.iteritems():
            values = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                    measurement)
            self.assertTrue(values == expected_values)

        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(np.all(pixel_data[labels == 0, :] == 0))
        colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
        for i, color in enumerate(colors + [colors[1]]):
            self.assertTrue(np.all(pixel_data[labels == i + 1, :] == color))

        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 9)
        self.assertEqual(len(set([column[1] for column in columns])), 9)  # no duplicates
        for column in columns:
            if column[0] != OBJECTS_NAME:  # Must be image
                self.assertEqual(column[0], cpmeas.IMAGE)
                self.assertTrue(column[1] in expected_img.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER if column[1].endswith(
                        C.F_NUM_PER_BIN) else cpmeas.COLTYPE_FLOAT)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in expected_obj.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)

        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, cpmeas.IMAGE, "foo")
        self.assertEqual(len(names), 0)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", C.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_obj.keys()
                             for name in names]))
        names = module.get_measurements(None, cpmeas.IMAGE, C.M_CATEGORY)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_img.keys()
                             for name in names]))

    def test_02_03_classify_single_custom(self):
        m = np.array((.5, 0, 1, .1))
        labels = np.zeros((20, 10), int)
        labels[2:5, 3:7] = 1
        labels[12:15, 1:4] = 2
        labels[6:11, 5:9] = 3
        labels[16:19, 5:9] = 4
        workspace, module = self.make_workspace(labels,
                                                C.BY_SINGLE_MEASUREMENT, m)
        module.single_measurements[0].bin_choice.value = C.BC_CUSTOM
        module.single_measurements[0].custom_thresholds.value = ".2,.7"
        module.single_measurements[0].bin_count.value = 14  # should ignore
        module.single_measurements[0].wants_custom_names.value = True
        module.single_measurements[0].wants_low_bin.value = True
        module.single_measurements[0].wants_high_bin.value = True
        module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
        module.single_measurements[0].wants_images.value = True

        expected_img = dict(Classify_Three_NumObjectsPerBin=2,
                            Classify_Three_PctObjectsPerBin=50.0,
                            Classify_Blind_NumObjectsPerBin=1,
                            Classify_Blind_PctObjectsPerBin=25.0,
                            Classify_Mice_NumObjectsPerBin=1,
                            Classify_Mice_PctObjectsPerBin=25.0)
        expected_obj = dict(Classify_Three=(0, 1, 0, 1),
                            Classify_Blind=(1, 0, 0, 0),
                            Classify_Mice=(0, 0, 1, 0))
        module.run(workspace)
        for measurement, expected_values in expected_obj.iteritems():
            values = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                    measurement)
            self.assertEqual(len(values), 4)
            self.assertTrue(np.all(values == np.array(expected_values)))
        for measurement, expected_values in expected_img.iteritems():
            values = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                    measurement)
            self.assertTrue(values == expected_values)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertTrue(np.all(pixel_data[labels == 0, :] == 0))
        colors = [pixel_data[x, y, :] for x, y in ((2, 3), (12, 1), (6, 5))]
        for i, color in enumerate(colors + [colors[1]]):
            self.assertTrue(np.all(pixel_data[labels == i + 1, :] == color))

        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 9)
        self.assertEqual(len(set([column[1] for column in columns])), 9)  # no duplicates
        for column in columns:
            if column[0] != OBJECTS_NAME:  # Must be image
                self.assertEqual(column[0], cpmeas.IMAGE)
                self.assertTrue(column[1] in expected_img.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER if column[1].endswith(
                        C.F_NUM_PER_BIN) else cpmeas.COLTYPE_FLOAT)
            else:
                self.assertEqual(column[0], OBJECTS_NAME)
                self.assertTrue(column[1] in expected_obj.keys())
                self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)

        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        categories = module.get_categories(None, OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], C.M_CATEGORY)
        names = module.get_measurements(None, OBJECTS_NAME, "foo")
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, "foo", C.M_CATEGORY)
        self.assertEqual(len(names), 0)
        names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_obj.keys()
                             for name in names]))
        names = module.get_measurements(None, cpmeas.IMAGE, C.M_CATEGORY)
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertTrue(all(['_'.join((C.M_CATEGORY, name)) in expected_img.keys()
                             for name in names]))

    def test_02_04_last_is_nan(self):
        # regression test for issue #1553
        #
        # Test that classify objects classifies an object whose measurement
        # is NaN as none of the categories. Test for no exception thrown
        # if showing the figure and last object has a measurement of NaN
        #
        for leave_last_out in (False, True):
            m = np.array((.5, 0, 1, np.NaN))
            if leave_last_out:
                m = m[:-1]
            labels = np.zeros((20, 10), int)
            labels[2:5, 3:7] = 1
            labels[12:15, 1:4] = 2
            labels[6:11, 5:9] = 3
            labels[16:19, 5:9] = 4
            workspace, module = self.make_workspace(
                    labels, C.BY_SINGLE_MEASUREMENT, m)
            module.single_measurements[0].bin_choice.value = C.BC_CUSTOM
            module.single_measurements[0].custom_thresholds.value = ".2,.7"
            module.single_measurements[0].bin_count.value = 14  # should ignore
            module.single_measurements[0].wants_custom_names.value = True
            module.single_measurements[0].wants_low_bin.value = True
            module.single_measurements[0].wants_high_bin.value = True
            module.single_measurements[0].bin_names.value = "Three,Blind,Mice"
            module.single_measurements[0].wants_images.value = True

            expected_img = dict(Classify_Three_NumObjectsPerBin=1,
                                Classify_Three_PctObjectsPerBin=25.0,
                                Classify_Blind_NumObjectsPerBin=1,
                                Classify_Blind_PctObjectsPerBin=25.0,
                                Classify_Mice_NumObjectsPerBin=1,
                                Classify_Mice_PctObjectsPerBin=25.0)
            expected_obj = dict(Classify_Three=(0, 1, 0, 0),
                                Classify_Blind=(1, 0, 0, 0),
                                Classify_Mice=(0, 0, 1, 0))
            module.run(workspace)
            for measurement, expected_values in expected_obj.iteritems():
                values = workspace.measurements.get_current_measurement(
                        OBJECTS_NAME, measurement)
                self.assertEqual(len(values), 4)
                self.assertTrue(np.all(values == np.array(expected_values)))
            for measurement, expected_values in expected_img.iteritems():
                values = workspace.measurements.get_current_measurement(
                        cpmeas.IMAGE, measurement)
                self.assertTrue(values == expected_values)
            image = workspace.image_set.get_image(IMAGE_NAME)
            pixel_data = image.pixel_data
            self.assertTrue(np.all(pixel_data[labels == 0, :] == 0))
            colors = [pixel_data[x, y, :] for x, y in
                      ((2, 3), (12, 1), (6, 5), (16, 5))]
            for i, color in enumerate(colors + [colors[1]]):
                self.assertTrue(np.all(pixel_data[labels == i + 1, :] == color))

    def test_03_01_two_none(self):
        workspace, module = self.make_workspace(
                np.zeros((10, 10), int),
                C.BY_TWO_MEASUREMENTS,
                np.zeros((0,), float), np.zeros((0,), float))
        module.run(workspace)
        for lh1 in ("low", "high"):
            for lh2 in ("low", "high"):
                m_name = ("Classify_Measurement1_%s_Measurement2_%s" %
                          (lh1, lh2))
                m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                   m_name)
                self.assertEqual(len(m), 0)

    def test_03_02_two(self):
        np.random.seed(0)
        labels = np.zeros((10, 20), int)
        index = 1
        for i_min, i_max in ((1, 4), (6, 9)):
            for j_min, j_max in ((2, 6), (8, 11), (13, 18)):
                labels[i_min:i_max, j_min:j_max] = index
                index += 1
        num_labels = index - 1
        exps = np.exp(np.arange(np.max(labels)))
        m1 = np.random.permutation(exps)
        m2 = np.random.permutation(exps)
        for wants_custom_names in (False, True):
            for tm1 in (C.TM_MEAN, C.TM_MEDIAN, C.TM_CUSTOM):
                for tm2 in (C.TM_MEAN, C.TM_MEDIAN, C.TM_CUSTOM):
                    workspace, module = self.make_workspace(labels,
                                                            C.BY_TWO_MEASUREMENTS,
                                                            m1, m2)
                    self.assertTrue(isinstance(module, C.ClassifyObjects))
                    module.first_threshold_method.value = tm1
                    module.first_threshold.value = 8
                    module.second_threshold_method.value = tm2
                    module.second_threshold.value = 70
                    module.wants_image.value = True

                    def cutoff(method, custom_cutoff):
                        if method == C.TM_MEAN:
                            return np.mean(exps)
                        elif method == C.TM_MEDIAN:
                            return np.median(exps)
                        else:
                            return custom_cutoff

                    c1 = cutoff(tm1, module.first_threshold.value)
                    c2 = cutoff(tm2, module.second_threshold.value)
                    m1_over = m1 >= c1
                    m2_over = m2 >= c2
                    if wants_custom_names:
                        f_names = ("TL", "TR", "BL", "BR")
                        module.wants_custom_names.value = True
                        module.low_low_custom_name.value = f_names[0]
                        module.low_high_custom_name.value = f_names[1]
                        module.high_low_custom_name.value = f_names[2]
                        module.high_high_custom_name.value = f_names[3]
                    else:
                        f_names = ("Measurement1_low_Measurement2_low",
                                   "Measurement1_low_Measurement2_high",
                                   "Measurement1_high_Measurement2_low",
                                   "Measurement1_high_Measurement2_high")
                    m_names = ["_".join((C.M_CATEGORY, name))
                               for name in f_names]

                    module.run(workspace)
                    columns = module.get_measurement_columns(None)
                    for column in columns:
                        if column[0] != OBJECTS_NAME:  # Must be image
                            self.assertEqual(column[0], cpmeas.IMAGE)
                            self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER if column[1].endswith(
                                    C.F_NUM_PER_BIN) else cpmeas.COLTYPE_FLOAT)
                        else:
                            self.assertEqual(column[0], OBJECTS_NAME)
                            self.assertTrue(column[2] == cpmeas.COLTYPE_INTEGER)

                    self.assertEqual(len(columns), 12)
                    self.assertEqual(len(set([column[1] for column in columns])), 12)  # no duplicates

                    categories = module.get_categories(None, cpmeas.IMAGE)
                    self.assertEqual(len(categories), 1)
                    categories = module.get_categories(None, OBJECTS_NAME)
                    self.assertEqual(len(categories), 1)
                    self.assertEqual(categories[0], C.M_CATEGORY)
                    names = module.get_measurements(None, OBJECTS_NAME, "foo")
                    self.assertEqual(len(names), 0)
                    names = module.get_measurements(None, "foo", C.M_CATEGORY)
                    self.assertEqual(len(names), 0)
                    names = module.get_measurements(None, OBJECTS_NAME, C.M_CATEGORY)
                    self.assertEqual(len(names), 4)

                    for m_name, expected in zip(m_names,
                                                ((~m1_over) & (~m2_over),
                                                 (~m1_over) & m2_over,
                                                 m1_over & ~m2_over,
                                                 m1_over & m2_over)):
                        m = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                           '_'.join((m_name, C.F_NUM_PER_BIN)))
                        self.assertTrue(m == expected.astype(int).sum())
                        m = workspace.measurements.get_current_measurement(cpmeas.IMAGE,
                                                                           '_'.join((m_name, C.F_PCT_PER_BIN)))
                        self.assertTrue(m == 100.0 * float(expected.astype(int).sum()) / num_labels)
                        m = workspace.measurements.get_current_measurement(OBJECTS_NAME,
                                                                           m_name)
                        self.assertTrue(np.all(m == expected.astype(int)))
                        self.assertTrue(m_name in [column[1] for column in columns])
                        self.assertTrue(m_name in ["_".join((C.M_CATEGORY, name))
                                                   for name in names])
                    image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
                    self.assertTrue(np.all(image[labels == 0, :] == 0))
                    colors = image[(labels > 0) & (m[labels - 1] == 1), :]
                    if colors.shape[0] > 0:
                        self.assertTrue(all([np.all(colors[:, i] == colors[0, i])
                                             for i in range(3)]))

    def test_03_04_nans(self):
        # Test for NaN values in two measurements.
        #
        labels = np.zeros((10, 15), int)
        labels[3:5, 3:5] = 1
        labels[6:8, 3:5] = 3
        labels[3:5, 6:8] = 4
        labels[6:8, 6:8] = 5
        labels[3:5, 10:12] = 2

        m1 = np.array((1, 2, np.NaN, 1, np.NaN))
        m2 = np.array((1, 2, 1, np.NaN, np.NaN))
        for leave_last_out in (False, True):
            end = np.max(labels) - 1 if leave_last_out else np.max(labels)
            workspace, module = self.make_workspace(
                    labels,
                    C.BY_TWO_MEASUREMENTS,
                    m1[:end], m2[:end])
            self.assertTrue(isinstance(module, C.ClassifyObjects))
            module.first_threshold_method.value = C.TM_MEAN
            module.first_threshold.value = 2
            module.second_threshold_method.value = C.TM_MEAN
            module.second_threshold.value = 2
            module.wants_image.value = True
            module.wants_custom_names.value = False
            module.run(workspace)
            f_names = ("Measurement1_low_Measurement2_low",
                       "Measurement1_low_Measurement2_high",
                       "Measurement1_high_Measurement2_low",
                       "Measurement1_high_Measurement2_high")
            m_names = ["_".join((C.M_CATEGORY, name))
                       for name in f_names]
            m = workspace.measurements
            for m_name, expected in zip(
                    m_names,
                    [np.array((1, 0, 0, 0, 0)),
                     np.array((0, 0, 0, 0, 0)),
                     np.array((0, 0, 0, 0, 0)),
                     np.array((0, 1, 0, 0, 0))]):
                values = m[OBJECTS_NAME, m_name]
                np.testing.assert_array_equal(values, expected)

    def test_03_05_nan_offset_by_1(self):
        # Regression test of 1636
        labels = np.zeros((10, 15), int)
        labels[3:5, 3:5] = 1
        labels[6:8, 3:5] = 2

        m1 = np.array((4, np.NaN))
        m2 = np.array((4, 4))
        workspace, module = self.make_workspace(
                labels,
                C.BY_TWO_MEASUREMENTS,
                m1, m2)
        self.assertTrue(isinstance(module, C.ClassifyObjects))
        module.first_threshold_method.value = C.TM_MEAN
        module.first_threshold.value = 2
        module.second_threshold_method.value = C.TM_MEAN
        module.second_threshold.value = 2
        module.wants_image.value = True
        module.wants_custom_names.value = False
        module.run(workspace)
        image = workspace.image_set.get_image(IMAGE_NAME).pixel_data
        colors = module.get_colors(4)
        reverse = np.zeros(image.shape[:2], int)
        for idx, color in enumerate(colors):
            reverse[
                np.all(image == color[np.newaxis, np.newaxis, :3], 2)] = idx
        self.assertTrue(np.all(reverse[labels == 1] == 4))
