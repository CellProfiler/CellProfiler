'''test_flagimages.py - Test the FlagImages module
'''

import base64
import contextlib
import os
import tempfile
import unittest
import zlib
from StringIO import StringIO

import PIL.Image as PILImage
import numpy as np
import scipy.ndimage

from cellprofiler.preferences import set_headless
from .test_filterobjects import make_classifier_pickle

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.image as cpi
import cellprofiler.workspace as cpw
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.preferences as cpprefs

import cellprofiler.modules.flagimage as F


def image_measurement_name(index):
    return "Metadata_ImageMeasurement_%d" % index


OBJECT_NAME = "object"


def object_measurement_name(index):
    return "Measurement_Measurement_%d" % index


MEASUREMENT_CATEGORY = "MyCategory"
MEASUREMENT_FEATURE = "MyFeature"
MEASUREMENT_NAME = '_'.join((MEASUREMENT_CATEGORY, MEASUREMENT_FEATURE))


class TestFlagImages(unittest.TestCase):
    def test_01_01_00_load_matlab_v1(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUnArylRwLE1XMDJUMDSwMjWzMrZQMDIwsFQgGTAwevryMzAwxDMx'
                'MFTMeTvdP/+ygciBuSZzrVpubRSe1fJEWXyBsNOjE9cCAzNfh1n5Pmq5J2W+'
                'ecqR0Ec5uup2bPNZCp8kF9xoPxg32cwwU3bTmpgNKa7nK99//3elzm82x8J4'
                '3ZZtRzYrLut8tljutFuWV6WHxfJ9wtcmts479tr+6J6GhkNLvfUO9Hp0iClH'
                'vv+ZXGY3WyduroF1kaf8ni2bJP6e2K3iWGHlx/P4enqJ1x3Fv04BZSpZ8idN'
                'eTU/J/+esX6d3h+jdT9yu37mVkeyO6909j1z98quK/Xt036F+d0tC/pl/dFG'
                'Pfjfo78ibjLVB9zDJ30OO5ypIaM5n+2KpcjSdYf2rd243eibmJWVxWvRk/XR'
                'H7WfNjcc+MCldvJ9RMvniIceO583OrMJ7phyXujcstcKxZpC/5WPrp/Sf5Pj'
                'jp60a+JHqyeFfRLfJI8HekyTiZVLa0te9Les8n9tm+2rYP1Pq84vzrExOHts'
                '8Wp33cMrF5q/LVtaWXOhYB7Q+vIj9x6xMH87ySu3+8GC9sehU5f3Tb7O78/B'
                'Lm1RY5dm7no+2PvhlktzHJdrb779pDLodxPPA7uy0KNr7Uo/qhSfM73wWnRO'
                'e9f6M/uW7fNdv1JXj8nn6VSBZwXdz9Z/O2q3elmfvvc1w/f/SphmtD56pWC5'
                'Z9HHlX/fTV5xa/+a6VL9/7mKLOPPfr2f/rLqbvn+e7lq6gkXl1SdW7W8b/51'
                '++k1On1MM9c8/v89fKJ98le7kO/FP987b6he2RbWcjnEImha+SnZ5aebHJdH'
                '+f2Kyc98lcCVuf2ixlX774GW9aWHfzyubzx3rqt0n/3uu7vvzv79t/Hfu+91'
                '+9zsJWdf+s9r7/LJDgCl2lmX')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, F.FlagImage))
        self.assertEqual(len(module.flags), 1)
        flag = module.flags[0]
        self.assertTrue(isinstance(flag, cps.SettingsGroup))
        self.assertEqual(len(flag.measurement_settings), 1)
        ms = flag.measurement_settings[0]
        self.assertTrue(isinstance(ms, cps.SettingsGroup))
        self.assertEqual(ms.measurement.value, "AreaShape_Area_OrigBlue")
        self.assertEqual(ms.source_choice, F.S_IMAGE)
        self.assertTrue(ms.wants_minimum.value)
        self.assertEqual(ms.minimum_value, 10)
        self.assertFalse(ms.wants_maximum.value)
        self.assertEqual(flag.category, "Metadata")
        self.assertEqual(flag.feature_name, "LowArea")

        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, F.FlagImage))
        self.assertEqual(len(module.flags), 1)
        flag = module.flags[0]
        self.assertTrue(isinstance(flag, cps.SettingsGroup))
        self.assertEqual(len(flag.measurement_settings), 1)
        ms = flag.measurement_settings[0]
        self.assertTrue(isinstance(ms, cps.SettingsGroup))
        self.assertEqual(ms.measurement.value, "ImageQuality_LocalFocus_OrigBlue")
        self.assertEqual(ms.source_choice, F.S_IMAGE)
        self.assertFalse(ms.wants_minimum.value)
        self.assertTrue(ms.wants_maximum.value)
        self.assertEqual(ms.maximum_value.value, 500)
        self.assertEqual(flag.category, "Metadata")
        self.assertEqual(flag.feature_name, "QCFlag")

    def test_01_01_02_load_matlab_v2(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUvDNz1PwSsxTMDBRMLC0MjW0MjJVMDIwNFAgGTAwevryMzAw5DAx'
                'MFTMeRvmmH/ZQKTsUskWmVVGnJyKKkVPDt1wWCArqSs6NWPXxiwh9/WFzUuP'
                'rBNKz7b4ONui8En7B+Yfkjcib6nF3bKIvGGRcHLTvM/3vt/7nbN/Dx9Dwn7u'
                'A3zx0TWvHmSZCEd6+yrOVzpmUNikZnBgccp/9fnpF45/uOOafoE3kCEseae9'
                'irFV57PV/spZ7XqbFkwPe3aYa2/xE/bHNy8Z/KzpET9hezRe8ETf0lvqJY3M'
                'Pp5J8x+c3q29f4pnnXXNv7z1lRdU5KRq3i371vovesM186KK18u+7D45ZwFr'
                '9b86xuVnvijMrhO0+M1q0cvjx/xl7eIb/xMbY7lT/nJXvkl+LmuvJLrf7dg9'
                'hr+LHy/smjtt4+WkWtF9t4L2OMkJtXk0RE55tFcrPvWxXUDPbuWjuk84Uy/4'
                'sq5vKo4/JFg0f1aLkTHPhKp/s3/ob7psXGZlURGZfye0QCerakaC/pcrn6d/'
                'vlH9N/8Jo/hNm3dC6vrr+Vkt+FVkFL/XsEgaLfS8sNSyeCHzP7m42Eqxq2er'
                'hfom6JZYClwMWbt78Ysl6gx5m/lOfI2+Wh8nK+eafF0u5fziHNmqqTcYM9d2'
                '81euLfpvsPEj6zbnF7NuNJ9b5X3isH50u04Ur/uzwvVK/ZXPX/0w4Fp2wdWy'
                'YtXHk/9vn5+zdfcT7772PQ07Le5/efs3btb62feu/zCytON9/vVxTlT3h8+/'
                'LpvuOzmfaVb+4f/v18vV6+Z+tp3yvvhq9eeJTtUztDK0vn5c2KL8RLDJ8cVi'
                'Ca27s8PfTj6qr8omHfB/Sfnf2vciHxYWZ17PuL9iv/7r9StfX//VoGStfLHy'
                '85ffk+P71//5x/D03N+bYn+f7tD9a7/C4c00AGyvW4A=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, F.FlagImage))
        self.assertEqual(len(module.flags), 1)
        flags = module.flags[0]
        self.assertEqual(len(flags.measurement_settings), 1)
        ms = flags.measurement_settings[0]
        self.assertEqual(ms.measurement.value, "Intensity_TotalIntensity_OrigBlue_12")
        self.assertEqual(ms.source_choice, F.S_IMAGE)
        self.assertTrue(ms.wants_minimum.value)
        self.assertFalse(ms.wants_maximum.value)
        self.assertEqual(ms.minimum_value.value, .5)
        self.assertEqual(flags.category, "Metadata")
        self.assertEqual(flags.feature_name, "MyQCFlag")

    def test_01_02_load_v1(self):
        data = ('eJztW0Fv2zYUphInaFZgyC5r1124Q4FkqwXJXVAnGFJ59ooYqzOvCboVRdcx'
                'Nm1zoCRDorp4Q4H+rB33k3bccaIiWxIrW7IrK3YmAYT9KH7vfXx8fKREqFU7'
                'f1r7Fh7ICmzVzss9QjFsU8R6pqUfQYM9gHULI4a70DSO4BOLwJrThxUVquqR'
                'Wj16+AhWFOUQLHZJzdbH7o9SBWDb/b3llg3/1pYvS6HC5TPMGDH69hYogbt+'
                '/d9ueY4sgi4ofo6og+3AxLi+afTM89Fwcqtldh2KT5Eebuxep45+gS37h94Y'
                '6N9uk0tMz8gfWOjCuNkz/IbYxDR8vK9frJ3YNZlgl/vh33uBHyTBD5tuuR+q'
                '5+1PQNC+FOO3T0Ltd32ZGF3yhnQdRCHRUX/CwhuHBH2bEX2boHFaS4WTIjgJ'
                'VHx7WgJuV+DPyzm+ZOXvLlGHQR2xziCN/Y2Ing1waqbrr8j7YUo/LYorRXAl'
                'l6eBOa6agLsFon7icgsz1EUMpfHzRwKeyw0TGiaDju0HfBo9O4IeLtdHzBxS'
                'ZOsg0JPUn21BD5d/rD+hqJ/Oj4uO9zRcUr/j4pSzhaQHkTGCPURomvl6R9DD'
                '5QbuIYcy2OSTFTaIhTvMtEZLjactgQeXPfvgfT9uC/jxNcbv+L9Z4ebp57z5'
                '7IWbDT+EZzvB3j0Q9SuXmwbDhk3Y6HULXQYCT62p+58m3rP0d9bza1k8xfFV'
                'ZGUl43daHC6S9+sDZBiYVvKcb4p8eJDV/mEenlnmsTTruBqDW4U8lgaXZR6D'
                'IOpXLofyGDECYbL8c73vEvR+L+jl8i97j9vf8AcSfCx/tf+aSz9hSp+Zvx+/'
                'rJXbr/bHNXWTOrpx/FIpH776U31QeXvV+Iy4SK9yP3Yc8pjHgwRcVeg3lzn3'
                'FxhZfoe+frtf5lUt02ADv67i1zXQKKjJd94rBx+Yp9TrWKe0BFzafWxecaTG'
                'rFvzPGetan7NOm8tau9dAm7V81JSvv5c4M/lUL7GyHh/43mT8lbW+/z89qvq'
                'WvBcH38eriTPZe4zs8z3cet4+D1T3nxPEvh+KvDl8gnpDyZr+STtXY+/tQT+'
                'ad8rrVqc3MT3R3nw/DmB5xcg6lcuT1vHl7Vfzfs90irwXPf3SKvGUxx3uXqF'
                '2/0swEkCLu78a1l5N+48xDss61umM8zWPzcRp4HZ/r0Nov7lsnnxG+6wwMGr'
                'yLuIi/9nXBT9LXAFbv1xGpg9H+Oet4L8DonRxcN16m+RtwpcgZuO08DsOC/y'
                'wXrhNDB7PIu8VeAKXIErcAWuwE3HaSFcsY4WuAJ3vbiBFODEcxEuh899ePtf'
                'wez5+yWIzl8udzClQ8vk339Zsu59pGTL1ETdq6+E5Kfu32bogyFuZ5hgRxPs'
                'aNPs6BjZjoU9U2R8hCm3rmo9q5ODzTTnpHuC3b1pdnsU9T2jMj9W9wyJ47QT'
                'oz/s7w1XurN9f+b4iuMajPc/jxexV5Ikz174HP52Aq4U4sQvjv8LzBdXezPa'
                'j/uYV/v/ACB4EDk=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, F.FlagImage))
        #
        # The module defines two flags:
        #
        # Metadata_QCFlag: flag if any fail
        #     Intensity_MaxIntensity_DNA: max = .95
        #     Intensity_MinIntensity_Cytoplasm: min = .05
        #     Intensity_MeanIntensity_DNA: min=.1, max=.9
        # Metadata_HighCytoplasmIntensity
        #     Intensity_MeanIntensity_Cytoplasm: max = .8
        #
        expected = (("QCFlag", F.C_ANY,
                     (("Intensity_MaxIntensity_DNA", None, .95),
                      ("Intensity_MinIntensity_Cytoplasm", .05, None),
                      ("Intensity_MeanIntensity_DNA", .1, .9))),
                    ("HighCytoplasmIntensity", None,
                     (("Intensity_MeanIntensity_Cytoplasm", None, .8),)))
        self.assertEqual(len(expected), module.flag_count.value)
        for flag, (feature_name, combine, measurements) \
                in zip(module.flags, expected):
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            self.assertEqual(flag.category, "Metadata")
            self.assertEqual(flag.feature_name, feature_name)
            if combine is not None:
                self.assertEqual(flag.combination_choice, combine)
            self.assertEqual(len(measurements), flag.measurement_count.value)
            for measurement, (measurement_name, min_value, max_value) \
                    in zip(flag.measurement_settings, measurements):
                self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                self.assertEqual(measurement.source_choice, F.S_IMAGE)
                self.assertEqual(measurement.measurement, measurement_name)
                self.assertEqual(measurement.wants_minimum.value, min_value is not None)
                if measurement.wants_minimum.value:
                    self.assertAlmostEqual(measurement.minimum_value.value,
                                           min_value)
                self.assertEqual(measurement.wants_maximum.value, max_value is not None)
                if measurement.wants_maximum.value:
                    self.assertAlmostEqual(measurement.maximum_value.value,
                                           max_value)

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9889

FlagImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Hidden:2
    Hidden:3
    Name the flag\'s category:Metadata
    Name the flag:QCFlag
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:No
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MaxIntensity_DNA
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:0.95
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MinIntensity_Cytoplasm
    Flag images based on low values?:Yes
    Minimum value:0.05
    Flag images based on high values?:No
    Maximum value:1.0
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_DNA
    Flag images based on low values?:Yes
    Minimum value:0.1
    Flag images based on high values?:Yes
    Maximum value:0.9
    Hidden:1
    Name the flag\'s category:Metadata
    Name the flag:HighCytoplasmIntensity
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:Yes
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_Cytoplasm
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:.8
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, F.FlagImage))
        expected = (("QCFlag", F.C_ANY, False,
                     (("Intensity_MaxIntensity_DNA", None, .95),
                      ("Intensity_MinIntensity_Cytoplasm", .05, None),
                      ("Intensity_MeanIntensity_DNA", .1, .9))),
                    ("HighCytoplasmIntensity", None, True,
                     (("Intensity_MeanIntensity_Cytoplasm", None, .8),)))
        self.assertEqual(len(expected), module.flag_count.value)
        for flag, (feature_name, combine, skip, measurements) \
                in zip(module.flags, expected):
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            self.assertEqual(flag.category, "Metadata")
            self.assertEqual(flag.feature_name, feature_name)
            self.assertEqual(flag.wants_skip, skip)
            if combine is not None:
                self.assertEqual(flag.combination_choice, combine)
            self.assertEqual(len(measurements), flag.measurement_count.value)
            for measurement, (measurement_name, min_value, max_value) \
                    in zip(flag.measurement_settings, measurements):
                self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                self.assertEqual(measurement.source_choice, F.S_IMAGE)
                self.assertEqual(measurement.measurement, measurement_name)
                self.assertEqual(measurement.wants_minimum.value, min_value is not None)
                if measurement.wants_minimum.value:
                    self.assertAlmostEqual(measurement.minimum_value.value,
                                           min_value)
                self.assertEqual(measurement.wants_maximum.value, max_value is not None)
                if measurement.wants_maximum.value:
                    self.assertAlmostEqual(measurement.maximum_value.value,
                                           max_value)

    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120306205005

FlagImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Hidden:2
    Hidden:3
    Name the flag\'s category:Metadata
    Name the flag:QCFlag
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:No
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MaxIntensity_DNA
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:0.95
    Rules file location:Default Input Folder\x7CNone
    Rules file name:foo.txt
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MinIntensity_Cytoplasm
    Flag images based on low values?:Yes
    Minimum value:0.05
    Flag images based on high values?:No
    Maximum value:1.0
    Rules file location:Default Input Folder\x7CNone
    Rules file name:bar.txt
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_DNA
    Flag images based on low values?:Yes
    Minimum value:0.1
    Flag images based on high values?:Yes
    Maximum value:0.9
    Rules file location:Default Input Folder\x7CNone
    Rules file name:baz.txt
    Hidden:1
    Name the flag\'s category:Metadata
    Name the flag:HighCytoplasmIntensity
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:Yes
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_Cytoplasm
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:.8
    Rules file location:Default Input Folder\x7CNone
    Rules file name:dunno.txt
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, F.FlagImage))
        expected = (("QCFlag", F.C_ANY, False,
                     (("Intensity_MaxIntensity_DNA", None, .95, "foo.txt"),
                      ("Intensity_MinIntensity_Cytoplasm", .05, None, "bar.txt"),
                      ("Intensity_MeanIntensity_DNA", .1, .9, "baz.txt"))),
                    ("HighCytoplasmIntensity", None, True,
                     (("Intensity_MeanIntensity_Cytoplasm", None, .8, "dunno.txt"),)))
        self.assertEqual(len(expected), module.flag_count.value)
        for flag, (feature_name, combine, skip, measurements) \
                in zip(module.flags, expected):
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            self.assertEqual(flag.category, "Metadata")
            self.assertEqual(flag.feature_name, feature_name)
            self.assertEqual(flag.wants_skip, skip)
            if combine is not None:
                self.assertEqual(flag.combination_choice, combine)
            self.assertEqual(len(measurements), flag.measurement_count.value)
            for measurement, (measurement_name, min_value, max_value, rules_file) \
                    in zip(flag.measurement_settings, measurements):
                self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                self.assertEqual(measurement.source_choice, F.S_IMAGE)
                self.assertEqual(measurement.measurement, measurement_name)
                self.assertEqual(measurement.wants_minimum.value, min_value is not None)
                if measurement.wants_minimum.value:
                    self.assertAlmostEqual(measurement.minimum_value.value,
                                           min_value)
                self.assertEqual(measurement.wants_maximum.value, max_value is not None)
                if measurement.wants_maximum.value:
                    self.assertAlmostEqual(measurement.maximum_value.value,
                                           max_value)
                self.assertEqual(measurement.rules_file_name, rules_file)
                self.assertEqual(measurement.rules_class, "1")

    def test_01_05_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120306205005

FlagImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Hidden:2
    Hidden:3
    Name the flag\'s category:Metadata
    Name the flag:QCFlag
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:No
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MaxIntensity_DNA
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:0.95
    Rules file location:Default Input Folder\x7CNone
    Rules file name:foo.txt
    Rules class:4
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MinIntensity_Cytoplasm
    Flag images based on low values?:Yes
    Minimum value:0.05
    Flag images based on high values?:No
    Maximum value:1.0
    Rules file location:Default Input Folder\x7CNone
    Rules file name:bar.txt
    Rules class:2
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_DNA
    Flag images based on low values?:Yes
    Minimum value:0.1
    Flag images based on high values?:Yes
    Maximum value:0.9
    Rules file location:Default Input Folder\x7CNone
    Rules file name:baz.txt
    Rules class:1
    Hidden:1
    Name the flag\'s category:Metadata
    Name the flag:HighCytoplasmIntensity
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:Yes
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_Cytoplasm
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:.8
    Rules file location:Default Input Folder\x7CNone
    Rules file name:dunno.txt
    Rules class:3
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, F.FlagImage))
        expected = (("QCFlag", F.C_ANY, False,
                     (("Intensity_MaxIntensity_DNA", None, .95, "foo.txt", "4"),
                      ("Intensity_MinIntensity_Cytoplasm", .05, None, "bar.txt", "2"),
                      ("Intensity_MeanIntensity_DNA", .1, .9, "baz.txt", "1"))),
                    ("HighCytoplasmIntensity", None, True,
                     (("Intensity_MeanIntensity_Cytoplasm", None, .8, "dunno.txt", "3"),)))
        self.assertEqual(len(expected), module.flag_count.value)
        for flag, (feature_name, combine, skip, measurements) \
                in zip(module.flags, expected):
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            self.assertEqual(flag.category, "Metadata")
            self.assertEqual(flag.feature_name, feature_name)
            self.assertEqual(flag.wants_skip, skip)
            if combine is not None:
                self.assertEqual(flag.combination_choice, combine)
            self.assertEqual(len(measurements), flag.measurement_count.value)
            for measurement, (
                    measurement_name, min_value, max_value, rules_file, rules_class) \
                    in zip(flag.measurement_settings, measurements):
                self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                self.assertEqual(measurement.source_choice, F.S_IMAGE)
                self.assertEqual(measurement.measurement, measurement_name)
                self.assertEqual(measurement.wants_minimum.value, min_value is not None)
                if measurement.wants_minimum.value:
                    self.assertAlmostEqual(measurement.minimum_value.value,
                                           min_value)
                self.assertEqual(measurement.wants_maximum.value, max_value is not None)
                if measurement.wants_maximum.value:
                    self.assertAlmostEqual(measurement.maximum_value.value,
                                           max_value)
                self.assertEqual(measurement.rules_file_name, rules_file)
                self.assertEqual(measurement.rules_class, rules_class)

    def make_workspace(self, image_measurements, object_measurements):
        '''Make a workspace with a FlagImage module and the given measurements

        image_measurements - a sequence of single image measurements. Use
                             image_measurement_name(i) to get the name of
                             the i th measurement
        object_measurements - a seequence of sequences of object measurements.
                              These are stored under object, OBJECT_NAME with
                              measurement name object_measurement_name(i) for
                              the i th measurement.

        returns module, workspace
        '''
        module = F.FlagImage()
        measurements = cpmeas.Measurements()
        for i in range(len(image_measurements)):
            measurements.add_image_measurement(image_measurement_name(i),
                                               image_measurements[i])
        for i in range(len(object_measurements)):
            measurements.add_measurement(OBJECT_NAME,
                                         object_measurement_name(i),
                                         np.array(object_measurements[i]))
        flag = module.flags[0]
        self.assertTrue(isinstance(flag, cps.SettingsGroup))
        flag.category.value = MEASUREMENT_CATEGORY
        flag.feature_name.value = MEASUREMENT_FEATURE
        module.module_num = 1
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  measurements, image_set_list)
        return module, workspace

    @contextlib.contextmanager
    def make_classifier(self, module, answer,
                        classes = None,
                        class_names = None,
                        rules_classes = None,
                        name = "Classifier",
                        n_features = 1):
        assert isinstance(module, F.FlagImage)
        feature_names = [image_measurement_name(i) for i in range(n_features)]
        if classes is None:
            classes = np.arange(1, max(3, answer+1))
        if class_names is None:
            class_names = ["Class%d" for _ in classes]
        if rules_classes is None:
            rules_classes = [class_names[0]]
        s = make_classifier_pickle(
            np.array([answer]), classes, class_names, name, feature_names)
        fd, filename = tempfile.mkstemp(".model")
        os.write(fd, s)
        os.close(fd)
        measurement = module.flags[0].measurement_settings[0]
        measurement.source_choice.value = F.S_CLASSIFIER
        measurement.rules_directory.set_custom_path(
            os.path.dirname(filename))
        measurement.rules_file_name.value = os.path.split(filename)[1]
        measurement.rules_class.value = rules_classes
        yield
        try:
            os.remove(filename)
        except:
            pass

    def test_02_01_positive_image_measurement(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        self.assertTrue(isinstance(flag, cps.SettingsGroup))
        measurement = flag.measurement_settings[0]
        self.assertTrue(isinstance(measurement, cps.SettingsGroup))
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = False
        measurement.wants_maximum.value = True
        measurement.maximum_value.value = .95
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
        self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 1)
        self.assertEqual(workspace.disposition, cpw.DISPOSITION_CONTINUE)

    def test_02_02_negative_image_measurement(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        self.assertTrue(isinstance(flag, cps.SettingsGroup))
        measurement = flag.measurement_settings[0]
        self.assertTrue(isinstance(measurement, cps.SettingsGroup))
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = True
        measurement.minimum_value.value = .1
        measurement.wants_maximum.value = False
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
        self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 0)

    def test_03_00_no_ave_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[]])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            measurement = flag.measurement_settings[0]
            self.assertTrue(isinstance(measurement, cps.SettingsGroup))
            measurement.source_choice.value = F.S_AVERAGE_OBJECT
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "minimum":
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = .3
            else:
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .2
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 1)

    def test_03_01_positive_ave_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[.1, .2, .3, .4]])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            measurement = flag.measurement_settings[0]
            self.assertTrue(isinstance(measurement, cps.SettingsGroup))
            measurement.source_choice.value = F.S_AVERAGE_OBJECT
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "minimum":
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = .3
            else:
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .2
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 1)

    def test_03_02_negative_ave_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[.1, .2, .3, .4]])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            measurement = flag.measurement_settings[0]
            self.assertTrue(isinstance(measurement, cps.SettingsGroup))
            measurement.source_choice.value = F.S_AVERAGE_OBJECT
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "minimum":
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = .2
            else:
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .3
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 0)

    def test_04_00_no_object_measurements(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[]])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            measurement = flag.measurement_settings[0]
            self.assertTrue(isinstance(measurement, cps.SettingsGroup))
            measurement.source_choice.value = F.S_ALL_OBJECTS
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "maximum":
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .35
            else:
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = .15
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 1)

    def test_04_01_positive_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[.1, .2, .3, .4]])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            measurement = flag.measurement_settings[0]
            self.assertTrue(isinstance(measurement, cps.SettingsGroup))
            measurement.source_choice.value = F.S_ALL_OBJECTS
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "maximum":
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .35
            else:
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = .15
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 1)

    def test_04_02_negative_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[.1, .2, .3, .4]])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            measurement = flag.measurement_settings[0]
            self.assertTrue(isinstance(measurement, cps.SettingsGroup))
            measurement.source_choice.value = F.S_ALL_OBJECTS
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "maximum":
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .45
            else:
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = .05
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 0)

    def test_05_01_two_measurements_any(self):
        for measurements, expected in (((0, 0), 0),
                                       ((0, 1), 1),
                                       ((1, 0), 1),
                                       ((1, 1), 1)):
            module, workspace = self.make_workspace(measurements, [])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            flag.combination_choice.value = F.C_ANY
            module.add_measurement(flag)
            for i in range(2):
                measurement = flag.measurement_settings[i]
                self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                measurement.measurement.value = image_measurement_name(i)
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .5
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME),
                             expected)

    def test_05_02_two_measurements_all(self):
        for measurements, expected in (((0, 0), 0),
                                       ((0, 1), 0),
                                       ((1, 0), 0),
                                       ((1, 1), 1)):
            module, workspace = self.make_workspace(measurements, [])
            flag = module.flags[0]
            self.assertTrue(isinstance(flag, cps.SettingsGroup))
            flag.combination_choice.value = F.C_ALL
            module.add_measurement(flag)
            for i in range(2):
                measurement = flag.measurement_settings[i]
                self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                measurement.measurement.value = image_measurement_name(i)
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = .5
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpmeas.Measurements))
            self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
            self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME),
                             expected)

    def test_06_01_get_measurement_columns(self):
        module = F.FlagImage()
        module.add_flag()
        module.flags[0].category.value = 'Foo'
        module.flags[0].feature_name.value = 'Bar'
        module.flags[1].category.value = 'Hello'
        module.flags[1].feature_name.value = 'World'
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 2)
        self.assertTrue(all([column[0] == cpmeas.IMAGE and
                             column[1] in ("Foo_Bar", "Hello_World") and
                             column[2] == cpmeas.COLTYPE_INTEGER
                             for column in columns]))
        self.assertNotEqual(columns[0][1], columns[1][1])
        categories = module.get_categories(None, 'foo')
        self.assertEqual(len(categories), 0)
        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 2)
        self.assertTrue('Foo' in categories)
        self.assertTrue('Hello' in categories)
        self.assertEqual(len(module.get_measurements(None, cpmeas.IMAGE, 'Whatever')), 0)
        for category, feature in (('Foo', 'Bar'), ('Hello', 'World')):
            features = module.get_measurements(None, cpmeas.IMAGE, category)
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], feature)

    def test_07_01_skip(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        self.assertTrue(isinstance(flag, cps.SettingsGroup))
        flag.wants_skip.value = True
        measurement = flag.measurement_settings[0]
        self.assertTrue(isinstance(measurement, cps.SettingsGroup))
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = False
        measurement.wants_maximum.value = True
        measurement.maximum_value.value = .95
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
        self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 1)
        self.assertEqual(workspace.disposition, cpw.DISPOSITION_SKIP)

    def test_07_02_dont_skip(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        self.assertTrue(isinstance(flag, cps.SettingsGroup))
        flag.wants_skip.value = True
        measurement = flag.measurement_settings[0]
        self.assertTrue(isinstance(measurement, cps.SettingsGroup))
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = True
        measurement.minimum_value.value = .1
        measurement.wants_maximum.value = False
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
        self.assertEqual(m.get_current_image_measurement(MEASUREMENT_NAME), 0)
        self.assertEqual(workspace.disposition, cpw.DISPOSITION_CONTINUE)

    def test_08_01_filter_by_rule(self):
        rules_file_contents = "IF (%s > 2.0, [1.0,-1.0], [-1.0,1.0])\n" % (
            '_'.join((cpmeas.IMAGE, image_measurement_name(0))))
        rules_path = tempfile.mktemp()
        rules_dir, rules_file = os.path.split(rules_path)
        fd = open(rules_path, 'wt')
        try:
            fd.write(rules_file_contents)
            fd.close()
            for value, choice, expected in ((1.0, 1, 0), (3.0, 1, 1),
                                            (1.0, 2, 1), (3.0, 2, 0)):
                module, workspace = self.make_workspace([value], [])
                flag = module.flags[0]
                self.assertTrue(isinstance(flag, cps.SettingsGroup))
                flag.wants_skip.value = False
                measurement = flag.measurement_settings[0]
                self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                measurement.source_choice.value = F.S_RULES
                measurement.rules_file_name.value = rules_file
                measurement.rules_directory.dir_choice = cpprefs.ABSOLUTE_FOLDER_NAME
                measurement.rules_directory.custom_path = rules_dir
                measurement.rules_class.set_value([str(choice)])
                module.run(workspace)
                m = workspace.measurements
                self.assertTrue(isinstance(m, cpmeas.Measurements))
                self.assertIn(MEASUREMENT_NAME, m.get_feature_names(cpmeas.IMAGE))
                self.assertEqual(
                        m.get_current_image_measurement(MEASUREMENT_NAME), expected)
        finally:
            os.remove(rules_path)

    def test_08_02_filter_by_3class_rule(self):
        f = '_'.join((cpmeas.IMAGE, image_measurement_name(0)))
        rules_file_contents = (
                                  "IF (%(f)s > 2.0, [1.0,-1.0,-1.0], [-0.5,0.5,0.5])\n"
                                  "IF (%(f)s > 1.6, [0.5,0.5,-0.5], [-1.0,-1.0,1.0])\n") % locals()
        measurement_values = [1.5, 2.3, 1.8]
        expected_classes = ["3", "1", "2"]
        rules_path = tempfile.mktemp()
        rules_dir, rules_file = os.path.split(rules_path)
        fd = open(rules_path, 'wt')
        fd.write(rules_file_contents)
        fd.close()
        try:
            for rules_classes in (["1"], ["2"], ["3"],
                                  ["1", "2"], ["1", "3"], ["2", "3"]):
                for expected_class, measurement_value in zip(
                        expected_classes, measurement_values):
                    module, workspace = self.make_workspace([measurement_value], [])
                    flag = module.flags[0]
                    self.assertTrue(isinstance(flag, cps.SettingsGroup))
                    flag.wants_skip.value = False
                    measurement = flag.measurement_settings[0]
                    self.assertTrue(isinstance(measurement, cps.SettingsGroup))
                    measurement.source_choice.value = F.S_RULES
                    measurement.rules_file_name.value = rules_file
                    measurement.rules_directory.dir_choice = cpprefs.ABSOLUTE_FOLDER_NAME
                    measurement.rules_directory.custom_path = rules_dir
                    measurement.rules_class.set_value(rules_classes)

                    m = workspace.measurements
                    self.assertTrue(isinstance(m, cpmeas.Measurements))
                    module.run(workspace)
                    self.assertTrue(MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE))
                    value = m.get_current_image_measurement(MEASUREMENT_NAME)
                    expected_value = 1 if expected_class in rules_classes else 0
                    self.assertEqual(value, expected_value)
        finally:
            os.remove(rules_path)

    def test_09_01_classify_true(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(module, 1):
            module.run(workspace)
            m = workspace.measurements
            self.assertEqual(m[cpmeas.IMAGE, MEASUREMENT_NAME], 1)

    def test_09_02_classify_false(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(module, 2):
            module.run(workspace)
            m = workspace.measurements
            self.assertEqual(m[cpmeas.IMAGE, MEASUREMENT_NAME], 0)

    def test_09_03_classify_multiple_select_true(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(module, 2,
                                  classes = [1, 2, 3],
                                  class_names = ["Foo", "Bar", "Baz"],
                                  rules_classes = ["Bar", "Baz"]):
            module.run(workspace)
            m = workspace.measurements
            self.assertEqual(m[cpmeas.IMAGE, MEASUREMENT_NAME], 1)

    def test_09_04_classify_multiple_select_false(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(module, 2, 
                                  classes = [1, 2, 3],
                                  class_names = ["Foo", "Bar", "Baz"],
                                  rules_classes = ["Foo", "Baz"]):
            module.run(workspace)
            m = workspace.measurements
            self.assertEqual(m[cpmeas.IMAGE, MEASUREMENT_NAME], 0)

    def test_09_01_batch(self):
        orig_path = '/foo/bar'

        def fn_alter_path(path, **varargs):
            self.assertEqual(path, orig_path)
            return '/imaging/analysis'

        module = F.FlagImage()
        rd = module.flags[0].measurement_settings[0].rules_directory
        rd.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        rd.custom_path = orig_path
        module.prepare_to_create_batch(None, fn_alter_path)
        self.assertEqual(rd.custom_path, '/imaging/analysis')
