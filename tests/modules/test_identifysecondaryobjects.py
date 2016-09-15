"""test_identifysecondary - test the IdentifySecondary module
"""

import StringIO
import base64
import unittest
import zlib

import numpy as np

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.modules.identifysecondaryobjects as cpmi2
import cellprofiler.modules.identify as cpmi
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.image as cpi
import cellprofiler.object as cpo
import cellprofiler.measurement as cpm

INPUT_OBJECTS_NAME = "input_objects"
OUTPUT_OBJECTS_NAME = "output_objects"
NEW_OBJECTS_NAME = "new_objects"
IMAGE_NAME = "image"
THRESHOLD_IMAGE_NAME = "threshold"


class TestIdentifySecondaryObjects(unittest.TestCase):
    def test_01_01_load_matlab(self):
        u64data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBUaHUgRmViIDE5IDE1OjQyOjQyIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAyAIAAHic7VjLjtMwFHWfGqAalRmBYFZZshiGRIDEkqegErQVHY3E0k3cYOTEVeIMLV/DZ7BgMWu+iQVOm7SuJ2ncJM1ssGRF1/U5vvf4+sbpIQBAvw9Amz8PeK+DZWtFdk3ooT1CjGHX9lugCR5E41e8X0APwzFBF5AEyAerFo/33Ak9n09XP32iVkBQHzriZN76gTNGnj+YxMDo5yGeITLCPxDYbPG0z+gS+5i6ET7il0dX61ImrXvI+8/OWoeapEOD92NhPJz/EqznNxN06wrzu1E/RzP2+N0MmkxzIDO/hjwvMngOJJ7QHnjYfs2lVsG3JXx7obNJEF7GoWfg6xv4OvhuLHFF183CtyR8aL9BhPhADV9Ut6T1Df30mQ7U9v+OhA/toUen0IaMJ+ViXIXntsQT2m+p5lKmBX50IFT2sbHB0wBf+BkoQ8eq/JfzsE/z78OA+YH2ntAxJKv1q4ojbR+ycLUNXA0YJftdlo67xq+fGpXqn6ZjFq65gWsC/Uw3bjJvYt12PTeGnk+npwV1yoF7XmV+35J4QrvnMuT6mM2BOo+qPx8yeO5KPKGNXQtfYiuARMMOtFe3mDLjy1uH91W/k/x+FTDKL1DYFPxW9V/O67MC+m3zIw+f7cG5b0KCBJ6875Oi+Cz/VfNc1Y+8+VG2H7Ie7hNYyj4U1SHGX7W2f6eIdWOXfEzScVFkbI8GU3WepHsCHX9DJlsTiX7lqYMCn8ZrIprukS8r3qTvvLVuS7q0/W9LfHGL+eoCTn5m+VXWft5UfHl5/sdTTTxpz6zzdyT5G9o0YAS76NoB3GXdfT33FU+3ll7H5fdJ3vP+kUKrJ1wUVeK5J/GEds9CLsOT+dDDjnjnyVtvY74RMqlrQS/t/innp6xLnVtHx53SvvtU1mt1Gtf+N8nCNaM5f09+n/x5uFz3F9ht/x9tmR+3qub/A5FILLM='
        data = base64.b64decode(u64data)
        p = cpp.Pipeline()
        fd = StringIO.StringIO(data)
        p.load(fd)
        self.assertTrue(len(p.modules()) == 3)
        module = p.modules()[2]
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        self.assertEqual(module.primary_objects.value, "Nuclei")
        self.assertEqual(module.objects_name.value, "Cells")
        self.assertEqual(module.method.value, cpmi2.M_PROPAGATION)
        self.assertEqual(module.image_name.value, "OrigBlue")
        self.assertEqual(module.threshold_method.value, cpmi.TM_OTSU)
        self.assertEqual(module.threshold_scope, cpmi.TM_GLOBAL)
        self.assertEqual(module.threshold_correction_factor.value, 1)
        self.assertEqual(module.threshold_range.min, 0)
        self.assertEqual(module.threshold_range.max, 1)
        self.assertEqual(module.distance_to_dilate.value, 10)
        self.assertEqual(module.regularization_factor.value, 0.05)
        self.assertEqual(module.threshold_smoothing_choice, cpmi.TSM_NONE)

    def test_01_02_load_v2(self):
        data = ('eJztWt1u2zYUlh0nSFZ0a9OLDugNL5suNiQ3xtJgSO3G/fEWu0bjtS'
                'iKbmMk2uZAk4ZEpXGLAnuUPUYv+zi97CNUdCRLZpVIkeM/QAQI+Rzx'
                'O38kzxFl1Sutw8ojUCqooF5p5duYINAkkLeZ2dsDlG+DAxNBjgzA6B'
                '6oMwp+tylQd4FW3Ctpe6UdUFTVB0qylqnVf3Qun+8pyppzXXd61r21'
                '6tKZQBf0EeIc0461quSUn13+Z6e/hCaGxwS9hMRGlq/C49dom7UG/d'
                'GtOjNsghqwFxzstIbdO0am9bztAd3bTXyKyBF+jyQXvGEv0Am2MKMu'
                '3pUvc0d6GZf0ijh8ueXHISPFYcXpdwJ8Mf6Z4o/PhcTtZmD8DZfG1M'
                'An2LAhAbgHOyMrhDw1Qt7KmLwVpdqoDHG7Ebg1yY61YZx1gvCZ3nIE'
                '/oaEF72FTnn+8SnUOehBrnevwo4o/KqEF/QBIsSKGb/MGD6j3FeS69'
                'XU7R1ViRe/axJe0E2T9WEHcmdxDvlJ7agdHv5Zj+m/vH5eO6tvknUX'
                'hcuO4bJKg8Wz8zxckjg/55YNnhJ2DMkozlH79rYkR9BV1IY24aAmNi'
                '2oYhPpnJmDmcZdK6jf4dYknNc83IZ7nWX8wvKeWlCHbVtzfwTsmmf8'
                'wnC5MVxO2K5NYues4yXvH02Nl1/WlXE7BH3QhZQiUpxm3EJwpUnyUj'
                'kCtyH5Kega5YhamA8CcY6S84MkR9BVBijjwLaQL+eydUmLqT+uH5eN'
                'o5pwnzQYRZOsz6T6/ovA/SHFSdB/3X3Y/E08aKP9wi9bfwvqlfMo8Y'
                'K9239TyTffbnmcA0bsHt1/o+YfvP2gbRc/ng0+wg5yyNyKHWd5nn+N'
                'iTuvHnYjcLuS34IWtr9G0HQd2vm4lRcs52DBuy6v6PKqcOBzJsl/5Q'
                'hcWN1pvWNAJ9Cy3Cfkada9uPs4SR5/hXCnK45vJ+KgQnXv/DKJP9Oa'
                'h7A4PGEm6pjMpsb87L4Kfamd4Xl9EeyMU0cWwc4455lFsHN512dpLn'
                'aWI+y8roznRUG3uiZCgQo1D7sXsR5Nuy5fZT36f/Ny7/tm6efw5aBw'
                'tB9fTthzFDv+F+ncFzTr9RbQDzA1UH+K8uaZ3xYZV1auZv0ti78pLs'
                'WluMlx5QAu7v9Cft44S8/TrDebkn5BM5sTTNF3BWKZ4r5o85zWh+XC'
                'pfsmxaW4NF+muBSX4pYfd5rxcfJ7Kvk9qhj/T0BPWH66p4znJ0HriJ'
                'C+ycT3h2ahN/xIzioQBo2zr9QKh87PWuCDNaGnH6GnLOkpn6cHG4hy'
                '3B70TUebzVkPcqwXai636XArHjfp/54X6rWQzqgBzcFI55HHkedtI0'
                'RfMP5Zh/rpzs0L51ueZ3/+vz5Moi+7kh3qC37fcS0ClwvYJJrAf1Iu'
                't87uXjDe83FW478B0PjACw==')
        p = cpp.Pipeline()
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        p.load(fd)
        self.assertTrue(len(p.modules()) == 3)
        module = p.modules()[2]
        self.assertEqual(module.threshold_method.value, cpmi.TM_OTSU)
        self.assertEqual(module.threshold_scope, cpmi.TM_GLOBAL)
        self.assertEqual(module.two_class_otsu.value, cpmi.O_TWO_CLASS)
        self.assertEqual(module.use_weighted_variance.value,
                         cpmi.O_WEIGHTED_VARIANCE)
        self.assertFalse(module.wants_discard_edge)
        self.assertFalse(module.wants_discard_primary)

    def test_01_03_load_v3(self):
        data = ('eJztW+Fv2kYUPxKSLauUZdK0TtMq3Yd+aKLENbRR02hroaHZkBqCCmo3pVnr'
                '2Afcau4s+5zCpv0f+xP3sX/CfGCwOTmxMQac1lYs8o77vd+7d++9Oxv7pNx8'
                'UX4G9yUZnpSbey2sI1jXFdaiZvcQErYLj0ykMKRBSg5h00awglRY2IeFwuED'
                '5+8hLMryYxDvyFVPNp2P0s8ArDufXzrnivvVmivnfCeXG4gxTNrWGsiD7932'
                'j875SjGxcqGjV4puI8ujGLVXSYs2+8b4qxOq2TqqKV1/Z+eo2d0LZFqnrRHQ'
                '/bqOe0hv4L+QMIRRt5foEluYEhfv6hdbx7yUCbyNDv1wbDrmCPqfKUztNJgz'
                'A5Pt3G+9Hz2/5QS/5Z3zjq+d9/8VeP3zAX7+xtd/y5Ux0fAl1mxFh7irtMdW'
                'c31yiL7VCX2roFIrD3AHIbh1wQ4u12xVR3jIWwrBbwp4Lh9jnSETadPo2RL0'
                '8LOJemzveU9RGezyqUliPGH4NQHP5SOk6xaI74/nPUMh2qQ/wuYzN6EnBx6A'
                '+PYX5N2HckT7bwl4LtdNaihthTnJNWjnet6F6JEEPdLYD5Be/IlUZsGLPlSg'
                'ZSAVt7BT88ggbyFtQYPnvxXN3q8EHi5XKCSUQdty8zhO/vzuZF8U/g2Bn8tH'
                'fUYNXbG6IDr/VfMdhluZwK2AGp0NFzbeoPg4ZZYNf9HphaJHHu9V/GF187bA'
                'z+UKaim2zmCVF01YwaYTXtTszzTv0+IKkpzYfK0LuNExwm24n0nOV5z1Spbk'
                'wbFbcP/x2RWm7ztBH5cbHdMm7xEZ1slTm+mYIGu2/J3XPOYncHnui8Is8ziv'
                'fJl23qa1oyDPFr/zqm9R5icibj+p8SWZ10HrTpUwRCzM+j49ceKjgVRKNMXs'
                'eyk483ji7Ns0WWK4law/pq0HckA9SDK+r9q3J5GHUeK7Rgma5/jEfU0hJu5R'
                'RFzS62sS/kxyvxm0njc/UKg6+03LzdRZxtsJ4X8k8HP5j3tP6z/xGxnoibSz'
                '/ZZLr51Lpidn5b36+Zm89/j87+I/228t/kUDO70GbduRxvu1wMfluulcHvuq'
                'U9w69xrhdoffbrnkNxaIOrrsn8V/v4XYcVewg8vSztmbN/fPuXsqrhPHDS9t'
                'wuW7QXYlGVdB11HH1ERtk9pEm90vsep/0av/i9r/BeHmvW7HuV5My3inrePF'
                'mOMbrcNh/o16Pypt68E81+U01/9lruefQl1ftB8WtQ+Wpf2l2znv+zBJ7tsW'
                'jUvLfitt8zzvfVTa8/ZTq0vi/mV/SXb++4OHywm4oN8fFxnfgx8reYAb0fUE'
                '1cPhD0OeoiT1LLLO+fghJhoybpC+m5i3acaVQDL5dFPGm+E+T1wJXB/nW2Ay'
                'zvnpxfmwDN2k8Wa4DJfl1c3HlcBy/RvGn+0PMtw8cCWw3LjPcJ8nrgSyuEvb'
                'fdk03Q8ohdifrYcZLsNluAyX4TJcenD/5TxcTsBx2f98C+//zscTtM7v+Ppv'
                'ubKKdN0wKX9v0pS6g5f7LEmnijZ8W0564fxb9b04x3mMEJ6SwFO6igdriDDc'
                '6hv84UOb0a7CsCpV3Vb+SGJ51Mp5OyG8BwLvQRivNXooe8w5fkw7Cl9R4Cte'
                'xYcGL2lR0+qYmLyXhu9snZqNgej51R8nGwF8/vlecaTbd7794rr4AmAyrrx4'
                '+/g0Dl8+v5rbBJPPHd4KweXBZJwP4hpMF9f3ruk/GmOa+0/r55xzzOonjyc/'
                'tmmoP539/we1OWvt')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        self.assertTrue(module.wants_discard_edge)
        self.assertTrue(module.wants_discard_primary)
        self.assertEqual(module.new_primary_objects_name, "FilteredNuclei")

    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9210

IdentifySecondaryObjects:[module_num:1|svn_version:\'9194\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D]
    Select the input objects:Primary
    Name the identified objects:Secondary
    Select the method to identify the secondary objects:Watershed - Image
    Select the input image:Cytoplasm
    Select the thresholding method:Otsu Adaptive
    Threshold correction factor:1.2
    Lower and upper bounds on threshold\x3A:0.05,0.95
    Approximate fraction of image covered by objects?:0.02
    Number of pixels by which to expand the primary objects\x3A:12
    Regularization factor\x3A:0.08
    Name the outline image:CellOutlines
    Enter manual threshold\x3A:0.01
    Select binary image\x3A:MyMask
    Save outlines of the identified objects?:No
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Entropy
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Do you want to discard objects that touch the edge of the image?:Yes
    Do you want to discard associated primary objects?:Yes
    New primary objects name\x3A:FilteredPrimary
    Do you want to save outlines of the new primary objects?:Yes
    New primary objects outlines name\x3A:FilteredPrimaryOutlines
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        self.assertEqual(module.primary_objects, "Primary")
        self.assertEqual(module.objects_name, "Secondary")
        self.assertEqual(module.method, cpmi2.M_WATERSHED_I)
        self.assertEqual(module.image_name, "Cytoplasm")
        self.assertEqual(module.threshold_method, cpmi2.cpthresh.TM_OTSU)
        self.assertEqual(module.threshold_scope, cpmi.TS_ADAPTIVE)
        self.assertAlmostEqual(module.threshold_correction_factor.value, 1.2)
        self.assertAlmostEqual(module.threshold_range.min, 0.05)
        self.assertAlmostEqual(module.threshold_range.max, 0.95)
        self.assertEqual(module.object_fraction, "0.02")
        self.assertEqual(module.distance_to_dilate, 12)
        self.assertAlmostEqual(module.regularization_factor.value, 0.08)
        self.assertEqual(module.outlines_name, "CellOutlines")
        self.assertAlmostEqual(module.manual_threshold.value, 0.01)
        self.assertEqual(module.binary_image, "MyMask")
        self.assertFalse(module.use_outlines)
        self.assertEqual(module.two_class_otsu, cpmi.O_THREE_CLASS)
        self.assertEqual(module.use_weighted_variance, cpmi.O_ENTROPY)
        self.assertEqual(module.assign_middle_to_foreground, cpmi.O_BACKGROUND)
        self.assertTrue(module.wants_discard_edge)
        self.assertTrue(module.wants_discard_primary)
        self.assertEqual(module.new_primary_objects_name, "FilteredPrimary")
        self.assertTrue(module.wants_primary_outlines)
        self.assertEqual(module.new_primary_outlines_name, "FilteredPrimaryOutlines")
        self.assertTrue(module.fill_holes)

    def test_01_07_load_v7(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10231

IdentifySecondaryObjects:[module_num:1|svn_version:\'10220\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D]
    Select the input objects:Nuclei
    Name the objects to be identified:PropCells
    Select the method to identify the secondary objects:Propagation
    Select the input image:CorrGreen
    Select the thresholding method:Otsu Global
    Threshold correction factor:1
    Lower and upper bounds on threshold:0.02,1
    Approximate fraction of image covered by objects?:15%
    Number of pixels by which to expand the primary objects:12
    Regularization factor:0.05
    Name the outline image:MyOutline
    Manual threshold:0
    Select binary image:MyMask
    Retain outlines of the identified secondary objects?:No
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Discard secondary objects that touch the edge of the image?:No
    Discard the associated primary objects?:No
    Name the new primary objects:FilteredNuclei
    Retain outlines of the new primary objects?:No
    Name the new primary object outlines:FilteredNucleiOutlines
    Select the measurement to threshold with:None
    Fill holes in identified objects?:No
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        self.assertEqual(module.primary_objects, "Nuclei")
        self.assertEqual(module.objects_name, "PropCells")
        self.assertEqual(module.method, cpmi2.M_PROPAGATION)
        self.assertEqual(module.image_name, "CorrGreen")
        self.assertEqual(module.threshold_method, cpmi2.cpthresh.TM_OTSU)
        self.assertEqual(module.threshold_scope, cpmi.TS_GLOBAL)
        self.assertAlmostEqual(module.threshold_correction_factor.value, 1)
        self.assertAlmostEqual(module.threshold_range.min, 0.02)
        self.assertAlmostEqual(module.threshold_range.max, 1)
        self.assertEqual(module.object_fraction, "15%")
        self.assertEqual(module.distance_to_dilate, 12)
        self.assertAlmostEqual(module.regularization_factor.value, 0.05)
        self.assertEqual(module.outlines_name, "MyOutline")
        self.assertAlmostEqual(module.manual_threshold.value, 0)
        self.assertEqual(module.binary_image, "MyMask")
        self.assertFalse(module.use_outlines)
        self.assertEqual(module.two_class_otsu, cpmi.O_TWO_CLASS)
        self.assertEqual(module.use_weighted_variance, cpmi.O_WEIGHTED_VARIANCE)
        self.assertEqual(module.assign_middle_to_foreground, cpmi.O_FOREGROUND)
        self.assertFalse(module.wants_discard_edge)
        self.assertFalse(module.wants_discard_primary)
        self.assertEqual(module.new_primary_objects_name, "FilteredNuclei")
        self.assertFalse(module.wants_primary_outlines)
        self.assertEqual(module.new_primary_outlines_name, "FilteredNucleiOutlines")
        self.assertFalse(module.fill_holes)

    def test_01_09_load_v9(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130226215424
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    :
    Filter based on rules:No
    Filter:or (file does contain "")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Extract metadata?:No
    Extraction method count:1
    Extraction method:Automatic
    Source:From file name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Assignment method:Assign all images
    Load as:Grayscale image
    Image name:DNA
    :\x5B\x5D
    Assign channels by:Order
    Assignments count:1
    Match this rule:or (file does contain "")
    Image name:DNA
    Objects name:Cell
    Load as:Grayscale image

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

IdentifySecondaryObjects:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:9|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input objects:ChocolateChips
    Name the objects to be identified:Cookies
    Select the method to identify the secondary objects:Propagation
    Select the input image:BakingSheet
    Number of pixels by which to expand the primary objects:11
    Regularization factor:0.125
    Name the outline image:CookieEdges
    Retain outlines of the identified secondary objects?:No
    Discard secondary objects touching the border of the image?:Yes
    Discard the associated primary objects?:No
    Name the new primary objects:FilteredChocolateChips
    Retain outlines of the new primary objects?:No
    Name the new primary object outlines:FilteredChocolateChipOutlines
    Fill holes in identified objects?:Yes
    Threshold setting version:1
    Threshold strategy:Automatic
    Threshold method:Otsu
    Smoothing for threshold:Automatic
    Threshold smoothing scale:1.5
    Threshold correction factor:.95
    Lower and upper bounds on threshold:0.01,.95
    Approximate fraction of image covered by objects?:0.02
    Manual threshold:0.3
    Select the measurement to threshold with:Count_Cookies
    Select binary image:CookieMask
    Masking objects:CookieMonsters
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Method to calculate adaptive window size:Image size
    Size of adaptive window:9
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        self.assertEqual(module.primary_objects, "ChocolateChips")
        self.assertEqual(module.objects_name, "Cookies")
        self.assertEqual(module.image_name, "BakingSheet")
        self.assertEqual(module.method, cpmi2.M_PROPAGATION)
        self.assertEqual(module.distance_to_dilate, 11)
        self.assertEqual(module.regularization_factor, .125)
        self.assertEqual(module.outlines_name, "CookieEdges")
        self.assertFalse(module.use_outlines)
        self.assertTrue(module.wants_discard_edge)
        self.assertFalse(module.wants_discard_primary)
        self.assertEqual(module.new_primary_objects_name, "FilteredChocolateChips")
        self.assertFalse(module.wants_primary_outlines)
        self.assertEqual(module.new_primary_outlines_name, "FilteredChocolateChipOutlines")
        self.assertTrue(module.fill_holes)
        self.assertEqual(module.threshold_scope, cpmi.TS_AUTOMATIC)
        self.assertEqual(module.threshold_method, cpmi.TM_OTSU)
        self.assertEqual(module.threshold_smoothing_choice, cpmi.TSM_AUTOMATIC)
        self.assertEqual(module.threshold_smoothing_scale, 1.5)
        self.assertEqual(module.threshold_correction_factor, .95)
        self.assertEqual(module.threshold_range.min, .01)
        self.assertEqual(module.threshold_range.max, .95)
        self.assertEqual(module.object_fraction, .02)
        self.assertEqual(module.manual_threshold, .3)
        self.assertEqual(module.thresholding_measurement, "Count_Cookies")
        self.assertEqual(module.binary_image, "CookieMask")
        self.assertEqual(module.masking_objects, "CookieMonsters")
        self.assertEqual(module.two_class_otsu, cpmi.O_TWO_CLASS)
        self.assertEqual(module.use_weighted_variance, cpmi.O_WEIGHTED_VARIANCE)
        self.assertEqual(module.assign_middle_to_foreground, cpmi.O_FOREGROUND)
        self.assertEqual(module.adaptive_window_method, cpmi.FI_IMAGE_SIZE)
        self.assertEqual(module.adaptive_window_size, 9)

    def make_workspace(self, image, segmented, unedited_segmented=None,
                       small_removed_segmented=None):
        p = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        p.add_listener(callback)
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(image)
        objects = cpo.Objects()
        if unedited_segmented is not None:
            objects.unedited_segmented = unedited_segmented
        if small_removed_segmented is not None:
            objects.small_removed_segmented = small_removed_segmented
        objects.segmented = segmented
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.use_outlines.value = False
        module.outlines_name.value = "my_outlines"
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        return workspace, module

    def test_02_01_zeros_propagation(self):
        workspace, module = self.make_workspace(np.zeros((10, 10)),
                                                np.zeros((10, 10), int))
        module.method.value = cpmi2.M_PROPAGATION
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 0)
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name in (cpm.IMAGE, OUTPUT_OBJECTS_NAME, INPUT_OBJECTS_NAME):
            ocolumns = [x for x in columns if x[0] == object_name]
            features = m.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))
        self.assertTrue("my_outlines" not in workspace.get_outline_names())

    def test_02_02_one_object_propagation(self):
        img = np.zeros((10, 10))
        img[2:7, 2:7] = .5
        labels = np.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cpmi2.M_PROPAGATION
        module.threshold_scope.value = cpmi.TS_MANUAL
        module.manual_threshold.value = .25
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        expected = np.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(np.all(objects_out.segmented == expected))
        child_counts = m.get_current_measurement(INPUT_OBJECTS_NAME,
                                                 "Children_%s_Count" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 1)
        parents = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Parent_%s" % INPUT_OBJECTS_NAME)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0], 1)

    def test_02_03_two_objects_propagation_image(self):
        img = np.zeros((10, 20))
        img[2:7, 2:7] = .3
        img[2:7, 7:17] = .5
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cpmi2.M_PROPAGATION
        module.regularization_factor.value = 0  # propagate by image
        module.threshold_scope.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .2
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        expected[2:7, 7:17] = 2
        mask = np.ones((10, 10), bool)
        mask[:, 7:9] = False
        self.assertTrue(np.all(objects_out.segmented[:10, :10][mask] == expected[mask]))

    def test_02_04_two_objects_propagation_distance(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 20))
        img[2:7, 2:7] = .3
        img[2:7, 7:17] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_PROPAGATION
        module.regularization_factor.value = 1000  # propagate by distance
        module.threshold_scope.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .2
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 20), int)
        expected[2:7, 2:10] = 1
        expected[2:7, 10:17] = 2
        mask = np.ones((10, 20), bool)
        mask[:, 9:11] = False
        self.assertTrue(np.all(objects_out.segmented[mask] == expected[mask]))

    def test_02_05_propagation_wrong_size(self):
        '''Regression test of img-961: different image / object sizes'''
        img = np.zeros((10, 20))
        img[2:7, 2:7] = .5
        labels = np.zeros((20, 10), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cpmi2.M_PROPAGATION
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        expected = np.zeros((10, 20), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(np.all(objects_out.segmented == expected))
        child_counts = m.get_current_measurement(INPUT_OBJECTS_NAME, "Children_%s_Count" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(child_counts), 1)
        self.assertEqual(child_counts[0], 1)
        parents = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Parent_%s" % INPUT_OBJECTS_NAME)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0], 1)

    def test_03_01_zeros_watershed_gradient(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(np.zeros((10, 10)))
        objects = cpo.Objects()
        objects.unedited_segmented = np.zeros((10, 10), int)
        objects.small_removed_segmented = np.zeros((10, 10), int)
        objects.segmented = np.zeros((10, 10), int)
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_WATERSHED_G
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 0)

    def test_03_02_one_object_watershed_gradient(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_WATERSHED_G
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(np.all(objects_out.segmented == expected))
        self.assertTrue("Location_Center_X" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
        self.assertEqual(np.product(values.shape), 1)
        self.assertEqual(values[0], 4)
        self.assertTrue("Location_Center_Y" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
        self.assertEqual(np.product(values.shape), 1)
        self.assertEqual(values[0], 4)

    def test_03_03_two_objects_watershed_gradient(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 20))
        # There should be a gradient at :,7 which should act
        # as the watershed barrier
        img[2:7, 2:7] = .3
        img[2:7, 7:17] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_WATERSHED_G
        module.threshold_scope.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .2
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 20), int)
        expected[2:7, 2:7] = 1
        expected[2:7, 7:17] = 2
        mask = np.ones((10, 20), bool)
        mask[:, 7:9] = False
        self.assertTrue(np.all(objects_out.segmented[mask] == expected[mask]))

    def test_03_04_watershed_gradient_wrong_size(self):
        img = np.zeros((20, 10))
        img[2:7, 2:7] = .5
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cpmi2.M_WATERSHED_G
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((20, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(np.all(objects_out.segmented == expected))
        self.assertTrue("Location_Center_X" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_X")
        self.assertEqual(np.product(values.shape), 1)
        self.assertEqual(values[0], 4)
        self.assertTrue("Location_Center_Y" in m.get_feature_names(OUTPUT_OBJECTS_NAME))
        values = m.get_current_measurement(OUTPUT_OBJECTS_NAME, "Location_Center_Y")
        self.assertEqual(np.product(values.shape), 1)
        self.assertEqual(values[0], 4)

    def test_04_01_zeros_watershed_image(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(np.zeros((10, 10)))
        objects = cpo.Objects()
        objects.unedited_segmented = np.zeros((10, 10), int)
        objects.small_removed_segmented = np.zeros((10, 10), int)
        objects.segmented = np.zeros((10, 10), int)
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_WATERSHED_I
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 0)

    def test_04_02_one_object_watershed_image(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_WATERSHED_I
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(np.all(objects_out.segmented == expected))

    def test_04_03_two_objects_watershed_image(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 20))
        # There should be a saddle at 7 which should serve
        # as the watershed barrier
        x, y = np.mgrid[0:10, 0:20]
        img[2:7, 2:7] = .05 * (7 - y[2:7, 2:7])
        img[2:7, 7:17] = .05 * (y[2:7, 7:17] - 6)
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_WATERSHED_I
        module.threshold_scope.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .01
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 20), int)
        expected[2:7, 2:7] = 1
        expected[2:7, 7:17] = 2
        mask = np.ones((10, 20), bool)
        mask[:, 7] = False
        self.assertTrue(np.all(objects_out.segmented[mask] == expected[mask]))

    def test_04_04_watershed_image_wrong_size(self):
        img = np.zeros((20, 10))
        img[2:7, 2:7] = .5
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cpmi2.M_WATERSHED_I
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((20, 10), int)
        expected[2:7, 2:7] = 1
        self.assertTrue(np.all(objects_out.segmented == expected))

    def test_05_01_zeros_distance_n(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(np.zeros((10, 10)))
        objects = cpo.Objects()
        objects.unedited_segmented = np.zeros((10, 10), int)
        objects.small_removed_segmented = np.zeros((10, 10), int)
        objects.segmented = np.zeros((10, 10), int)
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_DISTANCE_N
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 0)

    def test_05_02_one_object_distance_n(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 10))
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_DISTANCE_N
        module.distance_to_dilate.value = 1
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        for x in (2, 6):
            for y in (2, 6):
                expected[x, y] = 0
        self.assertTrue(np.all(objects_out.segmented == expected))

    def test_05_03_two_objects_distance_n(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 20))
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        labels[3:6, 13:16] = 2
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_DISTANCE_N
        module.distance_to_dilate.value = 100
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 2)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((10, 20), int)
        expected[:, :10] = 1
        expected[:, 10:] = 2
        self.assertTrue(np.all(objects_out.segmented == expected))

    def test_05_04_distance_n_wrong_size(self):
        img = np.zeros((20, 10))
        labels = np.zeros((10, 20), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cpmi2.M_DISTANCE_N
        module.distance_to_dilate.value = 1
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        expected = np.zeros((20, 10), int)
        expected[2:7, 2:7] = 1
        for x in (2, 6):
            for y in (2, 6):
                expected[x, y] = 0
        self.assertTrue(np.all(objects_out.segmented == expected))

    def test_06_01_save_outlines(self):
        '''Test the "save_outlines" feature'''
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.use_outlines.value = True
        module.outlines_name.value = "my_outlines"
        module.method.value = cpmi2.M_WATERSHED_I
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_%s" % OUTPUT_OBJECTS_NAME in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_%s" % OUTPUT_OBJECTS_NAME)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        outlines_out = workspace.image_set.get_image("my_outlines",
                                                     must_be_binary=True).pixel_data
        expected = np.zeros((10, 10), int)
        expected[2:7, 2:7] = 1
        outlines = expected == 1
        outlines[3:6, 3:6] = False
        self.assertTrue(np.all(objects_out.segmented == expected))
        self.assertTrue(np.all(outlines == outlines_out))

    def test_06_02_save_primary_outlines(self):
        '''Test saving new primary outlines'''
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10, 10))
        img[2:7, 2:7] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.use_outlines.value = True
        module.outlines_name.value = "my_outlines"
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.wants_primary_outlines.value = True
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME
        module.new_primary_outlines_name.value = "newprimaryoutlines"
        module.method.value = cpmi2.M_WATERSHED_I
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        self.assertTrue(OUTPUT_OBJECTS_NAME in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        count_feature = "Count_%s" % OUTPUT_OBJECTS_NAME
        self.assertTrue(count_feature in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", count_feature)
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts, 1)
        objects_out = o_s.get_objects(OUTPUT_OBJECTS_NAME)
        outlines_out = workspace.image_set.get_image("newprimaryoutlines",
                                                     must_be_binary=True)
        expected = np.zeros((10, 10), bool)
        expected[3:6, 3:6] = True
        expected[4, 4] = False
        self.assertTrue(np.all(outlines_out.pixel_data == expected))

    def test_07_01_measurements_no_new_primary(self):
        module = cpmi2.IdentifySecondary()
        for discard_edge in (True, False):
            module.wants_discard_edge.value = discard_edge
            module.wants_discard_primary.value = False
            module.primary_objects.value = INPUT_OBJECTS_NAME
            module.objects_name.value = OUTPUT_OBJECTS_NAME
            module.new_primary_objects_name.value = NEW_OBJECTS_NAME

            categories = module.get_categories(None, cpm.IMAGE)
            self.assertEqual(len(categories), 2)
            self.assertTrue(all([any([x == y for x in categories])
                                 for y in ("Count", "Threshold")]))
            categories = module.get_categories(None, OUTPUT_OBJECTS_NAME)
            self.assertEqual(len(categories), 3)
            self.assertTrue(all([any([x == y for x in categories])
                                 for y in ("Location", "Parent", "Number")]))
            categories = module.get_categories(None, INPUT_OBJECTS_NAME)
            self.assertEqual(len(categories), 1)
            self.assertEqual(categories[0], "Children")

            categories = module.get_categories(None, NEW_OBJECTS_NAME)
            self.assertEqual(len(categories), 0)

            features = module.get_measurements(None, cpm.IMAGE, "Count")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], OUTPUT_OBJECTS_NAME)

            features = module.get_measurements(None, cpm.IMAGE, "Threshold")
            threshold_features = ("OrigThreshold", "FinalThreshold",
                                  "WeightedVariance", "SumOfEntropies")
            self.assertEqual(len(features), 4)
            self.assertTrue(all([any([x == y for x in features])
                                 for y in threshold_features]))
            for threshold_feature in threshold_features:
                objects = module.get_measurement_objects(None, cpm.IMAGE,
                                                         "Threshold",
                                                         threshold_feature)
                self.assertEqual(len(objects), 1)
                self.assertEqual(objects[0], OUTPUT_OBJECTS_NAME)

            features = module.get_measurements(None, INPUT_OBJECTS_NAME, "Children")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], OUTPUT_OBJECTS_NAME + "_Count")

            features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Parent")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], INPUT_OBJECTS_NAME)

            features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Location")
            self.assertEqual(len(features), 2)
            self.assertTrue(all([any([x == y for x in features])
                                 for y in ("Center_X", "Center_Y")]))
            features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Number")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], "Object_Number")

            columns = module.get_measurement_columns(None)
            expected_columns = [(cpm.IMAGE,
                                 "Threshold_%s_%s" % (f, OUTPUT_OBJECTS_NAME),
                                 cpm.COLTYPE_FLOAT)
                                for f in threshold_features]
            expected_columns += [(cpm.IMAGE, "Count_%s" % OUTPUT_OBJECTS_NAME,
                                  cpm.COLTYPE_INTEGER),
                                 (INPUT_OBJECTS_NAME,
                                  "Children_%s_Count" % OUTPUT_OBJECTS_NAME,
                                  cpm.COLTYPE_INTEGER),
                                 (OUTPUT_OBJECTS_NAME, "Location_Center_X", cpm.COLTYPE_FLOAT),
                                 (OUTPUT_OBJECTS_NAME, "Location_Center_Y", cpm.COLTYPE_FLOAT),
                                 (OUTPUT_OBJECTS_NAME, "Number_Object_Number", cpm.COLTYPE_INTEGER),
                                 (OUTPUT_OBJECTS_NAME,
                                  "Parent_%s" % INPUT_OBJECTS_NAME,
                                  cpm.COLTYPE_INTEGER)]
            self.assertEqual(len(columns), len(expected_columns))
            for column in expected_columns:
                self.assertTrue(any([all([fa == fb
                                          for fa, fb
                                          in zip(column, expected_column)])
                                     for expected_column in expected_columns]))

    def test_07_02_measurements_new_primary(self):
        module = cpmi2.IdentifySecondary()
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME

        categories = module.get_categories(None, cpm.IMAGE)
        self.assertEqual(len(categories), 2)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Count", "Threshold")]))
        categories = module.get_categories(None, OUTPUT_OBJECTS_NAME)
        self.assertEqual(len(categories), 3)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Location", "Parent", "Number")]))
        categories = module.get_categories(None, INPUT_OBJECTS_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], "Children")

        categories = module.get_categories(None, NEW_OBJECTS_NAME)
        self.assertEqual(len(categories), 4)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Location", "Parent", "Children", "Number")]))

        features = module.get_measurements(None, cpm.IMAGE, "Count")
        self.assertEqual(len(features), 2)
        self.assertTrue(OUTPUT_OBJECTS_NAME in features)
        self.assertTrue(NEW_OBJECTS_NAME in features)

        features = module.get_measurements(None, cpm.IMAGE, "Threshold")
        threshold_features = ("OrigThreshold", "FinalThreshold",
                              "WeightedVariance", "SumOfEntropies")
        self.assertEqual(len(features), 4)
        self.assertTrue(all([any([x == y for x in features])
                             for y in threshold_features]))
        for threshold_feature in threshold_features:
            objects = module.get_measurement_objects(None, cpm.IMAGE,
                                                     "Threshold",
                                                     threshold_feature)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0], OUTPUT_OBJECTS_NAME)

        features = module.get_measurements(None, INPUT_OBJECTS_NAME, "Children")
        self.assertEqual(len(features), 2)
        self.assertTrue(all([any([x == y for x in features])
                             for y in ("%s_Count" % OUTPUT_OBJECTS_NAME,
                                       "%s_Count" % NEW_OBJECTS_NAME)]))

        features = module.get_measurements(None, OUTPUT_OBJECTS_NAME, "Parent")
        self.assertEqual(len(features), 2)
        self.assertTrue(all([any([x == y for x in features])
                             for y in (INPUT_OBJECTS_NAME, NEW_OBJECTS_NAME)]))

        for oname in (OUTPUT_OBJECTS_NAME, NEW_OBJECTS_NAME):
            features = module.get_measurements(None, oname, "Location")
            self.assertEqual(len(features), 2)
            self.assertTrue(all([any([x == y for x in features])
                                 for y in ("Center_X", "Center_Y")]))

        columns = module.get_measurement_columns(None)
        expected_columns = [(cpm.IMAGE,
                             "Threshold_%s_%s" % (f, OUTPUT_OBJECTS_NAME),
                             cpm.COLTYPE_FLOAT)
                            for f in threshold_features]
        for oname in (NEW_OBJECTS_NAME, OUTPUT_OBJECTS_NAME):
            expected_columns += [(cpm.IMAGE, "Count_%s" % oname, cpm.COLTYPE_INTEGER),
                                 (INPUT_OBJECTS_NAME, "Children_%s_Count" % oname, cpm.COLTYPE_INTEGER),
                                 (oname, "Location_Center_X", cpm.COLTYPE_FLOAT),
                                 (oname, "Location_Center_Y", cpm.COLTYPE_FLOAT),
                                 (oname, "Number_Object_Number", cpm.COLTYPE_INTEGER),
                                 (oname, "Parent_Primary", cpm.COLTYPE_INTEGER)]
        expected_columns += [(NEW_OBJECTS_NAME,
                              "Children_%s_Count" % OUTPUT_OBJECTS_NAME,
                              cpm.COLTYPE_INTEGER),
                             (OUTPUT_OBJECTS_NAME,
                              "Parent_%s" % NEW_OBJECTS_NAME,
                              cpm.COLTYPE_INTEGER)]
        self.assertEqual(len(columns), len(expected_columns))
        for column in expected_columns:
            self.assertTrue(any([all([fa == fb
                                      for fa, fb
                                      in zip(column, expected_column)])
                                 for expected_column in expected_columns]))

    def test_08_01_filter_edge(self):
        labels = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
        image = np.array([[0, 0, .5, 0, 0],
                          [0, .5, .5, .5, 0],
                          [0, .5, .5, .5, 0],
                          [0, .5, .5, .5, 0],
                          [0, 0, 0, 0, 0]])
        expected_unedited = np.array([[0, 0, 1, 0, 0],
                                      [0, 1, 1, 1, 0],
                                      [0, 1, 1, 1, 0],
                                      [0, 1, 1, 1, 0],
                                      [0, 0, 0, 0, 0]])

        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(image)
        objects = cpo.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_PROPAGATION
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        self.assertTrue(np.all(object_out.segmented == 0))
        self.assertTrue(np.all(object_out.unedited_segmented == expected_unedited))

        object_out = workspace.object_set.get_objects(NEW_OBJECTS_NAME)
        self.assertTrue(np.all(object_out.segmented == 0))
        self.assertTrue(np.all(object_out.unedited_segmented == labels))

    def test_08_02_filter_unedited(self):
        labels = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0]])
        labels_unedited = np.array([[0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 2, 0, 0],
                                    [0, 0, 0, 0, 0]])
        image = np.array([[0, 0, .5, 0, 0],
                          [0, .5, .5, .5, 0],
                          [0, .5, .5, .5, 0],
                          [0, .5, .5, .5, 0],
                          [0, 0, 0, 0, 0]])
        expected = np.array([[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0]])
        expected_unedited = np.array([[0, 0, 1, 0, 0],
                                      [0, 1, 1, 1, 0],
                                      [0, 2, 2, 2, 0],
                                      [0, 2, 2, 2, 0],
                                      [0, 0, 0, 0, 0]])

        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(image)
        objects = cpo.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.unedited_segmented = labels_unedited
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_PROPAGATION
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.new_primary_objects_name.value = NEW_OBJECTS_NAME
        module.threshold_scope.value = cpmi.TS_GLOBAL
        module.threshold_method.value = cpmi.TM_OTSU
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        self.assertTrue(np.all(object_out.segmented == expected))
        self.assertTrue(np.all(object_out.unedited_segmented == expected_unedited))

        object_out = workspace.object_set.get_objects(NEW_OBJECTS_NAME)
        self.assertTrue(np.all(object_out.segmented == labels))
        self.assertTrue(np.all(object_out.unedited_segmented == labels_unedited))

    def test_08_03_small(self):
        '''Regression test of IMG-791

        A small object in the seed mask should not attract any of the
        secondary object.
        '''
        labels = np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

        labels_unedited = np.array([[0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 2, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])

        image = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 0, 0, 0, 0, 0]], float)
        expected = image.astype(int)

        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(image)
        objects = cpo.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.unedited_segmented = labels_unedited
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_PROPAGATION
        module.threshold_scope.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .5
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        self.assertTrue(np.all(object_out.segmented == expected))

    def test_08_04_small_touching(self):
        '''Test of logic added for IMG-791

        A small object in the seed mask touching the edge should attract
        some of the secondary object
        '''
        labels = np.array([[0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])

        labels_unedited = np.array([[0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 2, 0, 0, 0]])

        image = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 0]], float)

        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(image)
        objects = cpo.Objects()
        objects.unedited_segmented = labels
        objects.small_removed_segmented = labels
        objects.unedited_segmented = labels_unedited
        objects.segmented = labels
        o_s.add_objects(objects, INPUT_OBJECTS_NAME)
        i_s = i_l.get_image_set(0)
        i_s.add(IMAGE_NAME, image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value = INPUT_OBJECTS_NAME
        module.objects_name.value = OUTPUT_OBJECTS_NAME
        module.image_name.value = IMAGE_NAME
        module.method.value = cpmi2.M_PROPAGATION
        module.threshold_scope.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .5
        module.module_num = 1
        p.add_module(module)
        workspace = cpw.Workspace(p, module, i_s, o_s, m, i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        i, j = np.argwhere(labels_unedited == 2)[0]
        self.assertTrue(np.all(object_out.segmented[i - 1:, j - 1:j + 2] == 0))
        self.assertEqual(len(np.unique(object_out.unedited_segmented)), 3)
        self.assertEqual(len(np.unique(object_out.unedited_segmented[i - 1:, j - 1:j + 2])), 1)

    def test_09_01_binary_threshold(self):
        '''Test segmentation using a binary image for thresholding'''
        np.random.seed(91)
        image = np.random.uniform(size=(20, 10))
        labels = np.zeros((20, 10), int)
        labels[5, 5] = 1
        labels[15, 5] = 2
        threshold = np.zeros((20, 10), bool)
        threshold[4:7, 4:7] = True
        threshold[14:17, 4:7] = True
        expected = np.zeros((20, 10), int)
        expected[4:7, 4:7] = 1
        expected[14:17, 4:7] = 2

        workspace, module = self.make_workspace(image, labels)
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondaryObjects))
        module.threshold_scope.value = cpmi.TS_BINARY_IMAGE
        module.binary_image.value = "threshold"
        image_set = workspace.image_set
        self.assertTrue(isinstance(image_set, cpi.ImageSet))
        image_set.add("threshold", cpi.Image(threshold, convert=False))

        module.run(workspace)
        object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
        labels_out = object_out.segmented
        indexes = workspace.measurements.get_current_measurement(
                OUTPUT_OBJECTS_NAME, "Parent_" + INPUT_OBJECTS_NAME)
        self.assertEqual(len(indexes), 2)
        indexes = np.hstack(([0], indexes))
        self.assertTrue(np.all(indexes[labels_out] == expected))

    def test_10_01_holes_no_holes(self):
        np.random.seed(92)
        for wants_fill_holes in (True, False):
            for method in (cpmi2.M_DISTANCE_B,
                           cpmi2.M_PROPAGATION,
                           cpmi2.M_WATERSHED_G,
                           cpmi2.M_WATERSHED_I):
                image = np.random.uniform(size=(20, 10))
                labels = np.zeros((20, 10), int)
                labels[5, 5] = 1
                labels[15, 5] = 2
                threshold = np.zeros((20, 10), bool)
                threshold[1:7, 4:7] = True
                threshold[2, 5] = False
                threshold[14:17, 4:7] = True
                expected = np.zeros((20, 10), int)
                expected[1:7, 4:7] = 1
                expected[14:17, 4:7] = 2
                if not wants_fill_holes:
                    expected[2, 5] = 0
                workspace, module = self.make_workspace(image, labels)
                self.assertTrue(isinstance(module, cpmi2.IdentifySecondaryObjects))
                module.threshold_scope.value = cpmi.TM_BINARY_IMAGE
                module.binary_image.value = "threshold"
                module.method.value = method
                module.fill_holes.value = wants_fill_holes
                module.distance_to_dilate.value = 10000
                image_set = workspace.image_set
                self.assertTrue(isinstance(image_set, cpi.ImageSet))
                image_set.add("threshold", cpi.Image(threshold, convert=False))

                module.run(workspace)
                object_out = workspace.object_set.get_objects(OUTPUT_OBJECTS_NAME)
                labels_out = object_out.segmented
                indexes = workspace.measurements.get_current_measurement(
                        OUTPUT_OBJECTS_NAME, "Parent_" + INPUT_OBJECTS_NAME)
                self.assertEqual(len(indexes), 2)
                indexes = np.hstack(([0], indexes))
                self.assertTrue(np.all(indexes[labels_out] == expected))

    def test_11_00_relationships_zero(self):
        workspace, module = self.make_workspace(
                np.zeros((10, 10)), np.zeros((10, 10), int))
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpm.Measurements))
        result = m.get_relationships(
                module.module_num, cpmi2.R_PARENT,
                module.primary_objects.value, module.objects_name.value)
        self.assertEqual(len(result), 0)

    def test_11_01_relationships_one(self):
        img = np.zeros((10, 10))
        img[2:7, 2:7] = .5
        labels = np.zeros((10, 10), int)
        labels[3:6, 3:6] = 1
        workspace, module = self.make_workspace(img, labels)
        module.method.value = cpmi2.M_PROPAGATION
        module.threshold_scope.value = cpmi.TS_MANUAL
        module.manual_threshold.value = .25
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpm.Measurements))
        result = m.get_relationships(
                module.module_num, cpmi2.R_PARENT,
                module.primary_objects.value, module.objects_name.value)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[cpm.R_FIRST_IMAGE_NUMBER][0], 1)
        self.assertEqual(result[cpm.R_SECOND_IMAGE_NUMBER][0], 1)
        self.assertEqual(result[cpm.R_FIRST_OBJECT_NUMBER][0], 1)
        self.assertEqual(result[cpm.R_SECOND_OBJECT_NUMBER][0], 1)

    def test_11_02_relationships_missing(self):
        for missing in range(1, 4):
            img = np.zeros((10, 30))
            labels = np.zeros((10, 30), int)
            for i in range(3):
                object_number = i + 1
                center_j = i * 10 + 4
                labels[3:6, (center_j - 1):(center_j + 2)] = object_number
                if object_number != missing:
                    img[2:7, (center_j - 2):(center_j + 3)] = .5
                else:
                    img[0:7, (center_j - 2):(center_j + 3)] = .5
            workspace, module = self.make_workspace(img, labels)
            self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
            module.method.value = cpmi2.M_PROPAGATION
            module.threshold_scope.value = cpmi.TS_MANUAL
            module.wants_discard_edge.value = True
            module.wants_discard_primary.value = False
            module.manual_threshold.value = .25
            module.run(workspace)
            m = workspace.measurements
            self.assertTrue(isinstance(m, cpm.Measurements))
            result = m.get_relationships(
                    module.module_num, cpmi2.R_PARENT,
                    module.primary_objects.value, module.objects_name.value)
            self.assertEqual(len(result), 2)
            for i in range(2):
                object_number = i + 1
                if object_number >= missing:
                    object_number += 1
                self.assertEqual(result[cpm.R_FIRST_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cpm.R_SECOND_IMAGE_NUMBER][i], 1)
                self.assertEqual(result[cpm.R_FIRST_OBJECT_NUMBER][i],
                                 object_number)
                self.assertEqual(result[cpm.R_SECOND_OBJECT_NUMBER][i],
                                 object_number)
