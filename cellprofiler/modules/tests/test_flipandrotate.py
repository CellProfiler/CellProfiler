'''test_flipandrotate - test the FlipAndRotate module
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
import cellprofiler.measurement as cpmeas
import cellprofiler.modules.flipandrotate as F
from centrosome.cpmorphology import draw_line

IMAGE_NAME = 'my_image'
OUTPUT_IMAGE = 'my_output_image'


class TestFlipAndRotate(unittest.TestCase):
    def test_01_000_load_matlab_flip(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

Flip:[module_num:1|svn_version:\'8913\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    What did you call the image you want to flip?:MyImage
    What do you want to call the flipped image?:MyFlippedImage
    Do you want to flip from left to right?:Yes
    Do you want to flip from top to bottom?:Yes

Flip:[module_num:2|svn_version:\'8913\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    What did you call the image you want to flip?:MyImage
    What do you want to call the flipped image?:MyFlippedImage
    Do you want to flip from left to right?:Yes
    Do you want to flip from top to bottom?:No

Flip:[module_num:3|svn_version:\'8913\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    What did you call the image you want to flip?:MyImage
    What do you want to call the flipped image?:MyFlippedImage
    Do you want to flip from left to right?:No
    Do you want to flip from top to bottom?:Yes
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        for module, flip_choice in zip(pipeline.modules(),
                                       (F.FLIP_BOTH, F.FLIP_LEFT_TO_RIGHT,
                                        F.FLIP_TOP_TO_BOTTOM)):
            self.assertTrue(isinstance(module, F.FlipAndRotate))
            self.assertEqual(module.image_name, "MyImage")
            self.assertEqual(module.output_name, "MyFlippedImage")
            self.assertEqual(module.flip_choice, flip_choice)
            self.assertEqual(module.rotate_choice, F.ROTATE_NONE)

    def test_01_001_matlab_rotate(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

Rotate:[module_num:1|svn_version:\'8913\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    What did you call the image to be rotated?:MyImage
    What do you want to call the rotated image?:MyRotatedImage
    Choose rotation method\x3A:Angle
    Would you like to crop away the rotated edges?:Yes
    Do you want to determine the amount of rotation for each image individually as you cycle through, or do you want to define it only once (on the first image) and then apply it to all images?:Individually
    For COORDINATES or MOUSE, do you want to click on points that are aligned horizontally or vertically?:horizontally
    For COORDINATES and ONLY ONCE, what are the coordinates of one point (X,Y)?:1,10
    For COORDINATES and ONLY ONCE, what are the coordinates of the other point (X,Y)?:121,144
    For ANGLE and ONLY ONCE, by what angle would you like to rotate the image (in degrees, positive = counterclockwise and negative = clockwise)?:45
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, F.FlipAndRotate))
        self.assertEqual(module.image_name, "MyImage")
        self.assertEqual(module.output_name, "MyRotatedImage")
        self.assertEqual(module.rotate_choice, F.ROTATE_ANGLE)
        self.assertTrue(module.wants_crop)
        self.assertEqual(module.how_often.value, F.IO_INDIVIDUALLY)
        self.assertEqual(module.first_pixel.x, 1)
        self.assertEqual(module.first_pixel.y, 10)
        self.assertEqual(module.second_pixel.x, 121)
        self.assertEqual(module.second_pixel.y, 144)
        self.assertEqual(module.angle, 45)

    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUvDNz1MITi1QMLJQMLC0AiIjIwUjAwNLBZIBA6OnLz8DA4MgEwND'
                'xZy7Yd7+jw0E5rVMM39a+GiDh2XRg1A5IfcbrVfDmpoYFAtNej11d5ZNsVjx'
                'VPxDd5jM9+WJ+p6vIlptQiweZIj0Gt83f6wvbM3LcCOSYcaU75mZZnt6q95E'
                'z9L8L8XtOEP82kK21R//2Gdu11B4Me9RtIuyjJPVpJ9i1Q/Mme1rS87d7Xhk'
                '9urlwaIti+/Z8h3qbmeRsN+nUuuz/qRw2xuPQ+s+KHNyphvaPTNPv7wnRqJS'
                '+lLcsjariDY+r+miuXFH5Q8a7DNz2q6qW6e5TfqO+OzltVcE/fg7/cI/yjz7'
                '4CNtc2o/R8f3YK/YJX2SzOv/xMcekVh29qzQ60iTiin/ZwvdP54XxNeTbGtt'
                'WWHdvH3h2e6r9tlPQv9xlkuI+4k8WshX+VPdyEa4QL06c87Pz19WA42zf54v'
                'ct+z3WZeZbP7T91nk9wfuXVccgycd37N8tPMn47M/NFTdWrf/GfH8hTnTD0Y'
                'uSHc8P69Y/+mr7u+5rfCsb8nD+pK/5zqf8muyIO/8mXmxQvvQr+vXPX6ptGn'
                '5y+tsu5/ZzLhsO5I6l2eYfu+9n380vaZ83MX8/y78O/Hn2tfNRY/XvHvz32R'
                'uAPK+23ur/jfpDhfOZr/9+Mnhq+ly+feb/3Peu+9/Ob61d/OSrrned8O6Pqg'
                'I2FzKvx/7PEzex8xdT65c33//JNzfuVWrj0Tert9lujnq2s/7Zi159+6r/Ua'
                'd4ynVNu/lj+ivjjrj3bcoTPHAOAKLXI=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        #
        # Flip module (#2):
        # image name = OrigBlue
        # output image = FlippedOrigBlue
        # No flip left/right
        # Flip top/bottom
        # Rotate with mouse
        # Crop
        # Rotate images individually
        # Coordinates: horizontally, pt1 = 1,2 pt2 = 3,4
        # Angle: 5
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, F.FlipAndRotate))
        self.assertEqual(module.image_name, 'OrigBlue')
        self.assertEqual(module.output_name, 'FlippedOrigBlue')
        self.assertEqual(module.flip_choice, F.FLIP_TOP_TO_BOTTOM)
        self.assertEqual(module.rotate_choice, F.ROTATE_MOUSE)
        self.assertTrue(module.wants_crop.value)
        self.assertEqual(module.how_often, F.IO_INDIVIDUALLY)
        self.assertEqual(module.first_pixel.x, 1)
        self.assertEqual(module.first_pixel.y, 2)
        self.assertEqual(module.second_pixel.x, 3)
        self.assertEqual(module.second_pixel.y, 4)
        self.assertEqual(module.angle.value, 5)

    def test_01_02_load_v1(self):
        '''Load a variable_revision_number = 1 module'''
        data = ('eJztWM9PGkEUXhCtP5pWkyb1OEdpgSyojZJGRakpqSARYmOMbUd2gElmZ8iw'
                'a8XGpMf+WT322D+lxx47g7vsMkUXVkkPZclkeW/f974338wswxRz1YPcLlhP'
                '6aCYqybrmCBQJtCqM25mAbUSYI8jaCEDMJoF+xyDCmqBzDpIp7Orm9m1VyCj'
                '65tauCtSKD4Rt+VnmjYj7rOiRZ1H044d8TVpV5BlYdpoT2sxbdnxfxftGHIM'
                'zwk6hsRGbY/C9RdonVU7rd6jIjNsgkrQ9AeLq2Sb54i3D+su0HlcxpeIVPAV'
                'Urrghh2hC9zGjDp4J7/q7fEyS+GVOvyc9XSIKDpIXZZ8fhn/VvPiYwN088cv'
                'OjamBr7Ahg0JwCZs9KqQ+fSAfFN9+aa0fCnXxe0E4BaVOmSroksr+eYS1ixg'
                'QqvWHCbPUyWPtPcJbrWQcchxY1eM/FD9iPTliWirTv+D+BcUfmnnGaDMAnVR'
                'Rk/HoDzzSp55L4/dRt54bATkmVbySLvIblIMo0O0Dx/VSux+uKD5+FypV9p5'
                'VIc2sUBBTkaQxxzVLMY7Q+n4WMkn7UJvfpOOq6O/PzNKHvdy88w59zDrQU/o'
                'oXAnYhWGHW89kdb1Ieu9bdzC6NxkHF8xat2m80PON3W9Dupv2HEdBRek05yi'
                'k7T3mpBSRNLJB+AP+34eF586Lukx9y/WxxcT84ei+/B9DeB7p/WPp7Q/rGyX'
                'X8sNEtpKvYx/lNZ7RMgR+7x1mkuWz+KuZ48R26Rbp3py8+xLOpG5vgmuYIHs'
                'OuMD+z1K/c2A+jeU+qUtazhBkDuFrV3Hk9JVFEu56fgyji8PO57nPnX+eDTa'
                'vmZc63PQ7253E9TgzG6Nn3/QPsjjB2Jrhlr/4r02wU1wE9z/g9vx4SbvqQlu'
                'VNxvH079PVf3+TL+k3b3fHuh9c83adfEFqrFmTyP4imze2jSThEGjZtTi9SB'
                '+FrwHWAMw6MrPPptPPLfPKQGZ5bY56XkEUOOGkddS9VtbgCPv/9R8Vmav1tv'
                'VWdP/1/bYfhikb/5FgJwMUcxifumjTa+K3fEu30LG/8HWH8MEg==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, F.FlipAndRotate))
        self.assertEqual(module.image_name, 'DNA')
        self.assertEqual(module.output_name, 'FlippedOrigBlue')
        self.assertEqual(module.flip_choice, F.FLIP_NONE)
        self.assertEqual(module.rotate_choice, F.ROTATE_MOUSE)
        self.assertFalse(module.wants_crop.value)
        self.assertEqual(module.how_often, F.IO_INDIVIDUALLY)
        self.assertEqual(module.angle, 0)
        self.assertEqual(module.first_pixel.x, 0)
        self.assertEqual(module.first_pixel.y, 0)
        self.assertEqual(module.second_pixel.x, 0)
        self.assertEqual(module.second_pixel.y, 100)
        self.assertEqual(module.horiz_or_vert, F.C_HORIZONTALLY)

    def test_01_03_load_v2(self):
        '''Load a v2 pipeline'''
        data = ('eJztWFtPGkEUXhCtl6bVpEn7OI/SAlmstkoaFaWmpIJEaBtjbDuyA0wyO0N2'
                'Z1VsTPrYn9af4M/oYx87gwu7TFeXi6QPZc0Gz9nzfecyZ5bDFLKV/ewOWEvp'
                'oJCtJGuYIFAikNeYZWYA5QmwayHIkQEYzYACo+CgyoG+BtJ6Jv06s/oKrOj6'
                'hjbcFckXHomP5hNNmxGfs+KOuo+mXTniu6VcRpxjWrentZj2zNVfi/sjtDA8'
                'JegjJA6yPRcdfZ7WWKXV7D4qMMMhqAhNv7G4io55iiz7oNYBuo9L+AKRMr5E'
                'Sgods0N0hm3MqIt3+VVt1y/jit9yg53vWSIchV/W53rWq09EqY+s15JPL+3f'
                'aZ59LKCefvtFV8bUwGfYcCAB2IT1bnSSTw/hm+rhm9JyxWwbtx2CW1TikHcF'
                'XfDk2wsomsyEvNroh+exwiPlPYKbTWQcWLi+IzqirzwiPTwR7aWbf5j/BcW/'
                'lHMMUMZBTYTRrWMYz7zCM+/xODbS+uaZU3ik/MFGwGRdmr7qEe3hiWpFNhou'
                'rC+fKnFLOYdq0CEc5GVTghy2UJUzq9VXHR4qfFLOd/uctILqMKPwdK4Oz9wA'
                '9VP3hZ7Qh8Idid0oceshuGklXynribSu9xnvbes2TJ0bzMKXjPLb6nyf/abu'
                '26B8h13XQXBhdQral7sNSCki6eQ9+B/2PT0uf+q6pMecX6zHX0z0D0Wj+Pse'
                '4u+91rueUv68vFV6IwcotJl6Ef8ipU+IkEN2vnmcTZZO4h3NLiOOSTeP9eTG'
                'ybd0YuXqxriMBbKtjAfmPUj8jZD415X4pSxjOELQcgNbvYonpUoMfrzh6lZc'
                'XQ62PM0ocf58MNh8M679GfT92x6G6hZzmuP3HzQPef6BGNFQ81+81ya4CW6C'
                '+39w2z7c5D01wQ2K++3Dqd/n6pwv7b9qd/fbc62336RcFSNU02LyvMpKme1D'
                'FTtFGDRuTi9S++LfvO8gox8/uuJHv82P/FUPqWExLua8lDxqyFLjsC2pdZsL'
                '8OPPPyr+lubvrrdaZ6/+v7aG8ReL/u1vIQQXcysmcT+0wdZ3+Q77Tm6j2A+a'
                'f0QIfwBNmhIA')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, F.FlipAndRotate))
        self.assertEqual(module.image_name, 'DNA')
        self.assertEqual(module.output_name, 'FlippedOrigBlue')
        self.assertEqual(module.flip_choice, F.FLIP_NONE)
        self.assertEqual(module.rotate_choice, F.ROTATE_MOUSE)
        self.assertFalse(module.wants_crop.value)
        self.assertEqual(module.how_often, F.IO_INDIVIDUALLY)
        self.assertEqual(module.angle, 0)
        self.assertEqual(module.first_pixel.x, 0)
        self.assertEqual(module.first_pixel.y, 0)
        self.assertEqual(module.second_pixel.x, 0)
        self.assertEqual(module.second_pixel.y, 100)
        self.assertEqual(module.horiz_or_vert, F.C_HORIZONTALLY)

    def run_module(self, image, mask=None, fn=None):
        '''Run the FlipAndRotate module

        image - pixel data to be transformed
        mask  - optional mask on the pixel data
        fn    - function with signature, "fn(module)" that will be
                called with the FlipAndRotate module
        returns an Image object containing the flipped/rotated/masked/cropped
        image and the angle measurement.
        '''
        img = cpi.Image(image, mask)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, img)
        module = F.FlipAndRotate()
        module.image_name.value = IMAGE_NAME
        module.output_name.value = OUTPUT_IMAGE
        module.module_num = 1
        if fn is not None:
            fn(module)
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)

        def error_callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(error_callback)
        measurements = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  cpo.ObjectSet(), measurements,
                                  image_set_list)
        module.run(workspace)
        feature = F.M_ROTATION_F % OUTPUT_IMAGE
        self.assertTrue(feature in
                        measurements.get_feature_names(cpmeas.IMAGE))
        angle = measurements.get_current_image_measurement(feature)
        output_image = image_set.get_image(OUTPUT_IMAGE)
        return output_image, angle

    def test_02_01_flip_left_to_right(self):
        np.random.seed(0)
        image = np.random.uniform(size=(3, 3))
        mask = np.array([[True, True, True],
                         [False, True, True],
                         [True, False, True]])
        expected_mask = np.array([[True, True, True],
                                  [True, True, False],
                                  [True, False, True]])
        expected = image.copy()
        expected[:, 2] = image[:, 0]
        expected[:, 0] = image[:, 2]

        def fn(module):
            self.assertTrue(isinstance(module, F.FlipAndRotate))
            module.flip_choice.value = F.FLIP_LEFT_TO_RIGHT
            module.rotate_choice.value = F.ROTATE_NONE

        output_image, angle = self.run_module(image, mask=mask, fn=fn)
        self.assertEqual(angle, 0)
        self.assertTrue(np.all(output_image.mask == expected_mask))
        self.assertTrue(np.all(np.abs(output_image.pixel_data - expected) <=
                               np.finfo(np.float32).eps))

    def test_02_02_flip_top_to_bottom(self):
        np.random.seed(0)
        image = np.random.uniform(size=(3, 3)).astype(np.float32)
        mask = np.array([[True, True, True],
                         [False, True, True],
                         [True, False, True]])
        expected_mask = np.array([[True, False, True],
                                  [False, True, True],
                                  [True, True, True]])
        expected = image.copy()
        expected[2, :] = image[0, :]
        expected[0, :] = image[2, :]

        def fn(module):
            self.assertTrue(isinstance(module, F.FlipAndRotate))
            module.flip_choice.value = F.FLIP_TOP_TO_BOTTOM
            module.rotate_choice.value = F.ROTATE_NONE

        output_image, angle = self.run_module(image, mask=mask, fn=fn)
        self.assertEqual(angle, 0)
        self.assertTrue(np.all(output_image.mask == expected_mask))
        self.assertTrue(np.all(np.abs(output_image.pixel_data - expected) <=
                               np.finfo(float).eps))

    def test_02_03_flip_both(self):
        np.random.seed(0)
        image = np.random.uniform(size=(3, 3)).astype(np.float32)
        mask = np.array([[True, True, True],
                         [False, True, True],
                         [True, False, True]])
        expected_mask = np.array([[True, False, True],
                                  [True, True, False],
                                  [True, True, True]])
        expected = image[np.array([[2, 2, 2],
                                   [1, 1, 1],
                                   [0, 0, 0]]),
                         np.array([[2, 1, 0],
                                   [2, 1, 0],
                                   [2, 1, 0]])]

        def fn(module):
            self.assertTrue(isinstance(module, F.FlipAndRotate))
            module.flip_choice.value = F.FLIP_BOTH
            module.rotate_choice.value = F.ROTATE_NONE

        output_image, angle = self.run_module(image, mask=mask, fn=fn)
        self.assertEqual(angle, 0)
        self.assertTrue(np.all(output_image.mask == expected_mask))
        self.assertTrue(np.all(np.abs(output_image.pixel_data - expected) <=
                               np.finfo(float).eps))

    def test_03_01_rotate_angle(self):
        '''Rotate an image through an angle'''
        #
        # Draw a rectangle with intensity that varies monotonically according
        # to angle.
        #
        i, j = np.mgrid[-5:6, -9:10]
        angle = np.arctan2(i.astype(float) / 5.0, j.astype(float) / 9.0)
        img = (1 + np.cos(angle)) / 2
        self.assertAlmostEqual(img[5, 0], 0)
        self.assertAlmostEqual(img[5, 18], 1)
        self.assertAlmostEqual(img[0, 9], .5)
        self.assertAlmostEqual(img[10, 9], .5)
        #
        # The pixels with low values get masked out
        #
        mask = img > .5
        #
        # Rotate the rectangle from 10 to 350
        #
        for angle in range(10, 360, 10):
            def fn(module, angle=angle):
                self.assertTrue(isinstance(module, F.FlipAndRotate))
                module.flip_choice.value = F.FLIP_NONE
                module.rotate_choice.value = F.ROTATE_ANGLE
                module.wants_crop.value = False
                module.angle.value = angle

            output_image, measured_angle = self.run_module(img, mask, fn)
            self.assertAlmostEqual(measured_angle, angle, 3)
            rangle = float(angle) * np.pi / 180.0
            pixel_data = output_image.pixel_data
            #
            # Check that the output contains the four corners of the original
            #
            corners_in = np.array([[-5, -9], [-5, 9], [5, -9], [5, 9]], float)
            corners_out_i = np.sum(corners_in * np.array([np.cos(rangle), -np.sin(rangle)]), 1)
            corners_out_j = np.sum(corners_in * np.array([np.sin(rangle), np.cos(rangle)]), 1)
            i_width = np.max(corners_out_i) - np.min(corners_out_i)
            j_width = np.max(corners_out_j) - np.min(corners_out_j)
            self.assertTrue(i_width < pixel_data.shape[0])
            self.assertTrue(i_width > pixel_data.shape[0] - 2)
            self.assertTrue(j_width < pixel_data.shape[1])
            self.assertTrue(j_width > pixel_data.shape[1] - 2)
            # The maximum rotates clockwise - i starts at center and increases
            # and j starts at max and decreases
            #
            i_max = min(pixel_data.shape[0] - 1,
                        max(0, int(-np.sin(rangle) * 8 +
                                   float(pixel_data.shape[0]) / 2)))
            j_max = min(pixel_data.shape[1] - 1,
                        max(0, int(np.cos(rangle) * 8 +
                                   float(pixel_data.shape[1] / 2))))
            self.assertTrue(pixel_data[i_max, j_max] > .9)
            self.assertTrue(output_image.mask[i_max, j_max])
            i_min = min(pixel_data.shape[0] - 1,
                        max(0, int(np.sin(rangle) * 8 +
                                   float(pixel_data.shape[0]) / 2)))
            j_min = min(pixel_data.shape[1] - 1,
                        max(0, int(-np.cos(rangle) * 8 +
                                   float(pixel_data.shape[1]) / 2)))
            self.assertTrue(pixel_data[i_min, j_min] < .1)
            self.assertFalse(output_image.mask[i_min, j_min])
            #
            # The corners of the image should be masked except for angle
            # in 90,180,270
            #
            if angle not in (90, 180, 270):
                for ci, cj in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
                    self.assertFalse(output_image.mask[ci, cj])

    def test_03_02_rotate_coordinates(self):
        '''Test rotating a line to the horizontal and vertical'''

        img = np.zeros((20, 20))
        pt0 = (2, 2)
        pt1 = (6, 18)
        draw_line(img, pt0, pt1, 1)
        i, j = np.mgrid[0:20, 0:20]
        for option in (F.C_HORIZONTALLY, F.C_VERTICALLY):
            def fn(module):
                self.assertTrue(isinstance(module, F.FlipAndRotate))
                module.flip_choice.value = F.FLIP_NONE
                module.rotate_choice.value = F.ROTATE_COORDINATES
                module.horiz_or_vert.value = option
                module.wants_crop.value = False
                module.first_pixel.value = pt0
                module.second_pixel.value = pt1

            output_image, angle = self.run_module(img, fn=fn)
            pixels = output_image.pixel_data

            if option == F.C_HORIZONTALLY:
                self.assertAlmostEqual(angle,
                                       np.arctan2(pt1[0] - pt0[0],
                                                  pt1[1] - pt0[1]) * 180.0 /
                                       np.pi, 3)
                #
                # Account for extra pixels due to twisting
                #
                line_i = 4 + (pixels.shape[0] - 20) / 2
                line_j = 4 + (pixels.shape[1] - 20) / 2
                self.assertTrue(np.all(pixels[line_i, line_j:line_j + 12] > .2))
                self.assertTrue(np.all(pixels[:20, :20][np.abs(i - line_i) > 1] < .1))
            else:
                self.assertAlmostEqual(angle,
                                       -np.arctan2(pt1[1] - pt0[1],
                                                   pt1[0] - pt0[0]) * 180.0 /
                                       np.pi, 3)
                line_i = 4 + (pixels.shape[0] - 20) / 2
                line_j = 15 + (pixels.shape[1] - 20) / 2
                self.assertTrue(np.all(pixels[line_i:line_i + 12, line_j] > .2))
                self.assertTrue(np.all(pixels[:20, :20][np.abs(j - line_j) > 1] < .1))

    def test_04_01_crop(self):
        '''Turn cropping on and check that the cropping mask covers the mask'''
        image = np.random.uniform(size=(19, 21))
        i, j = np.mgrid[0:19, 0:21].astype(float)
        image = i / 100 + j / 10000
        for angle in range(10, 360, 10):
            #
            # Run the module with cropping to get the crop mask
            #
            def fn(module, angle=angle):
                self.assertTrue(isinstance(module, F.FlipAndRotate))
                module.flip_choice.value = F.FLIP_NONE
                module.rotate_choice.value = F.ROTATE_ANGLE
                module.angle.value = angle
                module.wants_crop.value = True

            crop_output_image, angle = self.run_module(image, fn=fn)
            crop_mask = crop_output_image.crop_mask
            crop_image = crop_output_image.pixel_data
            self.assertTrue(np.all(crop_output_image.mask[1:-1, 1:-1]))

            #
            # Run the module without cropping to get the mask
            #
            def fn(module, angle=angle):
                self.assertTrue(isinstance(module, F.FlipAndRotate))
                module.flip_choice.value = F.FLIP_NONE
                module.rotate_choice.value = F.ROTATE_ANGLE
                module.angle.value = angle
                module.wants_crop.value = False

            output_image, angle = self.run_module(image, fn=fn)
            self.assertTrue(isinstance(crop_output_image, cpi.Image))
            pixel_data = output_image.pixel_data
            slop = (np.array(pixel_data.shape) - np.array(image.shape)) / 2
            mask = output_image.mask
            pixel_data = pixel_data[slop[0]:image.shape[0] + slop[0],
                         slop[1]:image.shape[1] + slop[1]]
            mask = mask[slop[0]:image.shape[0] + slop[0],
                   slop[1]:image.shape[1] + slop[1]]
            #
            # Slight misregistration: rotate returns even # shape
            #
            # recrop_image = crop_output_image.crop_image_similarly(pixel_data)
            # self.assertTrue(np.all(recrop_image == crop_image))
            # self.assertTrue(np.all(crop_output_image.crop_image_similarly(mask)))

    def test_05_01_get_measurements(self):
        '''Test the get_measurements and allied methods'''
        module = F.FlipAndRotate()
        module.output_name.value = OUTPUT_IMAGE
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], cpmeas.IMAGE)
        self.assertEqual(columns[0][1], F.M_ROTATION_F % OUTPUT_IMAGE)
        self.assertEqual(columns[0][2], cpmeas.COLTYPE_FLOAT)

        categories = module.get_categories(None, cpmeas.IMAGE)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], F.M_ROTATION_CATEGORY)
        self.assertEqual(len(module.get_categories(None, 'Foo')), 0)

        measurements = module.get_measurements(None, cpmeas.IMAGE,
                                               F.M_ROTATION_CATEGORY)
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0], OUTPUT_IMAGE)
        self.assertEqual(len(module.get_measurements(None, cpmeas.IMAGE, 'Foo')), 0)
        self.assertEqual(len(module.get_measurements(None, 'Foo', F.M_ROTATION_CATEGORY)), 0)
