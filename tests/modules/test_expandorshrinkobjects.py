'''test_expandorshrink - test the ExpandOrShrink module
'''

import base64
import os
import unittest
import zlib
from StringIO import StringIO

import PIL.Image as PILImage
import numpy as np
import scipy.ndimage
from matplotlib.image import pil_to_array

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw

import centrosome.cpmorphology as morph
from centrosome.outline import outline

import cellprofiler.modules.expandorshrinkobjects as E

INPUT_NAME = "input"
OUTPUT_NAME = "output"
OUTLINES_NAME = "outlines"


class TestExpandOrShrinkObjects(unittest.TestCase):
    def test_01_02_load_v1(self):
        '''Load ExpandOrShrink modules, v1'''
        data = ('eJztW91u2zYUlhMnbVasSHezYr3hZbPFguS2aBMUqb24W73VjlEbDYqiP7RF'
                'x2xpUpCoJF5RoJd7lD3GHmeXe4SJshTJnFvJju1YqQQI8jnidw6/Qx6KEs1a'
                'ufW0/DO4p2qgVm4Vupgg0CCQd5nV3wWUb4N9C0GODMDoLjh0r785BBSLQNvZ'
                '1e/s6hooatqOMt2Rq9auu5c/7yvKunu96p4r/q01X85FTiE3EeeYHtlrSl65'
                '6ev/ds/n0MKwTdBzSBxkhy4CfZV2WWtgnt2qMcMhqA770cLuUXf6bWTZB90A'
                '6N9u4FNEmvgPJFEIij1Dx9jGjPp4376sPfPLuORXxOHqD2EcclIc8u55K6IX'
                '5Z8oYfn8mLjdiJTf9GVMDXyMDQcSgPvw6KwWwp4WY291xN6qUqmXPdyDGNy6'
                'VI91L84dgvDQbykGf13CC7nZsxz6HtFJ7GxKdsTZQqe88PgUdjjoQ97ppYnP'
                't5IdIVdE6yIjMJOoXXMjdnLKHR8XF4c1yb+QdW37rubj38bgVQkv5MenJqTu'
                'YNN+hzrcBu0BgMA2UQd3sTv2UC+PAOsCU+SjnczPj5IfIZcNA5jQ4thNBC8j'
                '3EEFEEyR6xPxE4RoUIchLkl7fCP5EXKFAco4cGwU2pk0z164WXqedpwUpyfE'
                'rYzgVpQ6Sxcubvz8XhltTyFXUBc6hIOqGDxBBVtuF2HWYK7tOs94rUu44Ahw'
                'G/41Sf+/JsVLyAfcdsCvhLUhSWwnaR7Nys682k3G6ap2rvjPAjcJv2nmF5qq'
                'ece27v+Ycf1nyXvavMqP4PKCs34R/Eox9RyXj/s9SCkixYKmJ7azIdkRcpVy'
                'RG3MBzPgsah+HcxXLyvfWT3HdW05+cl5V2cUTROX+zOs5yS4TzH1/F0Z7XdC'
                'fn37UeOheCFHe+pPW2+EdIgIecZO9l6WC41XW4FmnxGnT/deaoWdVx/07eLH'
                'YeEmdpGecitxnJM87xYRr15MPR9I8RKy4PwCQcsPxN2PWwWhqjHKe76u6Osq'
                'cBBq5jWfWHTeLGJekPHL+C0jP20J5teLft6lmd+s30+Xjd/Xl3/3Lrye835/'
                'ap0w0CHQtv0v52ni+ySG77j3+UOEj3pi+edYLHTQDorYSwvvUgzvcfO6X5iF'
                'jizmUOPi6v3XzRCXk3Dj1oMWGR9v8UgEyExuZ1w+DT+yh4ZmaWeReRLxDzA1'
                'kJkie2nJ4wyX4TLc+XFx48d3yuj4IWTmcLE6+r8BZB720hLHDJfhLiOuFMEl'
                '/R9NOB8cZnOa+Ga4DHcZcdlzOcNluAy3bPOHpN+T0sI3w2W4DJfhvnbcP7kQ'
                'J69XCDm6ni3Kv434Sfq//Q4ixLSY2KdkqX1vM42tEgaN4W4W9an7sxrZ2CL8'
                'mDF+SpKf0uf8YANRjrsD03K9OZz1IccdteprG662HGiF316M36Lkt/g5v8jb'
                'FMEsu2dh+l4d7pE4sJqeGPJctL9oP9kY4y/a3iuutHnrypUv9S9FGe1XYX/7'
                '99E0/lZXczmBi+5juRaDyyuj/dzr18pk/fr2F8oHHJe1/H/d4QwI')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, E.ExpandOrShrink))
        self.assertEqual(module.object_name, "Nuclei")
        self.assertEqual(module.output_object_name, "ShrunkenNuclei")
        self.assertEqual(module.operation, E.O_EXPAND)
        self.assertEqual(module.iterations, 3)
        self.assertFalse(module.wants_outlines.value)

        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, E.ExpandOrShrink))
        self.assertEqual(module.object_name, "ShrunkenNuclei")
        self.assertEqual(module.output_object_name, "DividedNuclei")
        self.assertEqual(module.operation, E.O_DIVIDE)

    def make_workspace(self,
                       labels,
                       operation,
                       iterations=1,
                       wants_outlines=False,
                       wants_fill_holes=False):
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = labels
        object_set.add_objects(objects, INPUT_NAME)
        module = E.ExpandOrShrink()
        module.object_name.value = INPUT_NAME
        module.output_object_name.value = OUTPUT_NAME
        module.outlines_name.value = OUTLINES_NAME
        module.operation.value = operation
        module.iterations.value = iterations
        module.wants_outlines.value = wants_outlines
        module.wants_fill_holes.value = wants_fill_holes
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set_list.get_image_set(0),
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        return workspace, module

    def test_02_01_expand(self):
        '''Expand an object once'''
        labels = np.zeros((10, 10), int)
        labels[4, 4] = 1
        expected = np.zeros((10, 10), int)
        expected[np.array([4, 3, 4, 5, 4], int), np.array([3, 4, 4, 4, 5], int)] = 1
        workspace, module = self.make_workspace(labels, E.O_EXPAND)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))
        self.assertTrue(OUTLINES_NAME not in workspace.get_outline_names())
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        count = m.get_current_image_measurement("Count_" + OUTPUT_NAME)
        if not np.isscalar(count):
            count = count[0]
        self.assertEqual(count, 1)
        location_x = m.get_current_measurement(OUTPUT_NAME, "Location_Center_X")
        self.assertEqual(len(location_x), 1)
        self.assertEqual(location_x[0], 4)
        location_y = m.get_current_measurement(OUTPUT_NAME, "Location_Center_Y")
        self.assertEqual(len(location_y), 1)
        self.assertEqual(location_y[0], 4)

    def test_02_02_expand_twice(self):
        '''Expand an object "twice"'''
        labels = np.zeros((10, 10), int)
        labels[4, 4] = 1
        i, j = np.mgrid[0:10, 0:10] - 4
        expected = (i ** 2 + j ** 2 <= 4).astype(int)
        workspace, module = self.make_workspace(labels, E.O_EXPAND, 2)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_02_03_expand_two(self):
        '''Expand two objects once'''
        labels = np.zeros((10, 10), int)
        labels[2, 3] = 1
        labels[6, 5] = 2
        i, j = np.mgrid[0:10, 0:10]
        expected = (((i - 2) ** 2 + (j - 3) ** 2 <= 1).astype(int) +
                    ((i - 6) ** 2 + (j - 5) ** 2 <= 1).astype(int) * 2)
        workspace, module = self.make_workspace(labels, E.O_EXPAND, 1)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_03_01_expand_inf(self):
        '''Expand two objects infinitely'''
        labels = np.zeros((10, 10), int)
        labels[2, 3] = 1
        labels[6, 5] = 2
        i, j = np.mgrid[0:10, 0:10]
        distance = (((i - 2) ** 2 + (j - 3) ** 2) -
                    ((i - 6) ** 2 + (j - 5) ** 2))
        workspace, module = self.make_workspace(labels, E.O_EXPAND_INF)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented[distance < 0] == 1))
        self.assertTrue(np.all(objects.segmented[distance > 0] == 2))

    def test_04_01_divide(self):
        '''Divide two touching objects'''
        labels = np.ones((10, 10), int)
        labels[5:, :] = 2
        expected = labels.copy()
        expected[4:6, :] = 0
        workspace, module = self.make_workspace(labels, E.O_DIVIDE)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_04_02_dont_divide(self):
        '''Don't divide an object that would disappear'''
        labels = np.ones((10, 10), int)
        labels[9, 9] = 2
        expected = labels.copy()
        expected[8, 9] = 0
        expected[8, 8] = 0
        expected[9, 8] = 0
        workspace, module = self.make_workspace(labels, E.O_DIVIDE)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_05_01_shrink(self):
        '''Shrink once'''
        labels = np.zeros((10, 10), int)
        labels[1:9, 1:9] = 1
        expected = morph.thin(labels, iterations=1)
        workspace, module = self.make_workspace(labels, E.O_SHRINK, 1)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_06_01_shrink_inf(self):
        '''Shrink infinitely'''
        labels = np.zeros((10, 10), int)
        labels[1:8, 1:8] = 1
        expected = np.zeros((10, 10), int)
        expected[4, 4] = 1
        workspace, module = self.make_workspace(labels, E.O_SHRINK_INF)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_06_02_shrink_inf_fill_holes(self):
        '''Shrink infinitely after filling a hole'''
        labels = np.zeros((10, 10), int)
        labels[1:8, 1:8] = 1
        labels[4, 4] = 0
        expected = np.zeros((10, 10), int)
        expected[4, 4] = 1
        # Test failure without filling the hole
        workspace, module = self.make_workspace(labels, E.O_SHRINK_INF)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertFalse(np.all(objects.segmented == expected))
        # Test success after filling the hole
        workspace, module = self.make_workspace(labels, E.O_SHRINK_INF,
                                                wants_fill_holes=True)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))

    def test_07_01_outlines(self):
        '''Create an outline of the resulting objects'''
        labels = np.zeros((10, 10), int)
        labels[4, 4] = 1
        i, j = np.mgrid[0:10, 0:10] - 4
        expected = (i ** 2 + j ** 2 <= 4).astype(int)
        expected_outlines = outline(expected)
        workspace, module = self.make_workspace(labels, E.O_EXPAND, 2,
                                                wants_outlines=True)
        module.run(workspace)
        objects = workspace.object_set.get_objects(OUTPUT_NAME)
        self.assertTrue(np.all(objects.segmented == expected))
        self.assertTrue(OUTLINES_NAME in workspace.image_set.names)
        outlines = workspace.image_set.get_image(OUTLINES_NAME).pixel_data
        self.assertTrue(np.all(outlines == expected_outlines))
