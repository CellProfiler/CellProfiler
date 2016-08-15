'''test_morph - test the morphology module
'''

import StringIO
import base64
import unittest
import zlib

import numpy as np
import scipy.ndimage as scind

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw
import cellprofiler.modules.morph as morph
import centrosome.cpmorphology as cpmorph
import centrosome.filter as cpfilter


class TestMorph(unittest.TestCase):
    def test_01_01_load_matlab(self):
        '''Load a pipeline that has a morph module in it'''
        #
        # Pipeline has Morph as second module
        # input_name = OrigBlue
        # output_name = MorphBlue
        # dilate once
        # erode twice
        # fill forever
        #
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrV'
                'SCHAO9/TTUXAuSk0sSU1RyM+zUggpTVXwKs1TMDBSMDS'
                '0MjKyMjZTMDIwsFQgGTAwevryMzAw/GdkYKiYczfsrtd'
                'hAxG7t7sMWdurVBdd8M7/s/9jkOvBxjzn4w4f1NZF8qa'
                'cfXF+hl1j/2PlHxlqm29f2jFzq7awD+fTJc9/nzs5fbK'
                'JKMOPGGaNj+Uzy4X3RRtu413J/ZuxXfHjnQUcYjlHH8/'
                'fs32C/wXjo7ouyjIHdvQeWx3Xbhpf/di//EynxrqF696'
                'cidZ9ZZdsv6Lc7cCP4mYeg1qFfIlJuU/K+n+rKJzp/mi'
                '438qq77STap7rEQ3/EgMR2Ra92Zd2fLBv79xv6rq99GS'
                '97TtfG47v69lXyN6KFdkWz3J+ZtvyNR84dr8T/j7JW/e'
                'Du9SeqQeiC0ovrjg3+X+XUHnqivn3qwUbVWXFzsa362t'
                'WJFq3LBI4v51vNVfcjfAz3yIy3BnnnXFevnjiA8MHEp9'
                '6119euOIDh/TvMwctn08Ma5+sejz89U5njdi0P05Nyxn'
                '+5Jwu/Xd6VvuuvuWShZXvm/dU5JnYSHP1b+h8PsGuUMk'
                'l95Lrux/6Gzv0rObMPDAzILk+mp2d+1z49O9b/ne9Xnx'
                '9Qe+KP/EzzetL7e0+hXo3z19af+3HPUWvB+fP3H8vukn'
                'OyG/54y1f5cXWvt68Yvq8/Fu1qYfSP69trK78tOLn07n'
                'P6x/+b7j5dfG1/Gu7dqaeWOnFHnft94kPdRz23309J28'
                '/Of+sV93zNzYfu9yq5hydv3zu29jG6vaS15u+d/9fkfr'
                '1P9NGrROuAE1uKvg=')
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, morph.Morph))
        self.assertEqual(module.image_name.value, "OrigBlue")
        self.assertEqual(module.output_image_name.value, "MorphBlue")
        self.assertEqual(len(module.functions), 3)
        self.assertEqual(module.functions[0].function, morph.F_DILATE)
        self.assertEqual(module.functions[0].repeats_choice.value, morph.R_ONCE)
        self.assertEqual(module.functions[1].function, morph.F_ERODE)
        self.assertEqual(module.functions[1].repeats_choice.value, morph.R_CUSTOM)
        self.assertEqual(module.functions[1].custom_repeats.value, 2)
        self.assertEqual(module.functions[2].function, morph.F_FILL)
        self.assertEqual(module.functions[2].repeats_choice.value, morph.R_FOREVER)

    def test_01_02_load_v1(self):
        '''Load a revision 1 pipeline that has a morph module in it'''
        #
        # Pipeline has Morph as second module
        # input_name = OrigBlue
        # output_name = MorphBlue
        # dilate once
        # erode twice
        # fill forever
        #
        data = ('eJztWN1u2jAUdhDQskpre7Vd+nKa1ghYJ23ctLRsGlP50Yp6PZccqC'
                'UnjhwH0T3BHmOPssfp5R5hMU1IcClJI7qrBFnm2Oc7n89nJ47Ta48u'
                '2mf4g1nHvfboaEIZ4CEjcsKF3cKOfIfPBRAJFuZOC498wN98B9ebuN'
                'FoHTdbxw3crNc/oXyX0e29DKrfhwhVg3o3KKWwqxLaRqIo+xKkpM7U'
                'q6Ayeh22/wnKFRGUXDO4IswHL6aI2rvOhI9u3WVXj1s+gz6xk87B1f'
                'ftaxDeYBIBw+4hnQO7pD9BSyFy+w4z6lHuhPgwvt665OVS41U6fN2J'
                'dTA0HZQu+4n2hT+K/ctrdDtM+B+ENnUsOqOWTximNpkuR6HifUyJt6'
                'vFU/ZA0OlZILnCn6bgDzS8KiOYy6PPczKW2CZyfJMlTk2LU1voKtwb'
                'NZBEPvWUOMZKHAO9z6hDVeNXtkWDmwbu8Wnjf6Hhld3h2OES+x5kH3'
                '95JU4ZDZwxZMGVVnAl1Of59GqgbOvwlZavsjswIT6TuKsWIe5QAWPJ'
                'xW0m/StaPGWD4BZk1E3Pw0T55/3c9yS3s/FuS/dmxjzz8unrKtgUWB'
                '5cpz3sZtF1R9NV2V+4gBmIrTyXnkvfx/JN4qoaLroiXC2s8/D1ufMg'
                'v4Iv5rurPm0/zctzmpLXuuf9YvOdCu67z8+/bt+N+XHwSgDutua1wB'
                'W4AlfcxwXu/+PuEjh9v9PfB5X/D7R5vb1Fq+tN2WNgzBVcfScQpr04'
                'zHom48S6P02aF8HfbuJgqXiGKTxY48GP8djqkGcujnq6TrU1cZP5lo'
                'Lf/t5mfXVdY73/nuThKxkP+fZScOVQIYX7hZ42n282+Ee55fX/B9HP'
                '158=')
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, morph.Morph))
        self.assertEqual(module.image_name.value, "OrigBlue")
        self.assertEqual(module.output_image_name.value, "MorphBlue")
        self.assertEqual(len(module.functions), 3)
        self.assertEqual(module.functions[0].function, morph.F_DILATE)
        self.assertEqual(module.functions[0].repeats_choice.value, morph.R_ONCE)
        self.assertEqual(module.functions[1].function, morph.F_ERODE)
        self.assertEqual(module.functions[1].repeats_choice.value, morph.R_CUSTOM)
        self.assertEqual(module.functions[1].custom_repeats.value, 2)
        self.assertEqual(module.functions[2].function, morph.F_FILL)
        self.assertEqual(module.functions[2].repeats_choice.value, morph.R_FOREVER)

    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10016

Morph:[module_num:1|svn_version:\'9935\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Select the input image:InputImage
    Name the output image:MorphImage
    Select the operation to perform:bothat
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:branchpoints
    Repeat operation:Forever
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:bridge
    Repeat operation:Custom
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:clean
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:close
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:convex hull
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:diag
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:dilate
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:distance
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:endpoints
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:erode
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:fill
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:fill small holes
    Repeat operation:Once
    Maximum hole area:2
    Scale\x3A:3
    Select the operation to perform:hbreak
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:invert
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:majority
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:open
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:remove
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:shrink
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:skel
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:skelpe
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:spur
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:thicken
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:thin
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:tophat
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
    Select the operation to perform:vbreak
    Repeat operation:Once
    Custom # of repeats:2
    Scale\x3A:3
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        ops = [morph.F_BOTHAT, morph.F_BRANCHPOINTS, morph.F_BRIDGE,
               morph.F_CLEAN, morph.F_CLOSE, morph.F_CONVEX_HULL,
               morph.F_DIAG, morph.F_DILATE, morph.F_DISTANCE,
               morph.F_ENDPOINTS, morph.F_ERODE, morph.F_FILL,
               morph.F_FILL_SMALL, morph.F_HBREAK, morph.F_INVERT,
               morph.F_MAJORITY, morph.F_OPEN, morph.F_REMOVE,
               morph.F_SHRINK, morph.F_SKEL, morph.F_SKELPE, morph.F_SPUR,
               morph.F_THICKEN, morph.F_THIN, morph.F_TOPHAT, morph.F_VBREAK]
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, morph.Morph))
        self.assertEqual(module.image_name, "InputImage")
        self.assertEqual(module.output_image_name, "MorphImage")
        self.assertEqual(len(module.functions), len(ops))
        for op, function in zip(ops, module.functions):
            self.assertEqual(function.function, op)
            if op == morph.F_BRANCHPOINTS:
                self.assertEqual(function.repeats_choice, morph.R_FOREVER)
            elif op == morph.F_BRIDGE:
                self.assertEqual(function.repeats_choice, morph.R_CUSTOM)
            else:
                self.assertEqual(function.repeats_choice, morph.R_ONCE)
            self.assertEqual(function.custom_repeats, 2)
            self.assertEqual(function.scale, 3)
        fn0 = module.functions[0]
        self.assertEqual(fn0.structuring_element, morph.SE_DISK)
        self.assertEqual(fn0.x_offset, 1)
        self.assertEqual(fn0.y_offset, 1)
        self.assertEqual(fn0.angle, 0)
        self.assertEqual(fn0.width, 3)
        self.assertEqual(fn0.height, 3)
        strel = np.array(fn0.strel.get_matrix())
        self.assertEqual(strel.shape[0], 3)
        self.assertEqual(strel.shape[1], 3)
        self.assertTrue(np.all(strel))

    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20130402163959
ModuleCount:1
HasImagePlaneDetails:False

Morph:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Select the input image:Thresh
    Name the output image:MorphedThresh
    Select the operation to perform:dilate
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:7.0
    Structuring element:Custom
    X offset:1
    Y offset:1
    Angle:45
    Width:3
    Height:3
    Custom:5,7,11111110000000011111000000001111111
    Select the operation to perform:close
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:3
    Structuring element:Diamond
    X offset:1
    Y offset:1
    Angle:0
    Width:3
    Height:3
    Custom:5,5,1111111111111111111111111
    Select the operation to perform:open
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:4.0
    Structuring element:Line
    X offset:1
    Y offset:1
    Angle:60.0
    Width:3
    Height:3
    Custom:5,5,1111111111111111111111111
    Select the operation to perform:erode
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:3
    Structuring element:Octagon
    X offset:1
    Y offset:1
    Angle:0
    Width:3
    Height:3
    Custom:5,5,1111111111111111111111111
    Select the operation to perform:close
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:3
    Structuring element:Pair
    X offset:4.0
    Y offset:2.0
    Angle:0
    Width:3
    Height:3
    Custom:5,5,1111111111111111111111111
    Select the operation to perform:dilate
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:15.0
    Structuring element:Periodic line
    X offset:-1.0
    Y offset:-3.0
    Angle:0
    Width:3
    Height:3
    Custom:5,5,1111111111111111111111111
    Select the operation to perform:erode
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:3
    Structuring element:Rectangle
    X offset:1
    Y offset:1
    Angle:0
    Width:5.0
    Height:8.0
    Custom:5,5,1111111111111111111111111
    Select the operation to perform:open
    Number of times to repeat operation:Once
    Repetition number:2
    Scale:9.0
    Structuring element:Square
    X offset:1
    Y offset:1
    Angle:0
    Width:3
    Height:3
    Custom:5,5,1111111111111111111111111
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        module = pipeline.modules()[0]
        assert isinstance(module, morph.Morph)
        self.assertEqual(module.image_name, "Thresh")
        self.assertEqual(module.output_image_name, "MorphedThresh")
        self.assertEqual(len(module.functions), 8)
        f0 = module.functions[0]
        matrix = f0.strel.get_matrix()
        self.assertEqual(len(matrix), 5)
        self.assertEqual(len(matrix[0]), 7)
        expected = [[1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1]]
        np.testing.assert_array_equal(matrix, np.array(expected, bool))
        for f, se in zip(module.functions, (
                morph.SE_ARBITRARY, morph.SE_DIAMOND, morph.SE_LINE,
                morph.SE_OCTAGON, morph.SE_PAIR, morph.SE_PERIODIC_LINE,
                morph.SE_RECTANGLE, morph.SE_SQUARE)):
            self.assertEqual(f.structuring_element, se)
            if se == morph.SE_LINE:
                self.assertEqual(f.angle, 60)
                self.assertEqual(f.scale, 4.0)
            elif se == morph.SE_PERIODIC_LINE:
                self.assertEqual(f.x_offset, -1)
                self.assertEqual(f.y_offset, -3)
            elif se == morph.SE_RECTANGLE:
                self.assertEqual(f.width, 5)
                self.assertEqual(f.height, 8)

    def execute(self, image, function, mask=None, custom_repeats=None,
                scale=None, module=None):
        '''Run the morph module on an input and return the resulting image'''
        INPUT_IMAGE_NAME = 'input'
        OUTPUT_IMAGE_NAME = 'output'
        if module is None:
            module = morph.Morph()
        module.functions[0].function.value = function
        module.image_name.value = INPUT_IMAGE_NAME
        module.output_image_name.value = OUTPUT_IMAGE_NAME
        if custom_repeats is None:
            module.functions[0].repeats_choice.value = morph.R_ONCE
        elif custom_repeats == -1:
            module.functions[0].repeats_choice.value = morph.R_FOREVER
        else:
            module.functions[0].repeats_choice.value = morph.R_CUSTOM
            module.functions[0].custom_repeats.value = custom_repeats
        if scale is not None:
            module.functions[0].scale.value = scale
        pipeline = cpp.Pipeline()
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        image_set.add(INPUT_IMAGE_NAME, cpi.Image(image, mask=mask))
        module.run(workspace)
        output = image_set.get_image(OUTPUT_IMAGE_NAME)
        return output.pixel_data

    def binary_tteesstt(self, function_name, function,
                        gray_out=False, scale=None, custom_repeats=None):
        np.random.seed(map(ord, function_name))
        input = np.random.uniform(size=(20, 20)) > .7
        output = self.execute(input, function_name, scale=scale, custom_repeats=custom_repeats)
        if scale is None:
            expected = function(input)
        else:
            footprint = cpmorph.strel_disk(float(scale) / 2.0)
            expected = function(input, footprint=footprint)
        if not gray_out:
            expected = expected > 0
            self.assertTrue(np.all(output == expected))
        else:
            self.assertTrue(np.all(np.abs(output - expected) < np.finfo(np.float32).eps))

    def test_02_01_binary_bothat(self):
        self.binary_tteesstt('bothat', cpmorph.black_tophat, scale=5)

    def test_02_015_binary_branchpoints(self):
        self.binary_tteesstt('branchpoints', cpmorph.branchpoints)

    def test_02_02_binary_bridge(self):
        self.binary_tteesstt('bridge', cpmorph.bridge)

    def test_02_03_binary_clean(self):
        self.binary_tteesstt('clean', cpmorph.clean)

    def test_02_04_binary_close(self):
        self.binary_tteesstt(
                'close', lambda x, footprint: scind.binary_closing(x, footprint),
                scale=4)

    def test_02_05_binary_diag(self):
        self.binary_tteesstt('diag', cpmorph.diag)

    def test_02_06_binary_dilate(self):
        self.binary_tteesstt(
                'dilate', lambda x, footprint: scind.binary_dilation(x, footprint),
                scale=7)

    def test_02_065_binary_endpoints(self):
        self.binary_tteesstt('endpoints', cpmorph.endpoints)

    def test_02_07_binary_erode(self):
        self.binary_tteesstt('erode', lambda x: scind.binary_erosion(x, np.ones((3, 3), bool)))

    def test_02_08_binary_fill(self):
        self.binary_tteesstt('fill', cpmorph.fill)

    def test_02_08a_binary_fill_small(self):
        def small_hole_fn(area, is_foreground):
            return area <= 2

        def fun(im, footprint=None):
            return cpmorph.fill_labeled_holes(im, size_fn=small_hole_fn)

        self.binary_tteesstt('fill small holes', fun, custom_repeats=2)

    def test_02_09_binary_hbreak(self):
        self.binary_tteesstt('hbreak', cpmorph.hbreak)

    def test_02_10_binary_majority(self):
        self.binary_tteesstt('majority', cpmorph.majority)

    def test_02_11_binary_open(self):
        self.binary_tteesstt(
                'open', lambda x, footprint: scind.binary_opening(x, footprint),
                scale=5)

    def test_02_12_binary_remove(self):
        self.binary_tteesstt('remove', cpmorph.remove)

    def test_02_13_binary_shrink(self):
        self.binary_tteesstt('shrink', lambda x: cpmorph.binary_shrink(x, 1))

    def test_02_14_binary_skel(self):
        self.binary_tteesstt('skel', cpmorph.skeletonize)

    def test_02_15_binary_spur(self):
        self.binary_tteesstt('spur', cpmorph.spur)

    def test_02_16_binary_thicken(self):
        self.binary_tteesstt('thicken', cpmorph.thicken)

    def test_02_17_binary_thin(self):
        self.binary_tteesstt('thin', cpmorph.thin)

    def test_02_18_binary_tophat(self):
        self.binary_tteesstt('tophat', cpmorph.white_tophat)

    def test_02_19_binary_vbreak(self):
        self.binary_tteesstt('vbreak', cpmorph.vbreak)

    def test_02_20_binary_distance(self):
        def distance(x):
            y = scind.distance_transform_edt(x)
            if np.max(y) == 0:
                return y
            else:
                return y / np.max(y)

        self.binary_tteesstt('distance', distance, True)

    def test_02_21_binary_convex_hull(self):
        #
        # Set the four points of a square to True
        #
        image = np.zeros((20, 15), bool)
        image[2, 3] = True
        image[17, 3] = True
        image[2, 12] = True
        image[17, 12] = True
        expected = np.zeros((20, 15), bool)
        expected[2:18, 3:13] = True
        result = self.execute(image, 'convex hull')
        self.assertTrue(np.all(result == expected))

    def test_02_22_binary_invert(self):
        def invert(x):
            return ~ x

        self.binary_tteesstt('invert', invert, True)

    def test_02_23_gray_invert(self):
        np.random.seed(0)
        image = np.random.uniform(size=(20, 15)).astype(np.float32)
        result = self.execute(image, 'invert')
        self.assertTrue(np.all(result == (1 - image)))

    def test_02_24_gray_open(self):
        np.random.seed(0)
        image = np.random.uniform(size=(20, 15)).astype(np.float32)
        result = self.execute(image, 'open', mask=np.ones(image.shape, np.bool))
        self.assertTrue(np.all(result <= image))

    def test_02_25_gray_close(self):
        np.random.seed(0)
        image = np.random.uniform(size=(20, 15)).astype(np.float32)
        result = self.execute(image, 'close', mask=np.ones(image.shape, np.bool))
        self.assertTrue(np.all(result >= image))

    def test_02_26_binary_skelpe(self):
        def fn(x):
            d = scind.distance_transform_edt(x)
            pe = cpfilter.poisson_equation(x)
            return cpmorph.skeletonize(x, ordering=pe * d)

        self.binary_tteesstt('skelpe', fn)

    def test_03_01_color(self):
        # Regression test for issue # 324
        np.random.seed(0)
        image = np.random.uniform(size=(20, 15)).astype(np.float32)
        image = image[:, :, np.newaxis] * np.ones(3)[np.newaxis, np.newaxis, :]
        result = self.execute(image, morph.F_ERODE,
                              mask=np.ones(image.shape[:2], np.bool))
        self.assertTrue(np.all(result < image[:, :, 0] + np.finfo(np.float32).eps))

    def test_04_01_strel_arbitrary(self):
        r = np.random.RandomState()
        r.seed(41)
        strel = r.uniform(size=(5, 3)) > .5
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_ARBITRARY
        module.functions[0].strel.value_text = cps.BinaryMatrix.to_value(strel)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_02_strel_diamond(self):
        r = np.random.RandomState()
        r.seed(42)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_DIAMOND
        strel = cpmorph.strel_diamond(3.5)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, scale=7,
                              module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_03_strel_disk(self):
        r = np.random.RandomState()
        r.seed(43)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_DISK
        strel = cpmorph.strel_disk(3.5)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, scale=7,
                              module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_04_strel_line(self):
        r = np.random.RandomState()
        r.seed(44)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_LINE
        module.functions[0].angle.value = 75
        strel = cpmorph.strel_line(15, 75)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, scale=15,
                              module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_05_strel_octagon(self):
        r = np.random.RandomState()
        r.seed(45)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_OCTAGON
        strel = cpmorph.strel_octagon(7.5)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, scale=15,
                              module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_06_strel_pair(self):
        r = np.random.RandomState()
        r.seed(46)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_PAIR
        module.functions[0].x_offset.value = -1
        module.functions[0].y_offset.value = 4
        strel = cpmorph.strel_pair(-1, 4)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_07_strel_periodicline(self):
        r = np.random.RandomState()
        r.seed(43)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_PERIODIC_LINE
        module.functions[0].x_offset.value = 4
        module.functions[0].y_offset.value = -3
        strel = cpmorph.strel_periodicline(4, -3, 3)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, scale=30,
                              module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_08_strel_disk(self):
        r = np.random.RandomState()
        r.seed(48)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_RECTANGLE
        module.functions[0].width.value = 5
        module.functions[0].height.value = 9
        strel = cpmorph.strel_rectangle(5, 9)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, module=module)
        np.testing.assert_array_equal(expected, result)

    def test_04_09_strel_square(self):
        r = np.random.RandomState()
        r.seed(49)
        module = morph.Morph()
        module.functions[0].structuring_element.value = morph.SE_SQUARE
        strel = cpmorph.strel_square(7)
        pixel_data = r.uniform(size=(20, 30)) > .5
        expected = scind.binary_dilation(pixel_data, strel)
        result = self.execute(pixel_data, morph.F_DILATE, scale=7,
                              module=module)
        np.testing.assert_array_equal(expected, result)
