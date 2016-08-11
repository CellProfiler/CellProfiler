import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.enhanceedges
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import centrosome.filter
import centrosome.kirsch
import centrosome.otsu
import numpy

cellprofiler.preferences.set_headless()

INPUT_IMAGE_NAME = 'inputimage'
OUTPUT_IMAGE_NAME = 'outputimage'


class TestEnhanceEdges(unittest.TestCase):
    def make_workspace(self, image, mask=None):
        """Make a workspace for testing FindEdges"""
        module = cellprofiler.modules.enhanceedges.FindEdges()
        module.image_name.value = INPUT_IMAGE_NAME
        module.output_image_name.value = OUTPUT_IMAGE_NAME
        pipeline = cellprofiler.pipeline.Pipeline()
        object_set = cellprofiler.region.Set()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     cellprofiler.measurement.Measurements(),
                                                     image_set_list)
        image_set.add(INPUT_IMAGE_NAME,
                      cellprofiler.image.Image(image) if mask is None
                      else cellprofiler.image.Image(image, mask))
        return workspace, module

    def test_01_02_load_v1(self):
        """Load a Python pipeline with a version 1 FindEdges module"""
        data = ('eJztWNFu2jAUNZR2o5um7qHqHv0IW4kS1m4UTW0ZdBpaYahF7aqq61wwJZIT'
                'o+C0ZVOlPe4T9jn7hH3CPqGfMBsSCC5rIC2TNiXICvfmHp/rk2tjXMpVt3Ov'
                '4aqiwlKummroBMMKQaxBLSMLTbYM8xZGDNchNbOw2rRhCXVgWoPaSlZ7mVVX'
                'YVpV10CwK1IsPeK3X0sAzPH7fd6izqNZx454mrB3MWO6edqeBTHwxPH/4G0P'
                'WTo6IXgPERu3BxSuv2g2aLXT6j8q0bpNcBkZ3mB+lW3jBFvt9w0X6Dyu6BeY'
                '7OqfsTQEN2wHn+ltnZoO3ulf9vZ5KZN4hQ4/5wc6RCQdhC6LHr+IfwsG8bER'
                'uj32xC84tm7W9TO9biMCdQOd9rMQ/ak+/cWG+ouBfV4jArfpg1uQ8hCtii9Y'
                'ausC1Rg0EKs1x+knLvUjbJHDVn2ycUSG+omA52PiokO4KCjTHi7jg5uV8hY2'
                'a/KSGJN3Zgg/A1Qlfat8J+XTFHWseluSxinsAm4gmzBYFMUGC7qFa4xancC6'
                '5ZFpdkboNifh3cvFx517kPrIjImTdTvgVRkElyNkqnx3VRfH5+PVoYx74dRT'
                '0PVmWuOT37sGgs5PLdD4ytTEt6nrrz5878DwfBL2x8RG5ZX4wcfryrPksbD2'
                'MSE79Hz9MJeqHCVdT54S2zDXD9XU2tEXbTl92Qve1Tmy60z28wiaf9Mn/4yU'
                'v7BFDgcYWU5iK5fJlHCVqMmaji/t+AqoM/BMc/24Pk+u18MkfH7r5D1JF2EX'
                'LNoimP3VdTI9AneXfOF8+Tfny/f4ZPvaoDybPnrMS3oIu7sJPrWo3Zo+/6h9'
                '8IAf8q05bt3V/AlxIS7EhfM4xP3/uE0Q1muIC4678uDk/Zn8f1rEfwI319tT'
                'MFxvwq7xLXHLouJ82VKM7iFoWyEU1XunkMo2/1r0HEgKng8+PAmJJ/EnnoYo'
                'cXFIqLzh37rHhbJe8RH9e8cd5Z/FhzfrLOs70P1qIwhfNHKd74EPLuYoJXDf'
                'wGTvNXFDvDu2oPG/AYtINWo=')
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        p = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        p.add_listener(callback)
        p.load(fd)
        self.assertEqual(len(p.modules()), 2)
        module = p.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceedges.FindEdges))
        self.assertEqual(module.image_name.value, "Worm")
        self.assertEqual(module.output_image_name.value, "WormEdges")
        self.assertFalse(module.wants_automatic_threshold.value)
        self.assertEqual(module.manual_threshold.value, 0.2)
        self.assertEqual(module.threshold_adjustment_factor.value, 1)
        self.assertEqual(module.method.value, cellprofiler.modules.enhanceedges.M_CANNY)
        self.assertFalse(module.wants_automatic_sigma.value)
        self.assertEqual(module.sigma.value, 6.0)
        self.assertFalse(module.wants_automatic_low_threshold.value)
        self.assertEqual(module.low_threshold.value, 0.1)

    def test_01_03_load_v2(self):
        data = ('eJztWOFOGkEQXhBt0aSxP4z9uT+hlctBa6OkUaloSiqUKNEao3aFRTbZuyXH'
                'nkobk/7sI/Rx+gh9hD6Cj9BdvJNjRY87iz+aO7I5Zm6++WbmZpdly4XaVuE9'
                'XNR0WC7UMk1CMaxSxJvMMvLQ5Atw3cKI4wZkZh5uWgSWURfmclBfzmf1vL4I'
                'c7q+DMJdsVL5mbj9mQdgStyfihF3Hk06cswzpLyDOSfmaWcSJMALR/9LjF1k'
                'EXRC8S6iNu70KVx9yWyyWrd986jMGjbFFWR4jcVVsY0TbHU+NV2g87hKLjDd'
                'IV+xkoJrto3PSIcw08E7/lXtDS/jCq+sw+/pfh1iSh1kXeY8emn/AfTtE0Pq'
                '9txjP+vIxGyQM9KwEYXEQKc3UUh/uo+/xIC/BNgTPSJxaz64WSUOOWr4gmc2'
                'LlCdQwPxemsUP0nFj5RlDBuNYHnEBvzEwOsRcfEBXBxU2DVuyQc3qcQtZd4S'
                'LTEi78QAfgLoWu5B8Qbly2r6SP02r+Qp5SJuIptyWJLNBovEwnXOrG7ouq0j'
                '0+wOqduUgncvF5907mHyL1AaCrcvuvIx39OofCrurfN+g+KOz8fbh3etN8Hn'
                'SzbUupAF4+0zNb8KM/FD+L778H0Eg/NJykep1eo7+YOPV7RX6WMp7WFKt9n5'
                'ykEhUz1Mu5p1Rm3DXDnQM8uH37ILuctr4x0ikD1lemjeQeJv+cS/pMQvZRnD'
                'PkaWE9iby3RGqsrM5C1Hl3N0RdTtax5z/Tg+v91/Qfj81sknSl2kXLRYm2I+'
                '1v5V50tuCC6aL9F8+ZkMtq8Ny7PmU49ppR5S7m2CTy1mt8fPP2wf3OeHYmuO'
                '2/9q/kS4CBfhonkc4f5/3BqI+jXChcddeXDq/kz9/y7tv4D7++0lGOw3KdfF'
                'lrhtMXm+bGlG7xC0o1GGGtenkNqW+FryHEhKns8+PCmFJ3UXT1O2uDwk1DbF'
                't95xoVqv5BD/3rzj4jM3c3+d1fr26361GoYvHr/NN+ODSziVkrgfINh7Td1j'
                '7+YW1v4vvdw1GA==')
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        p = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        p.add_listener(callback)
        p.load(fd)
        self.assertEqual(len(p.modules()), 2)
        module = p.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.enhanceedges.FindEdges))
        self.assertEqual(module.image_name.value, "Worm")
        self.assertEqual(module.output_image_name.value, "WormEdges")
        self.assertFalse(module.wants_automatic_threshold.value)
        self.assertEqual(module.manual_threshold.value, 0.2)
        self.assertEqual(module.threshold_adjustment_factor.value, 1)
        self.assertEqual(module.method.value, cellprofiler.modules.enhanceedges.M_CANNY)
        self.assertFalse(module.wants_automatic_sigma.value)
        self.assertEqual(module.sigma.value, 6.0)
        self.assertFalse(module.wants_automatic_low_threshold.value)
        self.assertEqual(module.low_threshold.value, 0.1)

    def test_02_01_sobel_horizontal(self):
        """Test the Sobel horizontal transform"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_SOBEL
        module.direction.value = cellprofiler.modules.enhanceedges.E_HORIZONTAL
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == centrosome.filter.hsobel(image)))

    def test_02_02_sobel_vertical(self):
        """Test the Sobel vertical transform"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_SOBEL
        module.direction.value = cellprofiler.modules.enhanceedges.E_VERTICAL
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == centrosome.filter.vsobel(image)))

    def test_02_03_sobel_all(self):
        """Test the Sobel transform"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_SOBEL
        module.direction.value = cellprofiler.modules.enhanceedges.E_ALL
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == centrosome.filter.sobel(image)))

    def test_03_01_prewitt_horizontal(self):
        """Test the prewitt horizontal transform"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_PREWITT
        module.direction.value = cellprofiler.modules.enhanceedges.E_HORIZONTAL
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == centrosome.filter.hprewitt(image)))

    def test_03_02_prewitt_vertical(self):
        """Test the prewitt vertical transform"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_PREWITT
        module.direction.value = cellprofiler.modules.enhanceedges.E_VERTICAL
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == centrosome.filter.vprewitt(image)))

    def test_03_03_prewitt_all(self):
        """Test the prewitt transform"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_PREWITT
        module.direction.value = cellprofiler.modules.enhanceedges.E_ALL
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == centrosome.filter.prewitt(image)))

    def test_04_01_roberts(self):
        """Test the roberts transform"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_ROBERTS
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        self.assertTrue(numpy.all(output.pixel_data == centrosome.filter.roberts(image)))

    def test_05_01_log_automatic(self):
        """Test the laplacian of gaussian with automatic sigma"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_LOG
        module.sigma.value = 20
        module.wants_automatic_sigma.value = True
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        sigma = 2.0
        expected = centrosome.filter.laplacian_of_gaussian(image,
                                                           numpy.ones(image.shape, bool),
                                                           int(sigma * 4) + 1,
                                                           sigma).astype(numpy.float32)

        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_05_02_log_manual(self):
        """Test the laplacian of gaussian with manual sigma"""
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(20, 20)).astype(numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_LOG
        module.sigma.value = 4
        module.wants_automatic_sigma.value = False
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        sigma = 4.0
        expected = centrosome.filter.laplacian_of_gaussian(image,
                                                           numpy.ones(image.shape, bool),
                                                           int(sigma * 4) + 1,
                                                           sigma).astype(numpy.float32)

        self.assertTrue(numpy.all(output.pixel_data == expected))

    def test_06_01_canny(self):
        """Test the canny method"""
        i, j = numpy.mgrid[-20:20, -20:20]
        image = numpy.logical_and(i > j, i ** 2 + j ** 2 < 300).astype(numpy.float32)
        numpy.random.seed(0)
        image = image * .5 + numpy.random.uniform(size=image.shape) * .3
        image = numpy.ascontiguousarray(image, numpy.float32)
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_CANNY
        module.wants_automatic_threshold.value = True
        module.wants_automatic_low_threshold.value = True
        module.wants_automatic_sigma.value = True
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        t1, t2 = centrosome.otsu.otsu3(centrosome.filter.sobel(image))
        result = centrosome.filter.canny(image, numpy.ones(image.shape, bool), 1.0, t1, t2)
        self.assertTrue(numpy.all(output.pixel_data == result))

    def test_07_01_kirsch(self):
        r = numpy.random.RandomState([ord(_) for _ in "test_07_01_kirsch"])
        i, j = numpy.mgrid[-20:20, -20:20]
        image = (numpy.sqrt(i * i + j * j) <= 10).astype(float) * .5
        image = image + r.uniform(size=image.shape) * .1
        workspace, module = self.make_workspace(image)
        module.method.value = cellprofiler.modules.enhanceedges.M_KIRSCH
        module.run(workspace)
        output = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
        result = centrosome.kirsch.kirsch(image)
        numpy.testing.assert_almost_equal(output.pixel_data, result, decimal=4)
