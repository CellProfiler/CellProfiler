import StringIO
import base64
import unittest
import zlib

import numpy
import numpy.testing
import pytest
import skimage.util

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.imagemath
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace
from functools import reduce

cellprofiler.preferences.set_headless()

MEASUREMENT_NAME = 'mymeasurement'


@pytest.fixture(scope="function")
def module():
    return cellprofiler.modules.imagemath.ImageMath()


@pytest.fixture(scope="function")
def workspace(image_a, image_b, module):
    image_set_list = cellprofiler.image.ImageSetList()
    image_set = image_set_list.get_image_set(0)
    workspace = cellprofiler.workspace.Workspace(
        image_set=image_set,
        image_set_list=image_set_list,
        module=module,
        pipeline=cellprofiler.pipeline.Pipeline(),
        measurements=cellprofiler.measurement.Measurements(),
        object_set=cellprofiler.object.ObjectSet()
    )

    workspace.image_set.add("input_a", image_a)
    workspace.image_set.add("input_b", image_b)

    module.images[0].image_name.value = "input_a"
    module.images[0].factor.value = 1.0

    module.images[1].image_name.value = "input_b"
    module.images[1].factor.value = 1.0

    module.truncate_low.value = False
    module.truncate_high.value = False

    module.output_image_name.value = "output"

    return workspace


def run_operation(operation, expected, module, workspace):
    module.operation.value = operation
    module.run(workspace)
    output = workspace.image_set.get_image("output")
    actual = output.pixel_data
    numpy.testing.assert_array_equal(actual, expected)


class TestVolumes(object):
    @staticmethod
    @pytest.fixture(scope="function")
    def image_a():
        k, i, j = numpy.mgrid[-5:6, -5:6, -5:10]
        data_a = numpy.zeros((11, 11, 15))
        data_a[k ** 2 + i ** 2 + j ** 2 <= 25] = 1

        image_a = cellprofiler.image.Image()
        image_a.pixel_data = data_a
        image_a.dimensions = 3

        return image_a

    @staticmethod
    @pytest.fixture(scope="function")
    def image_b():
        k, i, j = numpy.mgrid[-5:6, -5:6, -10:5]
        data_b = numpy.zeros((11, 11, 15))
        data_b[k ** 2 + i ** 2 + j ** 2 <= 25] = 0.5

        image_b = cellprofiler.image.Image()
        image_b.pixel_data = data_b
        image_b.dimensions = 3

        return image_b

    @staticmethod
    def test_add(image_a, image_b, module, workspace):
        operation = "Add"
        expected = image_a.pixel_data + image_b.pixel_data
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_subtract(image_a, image_b, module, workspace):
        operation = "Subtract"
        expected = image_a.pixel_data - image_b.pixel_data
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_absolute_difference(image_a, image_b, module, workspace):
        operation = "Absolute Difference"
        expected = numpy.abs(image_a.pixel_data - image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_multiply(image_a, image_b, module, workspace):
        operation = "Multiply"
        expected = image_a.pixel_data * image_b.pixel_data
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_divide(image_a, image_b, module, workspace):
        operation = "Divide"
        expected = image_a.pixel_data / image_b.pixel_data
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_average(image_a, image_b, module, workspace):
        operation = "Average"
        expected = (image_a.pixel_data + image_b.pixel_data) / 2.0
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_minimum(image_a, image_b, module, workspace):
        operation = "Minimum"
        expected = numpy.minimum(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_maximum(image_a, image_b, module, workspace):
        operation = "Maximum"
        expected = numpy.maximum(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_invert(image_a, module, workspace):
        operation = "Invert"
        expected = skimage.util.invert(image_a.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_log_transform(image_a, module, workspace):
        operation = "Log transform (base 2)"
        expected = numpy.log2(image_a.pixel_data + 1)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_and(image_a, image_b, module, workspace):
        operation = "And"
        expected = 1.0 * numpy.logical_and(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_or(image_a, image_b, module, workspace):
        operation = "Or"
        expected = numpy.logical_or(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_not(image_a, module, workspace):
        operation = "Not"
        expected = numpy.logical_not(image_a.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_equals(image_a, image_b, module, workspace):
        operation = "Equals"
        expected = image_a.pixel_data == image_b.pixel_data
        run_operation(operation, expected, module, workspace)


class TestBinaryImages(object):
    @staticmethod
    @pytest.fixture()
    def image_a():
        data_a = numpy.random.rand(128, 128) > 0.5

        image_a = cellprofiler.image.Image()
        image_a.pixel_data = data_a
        image_a.dimensions = 2

        return image_a

    @staticmethod
    @pytest.fixture()
    def image_b():
        data_b = numpy.random.rand(128, 128) > 0.5

        image_b = cellprofiler.image.Image()
        image_b.pixel_data = data_b
        image_b.dimensions = 2

        return image_b

    @staticmethod
    def test_add(image_a, image_b, module, workspace):
        operation = "Add"
        expected = numpy.logical_or(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_subtract(image_a, image_b, module, workspace):
        operation = "Subtract"
        expected = numpy.logical_xor(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_absolute_difference(image_a, image_b, module, workspace):
        operation = "Absolute Difference"
        expected = numpy.logical_xor(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_multiply(image_a, image_b, module, workspace):
        operation = "Multiply"
        expected = numpy.logical_and(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_divide(image_a, image_b, module, workspace):
        operation = "Divide"
        expected = numpy.logical_and(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_average(image_a, image_b, module, workspace):
        operation = "Average"
        expected = numpy.logical_or(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_minimum(image_a, image_b, module, workspace):
        operation = "Minimum"
        expected = numpy.logical_and(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_maximum(image_a, image_b, module, workspace):
        operation = "Maximum"
        expected = numpy.logical_or(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_invert(image_a, module, workspace):
        operation = "Invert"
        expected = numpy.logical_not(image_a.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_log_transform(image_a, module, workspace):
        operation = "Log transform (base 2)"
        expected = image_a.pixel_data
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_and(image_a, image_b, module, workspace):
        operation = "And"
        expected = numpy.logical_and(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_or(image_a, image_b, module, workspace):
        operation = "Or"
        expected = numpy.logical_or(image_a.pixel_data, image_b.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_not(image_a, module, workspace):
        operation = "Not"
        expected = numpy.logical_not(image_a.pixel_data)
        run_operation(operation, expected, module, workspace)

    @staticmethod
    def test_equals(image_a, image_b, module, workspace):
        operation = "Equals"
        expected = image_a.pixel_data == image_b.pixel_data
        run_operation(operation, expected, module, workspace)


class TestImageMath(unittest.TestCase):
    def test_01_000_load_subtract(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

Subtract:[module_num:1|svn_version:\'8913\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Subtract this image\x3A:MySubtrahend
    From this image\x3A:MyMinuend
    What do you want to call the resulting image?:MyOutput
    Enter the factor to multiply the first image by before subtracting\x3A:1.5
    Enter the factor to multiply the second image by before subtracting\x3A:2.6
    Do you want negative values in the image to be set to zero?:Yes
    """
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(len(module.images), 2)
        self.assertEqual(module.images[0].image_name, "MyMinuend")
        self.assertEqual(module.images[1].image_name, "MySubtrahend")
        self.assertEqual(module.output_image_name, "MyOutput")
        self.assertEqual(module.operation, cellprofiler.modules.imagemath.O_SUBTRACT)
        self.assertAlmostEqual(module.images[0].factor.value, 2.6)
        self.assertAlmostEqual(module.images[1].factor.value, 1.5)
        self.assertTrue(module.truncate_low)
        self.assertFalse(module.truncate_high)

    def test_01_001_load_combine(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

Combine:[module_num:1|svn_version:\'8913\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    What did you call the first image to be combined?:MyFirstImage
    What did you call the second image to be combined?:MySecondImage
    What did you call the third image to be combined?:Do not use
    What do you want to call the combined image?:MyOutputImage
    Enter the weight you want to give the first image:0.2
    Enter the weight you want to give the second image:0.7
    Enter the weight you want to give the third image:.10
    """
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(len(module.images), 2)
        self.assertEqual(module.images[0].image_name, "MyFirstImage")
        self.assertEqual(module.images[1].image_name, "MySecondImage")
        self.assertAlmostEqual(module.images[0].factor.value, 0.2)
        self.assertAlmostEqual(module.images[1].factor.value, 0.7)
        self.assertAlmostEqual(module.after_factor.value, 1.0 / 0.9)
        self.assertEqual(module.operation.value, cellprofiler.modules.imagemath.O_ADD)
        self.assertEqual(module.output_image_name, "MyOutputImage")

    def test_01_002_load_invert_intensity(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

InvertIntensity:[module_num:1|svn_version:\'8913\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    What did you call the image to be inverted (made negative)?:MyImage
    What do you want to call the inverted image?:MyInvertedImage
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(len(module.images), 2)  # there are two, but only 1 shown
        self.assertEqual(module.images[0].image_name, "MyImage")
        self.assertEqual(module.output_image_name, "MyInvertedImage")
        self.assertEqual(module.operation, cellprofiler.modules.imagemath.O_INVERT)

    def test_01_003_load_multiply(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

Multiply:[module_num:1|svn_version:\'8913\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    What is the name of the first image you would like to use:MyFirstImage
    What is the name of the second image you would like to use:MySecondImage
    What do you want to call the resulting image?:MyOutputImage
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(len(module.images), 2)
        self.assertEqual(module.images[0].image_name, "MyFirstImage")
        self.assertEqual(module.images[1].image_name, "MySecondImage")
        self.assertEqual(module.images[0].factor, 1)
        self.assertEqual(module.images[1].factor, 1)
        self.assertEqual(module.output_image_name, "MyOutputImage")

    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUgjJKFXwKs1RMDQDIitDCytjMwUjAwNLBZIBA6OnLz8DA0MWEwND'
                'xZy3EXfzLhuIlG1aIr0i5HaehJD4zidt9RN25EQ4dh+5lTVbTbHwsVleumbn'
                '4iMro+Z1HH9QWMHzZtKWzqkm9ryKHDGRFwvPfa+1O26/8TYzQ48804XE88ph'
                'qR9Wr1ioe1Rr+3rBpTdm/tYKOBB549/cq9EvEi02XtDdwvSk2bfzFm+6W17Y'
                'u2dXVQ2ZuXkXypanpUrcEH7yc3EkC098/0ehyOs/tJlOCu2MXcj6zbDQqeaE'
                '/ra8Ek7Rkuo1u/i3VU5osdFYM+/YrZ318jl/p07ZI3v8T//hquZpX162q84J'
                'OttWUnW8Mu7JBxmVCr4DCSuazR5NYfFjZK7NebN4+vT71U51O2v3LTqydUF+'
                '+9vST1t070fuKLe0CO2YfvvXFN3XXYV1Mr1cL8tYYvOkzzFVbpm1Lm9Wawk/'
                'a2zefOHqJtkrL/5MELIRqrgZNGfG8++/Qhf/rdmaXrO/+t910eevbb/+stzx'
                '22nOHTmBs4oh3H8+ON9K2ZP/b2PFfbM14SV2na9/G1tzix48yTmHpy3Ffm8d'
                '37p1H3O894hb7hBWjMtLPn6EO4ftwqUNvM9nB2dP/FPKape32bjJbvGJ3S1f'
                '1mn4+z94uO2gtXDyqsnLX+1w+dhxftPrF5Fs/hdrls4VfyYX8zM4Y6HIvk1/'
                'z86KD/356tzyG/bPLe+c/O57t/Z91/nHRo8nbKtfI37xjnLj7bh5276t1py3'
                'pfrzNPfkaKWi/8d2nOqrv7Z53R/Rq7/X/0//fdNsHrdm9f5pDp/qF6x6/Ui4'
                'cZmg9YeQv+/WWS467Odd+ej+ZT8/u9S0uWW77+56J+r6JSRy2tHqUjk/ztv7'
                'beq2/Cv8wx+Qd8/m5kP5qdt+/WfJeirzEgCAxWF+')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        #
        # There are 3 ImageMath modules:
        # 1)
        #   image_name[0] = DNA, factor[0] = 1, operation = invert, truncation
        #   output_image_name = DNAAfterMath
        # 2)
        #   image_name[0] = DNA, factor[0] = 1
        #   image_name[1] = Cytoplasm, factor[1] = 2
        #   operation = add, exponent = 4, multiply = 5, no truncation
        #   output_image_name = ImageAfterMath
        # 3)  DNA, Cytoplasm, Actin
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(len(module.images), 2)
        self.assertEqual(module.images[0].image_name.value, 'DNA')
        self.assertEqual(module.output_image_name.value, 'DNAAfterMath')
        self.assertEqual(module.operation.value, cellprofiler.modules.imagemath.O_INVERT)
        self.assertTrue(module.truncate_low.value)
        self.assertTrue(module.truncate_high.value)

        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(len(module.images), 2)
        self.assertEqual(module.images[0].image_name.value, 'DNA')
        self.assertEqual(module.images[0].factor.value, 1)
        self.assertEqual(module.images[1].image_name.value, 'Cytoplasm')
        self.assertEqual(module.images[1].factor.value, 2)
        self.assertEqual(module.output_image_name.value, 'ImageAfterMath')
        self.assertEqual(module.operation.value, cellprofiler.modules.imagemath.O_ADD)
        self.assertEqual(module.exponent.value, 4)
        self.assertEqual(module.after_factor.value, 5)
        self.assertFalse(module.truncate_low.value)
        self.assertFalse(module.truncate_high.value)

        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(len(module.images), 3)
        self.assertEqual(module.images[0].image_name.value, 'DNA')
        self.assertEqual(module.images[1].image_name.value, 'Cytoplasm')
        self.assertEqual(module.images[2].image_name.value, 'Actin')
        self.assertEqual(module.output_image_name.value, 'ImageAfterMath')
        self.assertEqual(module.operation.value, cellprofiler.modules.imagemath.O_ADD)

    def test_01_02_load_wierd_matlab(self):
        '''Load some ImageMath modules with constants typed in instead of images'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUgjJKFXwKs1RMDQDIisTSysTMwUjAwNLBZIBA6OnLz8DA8M2JgaG'
                'ijlvIvzzbxmI1C88cCNk4brdOZf4LN9teuevbz1pmq+rVMDW0KWeuyWSDgQJ'
                'uXcGH7S72/eb47dFt2LjEe5ez5NebnKb1jz/8f1z9flzj4/6MvxYynTl776H'
                '27l0loi98FpS+FfK2G3NxuhmrT08x+p/vRWzZOHlqenm4d+9YUrBnuTeF2fP'
                'xNm9CrprXMA/8+LJfV9sihtkt9tzNR587G109Jjpz0ZFvYdbtGvcY5zWMsh/'
                '6F58ulRT+FruL9/ajV3NSkca++Xdj/38I7bmWbjwozKRmfUiW6RtUvPt/S9O'
                'WKmWdJjzeeX8Q8UCjy0byvybhbZLRD9cdeGanrWRhWpb8APmhitVtw/VnpEM'
                'n/L3ttm7YK51Yod1S3JfLDc+uSMs6QfXnDlpt/v1/9apzVn2OqjOTOHdc5v8'
                'auZnPyLVtfe9luy5ccyHLV78Q0z+ZOtFyRM2/e9seP7L0pT9yKGgyu8rgeyf'
                'PYG3HdZJ9ogr+r/R8rOZnn7rqt+tndo79ny+tO7Hz/WFa0XuGCuui8kPvrrI'
                '5+udBRN0Ju1PrfkuXa7VKfz5s/n/aHX/ORHiX188vnHT62Du6z3vsmv0gwrO'
                't/4omhthtf6PTcqKIy6Pn4ZvP1Mny1UT8z27NvfIhsuuOUqnL4tU7Lj38JjL'
                'jH/p89Y81+7Jj9qfKWc98Qnv7y0WKld/J7OfvaNoVMB03/lv5Pe6nG77938F'
                '3m6/t7D3h3rv7/cnF71/q/Z7xpT5zz2T3wSc16tMqSy7uq72qecu9riPK/7F'
                'q2zrfhEeWGn29vK1qXfvr3k9/dvT7PvJv0M/3Tn/OToyPl9DYH/+uVnq+4KW'
                'lCvJa0S+Pmfp/qMw6JPHR5f37q85bS7+Xcvy66Pi9Pa7+1Mit4c9/LT9f2ZF'
                'nvj67v3767b+af0Xn1Ox7/6BXz8Xnjj3n5mB4f//5AYmN42z7EekHM6wHZHx'
                'eLQm4F9COD8Dk26hOwCE1ox/')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        #
        # Modules:
        # 1) Add, 1
        # 2) Subtract, 1
        # 3) Multiply, 2
        # 4) Divide, 2
        # 5) Combine, .3
        self.assertEqual(len(pipeline.modules()), 6)
        for i in range(1, 5):
            module = pipeline.modules()[i]
            self.assertEqual(len(module.images), 2)
            self.assertEqual(module.operation.value, cellprofiler.modules.imagemath.O_NONE)
        self.assertEqual(pipeline.modules()[1].addend.value, 1)
        self.assertEqual(pipeline.modules()[2].addend.value, -1)
        self.assertAlmostEqual(pipeline.modules()[3].after_factor.value, 2)
        self.assertAlmostEqual(pipeline.modules()[4].after_factor.value, .5)
        module = pipeline.modules()[5]
        self.assertEqual(len(module.images), 2)
        self.assertAlmostEqual(module.addend.value, .3)

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140124151645
GitHash:0c7fb94
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'To begin creating your project, use the Images module to compile a list of files and/or folders that you want to analyze. You can also specify a set of rules to include only the desired files in your selected folders.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    :
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "\x5B\\\\\\\\\\\\\\\\/\x5D\\\\\\\\.")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\'The Metadata module optionally allows you to extract information describing your images (i.e, metadata) which will be stored along with your measurements. This information can be contained in the file name and/or location, or in an external file.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:
    Match file and image metadata:\x5B\x5D
    Use case insensitive matching?:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:5|show_window:False|notes:\x5B\'The NamesAndTypes module allows you to assign a meaningful name to each image by which other modules will refer to it.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Assign a name to:All images
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:\x5B\x5D
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:1
    Single images count:0
    Select the rule criteria:and (file does contain "")
    Name to assign these images:DNA
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedOutlines

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'The Groups module optionally allows you to split your list of images into image subsets (groups) which will be processed independently of each other. Examples of groupings include screening batches, microtiter plates, time-lapse movies, etc.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

ImageMath:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Operation:Log transform (base 2)
    Raise the power of the result by:1.5
    Multiply the result by:0.5
    Add to result:0.1
    Set values less than 0 equal to 0?:Yes
    Set values greater than 1 equal to 1?:No
    Ignore the image masks?:Yes
    Name the output image:LogTransformed
    Image or measurement?:Image
    Select the first image:DNA
    Multiply the first image by:1.2
    Measurement:
    Image or measurement?:Measurement
    Select the second image:
    Multiply the second image by:1.5
    Measurement:Count_Nuclei
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(module.operation, cellprofiler.modules.imagemath.O_LOG_TRANSFORM_LEGACY)
        self.assertEqual(module.exponent, 1.5)
        self.assertEqual(module.after_factor, 0.5)
        self.assertEqual(module.addend, 0.1)
        self.assertTrue(module.truncate_low)
        self.assertFalse(module.truncate_high)
        self.assertTrue(module.ignore_mask)
        self.assertEqual(module.output_image_name, "LogTransformed")
        self.assertEqual(module.images[0].image_or_measurement, cellprofiler.modules.imagemath.IM_IMAGE)
        self.assertEqual(module.images[0].image_name, "DNA")
        self.assertEqual(module.images[0].factor, 1.2)
        self.assertEqual(module.images[1].image_or_measurement, cellprofiler.modules.imagemath.IM_MEASUREMENT)
        self.assertEqual(module.images[1].measurement, "Count_Nuclei")
        self.assertEqual(module.images[1].factor, 1.5)

    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20140124151645
GitHash:0c7fb94
ModuleCount:5
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'To begin creating your project, use the Images module to compile a list of files and/or folders that you want to analyze. You can also specify a set of rules to include only the desired files in your selected folders.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    :
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage) (directory doesnot containregexp "\x5B\\\\\\\\\\\\\\\\/\x5D\\\\\\\\.")

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:4|show_window:False|notes:\x5B\'The Metadata module optionally allows you to extract information describing your images (i.e, metadata) which will be stored along with your measurements. This information can be contained in the file name and/or location, or in an external file.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Extract metadata?:No
    Metadata data type:Text
    Metadata types:{}
    Extraction method count:1
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Extract metadata from:All images
    Select the filtering criteria:and (file does contain "")
    Metadata file location:
    Match file and image metadata:\x5B\x5D
    Use case insensitive matching?:No

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:5|show_window:False|notes:\x5B\'The NamesAndTypes module allows you to assign a meaningful name to each image by which other modules will refer to it.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Assign a name to:All images
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Match metadata:\x5B\x5D
    Image set matching method:Order
    Set intensity range from:Image metadata
    Assignments count:1
    Single images count:0
    Select the rule criteria:and (file does contain "")
    Name to assign these images:DNA
    Name to assign these objects:Cell
    Select the image type:Grayscale image
    Set intensity range from:Image metadata
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedOutlines

Groups:[module_num:4|svn_version:\'Unknown\'|variable_revision_number:2|show_window:False|notes:\x5B\'The Groups module optionally allows you to split your list of images into image subsets (groups) which will be processed independently of each other. Examples of groupings include screening batches, microtiter plates, time-lapse movies, etc.\'\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Do you want to group your images?:No
    grouping metadata count:1
    Metadata category:None

ImageMath:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True]
    Operation:Log transform (base 2)
    Raise the power of the result by:1.5
    Multiply the result by:0.5
    Add to result:0.1
    Set values less than 0 equal to 0?:Yes
    Set values greater than 1 equal to 1?:No
    Ignore the image masks?:Yes
    Name the output image:LogTransformed
    Image or measurement?:Image
    Select the first image:DNA
    Multiply the first image by:1.2
    Measurement:
    Image or measurement?:Measurement
    Select the second image:
    Multiply the second image by:1.5
    Measurement:Count_Nuclei
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, cellprofiler.modules.imagemath.ImageMath))
        self.assertEqual(module.operation, cellprofiler.modules.imagemath.O_LOG_TRANSFORM)
        self.assertEqual(module.exponent, 1.5)
        self.assertEqual(module.after_factor, 0.5)
        self.assertEqual(module.addend, 0.1)
        self.assertTrue(module.truncate_low)
        self.assertFalse(module.truncate_high)
        self.assertTrue(module.ignore_mask)
        self.assertEqual(module.output_image_name, "LogTransformed")
        self.assertEqual(module.images[0].image_or_measurement, cellprofiler.modules.imagemath.IM_IMAGE)
        self.assertEqual(module.images[0].image_name, "DNA")
        self.assertEqual(module.images[0].factor, 1.2)
        self.assertEqual(module.images[1].image_or_measurement, cellprofiler.modules.imagemath.IM_MEASUREMENT)
        self.assertEqual(module.images[1].measurement, "Count_Nuclei")
        self.assertEqual(module.images[1].factor, 1.5)

    def run_imagemath(self, images, modify_module_fn=None, measurement=None):
        '''Run the ImageMath module, returning the image created

        images - a list of dictionaries. The dictionary has keys:
                 pixel_data - image pixel data
                 mask - mask for image
                 cropping - cropping mask for image
        modify_module_fn - a function of the signature, fn(module)
                 that allows the test to modify the module.
        measurement - an image measurement value
        '''
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = cellprofiler.modules.imagemath.ImageMath()
        module.module_num = 1
        for i, image in enumerate(images):
            pixel_data = image['pixel_data']
            mask = image.get('mask', None)
            cropping = image.get('cropping', None)
            if i >= 2:
                module.add_image()
            name = 'inputimage%s' % i
            module.images[i].image_name.value = name
            img = cellprofiler.image.Image(pixel_data, mask=mask, crop_mask=cropping)
            image_set.add(name, img)
        module.output_image_name.value = 'outputimage'
        if modify_module_fn is not None:
            modify_module_fn(module)
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        measurements = cellprofiler.measurement.Measurements()
        if measurement is not None:
            measurements.add_image_measurement(MEASUREMENT_NAME, str(measurement))
        workspace = cellprofiler.workspace.Workspace(pipeline, module, image_set, cellprofiler.object.ObjectSet(),
                                                     measurements, image_set_list)
        module.run(workspace)
        return image_set.get_image('outputimage')

    def check_expected(self, image, expected, mask=None, ignore=False):
        if mask is None and not image.has_crop_mask:
            numpy.testing.assert_array_almost_equal(image.pixel_data, expected)

            self.assertFalse(image.has_mask)
        elif mask is not None and ignore:
            numpy.testing.assert_array_almost_equal(image.pixel_data, expected)

            self.assertTrue(image.has_mask)
        elif mask is not None and not ignore:
            self.assertTrue(image.has_mask)

            if not image.has_crop_mask:
                self.assertTrue(numpy.all(mask == image.mask))

            numpy.testing.assert_array_almost_equal(image.pixel_data[image.mask], expected[image.mask])

    def test_02_01_exponent(self):
        '''Test exponentiation of an image'''

        def fn(module):
            module.exponent.value = 2
            module.operation.value = cellprofiler.modules.imagemath.O_NONE

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected = image ** 2
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_02_02_factor(self):
        '''Test multiplicative factor'''

        def fn(module):
            module.after_factor.value = .5
            module.operation.value = cellprofiler.modules.imagemath.O_NONE

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10))
        expected = image * .5
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_02_03_addend(self):
        '''Test adding a value to image'''

        def fn(module):
            module.addend.value = .5
            module.operation.value = cellprofiler.modules.imagemath.O_NONE

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)) * .5
        image = image.astype(numpy.float32)
        expected = image + .5
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_02_04_mask(self):
        '''Test a mask in the first image'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_NONE

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        mask = numpy.random.uniform(size=(10, 10)) > .3
        output = self.run_imagemath([{'pixel_data': image,
                                      'mask': mask
                                      }], fn)
        self.check_expected(output, image, mask)

    def test_03_01_add(self):
        '''Test adding'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_ADD
            module.truncate_high.value = False

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            expected = reduce(numpy.add, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_03_02_add_mask(self):
        '''Test adding masked images'''
        '''Test adding'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_ADD
            module.truncate_high.value = False

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(50, 50)).astype(numpy.float32),
                       'mask': (numpy.random.uniform(size=(50, 50)) > .1)}
                      for i in range(n)]
            expected = reduce(numpy.add, [x['pixel_data'] for x in images])
            mask = reduce(numpy.logical_and, [x['mask'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected, mask)

    def test_03_03_add_mask_truncate(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_ADD
            module.truncate_high.value = True

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(50, 50)).astype(numpy.float32),
                       'mask': (numpy.random.uniform(size=(50, 50)) > .1)}
                      for i in range(n)]
            expected = reduce(numpy.add, [x['pixel_data'] for x in images])
            expected[expected > 1] = 1
            mask = reduce(numpy.logical_and, [x['mask'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected, mask)

    def test_03_04_add_crop(self):
        '''Add images, cropping to border'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_ADD
            module.truncate_high.value = False

        numpy.random.seed(0)
        crop_mask = numpy.zeros((20, 20), bool)
        crop_mask[5:15, 5:15] = True
        for n in range(2, 3):
            for m in range(n):
                images = [{'pixel_data': numpy.random.uniform(size=(20, 20)).astype(numpy.float32)}
                          for i in range(n)]
                for i, img in enumerate(images):
                    img['cropped_data'] = img['pixel_data'][5:15, 5:15]
                    if m == i:
                        img['pixel_data'] = img['cropped_data']
                        img['cropping'] = crop_mask
                expected = reduce(numpy.add, [x['cropped_data'] for x in images])
                output = self.run_imagemath(images, fn)
                self.check_expected(output, expected)

    def test_03_05_add_factors(self):
        '''Test adding with factors'''
        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            factors = numpy.random.uniform(size=n)
            expected = reduce(numpy.add, [x['pixel_data'] * factor
                                          for x, factor in zip(images, factors)])

            def fn(module):
                module.operation.value = cellprofiler.modules.imagemath.O_ADD
                module.truncate_high.value = False
                for i in range(n):
                    module.images[i].factor.value = factors[i]

            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_03_06_ignore_mask(self):
        '''Test adding images with masks, but ignoring the masks'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_ADD
            module.truncate_high.value = False
            module.ignore_mask.value = True

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(50, 50)).astype(numpy.float32),
                       'mask': (numpy.random.uniform(size=(50, 50)) > .1)}
                      for i in range(n)]
            expected = reduce(numpy.add, [x['pixel_data'] for x in images])
            mask = reduce(numpy.logical_and, [x['mask'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected, mask, True)

    def test_04_01_subtract(self):
        '''Test subtracting'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_SUBTRACT
            module.truncate_low.value = False

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            expected = reduce(numpy.subtract, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_04_02_subtract_truncate(self):
        '''Test subtracting with truncation'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_SUBTRACT
            module.truncate_low.value = True

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            expected = reduce(numpy.subtract, [x['pixel_data'] for x in images])
            expected[expected < 0] = 0
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_05_01_multiply(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_MULTIPLY
            module.truncate_low.value = False

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            expected = reduce(numpy.multiply, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_05_02_multiply_binary(self):
        # Regression test of issue # 42
        #
        # Multiplying two binary images should yield a binary image
        #
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_MULTIPLY
            module.truncate_low.value = False

        r = numpy.random.RandomState()
        r.seed(52)
        images = [{'pixel_data': numpy.random.uniform(size=(10, 10)) > .5}
                  for i in range(2)]
        output = self.run_imagemath(images, fn)
        self.assertTrue(output.pixel_data.dtype == numpy.bool)

    def test_06_01_divide(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_DIVIDE
            module.truncate_low.value = False

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            expected = reduce(numpy.divide, [x['pixel_data'] for x in images])
            expected[expected < 0] = 0
            expected[expected > 1] = 1

            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_07_01_average(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_AVERAGE
            module.truncate_low.value = False

        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            expected = reduce(numpy.add, [x['pixel_data'] for x in images]) / n
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_07_02_average_factors(self):
        '''Test averaging with factors'''
        numpy.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': numpy.random.uniform(size=(10, 10)).astype(numpy.float32)}
                      for i in range(n)]
            factors = numpy.random.uniform(size=n)
            expected = reduce(numpy.add, [x['pixel_data'] * factor
                                          for x, factor in zip(images, factors)])
            expected /= numpy.sum(factors)

            def fn(module):
                module.operation.value = cellprofiler.modules.imagemath.O_AVERAGE
                module.truncate_high.value = False
                for i in range(n):
                    module.images[i].factor.value = factors[i]

            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_08_01_invert(self):
        '''Test invert of an image'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_INVERT

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected = skimage.util.invert(image)
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_09_01_log_transform(self):
        '''Test log transform of an image'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_LOG_TRANSFORM
            module.truncate_low.value = False

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected = numpy.log2(image + 1)
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_09_02_log_transform_legacy(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_LOG_TRANSFORM_LEGACY
            module.truncate_low.value = False

        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected = numpy.log2(image)
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_10_01_with_measurement(self):
        '''Test multiplying an image by a measurement'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_MULTIPLY
            module.images[1].image_or_measurement.value = cellprofiler.modules.imagemath.IM_MEASUREMENT
            module.images[1].measurement.value = MEASUREMENT_NAME

        numpy.random.seed(101)
        measurement = 1.23
        expected = numpy.random.uniform(size=(10, 20)).astype(numpy.float32)
        image = expected / measurement
        output = self.run_imagemath([{'pixel_data': image}],
                                    modify_module_fn=fn,
                                    measurement=measurement)
        self.check_expected(output, expected)

    def test_10_02_with_measurement_and_mask(self):
        '''Test a measurement operation on a masked image'''

        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_MULTIPLY
            module.images[1].image_or_measurement.value = cellprofiler.modules.imagemath.IM_MEASUREMENT
            module.images[1].measurement.value = MEASUREMENT_NAME

        numpy.random.seed(102)
        measurement = 1.52
        expected = numpy.random.uniform(size=(10, 20)).astype(numpy.float32)
        image = expected / measurement
        mask = numpy.random.uniform(size=(10, 20)) < .2
        output = self.run_imagemath([{'pixel_data': image, 'mask': mask}],
                                    modify_module_fn=fn,
                                    measurement=measurement)
        self.check_expected(output, expected, mask)

    def test_11_01_add_and_do_nothing(self):
        #
        # Regression for issue #1333 - add one, do nothing, input image
        # is changed
        #
        r = numpy.random.RandomState()
        r.seed(1101)
        m = cellprofiler.measurement.Measurements()
        pixel_data = r.uniform(size=(20, 20))
        m.add("inputimage", cellprofiler.image.Image(pixel_data))
        module = cellprofiler.modules.imagemath.ImageMath()
        module.images[0].image_name.value = "inputimage"
        module.output_image_name.value = "outputimage"
        module.operation.value = cellprofiler.modules.imagemath.O_NONE
        module.addend.value = 0.5
        module.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        workspace = cellprofiler.workspace.Workspace(pipeline, module, m, None, m, None)
        module.run(workspace)
        numpy.testing.assert_array_almost_equal(
                pixel_data, m.get_image("inputimage").pixel_data)

    def test_11_02_invert_binary_invert(self):
        #
        # Regression for issue #1329
        #
        r = numpy.random.RandomState()
        r.seed(1102)
        m = cellprofiler.measurement.Measurements()
        pixel_data = r.uniform(size=(20, 20)) > .5
        m.add("inputimage", cellprofiler.image.Image(pixel_data))
        module = cellprofiler.modules.imagemath.ImageMath()
        module.images[0].image_name.value = "inputimage"
        module.output_image_name.value = "intermediateimage"
        module.operation.value = cellprofiler.modules.imagemath.O_INVERT
        module.module_num = 1
        pipeline = cellprofiler.pipeline.Pipeline()
        pipeline.add_module(module)
        module = cellprofiler.modules.imagemath.ImageMath()
        module.images[0].image_name.value = "intermediateimage"
        module.output_image_name.value = "outputimage"
        module.operation.value = cellprofiler.modules.imagemath.O_INVERT
        module.module_num = 2
        pipeline = cellprofiler.pipeline.Pipeline()
        workspace = cellprofiler.workspace.Workspace(pipeline, module, m, None, m, None)
        for module in pipeline.modules():
            module.run(workspace)
        numpy.testing.assert_array_equal(
                pixel_data, m.get_image("inputimage").pixel_data > .5)

    def test_12_01_or_binary(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_OR

        numpy.random.seed(1201)
        for n in range(2, 5):
            images = [
                {'pixel_data': numpy.random.uniform(size=(10, 10)) > .5}
                for i in range(n)]
            expected = reduce(numpy.logical_or, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_12_02_or_numeric(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_OR

        numpy.random.seed(1201)
        images = []
        for _ in range(2):
            pixel_data = numpy.random.uniform(size=(10, 10))
            pixel_data[pixel_data < .5] = 0
            images.append({'pixel_data': pixel_data})
        expected = reduce(numpy.logical_or, [x['pixel_data'] for x in images])
        output = self.run_imagemath(images, fn)
        self.check_expected(output, expected)

    def test_13_01_and_binary(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_AND

        numpy.random.seed(1301)
        for n in range(2, 5):
            images = [
                {'pixel_data': numpy.random.uniform(size=(10, 10)) > .5}
                for i in range(n)]
            expected = reduce(numpy.logical_and, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_14_01_not(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_NOT

        numpy.random.seed(4201)
        pixel_data = numpy.random.uniform(size=(10, 10)) > .5
        expected = ~ pixel_data
        output = self.run_imagemath([{'pixel_data': pixel_data}], fn)
        self.check_expected(output, expected)

    def test_15_01_equals_binary(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_EQUALS

        numpy.random.seed(1501)

        for n in range(2, 5):
            image0 = numpy.random.uniform(size=(20, 20)) > .5
            images = [{'pixel_data': image0}]
            expected = numpy.ones(image0.shape, bool)
            for i in range(1, n):
                image = numpy.random.uniform(size=(20, 20)) > .5
                expected = expected & (image == image0)
                images.append(dict(pixel_data=image))
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_15_02_equals_numeric(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_EQUALS

        numpy.random.seed(1502)

        image0 = numpy.random.uniform(size=(20, 20))
        image1 = numpy.random.uniform(size=(20, 20))
        expected = numpy.random.uniform(size=(20, 20)) > .5
        image1[expected] = image0[expected]
        images = [{'pixel_data': image0},
                  {'pixel_data': image1}]
        output = self.run_imagemath(images, fn)
        self.check_expected(output, expected)

    def test_16_01_minimum(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_MINIMUM

        numpy.random.seed(1502)

        for n in range(2, 5):
            image0 = numpy.random.uniform(size=(20, 20))
            images = [{'pixel_data': image0}]
            expected = image0.copy()
            for i in range(1, n):
                image = numpy.random.uniform(size=(20, 20))
                expected = numpy.minimum(expected, image)
                images.append(dict(pixel_data=image))
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_17_01_maximum(self):
        def fn(module):
            module.operation.value = cellprofiler.modules.imagemath.O_MAXIMUM

        numpy.random.seed(1502)

        for n in range(2, 5):
            image0 = numpy.random.uniform(size=(20, 20))
            images = [{'pixel_data': image0}]
            expected = image0.copy()
            for i in range(1, n):
                image = numpy.random.uniform(size=(20, 20))
                expected = numpy.maximum(expected, image)
                images.append(dict(pixel_data=image))
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)
