'''test_imagemath.py - test the ImageMath module'''

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

import cellprofiler.modules.imagemath as I

MEASUREMENT_NAME = 'mymeasurement'


class TestImageMath(unittest.TestCase):
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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, I.ImageMath))
        self.assertEqual(module.operation, I.O_LOG_TRANSFORM_LEGACY)
        self.assertEqual(module.exponent, 1.5)
        self.assertEqual(module.after_factor, 0.5)
        self.assertEqual(module.addend, 0.1)
        self.assertTrue(module.truncate_low)
        self.assertFalse(module.truncate_high)
        self.assertTrue(module.ignore_mask)
        self.assertEqual(module.output_image_name, "LogTransformed")
        self.assertEqual(module.images[0].image_or_measurement, I.IM_IMAGE)
        self.assertEqual(module.images[0].image_name, "DNA")
        self.assertEqual(module.images[0].factor, 1.2)
        self.assertEqual(module.images[1].image_or_measurement, I.IM_MEASUREMENT)
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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, I.ImageMath))
        self.assertEqual(module.operation, I.O_LOG_TRANSFORM)
        self.assertEqual(module.exponent, 1.5)
        self.assertEqual(module.after_factor, 0.5)
        self.assertEqual(module.addend, 0.1)
        self.assertTrue(module.truncate_low)
        self.assertFalse(module.truncate_high)
        self.assertTrue(module.ignore_mask)
        self.assertEqual(module.output_image_name, "LogTransformed")
        self.assertEqual(module.images[0].image_or_measurement, I.IM_IMAGE)
        self.assertEqual(module.images[0].image_name, "DNA")
        self.assertEqual(module.images[0].factor, 1.2)
        self.assertEqual(module.images[1].image_or_measurement, I.IM_MEASUREMENT)
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
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        module = I.ImageMath()
        module.module_num = 1
        for i, image in enumerate(images):
            pixel_data = image['pixel_data']
            mask = image.get('mask', None)
            cropping = image.get('cropping', None)
            if i >= 2:
                module.add_image()
            name = 'inputimage%s' % i
            module.images[i].image_name.value = name
            img = cpi.Image(pixel_data, mask=mask, crop_mask=cropping)
            image_set.add(name, img)
        module.output_image_name.value = 'outputimage'
        if modify_module_fn is not None:
            modify_module_fn(module)
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
        if measurement is not None:
            measurements.add_image_measurement(MEASUREMENT_NAME, str(measurement))
        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  measurements, image_set_list)
        module.run(workspace)
        return image_set.get_image('outputimage')

    def check_expected(self, image, expected, mask=None, ignore=False):
        if mask is None and not image.has_crop_mask:
            self.assertTrue(np.all(np.abs(image.pixel_data - expected <
                                          np.sqrt(np.finfo(np.float32).eps))))
            self.assertFalse(image.has_mask)
        elif mask is not None and ignore == True:
            self.assertTrue(np.all(np.abs(image.pixel_data - expected <
                                          np.sqrt(np.finfo(np.float32).eps))))
            self.assertTrue(image.has_mask)
        elif mask is not None and ignore == False:
            self.assertTrue(image.has_mask)
            if not image.has_crop_mask:
                self.assertTrue(np.all(mask == image.mask))
            self.assertTrue(np.all(np.abs(image.pixel_data[image.mask] -
                                          expected[image.mask]) <
                                   np.sqrt(np.finfo(np.float32).eps)))

    def test_02_01_exponent(self):
        '''Test exponentiation of an image'''

        def fn(module):
            module.exponent.value = 2
            module.operation.value = I.O_NONE

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = image ** 2
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_02_02_factor(self):
        '''Test multiplicative factor'''

        def fn(module):
            module.after_factor.value = .5
            module.operation.value = I.O_NONE

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10))
        expected = image * .5
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_02_03_addend(self):
        '''Test adding a value to image'''

        def fn(module):
            module.addend.value = .5
            module.operation.value = I.O_NONE

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)) * .5
        image = image.astype(np.float32)
        expected = image + .5
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_02_04_mask(self):
        '''Test a mask in the first image'''

        def fn(module):
            module.operation.value = I.O_NONE

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        mask = np.random.uniform(size=(10, 10)) > .3
        output = self.run_imagemath([{'pixel_data': image,
                                      'mask': mask
                                      }], fn)
        self.check_expected(output, image, mask)

    def test_03_01_add(self):
        '''Test adding'''

        def fn(module):
            module.operation.value = I.O_ADD
            module.truncate_high.value = False

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            expected = reduce(np.add, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_03_02_add_mask(self):
        '''Test adding masked images'''
        '''Test adding'''

        def fn(module):
            module.operation.value = I.O_ADD
            module.truncate_high.value = False

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(50, 50)).astype(np.float32),
                       'mask': (np.random.uniform(size=(50, 50)) > .1)}
                      for i in range(n)]
            expected = reduce(np.add, [x['pixel_data'] for x in images])
            mask = reduce(np.logical_and, [x['mask'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected, mask)

    def test_03_03_add_mask_truncate(self):
        def fn(module):
            module.operation.value = I.O_ADD
            module.truncate_high.value = True

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(50, 50)).astype(np.float32),
                       'mask': (np.random.uniform(size=(50, 50)) > .1)}
                      for i in range(n)]
            expected = reduce(np.add, [x['pixel_data'] for x in images])
            expected[expected > 1] = 1
            mask = reduce(np.logical_and, [x['mask'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected, mask)

    def test_03_04_add_crop(self):
        '''Add images, cropping to border'''

        def fn(module):
            module.operation.value = I.O_ADD
            module.truncate_high.value = False

        np.random.seed(0)
        crop_mask = np.zeros((20, 20), bool)
        crop_mask[5:15, 5:15] = True
        for n in range(2, 3):
            for m in range(n):
                images = [{'pixel_data': np.random.uniform(size=(20, 20)).astype(np.float32)}
                          for i in range(n)]
                for i, img in enumerate(images):
                    img['cropped_data'] = img['pixel_data'][5:15, 5:15]
                    if m == i:
                        img['pixel_data'] = img['cropped_data']
                        img['cropping'] = crop_mask
                expected = reduce(np.add, [x['cropped_data'] for x in images])
                output = self.run_imagemath(images, fn)
                self.check_expected(output, expected)

    def test_03_05_add_factors(self):
        '''Test adding with factors'''
        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            factors = np.random.uniform(size=n)
            expected = reduce(np.add, [x['pixel_data'] * factor
                                       for x, factor in zip(images, factors)])

            def fn(module):
                module.operation.value = I.O_ADD
                module.truncate_high.value = False
                for i in range(n):
                    module.images[i].factor.value = factors[i]

            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_03_06_ignore_mask(self):
        '''Test adding images with masks, but ignoring the masks'''

        def fn(module):
            module.operation.value = I.O_ADD
            module.truncate_high.value = False
            module.ignore_mask.value = True

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(50, 50)).astype(np.float32),
                       'mask': (np.random.uniform(size=(50, 50)) > .1)}
                      for i in range(n)]
            expected = reduce(np.add, [x['pixel_data'] for x in images])
            mask = reduce(np.logical_and, [x['mask'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected, mask, True)

    def test_04_01_subtract(self):
        '''Test subtracting'''

        def fn(module):
            module.operation.value = I.O_SUBTRACT
            module.truncate_low.value = False

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            expected = reduce(np.subtract, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_04_02_subtract_truncate(self):
        '''Test subtracting with truncation'''

        def fn(module):
            module.operation.value = I.O_SUBTRACT
            module.truncate_low.value = True

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            expected = reduce(np.subtract, [x['pixel_data'] for x in images])
            expected[expected < 0] = 0
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_05_01_multiply(self):
        def fn(module):
            module.operation.value = I.O_MULTIPLY
            module.truncate_low.value = False

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            expected = reduce(np.multiply, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_05_02_multiply_binary(self):
        # Regression test of issue # 42
        #
        # Multiplying two binary images should yield a binary image
        #
        def fn(module):
            module.operation.value = I.O_MULTIPLY
            module.truncate_low.value = False

        r = np.random.RandomState()
        r.seed(52)
        images = [{'pixel_data': np.random.uniform(size=(10, 10)) > .5}
                  for i in range(2)]
        output = self.run_imagemath(images, fn)
        self.assertTrue(output.pixel_data.dtype == np.bool)

    def test_06_01_divide(self):
        def fn(module):
            module.operation.value = I.O_DIVIDE
            module.truncate_low.value = False

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            expected = reduce(np.divide, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_07_01_average(self):
        def fn(module):
            module.operation.value = I.O_AVERAGE
            module.truncate_low.value = False

        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            expected = reduce(np.add, [x['pixel_data'] for x in images]) / n
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_07_02_average_factors(self):
        '''Test averaging with factors'''
        np.random.seed(0)
        for n in range(2, 5):
            images = [{'pixel_data': np.random.uniform(size=(10, 10)).astype(np.float32)}
                      for i in range(n)]
            factors = np.random.uniform(size=n)
            expected = reduce(np.add, [x['pixel_data'] * factor
                                       for x, factor in zip(images, factors)])
            expected /= np.sum(factors)

            def fn(module):
                module.operation.value = I.O_AVERAGE
                module.truncate_high.value = False
                for i in range(n):
                    module.images[i].factor.value = factors[i]

            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_08_01_invert(self):
        '''Test invert of an image'''

        def fn(module):
            module.operation.value = I.O_INVERT

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = 1 - image
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_09_01_log_transform(self):
        '''Test log transform of an image'''

        def fn(module):
            module.operation.value = I.O_LOG_TRANSFORM
            module.truncate_low.value = False

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = np.log2(image + 1)
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_09_02_log_transform_legacy(self):
        def fn(module):
            module.operation.value = I.O_LOG_TRANSFORM_LEGACY
            module.truncate_low.value = False

        np.random.seed(0)
        image = np.random.uniform(size=(10, 10)).astype(np.float32)
        expected = np.log2(image)
        output = self.run_imagemath([{'pixel_data': image}], fn)
        self.check_expected(output, expected)

    def test_10_01_with_measurement(self):
        '''Test multiplying an image by a measurement'''

        def fn(module):
            module.operation.value = I.O_MULTIPLY
            module.images[1].image_or_measurement.value = I.IM_MEASUREMENT
            module.images[1].measurement.value = MEASUREMENT_NAME

        np.random.seed(101)
        measurement = 1.23
        expected = np.random.uniform(size=(10, 20)).astype(np.float32)
        image = expected / measurement
        output = self.run_imagemath([{'pixel_data': image}],
                                    modify_module_fn=fn,
                                    measurement=measurement)
        self.check_expected(output, expected)

    def test_10_02_with_measurement_and_mask(self):
        '''Test a measurement operation on a masked image'''

        def fn(module):
            module.operation.value = I.O_MULTIPLY
            module.images[1].image_or_measurement.value = I.IM_MEASUREMENT
            module.images[1].measurement.value = MEASUREMENT_NAME

        np.random.seed(102)
        measurement = 1.52
        expected = np.random.uniform(size=(10, 20)).astype(np.float32)
        image = expected / measurement
        mask = np.random.uniform(size=(10, 20)) < .2
        output = self.run_imagemath([{'pixel_data': image, 'mask': mask}],
                                    modify_module_fn=fn,
                                    measurement=measurement)
        self.check_expected(output, expected, mask)

    def test_11_01_add_and_do_nothing(self):
        #
        # Regression for issue #1333 - add one, do nothing, input image
        # is changed
        #
        r = np.random.RandomState()
        r.seed(1101)
        m = cpmeas.Measurements()
        pixel_data = r.uniform(size=(20, 20))
        m.add("inputimage", cpi.Image(pixel_data))
        module = I.ImageMath()
        module.images[0].image_name.value = "inputimage"
        module.output_image_name.value = "outputimage"
        module.operation.value = I.O_NONE
        module.addend.value = 0.5
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, m, None, m, None)
        module.run(workspace)
        np.testing.assert_array_almost_equal(
                pixel_data, m.get_image("inputimage").pixel_data)

    def test_11_02_invert_binary_invert(self):
        #
        # Regression for issue #1329
        #
        r = np.random.RandomState()
        r.seed(1102)
        m = cpmeas.Measurements()
        pixel_data = r.uniform(size=(20, 20)) > .5
        m.add("inputimage", cpi.Image(pixel_data))
        module = I.ImageMath()
        module.images[0].image_name.value = "inputimage"
        module.output_image_name.value = "intermediateimage"
        module.operation.value = I.O_INVERT
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        module = I.ImageMath()
        module.images[0].image_name.value = "intermediateimage"
        module.output_image_name.value = "outputimage"
        module.operation.value = I.O_INVERT
        module.module_num = 2
        pipeline = cpp.Pipeline()
        workspace = cpw.Workspace(pipeline, module, m, None, m, None)
        for module in pipeline.modules():
            module.run(workspace)
        np.testing.assert_array_equal(
                pixel_data, m.get_image("inputimage").pixel_data > .5)

    def test_12_01_or_binary(self):
        def fn(module):
            module.operation.value = I.O_OR

        np.random.seed(1201)
        for n in range(2, 5):
            images = [
                {'pixel_data': np.random.uniform(size=(10, 10)) > .5}
                for i in range(n)]
            expected = reduce(np.logical_or, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_12_02_or_numeric(self):
        def fn(module):
            module.operation.value = I.O_OR

        np.random.seed(1201)
        images = []
        for _ in range(2):
            pixel_data = np.random.uniform(size=(10, 10))
            pixel_data[pixel_data < .5] = 0
            images.append({'pixel_data': pixel_data})
        expected = reduce(np.logical_or, [x['pixel_data'] for x in images])
        output = self.run_imagemath(images, fn)
        self.check_expected(output, expected)

    def test_13_01_and_binary(self):
        def fn(module):
            module.operation.value = I.O_AND

        np.random.seed(1301)
        for n in range(2, 5):
            images = [
                {'pixel_data': np.random.uniform(size=(10, 10)) > .5}
                for i in range(n)]
            expected = reduce(np.logical_and, [x['pixel_data'] for x in images])
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_14_01_not(self):
        def fn(module):
            module.operation.value = I.O_NOT

        np.random.seed(4201)
        pixel_data = np.random.uniform(size=(10, 10)) > .5
        expected = ~ pixel_data
        output = self.run_imagemath([{'pixel_data': pixel_data}], fn)
        self.check_expected(output, expected)

    def test_15_01_equals_binary(self):
        def fn(module):
            module.operation.value = I.O_EQUALS

        np.random.seed(1501)

        for n in range(2, 5):
            image0 = np.random.uniform(size=(20, 20)) > .5
            images = [{'pixel_data': image0}]
            expected = np.ones(image0.shape, bool)
            for i in range(1, n):
                image = np.random.uniform(size=(20, 20)) > .5
                expected = expected & (image == image0)
                images.append(dict(pixel_data=image))
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_15_02_equals_numeric(self):
        def fn(module):
            module.operation.value = I.O_EQUALS

        np.random.seed(1502)

        image0 = np.random.uniform(size=(20, 20))
        image1 = np.random.uniform(size=(20, 20))
        expected = np.random.uniform(size=(20, 20)) > .5
        image1[expected] = image0[expected]
        images = [{'pixel_data': image0},
                  {'pixel_data': image1}]
        output = self.run_imagemath(images, fn)
        self.check_expected(output, expected)

    def test_16_01_minimum(self):
        def fn(module):
            module.operation.value = I.O_MINIMUM

        np.random.seed(1502)

        for n in range(2, 5):
            image0 = np.random.uniform(size=(20, 20))
            images = [{'pixel_data': image0}]
            expected = image0.copy()
            for i in range(1, n):
                image = np.random.uniform(size=(20, 20))
                expected = np.minimum(expected, image)
                images.append(dict(pixel_data=image))
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)

    def test_17_01_maximum(self):
        def fn(module):
            module.operation.value = I.O_MAXIMUM

        np.random.seed(1502)

        for n in range(2, 5):
            image0 = np.random.uniform(size=(20, 20))
            images = [{'pixel_data': image0}]
            expected = image0.copy()
            for i in range(1, n):
                image = np.random.uniform(size=(20, 20))
                expected = np.maximum(expected, image)
                images.append(dict(pixel_data=image))
            output = self.run_imagemath(images, fn)
            self.check_expected(output, expected)
