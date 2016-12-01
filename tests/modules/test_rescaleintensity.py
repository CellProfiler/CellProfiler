import StringIO
import base64
import unittest
import zlib

import numpy

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.injectimage
import cellprofiler.modules.rescaleintensity
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.workspace

cellprofiler.preferences.set_headless()

INPUT_NAME = 'input'
OUTPUT_NAME = 'output'
REFERENCE_NAME = 'reference'
MEASUREMENT_NAME = 'measurement'


class TestRescaleIntensity(unittest.TestCase):
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cellprofiler.modules.rescaleintensity.RescaleIntensity))
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
        self.assertEqual(module.rescale_method.value, cellprofiler.modules.rescaleintensity.M_MANUAL_IO_RANGE)
        self.assertEqual(module.wants_automatic_high.value, cellprofiler.modules.rescaleintensity.CUSTOM_VALUE)
        self.assertEqual(module.wants_automatic_low.value, cellprofiler.modules.rescaleintensity.CUSTOM_VALUE)
        self.assertAlmostEqual(module.source_low.value, .01)
        self.assertAlmostEqual(module.source_high.value, .99)
        self.assertAlmostEqual(module.source_scale.min, .1)
        self.assertAlmostEqual(module.source_scale.max, .9)
        self.assertEqual(module.matching_image_name.value, "Cytoplasm")
        self.assertAlmostEqual(module.divisor_value.value, 2)
        self.assertEqual(module.divisor_measurement.value, "Intensity_MeanIntensity_DNA")

    def make_workspace(self, input_image, input_mask=None,
                       reference_image=None, reference_mask=None,
                       measurement=None):
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        object_set = cellprofiler.object.ObjectSet()
        image_set_list = cellprofiler.image.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        measurements = cellprofiler.measurement.Measurements()
        module_number = 1
        module = cellprofiler.modules.rescaleintensity.RescaleIntensity()
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
            image = cellprofiler.image.Image(first_input_image)
        else:
            image = cellprofiler.image.Image(first_input_image, first_input_mask)
        ii = cellprofiler.modules.injectimage.InjectImage(INPUT_NAME, input_image, input_mask)
        ii.module_num = module_number
        module_number += 1
        pipeline.add_module(ii)
        image_set.add(INPUT_NAME, image)
        module.rescaled_image_name.value = OUTPUT_NAME
        if reference_image is not None:
            module.matching_image_name.value = REFERENCE_NAME
            if reference_mask is None:
                image = cellprofiler.image.Image(reference_image)
            else:
                image = cellprofiler.image.Image(reference_image, mask=reference_mask)
            image_set.add(REFERENCE_NAME, image)
            ii = cellprofiler.modules.injectimage.InjectImage(REFERENCE_NAME, reference_image, reference_mask)
            ii.module_num = module_number
            module_number += 1
            pipeline.add_module(ii)
        module.module_num = module_number
        pipeline.add_module(module)
        if measurement is not None:
            module.divisor_measurement.value = MEASUREMENT_NAME
            measurements.add_image_measurement(MEASUREMENT_NAME, measurement)
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set,
                                                     object_set,
                                                     measurements,
                                                     image_set_list)
        return workspace, module

    def test_03_01_stretch(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected[0, 0] = 1
        expected[9, 9] = 0
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_STRETCH
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_03_02_stretch_mask(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected[0, 0] = 1
        expected[9, 9] = 0
        mask = numpy.ones(expected.shape, bool)
        mask[3:5, 4:7] = False
        expected[~ mask] = 1.5
        workspace, module = self.make_workspace(expected / 2 + .1, mask)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_STRETCH
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_04_01_manual_input_range(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10))
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.source_scale.min = .1
        module.source_scale.max = .6
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_04_02_00_manual_input_range_auto_low(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected[0, 0] = 0
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.LOW_EACH_IMAGE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.source_high.value = .6
        self.assertFalse(module.is_aggregation_module())
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_04_02_01_manual_input_range_auto_low_all(self):
        numpy.random.seed(421)
        image1 = numpy.random.uniform(size=(10, 20)).astype(numpy.float32) * .5 + .5
        image2 = numpy.random.uniform(size=(10, 20)).astype(numpy.float32)
        expected = (image1 - numpy.min(image2)) / (1 - numpy.min(image2))
        workspace, module = self.make_workspace([image1, image2])
        self.assertTrue(isinstance(module, cellprofiler.modules.rescaleintensity.RescaleIntensity))
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.LOW_ALL_IMAGES
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.source_high.value = 1
        self.assertTrue(module.is_aggregation_module())
        module.prepare_group(workspace, {}, [1, 2])
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_04_03_00_manual_input_range_auto_high(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected[0, 0] = 1
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.HIGH_EACH_IMAGE
        module.source_low.value = .1
        self.assertFalse(module.is_aggregation_module())
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_04_03_01_manual_input_range_auto_high_all(self):
        numpy.random.seed(421)
        image1 = numpy.random.uniform(size=(10, 20)).astype(numpy.float32) * .5
        image2 = numpy.random.uniform(size=(10, 20)).astype(numpy.float32)
        expected = image1 / numpy.max(image2)
        workspace, module = self.make_workspace([image1, image2])
        self.assertTrue(isinstance(module, cellprofiler.modules.rescaleintensity.RescaleIntensity))
        image_set_2 = workspace.image_set_list.get_image_set(1)
        image_set_2.add(INPUT_NAME, cellprofiler.image.Image(image2))
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.HIGH_ALL_IMAGES
        module.source_low.value = 0
        self.assertTrue(module.is_aggregation_module())
        module.prepare_group(workspace, {}, [1, 2])
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_04_03_02_manual_input_range_auto_low_and_high(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected = expected - expected.min()
        expected = expected / expected.max()
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.LOW_EACH_IMAGE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.HIGH_EACH_IMAGE
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_04_04_manual_input_range_mask(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected[0, 0] = 1
        mask = numpy.ones(expected.shape, bool)
        mask[3:5, 4:7] = False
        expected[~ mask] = 1.5
        workspace, module = self.make_workspace(expected / 2 + .1, mask)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.HIGH_EACH_IMAGE
        module.source_low.value = .1
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_04_05_manual_input_range_truncate(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        expected_low_mask = numpy.zeros(expected.shape, bool)
        expected_low_mask[2:4, 1:3] = True
        expected[expected_low_mask] = -.05
        expected_high_mask = numpy.zeros(expected.shape, bool)
        expected_high_mask[6:8, 5:7] = True
        expected[expected_high_mask] = 1.05
        mask = ~(expected_low_mask | expected_high_mask)
        workspace, module = self.make_workspace(expected / 2 + .1)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_INPUT_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.source_scale.min = .1
        module.source_scale.max = .6
        module.run(workspace)
        image = workspace.image_set.get_image(OUTPUT_NAME)
        pixels = image.pixel_data
        numpy.testing.assert_almost_equal(pixels[mask], expected[mask])
        low_value = 0
        numpy.testing.assert_almost_equal(pixels[expected_low_mask], low_value)
        high_value = 1
        numpy.testing.assert_almost_equal(pixels[expected_high_mask], high_value)

    def test_05_01_manual_io_range(self):
        numpy.random.seed(0)
        expected = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        workspace, module = self.make_workspace(expected / 2 + .1)
        expected = expected * .75 + .05
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_MANUAL_IO_RANGE
        module.wants_automatic_low.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.wants_automatic_high.value = cellprofiler.modules.rescaleintensity.CUSTOM_VALUE
        module.source_scale.min = .1
        module.source_scale.max = .6
        module.dest_scale.min = .05
        module.dest_scale.max = .80
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_06_01_divide_by_image_minimum(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image[0, 0] = 0
        image = image / 2 + .25
        expected = image * 4
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_DIVIDE_BY_IMAGE_MINIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_06_02_divide_by_image_minimum_masked(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10))
        image[0, 0] = 0
        image = image / 2 + .25
        mask = numpy.ones(image.shape, bool)
        mask[3:6, 7:9] = False
        image[~mask] = .05
        expected = image * 4
        workspace, module = self.make_workspace(image, mask)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_DIVIDE_BY_IMAGE_MINIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_07_01_divide_by_image_maximum(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image = image / 2 + .1
        image[0, 0] = .8
        expected = image / .8
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_DIVIDE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_07_02_divide_by_image_minimum_masked(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image = image / 2 + .1
        image[0, 0] = .8
        mask = numpy.ones(image.shape, bool)
        mask[3:6, 7:9] = False
        image[~mask] = .9
        expected = image / .8
        workspace, module = self.make_workspace(image, mask)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_DIVIDE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_08_01_divide_by_value(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image = image / 2 + .1
        value = .9
        expected = image / value
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_DIVIDE_BY_VALUE
        module.divisor_value.value = value
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_09_01_divide_by_measurement(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image = image / 2 + .1
        value = .75
        expected = image / value
        workspace, module = self.make_workspace(image, measurement=value)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_DIVIDE_BY_MEASUREMENT
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_10_01_scale_by_image_maximum(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image[0, 0] = 1
        image = image / 2 + .1
        reference = numpy.random.uniform(size=(10, 10)).astype(numpy.float32) * .75
        reference[0, 0] = .75
        expected = image * .75 / .60
        workspace, module = self.make_workspace(image,
                                                reference_image=reference)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_SCALE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)

    def test_10_02_scale_by_image_maximum_mask(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10)).astype(numpy.float32)
        image[0, 0] = 1
        image = image / 2 + .1
        mask = numpy.ones(image.shape, bool)
        mask[3:6, 4:8] = False
        image[~mask] = .9
        reference = numpy.random.uniform(size=(10, 10)) * .75
        reference[0, 0] = .75
        rmask = numpy.ones(reference.shape, bool)
        rmask[7:9, 1:3] = False
        reference[~rmask] = .91
        expected = image * .75 / .60
        workspace, module = self.make_workspace(image,
                                                input_mask=mask,
                                                reference_image=reference,
                                                reference_mask=rmask)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_SCALE_BY_IMAGE_MAXIMUM
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels[mask], expected[mask])

    def test_11_01_convert_to_8_bit(self):
        numpy.random.seed(0)
        image = numpy.random.uniform(size=(10, 10))
        expected = (image * 255).astype(numpy.uint8)
        workspace, module = self.make_workspace(image)
        module.rescale_method.value = cellprofiler.modules.rescaleintensity.M_CONVERT_TO_8_BIT
        module.run(workspace)
        pixels = workspace.image_set.get_image(OUTPUT_NAME).pixel_data
        numpy.testing.assert_almost_equal(pixels, expected)
