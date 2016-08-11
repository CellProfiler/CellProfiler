import StringIO
import base64
import unittest
import zlib

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.calculatemath
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

OUTPUT_MEASUREMENTS = "outputmeasurements"
MATH_OUTPUT_MEASUREMENTS = "_".join(("Math", OUTPUT_MEASUREMENTS))
OBJECT = ["object%d" % i for i in range(2)]


class TestCalculateMath(unittest.TestCase):
    def test_01_000_load_calculate_ratios(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8925
FromMatlab:True

CalculateRatios:[module_num:1|svn_version:\'8913\'|variable_revision_number:6|show_window:False|notes:\x5B\x5D]
    What do you want to call the ratio calculated by this module?  The prefix 'Ratio_' will be applied to your entry, or simply leave as 'Automatic' and a sensible name will be generated:MyRatio
    Which object would you like to use for the numerator?:MyNumeratorObject
    Which category of measurements would you like to use?:AreaShape
    Which feature do you want to use?:Perimeter
    For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use?:
    For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?:
    Which object would you like to use for the denominator?:Image
    Which category of measurements would you like to use?:Texture
    Which feature do you want to use?:Fuzziness
    For INTENSITY, AREAOCCUPIED or TEXTURE features, which image's measurements would you like to use?:MyImage
    For TEXTURE, RADIAL DISTRIBUTION, OR NEIGHBORS features, what previously measured size scale (TEXTURE OR NEIGHBORS) or previously used number of bins (RADIALDISTRIBUTION) do you want to use?:10
    Do you want the log (base 10) of the ratio?:No
"""
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.calculatemath.CalculateMath))
        self.assertEqual(module.operation, cellprofiler.modules.calculatemath.O_DIVIDE)
        self.assertEqual(module.operands[0].operand_choice, cellprofiler.modules.calculatemath.MC_OBJECT)
        self.assertEqual(module.operands[1].operand_choice, cellprofiler.modules.calculatemath.MC_IMAGE)
        self.assertEqual(module.operands[0].operand_objects, "MyNumeratorObject")
        self.assertEqual(module.operands[0].operand_measurement, "AreaShape_Perimeter")
        self.assertEqual(module.operands[1].operand_measurement, "Texture_Fuzziness_MyImage_10")
        self.assertFalse(module.wants_log)

    def test_01_02_load_v1(self):
        data = ('eJztnN1u2zYUgCnnB806tO52sWG90cUGNFtiyGmyNsGQ2km2xVudGk3Qoii6'
                'jLHpmIMkGhKVxhsK7HKPtUfYY+xyjzBRliyJUSJZlpXIoQDBOjQ/Hp6jQ1LU'
                'D5v1o+f1HXmjosjN+tFqF6tIbqmQdomhbck6XZF3DQQp6shE35KbRJd/slR5'
                'TZGVza319a2qYh8rmyDdJjWa9+yfv1oALNq/d+y95P614MpSYGfyIaIU66fm'
                'ApgHn7vpf9v7K2hgeKKiV1C1kOmr8NIbepccDfqjv5qkY6noAGrBzPZ2YGkn'
                'yDBfdD3Q/buFz5F6iH9HnAletpfoDJuY6C7vls+njvQSyullfvj3K98PEucH'
                'lv4wkM7y7wM//3yE3x4E8pddGesdfIY7FlRlrMHTUS1YeUpMeXOh8ubA3kF9'
                'Iu5pDLfI1X/ROT9tFeFhfWsxfJnj2d5gRjcRNC0DaUinaf344uQ31KaBggJ+'
                'TFOvI3ROV78/h20qa5C2e0n8WgqVUwIHBGTi13HP5xs7ipLovQPCepnctFSK'
                '++ogCz6u3lKIl8BjkMze+RA3b/tZR0nqu8DVl8lVZWVdidC7yPHe5vFL7m9a'
                'vU7cJ+Sj4mMY78ni+yOOZ/IekXVCZctEvh1p7U8bn9PSlzY+Jm2Hl7X/admZ'
                'tb5WjL4vOP8wuaFTpJuYDo6b8NwX2JAyKrcWU+49rlwm1+2rnMMe7KNjdpRs'
                'fP2MK4fJe6gL7S5JdtqbvIcNu9UQYzBT8c7rW6soqbhqBJe2nuNwtZh63gXh'
                '88rkF9S05B9VcgLVifWP66fHGfr3JsbPpHFw3eNh3v5Nc72qVBRnW6m6B4Hy'
                'xLiYrZ22r6vTHBeD88CyKwfGRQT1iwNjkrj5lCuXyf642EIG1hBFRqRfbkK/'
                'vcTVfynolwz0j9vO13Put6Pm47Ps340p+5efL1anbB/fb1WV67+eTzuejTMe'
                'TMs+/vw9ieCmGZ/fThiff8ZwP4Nwe2TyL4+etb5jN3DRduWb5WMmvUaq+pK8'
                '335bX229W/ZSdolqafr2W2V1890f1ZW1D8PMh9gmncTlSLtnad7yZMLz04vh'
                'noLw+WEy8/EbBA3X8esflldZUpPotOemrblpe3Dgp0yzX8/zfs2sc3nH+axz'
                'wp/ZcsKf2XJJ/Klc032uInJ5XzfOOpf3/ZFZ50T/mS2XrP/cuPZ63mSuBq72'
                'Z9TzjKP3RG6r0DTdN0CKZG/e3D642r9R99tfI3zaY69NnbEXhPQ2KqDdNy2O'
                'o+apPxADnRrE0jvFs7f3pc9JHMfS+fe98vSr83IYc2w//3Ki+iviPG/zCyrS'
                'ec6z3wn4ScZ6B/ULaHee4+BtjKtZ52ogmzjIqpyi+E1wghOc4AQnxqsi+E1w'
                't5OrgavjvAzCcc52fx46nBYVyV7BCU5wgpsVbj/ARfXfn4Bw/81kYlEV6+jC'
                'ja0i2S04wQlOcIIrBlcLcJM87yqKvYITnOAEJzjBCW72uH9KPidxnOSWJQXy'
                '/xrQE3X983Ugf9mV20hV+wZh64gZFc1Z7MqsqAR2hqtNVZ7bh43AwlNMTz9G'
                'T43TU7tMjzZcjslRhb0PUyvuIk2O1tHnqpnqxR2kU9wd9A1btUWJBiluVxpu'
                'astOrXupTO95jN4dTu9OjL3DZzjQsCX2pbZn8HCJgtEH3MnPq8LpVy7T34Zq'
                '22LfGtqm9Sq7ntS0pTz1BNcTWIrQE4zrkiuX5x6W7t+/uj0BEG5Hfvv671la'
                'vXPSnCRJF+cXd2N45r+PwcXNuZ8mjdeuH4HL83s239b8/wMM1XyW')
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        #
        # 2 CalculateMath modules: 4 & 5
        # 4: ImageMeasurement, multiply
        #    Image - Intensity_MaxIntensity_DNA, multiply by 2, raise to 3
        #    Image - Intensity_MeanIntensity_DNA, multiply by 4, raise to 5
        #    multiply by 6 raise to 7
        # 5: ObjectMeasurement, multiply
        #    Nuclei - AreaShape_Area
        #    Nuclei - AreaShape_Perimeter
        #
        self.assertEqual(len(pipeline.modules()), 6)
        module = pipeline.modules()[4]
        module.output_feature_name.value = "ImageMeasurement"
        self.assertEqual(module.operands[0].operand_choice, cellprofiler.modules.calculatemath.MC_IMAGE)
        self.assertEqual(module.operands[1].operand_choice, cellprofiler.modules.calculatemath.MC_IMAGE)
        self.assertEqual(module.operands[0].operand_measurement, "Intensity_MaxIntensity_DNA")
        self.assertEqual(module.operands[1].operand_measurement, "Intensity_MeanIntensity_DNA")
        self.assertEqual(module.operands[0].multiplicand, 2)
        self.assertEqual(module.operands[0].exponent, 3)
        self.assertEqual(module.operands[1].multiplicand, 4)
        self.assertEqual(module.operands[1].exponent, 5)
        self.assertEqual(module.final_multiplicand, 6)
        self.assertEqual(module.final_exponent, 7)
        module = pipeline.modules()[5]
        self.assertEqual(module.operands[0].operand_choice, cellprofiler.modules.calculatemath.MC_OBJECT)
        self.assertEqual(module.operands[1].operand_choice, cellprofiler.modules.calculatemath.MC_OBJECT)
        self.assertEqual(module.operands[0].object, "Nuclei")
        self.assertEqual(module.operands[1].object, "Nuclei")
        self.assertEqual(module.operands[0].operand_measurement, "AreaShape_Area")
        self.assertEqual(module.operands[1].operand_measurement, "AreaShape_Perimeter")

    def run_workspace(self, operation, m1_is_image_measurement, m1_data, m2_is_image_measurement, m2_data, setup_fn=None):
        """Create and run a workspace, returning the measurements

        m<n>_is_image_measurement - true for an image measurement, false
                                    for object
        m<n>_data - either a single value or an array
        setup_fn - this gets called with the module before running
        """
        module = cellprofiler.modules.calculatemath.CalculateMath()
        module.operation.value = operation
        measurements = cellprofiler.measurement.Measurements()
        for i, operand, is_image_measurement, data in \
                ((0, module.operands[0], m1_is_image_measurement, m1_data),
                 (1, module.operands[1], m2_is_image_measurement, m2_data)):
            measurement = "measurement%d" % i
            if is_image_measurement:
                operand.operand_choice.value = cellprofiler.modules.calculatemath.MC_IMAGE
                measurements.add_image_measurement(measurement, data)
            else:
                operand.operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
                operand.operand_objects.value = OBJECT[i]
                measurements.add_measurement(OBJECT[i], measurement, data)
            operand.operand_measurement.value = measurement
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        pipeline = cellprofiler.pipeline.Pipeline()
        image_set_list = cellprofiler.image.ImageSetList()
        workspace = cellprofiler.workspace.Workspace(pipeline,
                                                     module,
                                                     image_set_list.get_image_set(0),
                                                     cellprofiler.region.Set(),
                                                     measurements,
                                                     image_set_list)
        if setup_fn is not None:
            setup_fn(module, workspace)
        module.run(workspace)
        return measurements

    def test_02_01_add_image_image(self):
        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 2, True, 2)
        self.assertTrue(measurements.has_feature(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        for i in range(2):
            self.assertFalse(measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS))
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 4)

    def test_02_02_add_image_object(self):
        """Add an image measurement to each of several object measurements"""
        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 2, False, numpy.array([1, 4, 9]))
        self.assertFalse(measurements.has_feature(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        self.assertTrue(measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS))
        data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
        self.assertTrue(numpy.all(data == numpy.array([3, 6, 11])))

    def test_02_03_add_object_image(self):
        """Add an image measurement to each of several object measurements (reverse)"""
        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, False, numpy.array([1, 4, 9]), True, 2)
        self.assertFalse(measurements.has_feature(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        self.assertTrue(measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS))
        data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
        self.assertTrue(numpy.all(data == numpy.array([3, 6, 11])))

    def test_02_04_add_premultiply(self):
        def fn(module, workspace):
            module.operands[0].multiplicand.value = 2
            module.operands[1].multiplicand.value = 3

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn)
        expected = 2 * 5 + 3 * 7
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_05_add_pre_exponentiate(self):
        def fn(module, workspace):
            module.operands[0].exponent.value = 2
            module.operands[1].exponent.value = 3

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn)
        expected = 5 ** 2 + 7 ** 3
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_06_add_postmultiply(self):
        def fn(module, workspace):
            module.final_multiplicand.value = 3

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn)
        expected = (5 + 7) * 3
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_07_add_postexponentiate(self):
        def fn(module, workspace):
            module.final_exponent.value = 3

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn)
        expected = (5 + 7) ** 3
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_08_add_log(self):
        def fn(module, workspace):
            module.wants_log.value = True

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn)
        expected = numpy.log10(5 + 7)
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_09_add_object_object(self):
        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, False, numpy.array([1, 2, 3]),
                                          False, numpy.array([1, 4, 9]))
        self.assertFalse(measurements.has_feature(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        for i in range(2):
            self.assertTrue(measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS))
            data = measurements.get_current_measurement(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
            self.assertTrue(numpy.all(data == numpy.array([2, 6, 12])))

    def test_03_01_subtract(self):
        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_SUBTRACT, True, 7, True, 5)
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 2)

    def test_04_01_multiply(self):
        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_MULTIPLY, True, 7, True, 5)
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 35)

    def test_04_01_divide(self):
        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_DIVIDE, True, 35, True, 5)
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 7)

    def test_05_01_measurement_columns_image(self):
        module = cellprofiler.modules.calculatemath.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        for operand in module.operands:
            operand.operand_choice.value = cellprofiler.modules.calculatemath.MC_IMAGE
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], cellprofiler.measurement.IMAGE)
        self.assertEqual(columns[0][1], MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(columns[0][2], cellprofiler.measurement.COLTYPE_FLOAT)
        self.assertEqual(module.get_categories(None, cellprofiler.measurement.IMAGE)[0], "Math")
        self.assertEqual(module.get_measurements(None, cellprofiler.measurement.IMAGE, "Math")[0],
                         OUTPUT_MEASUREMENTS)

    def test_05_02_measurement_columns_image_object(self):
        module = cellprofiler.modules.calculatemath.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = cellprofiler.modules.calculatemath.MC_IMAGE
        module.operands[1].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.operands[1].operand_objects.value = OBJECT[1]
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], OBJECT[1])
        self.assertEqual(columns[0][1], MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(columns[0][2], cellprofiler.measurement.COLTYPE_FLOAT)
        self.assertEqual(module.get_categories(None, OBJECT[1])[0], "Math")
        self.assertEqual(module.get_measurements(None, OBJECT[1], "Math")[0],
                         OUTPUT_MEASUREMENTS)
        self.assertEqual(len(module.get_categories(None, cellprofiler.measurement.IMAGE)), 0)

    def test_05_03_measurement_columns_object_image(self):
        module = cellprofiler.modules.calculatemath.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.operands[1].operand_choice.value = cellprofiler.modules.calculatemath.MC_IMAGE
        module.operands[0].operand_objects.value = OBJECT[0]
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], OBJECT[0])
        self.assertEqual(columns[0][1], MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(columns[0][2], cellprofiler.measurement.COLTYPE_FLOAT)
        self.assertEqual(module.get_categories(None, OBJECT[0])[0], "Math")
        self.assertEqual(module.get_measurements(None, OBJECT[0], "Math")[0],
                         OUTPUT_MEASUREMENTS)
        self.assertEqual(len(module.get_categories(None, cellprofiler.measurement.IMAGE)), 0)

    def test_05_04_measurement_columns_object_object(self):
        module = cellprofiler.modules.calculatemath.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.operands[1].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.operands[0].operand_objects.value = OBJECT[0]
        module.operands[1].operand_objects.value = OBJECT[1]
        columns = list(module.get_measurement_columns(None))
        self.assertEqual(len(columns), 2)
        if columns[0][0] == OBJECT[1]:
            columns = [columns[1], columns[0]]
        for i in range(2):
            self.assertEqual(columns[i][0], OBJECT[i])
            self.assertEqual(columns[i][1], MATH_OUTPUT_MEASUREMENTS)
            self.assertEqual(columns[i][2], cellprofiler.measurement.COLTYPE_FLOAT)
            self.assertEqual(module.get_categories(None, OBJECT[i])[0], "Math")
            self.assertEqual(module.get_measurements(None, OBJECT[i], "Math")[0],
                             OUTPUT_MEASUREMENTS)
        self.assertEqual(len(module.get_categories(None, cellprofiler.measurement.IMAGE)), 0)

    def test_06_01_add_object_object_same(self):
        """Regression test: add two measurements from the same object

        The bug was that the measurement gets added twice
        """

        def fn(module, workspace):
            module.operands[1].operand_objects.value = OBJECT[0]
            module.operands[1].operand_measurement.value = "measurement0"

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, False, numpy.array([5, 6]),
                                          False, numpy.array([-1, -1]), fn)
        data = measurements.get_current_measurement(OBJECT[0],
                                                    MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(len(data), 2)
        self.assertAlmostEqual(data[0], 10)
        self.assertAlmostEqual(data[1], 12)

    def test_07_01_img_379(self):
        """Regression test for IMG-379, divide by zero"""

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_DIVIDE, True, 35, True, 0)
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE,
                                                    MATH_OUTPUT_MEASUREMENTS)
        self.assertTrue(numpy.isnan(data))

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_DIVIDE,
                                          False, numpy.array([1.0]),
                                          False, numpy.array([0.0]))
        data = measurements.get_current_measurement(OBJECT[0],
                                                    MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(len(data), 1)
        self.assertTrue(numpy.isnan(data[0]))

    def test_08_01_none_operation(self):
        # In this case, just multiply the array by a constant
        def fn(module, workspace):
            module.operands[0].multiplicand.value = 2

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_NONE, False, numpy.array([1, 2, 3]),
                                          False, numpy.array([1, 4, 9]), fn)
        self.assertFalse(measurements.has_feature(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)) \
            # There should be only one operand and a measurement for that operand only
        self.assertTrue(len(OBJECT), 1)
        self.assertTrue(measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS))
        # Check the operation result
        data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data[0], 2)
        self.assertAlmostEqual(data[1], 4)
        self.assertAlmostEqual(data[2], 6)

    def test_09_01_img_919(self):
        """Regression test: one measurement, but both operands are from same object

        The bug was that the measurement gets added twice. It was fixed in run
        but not in get_measurement_columns
        """

        def fn(module):
            module.operands[1].operand_objects.value = OBJECT[0]
            module.operands[1].operand_measurement.value = "measurement0"

        module = cellprofiler.modules.calculatemath.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.operands[1].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.operands[0].operand_objects.value = OBJECT[0]
        module.operands[1].operand_objects.value = OBJECT[0]
        columns = module.get_measurement_columns(None)
        self.assertEqual(columns[0][0], OBJECT[0])
        self.assertEqual(len(columns), 1)

    def test_10_1_img_1566(self):
        """Regression test: different numbers of objects"""
        r = numpy.random.RandomState(1566)
        o0 = [numpy.array([1, 2, 3, 4, 5]), numpy.array([1, 1, 2, 2, 3]),
              numpy.array([1, 2, 4, 5]), numpy.array([1, 1, 1, 1])]
        o1 = [numpy.array([1, 1, 2, 2, 3]), numpy.array([1, 2, 3, 4, 5]),
              numpy.array([1, 1, 1, 1]), numpy.array([1, 2, 4, 5])]
        in0 = [numpy.array([0, 1, 2, 3, 4], float), numpy.array([2, 4, 8], float),
               numpy.array([0, 1, 2, 3, 4], float), numpy.array([5], float)]
        in1 = [numpy.array([2, 4, 8], float), numpy.array([0, 1, 2, 3, 4], float),
               numpy.array([5], float), numpy.array([0, 1, 2, 3, 4], float)]

        expected0 = [numpy.array([2, 3, 6, 7, 12]),
                     numpy.array([2.5, 6.5, 12]),
                     numpy.array([5, 6, numpy.nan, 8, 9]),
                     numpy.array([7])]
        expected1 = [numpy.array([2.5, 6.5, 12]),
                     numpy.array([2, 3, 6, 7, 12]),
                     numpy.array([7]),
                     numpy.array([5, 6, numpy.nan, 8, 9])]
        for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):
            for flip in (False, True):
                def setup_fn(module, workspace, oo0=oo0, oo1=oo1, flip=flip):
                    m = workspace.measurements
                    self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                    if not flip:
                        m.add_relate_measurement(
                                1, cellprofiler.modules.calculatemath.R_PARENT, OBJECT[0], OBJECT[1],
                                numpy.ones(len(oo0), int), oo0,
                                numpy.ones(len(oo1), int), oo1)
                    else:
                        m.add_relate_measurement(
                                1, cellprofiler.modules.calculatemath.R_PARENT, OBJECT[1], OBJECT[0],
                                numpy.ones(len(oo0), int), oo1,
                                numpy.ones(len(oo1), int), oo0)

                measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, False, ii0,
                                                  False, ii1, setup_fn)
                data = measurements.get_current_measurement(
                        OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
                numpy.testing.assert_almost_equal(e0, data)
                data = measurements.get_current_measurement(
                        OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
                numpy.testing.assert_almost_equal(e1, data)

    def test_10_01_02_different_image_sets(self):
        #
        # Relationship code was matching object numbers from any object
        # set to any other
        #
        r = numpy.random.RandomState(100102)
        o0 = [numpy.array([1, 2, 3, 4, 5]), numpy.array([1, 1, 2, 2, 3]),
              numpy.array([1, 2, 4, 5]), numpy.array([1, 1, 1, 1])]
        o1 = [numpy.array([1, 1, 2, 2, 3]), numpy.array([1, 2, 3, 4, 5]),
              numpy.array([1, 1, 1, 1]), numpy.array([1, 2, 4, 5])]
        in0 = [numpy.array([0, 1, 2, 3, 4], float), numpy.array([2, 4, 8], float),
               numpy.array([0, 1, 2, 3, 4], float), numpy.array([5], float)]
        in1 = [numpy.array([2, 4, 8], float), numpy.array([0, 1, 2, 3, 4], float),
               numpy.array([5], float), numpy.array([0, 1, 2, 3, 4], float)]

        expected0 = [numpy.array([2, 3, 6, 7, 12]),
                     numpy.array([2.5, 6.5, 12]),
                     numpy.array([5, 6, numpy.nan, 8, 9]),
                     numpy.array([7])]
        expected1 = [numpy.array([2.5, 6.5, 12]),
                     numpy.array([2, 3, 6, 7, 12]),
                     numpy.array([7]),
                     numpy.array([5, 6, numpy.nan, 8, 9])]
        for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):
            def setup_fn(module, workspace, oo0=oo0, oo1=oo1):
                m = workspace.measurements
                self.assertTrue(isinstance(m, cellprofiler.measurement.Measurements))
                m.add_relate_measurement(
                        1, cellprofiler.modules.calculatemath.R_PARENT, OBJECT[0], OBJECT[1],
                        numpy.ones(len(oo0), int), oo0,
                        numpy.ones(len(oo1), int), oo1)
                for i1, i2 in ((1, 2), (2, 1), (2, 2)):
                    m.add_relate_measurement(
                            1, cellprofiler.modules.calculatemath.R_PARENT, OBJECT[0], OBJECT[1],
                        numpy.ones(len(oo0), int) * i1, r.permutation(oo0),
                        numpy.ones(len(oo1), int) * i2, oo1)

            measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, False, ii0,
                                              False, ii1, setup_fn)
            data = measurements.get_current_measurement(
                    OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
            numpy.testing.assert_almost_equal(e0, data)
            data = measurements.get_current_measurement(
                    OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
            numpy.testing.assert_almost_equal(e1, data)

    def test_10_01_issue_422(self):
        # Regression test of issue # 422
        #
        # If no operation is chosen, get_measurement_columns and
        # get_categories report measurements for both operands when
        # they should report for only a single one
        #
        module = cellprofiler.modules.calculatemath.CalculateMath()
        module.operation.value = cellprofiler.modules.calculatemath.O_NONE
        module.operands[0].operand_objects.value = OBJECT[0]
        module.operands[1].operand_objects.value = OBJECT[1]
        module.operands[0].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.operands[1].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
        module.output_feature_name.value = OUTPUT_MEASUREMENTS

        c = module.get_measurement_columns(None)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0][0], OBJECT[0])
        self.assertEqual(c[0][1], MATH_OUTPUT_MEASUREMENTS)

        self.assertEqual(len(module.get_categories(None, OBJECT[0])), 1)
        self.assertEqual(len(module.get_categories(None, OBJECT[1])), 0)

        self.assertEqual(
                len(module.get_measurements(None, OBJECT[0], cellprofiler.modules.calculatemath.C_MATH)), 1)
        self.assertEqual(
                len(module.get_measurements(None, OBJECT[1], cellprofiler.modules.calculatemath.C_MATH)), 0)

    def test_11_01_postadd(self):
        """Test whether the addend is added to the result"""

        def fn(module, workspace):
            module.final_addend.value = 1.5

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn)
        expected = (5 + 7) + 1.5
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_11_02_constrain_lower(self):
        """Test whether the lower bound option works"""

        def fn(module, workspace):
            module.constrain_lower_bound.value = True
            module.lower_bound.value = 0

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_SUBTRACT, True, 5, True, 7, fn)
        expected = 0
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_11_03_constrain_upper(self):
        """Test whether the upper bound option works"""

        def fn(module, workspace):
            module.constrain_upper_bound.value = True
            module.upper_bound.value = 10

        measurements = self.run_workspace(cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn)
        expected = 10
        data = measurements.get_current_measurement(cellprofiler.measurement.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)
