'''test_calculatemath.py - Test the CalculateMath module'''

import base64
import os
import unittest
import zlib
from StringIO import StringIO

import PIL.Image as PILImage
import cellprofiler.measurement
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

import cellprofiler.modules.calculatemath as C

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
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CalculateMath))
        self.assertEqual(module.operation, C.O_DIVIDE)
        self.assertEqual(module.operands[0].operand_choice, C.MC_OBJECT)
        self.assertEqual(module.operands[1].operand_choice, C.MC_IMAGE)
        self.assertEqual(module.operands[0].operand_objects, "MyNumeratorObject")
        self.assertEqual(module.operands[0].operand_measurement, "AreaShape_Perimeter")
        self.assertEqual(module.operands[1].operand_measurement, "Texture_Fuzziness_MyImage_10")
        self.assertFalse(module.wants_log)

    def run_workspace(self, operation, m1_is_image_measurement, m1_data,
                      m2_is_image_measurement, m2_data,
                      setup_fn=None):
        '''Create and run a workspace, returning the measurements

        m<n>_is_image_measurement - true for an image measurement, false
                                    for object
        m<n>_data - either a single value or an array
        setup_fn - this gets called with the module before running
        '''
        module = C.CalculateMath()
        module.operation.value = operation
        measurements = cpmeas.Measurements()
        for i, operand, is_image_measurement, data in \
                ((0, module.operands[0], m1_is_image_measurement, m1_data),
                 (1, module.operands[1], m2_is_image_measurement, m2_data)):
            measurement = "measurement%d" % i
            if is_image_measurement:
                operand.operand_choice.value = C.MC_IMAGE
                measurements.add_image_measurement(measurement, data)
            else:
                operand.operand_choice.value = C.MC_OBJECT
                operand.operand_objects.value = OBJECT[i]
                measurements.add_measurement(OBJECT[i], measurement, data)
            operand.operand_measurement.value = measurement
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        pipeline = cpp.Pipeline()
        image_set_list = cpi.ImageSetList()
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set_list.get_image_set(0),
                                  cpo.ObjectSet(),
                                  measurements,
                                  image_set_list)
        if setup_fn is not None:
            setup_fn(module, workspace)
        module.run(workspace)
        return measurements

    def test_02_01_add_image_image(self):
        measurements = self.run_workspace(C.O_ADD, True, 2, True, 2)
        self.assertTrue(measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        for i in range(2):
            self.assertFalse(measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS))
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 4)

    def test_02_02_add_image_object(self):
        '''Add an image measurement to each of several object measurements'''
        measurements = self.run_workspace(C.O_ADD, True, 2, False, np.array([1, 4, 9]))
        self.assertFalse(measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        self.assertTrue(measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS))
        data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
        self.assertTrue(np.all(data == np.array([3, 6, 11])))

    def test_02_03_add_object_image(self):
        '''Add an image measurement to each of several object measurements (reverse)'''
        measurements = self.run_workspace(C.O_ADD, False, np.array([1, 4, 9]), True, 2)
        self.assertFalse(measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        self.assertTrue(measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS))
        data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
        self.assertTrue(np.all(data == np.array([3, 6, 11])))

    def test_02_04_add_premultiply(self):
        def fn(module, workspace):
            module.operands[0].multiplicand.value = 2
            module.operands[1].multiplicand.value = 3

        measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
        expected = 2 * 5 + 3 * 7
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_05_add_pre_exponentiate(self):
        def fn(module, workspace):
            module.operands[0].exponent.value = 2
            module.operands[1].exponent.value = 3

        measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
        expected = 5 ** 2 + 7 ** 3
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_06_add_postmultiply(self):
        def fn(module, workspace):
            module.final_multiplicand.value = 3

        measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
        expected = (5 + 7) * 3
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_07_add_postexponentiate(self):
        def fn(module, workspace):
            module.final_exponent.value = 3

        measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
        expected = (5 + 7) ** 3
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_08_add_log(self):
        def fn(module, workspace):
            module.wants_log.value = True

        measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
        expected = np.log10(5 + 7)
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_02_09_add_object_object(self):
        measurements = self.run_workspace(C.O_ADD, False, np.array([1, 2, 3]),
                                          False, np.array([1, 4, 9]))
        self.assertFalse(measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS))
        for i in range(2):
            self.assertTrue(measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS))
            data = measurements.get_current_measurement(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
            self.assertTrue(np.all(data == np.array([2, 6, 12])))

    def test_03_01_subtract(self):
        measurements = self.run_workspace(C.O_SUBTRACT, True, 7, True, 5)
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 2)

    def test_04_01_multiply(self):
        measurements = self.run_workspace(C.O_MULTIPLY, True, 7, True, 5)
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 35)

    def test_04_01_divide(self):
        measurements = self.run_workspace(C.O_DIVIDE, True, 35, True, 5)
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, 7)

    def test_05_01_measurement_columns_image(self):
        module = C.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        for operand in module.operands:
            operand.operand_choice.value = C.MC_IMAGE
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], cpmeas.IMAGE)
        self.assertEqual(columns[0][1], MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(columns[0][2], cpmeas.COLTYPE_FLOAT)
        self.assertEqual(module.get_categories(None, cpmeas.IMAGE)[0], "Math")
        self.assertEqual(module.get_measurements(None, cpmeas.IMAGE, "Math")[0],
                         OUTPUT_MEASUREMENTS)

    def test_05_02_measurement_columns_image_object(self):
        module = C.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = C.MC_IMAGE
        module.operands[1].operand_choice.value = C.MC_OBJECT
        module.operands[1].operand_objects.value = OBJECT[1]
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], OBJECT[1])
        self.assertEqual(columns[0][1], MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(columns[0][2], cpmeas.COLTYPE_FLOAT)
        self.assertEqual(module.get_categories(None, OBJECT[1])[0], "Math")
        self.assertEqual(module.get_measurements(None, OBJECT[1], "Math")[0],
                         OUTPUT_MEASUREMENTS)
        self.assertEqual(len(module.get_categories(None, cpmeas.IMAGE)), 0)

    def test_05_03_measurement_columns_object_image(self):
        module = C.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = C.MC_OBJECT
        module.operands[1].operand_choice.value = C.MC_IMAGE
        module.operands[0].operand_objects.value = OBJECT[0]
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0][0], OBJECT[0])
        self.assertEqual(columns[0][1], MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(columns[0][2], cpmeas.COLTYPE_FLOAT)
        self.assertEqual(module.get_categories(None, OBJECT[0])[0], "Math")
        self.assertEqual(module.get_measurements(None, OBJECT[0], "Math")[0],
                         OUTPUT_MEASUREMENTS)
        self.assertEqual(len(module.get_categories(None, cpmeas.IMAGE)), 0)

    def test_05_04_measurement_columns_object_object(self):
        module = C.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = C.MC_OBJECT
        module.operands[1].operand_choice.value = C.MC_OBJECT
        module.operands[0].operand_objects.value = OBJECT[0]
        module.operands[1].operand_objects.value = OBJECT[1]
        columns = list(module.get_measurement_columns(None))
        self.assertEqual(len(columns), 2)
        if columns[0][0] == OBJECT[1]:
            columns = [columns[1], columns[0]]
        for i in range(2):
            self.assertEqual(columns[i][0], OBJECT[i])
            self.assertEqual(columns[i][1], MATH_OUTPUT_MEASUREMENTS)
            self.assertEqual(columns[i][2], cpmeas.COLTYPE_FLOAT)
            self.assertEqual(module.get_categories(None, OBJECT[i])[0], "Math")
            self.assertEqual(module.get_measurements(None, OBJECT[i], "Math")[0],
                             OUTPUT_MEASUREMENTS)
        self.assertEqual(len(module.get_categories(None, cpmeas.IMAGE)), 0)

    def test_06_01_add_object_object_same(self):
        '''Regression test: add two measurements from the same object

        The bug was that the measurement gets added twice
        '''

        def fn(module, workspace):
            module.operands[1].operand_objects.value = OBJECT[0]
            module.operands[1].operand_measurement.value = "measurement0"

        measurements = self.run_workspace(C.O_ADD, False, np.array([5, 6]),
                                          False, np.array([-1, -1]), fn)
        data = measurements.get_current_measurement(OBJECT[0],
                                                    MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(len(data), 2)
        self.assertAlmostEqual(data[0], 10)
        self.assertAlmostEqual(data[1], 12)

    def test_07_01_img_379(self):
        '''Regression test for IMG-379, divide by zero'''

        measurements = self.run_workspace(C.O_DIVIDE, True, 35, True, 0)
        data = measurements.get_current_measurement(cpmeas.IMAGE,
                                                    MATH_OUTPUT_MEASUREMENTS)
        self.assertTrue(np.isnan(data))

        measurements = self.run_workspace(C.O_DIVIDE,
                                          False, np.array([1.0]),
                                          False, np.array([0.0]))
        data = measurements.get_current_measurement(OBJECT[0],
                                                    MATH_OUTPUT_MEASUREMENTS)
        self.assertEqual(len(data), 1)
        self.assertTrue(np.isnan(data[0]))

    def test_08_01_none_operation(self):
        # In this case, just multiply the array by a constant
        def fn(module, workspace):
            module.operands[0].multiplicand.value = 2

        measurements = self.run_workspace(C.O_NONE, False, np.array([1, 2, 3]),
                                          False, np.array([1, 4, 9]), fn)
        self.assertFalse(measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)) \
            # There should be only one operand and a measurement for that operand only
        self.assertTrue(len(OBJECT), 1)
        self.assertTrue(measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS))
        # Check the operation result
        data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data[0], 2)
        self.assertAlmostEqual(data[1], 4)
        self.assertAlmostEqual(data[2], 6)

    def test_09_01_img_919(self):
        '''Regression test: one measurement, but both operands are from same object

        The bug was that the measurement gets added twice. It was fixed in run
        but not in get_measurement_columns
        '''

        def fn(module):
            module.operands[1].operand_objects.value = OBJECT[0]
            module.operands[1].operand_measurement.value = "measurement0"

        module = C.CalculateMath()
        module.output_feature_name.value = OUTPUT_MEASUREMENTS
        module.operands[0].operand_choice.value = C.MC_OBJECT
        module.operands[1].operand_choice.value = C.MC_OBJECT
        module.operands[0].operand_objects.value = OBJECT[0]
        module.operands[1].operand_objects.value = OBJECT[0]
        columns = module.get_measurement_columns(None)
        self.assertEqual(columns[0][0], OBJECT[0])
        self.assertEqual(len(columns), 1)

    def test_10_1_img_1566(self):
        '''Regression test: different numbers of objects'''
        r = np.random.RandomState(1566)
        o0 = [np.array([1, 2, 3, 4, 5]), np.array([1, 1, 2, 2, 3]),
              np.array([1, 2, 4, 5]), np.array([1, 1, 1, 1])]
        o1 = [np.array([1, 1, 2, 2, 3]), np.array([1, 2, 3, 4, 5]),
              np.array([1, 1, 1, 1]), np.array([1, 2, 4, 5])]
        in0 = [np.array([0, 1, 2, 3, 4], float), np.array([2, 4, 8], float),
               np.array([0, 1, 2, 3, 4], float), np.array([5], float)]
        in1 = [np.array([2, 4, 8], float), np.array([0, 1, 2, 3, 4], float),
               np.array([5], float), np.array([0, 1, 2, 3, 4], float)]

        expected0 = [np.array([2, 3, 6, 7, 12]),
                     np.array([2.5, 6.5, 12]),
                     np.array([5, 6, np.nan, 8, 9]),
                     np.array([7])]
        expected1 = [np.array([2.5, 6.5, 12]),
                     np.array([2, 3, 6, 7, 12]),
                     np.array([7]),
                     np.array([5, 6, np.nan, 8, 9])]
        for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):
            for flip in (False, True):
                def setup_fn(module, workspace, oo0=oo0, oo1=oo1, flip=flip):
                    m = workspace.measurements
                    self.assertTrue(isinstance(m, cpmeas.Measurements))
                    if not flip:
                        m.add_relate_measurement(
                                1, cellprofiler.measurement.R_PARENT, OBJECT[0], OBJECT[1],
                                np.ones(len(oo0), int), oo0,
                                np.ones(len(oo1), int), oo1)
                    else:
                        m.add_relate_measurement(
                                1, cellprofiler.measurement.R_PARENT, OBJECT[1], OBJECT[0],
                                np.ones(len(oo0), int), oo1,
                                np.ones(len(oo1), int), oo0)

                measurements = self.run_workspace(C.O_ADD, False, ii0,
                                                  False, ii1, setup_fn)
                data = measurements.get_current_measurement(
                        OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
                np.testing.assert_almost_equal(e0, data)
                data = measurements.get_current_measurement(
                        OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
                np.testing.assert_almost_equal(e1, data)

    def test_10_01_02_different_image_sets(self):
        #
        # Relationship code was matching object numbers from any object
        # set to any other
        #
        r = np.random.RandomState(100102)
        o0 = [np.array([1, 2, 3, 4, 5]), np.array([1, 1, 2, 2, 3]),
              np.array([1, 2, 4, 5]), np.array([1, 1, 1, 1])]
        o1 = [np.array([1, 1, 2, 2, 3]), np.array([1, 2, 3, 4, 5]),
              np.array([1, 1, 1, 1]), np.array([1, 2, 4, 5])]
        in0 = [np.array([0, 1, 2, 3, 4], float), np.array([2, 4, 8], float),
               np.array([0, 1, 2, 3, 4], float), np.array([5], float)]
        in1 = [np.array([2, 4, 8], float), np.array([0, 1, 2, 3, 4], float),
               np.array([5], float), np.array([0, 1, 2, 3, 4], float)]

        expected0 = [np.array([2, 3, 6, 7, 12]),
                     np.array([2.5, 6.5, 12]),
                     np.array([5, 6, np.nan, 8, 9]),
                     np.array([7])]
        expected1 = [np.array([2.5, 6.5, 12]),
                     np.array([2, 3, 6, 7, 12]),
                     np.array([7]),
                     np.array([5, 6, np.nan, 8, 9])]
        for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):
            def setup_fn(module, workspace, oo0=oo0, oo1=oo1):
                m = workspace.measurements
                self.assertTrue(isinstance(m, cpmeas.Measurements))
                m.add_relate_measurement(
                        1, cellprofiler.measurement.R_PARENT, OBJECT[0], OBJECT[1],
                        np.ones(len(oo0), int), oo0,
                        np.ones(len(oo1), int), oo1)
                for i1, i2 in ((1, 2), (2, 1), (2, 2)):
                    m.add_relate_measurement(
                            1, cellprofiler.measurement.R_PARENT, OBJECT[0], OBJECT[1],
                            np.ones(len(oo0), int) * i1, r.permutation(oo0),
                            np.ones(len(oo1), int) * i2, oo1)

            measurements = self.run_workspace(C.O_ADD, False, ii0,
                                              False, ii1, setup_fn)
            data = measurements.get_current_measurement(
                    OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
            np.testing.assert_almost_equal(e0, data)
            data = measurements.get_current_measurement(
                    OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
            np.testing.assert_almost_equal(e1, data)

    def test_10_01_issue_422(self):
        # Regression test of issue # 422
        #
        # If no operation is chosen, get_measurement_columns and
        # get_categories report measurements for both operands when
        # they should report for only a single one
        #
        module = C.CalculateMath()
        module.operation.value = C.O_NONE
        module.operands[0].operand_objects.value = OBJECT[0]
        module.operands[1].operand_objects.value = OBJECT[1]
        module.operands[0].operand_choice.value = C.MC_OBJECT
        module.operands[1].operand_choice.value = C.MC_OBJECT
        module.output_feature_name.value = OUTPUT_MEASUREMENTS

        c = module.get_measurement_columns(None)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0][0], OBJECT[0])
        self.assertEqual(c[0][1], MATH_OUTPUT_MEASUREMENTS)

        self.assertEqual(len(module.get_categories(None, OBJECT[0])), 1)
        self.assertEqual(len(module.get_categories(None, OBJECT[1])), 0)

        self.assertEqual(
                len(module.get_measurements(None, OBJECT[0], C.C_MATH)), 1)
        self.assertEqual(
                len(module.get_measurements(None, OBJECT[1], C.C_MATH)), 0)

    def test_11_01_postadd(self):
        '''Test whether the addend is added to the result'''

        def fn(module, workspace):
            module.final_addend.value = 1.5

        measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
        expected = (5 + 7) + 1.5
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_11_02_constrain_lower(self):
        '''Test whether the lower bound option works'''

        def fn(module, workspace):
            module.constrain_lower_bound.value = True
            module.lower_bound.value = 0

        measurements = self.run_workspace(C.O_SUBTRACT, True, 5, True, 7, fn)
        expected = 0
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)

    def test_11_03_constrain_upper(self):
        '''Test whether the upper bound option works'''

        def fn(module, workspace):
            module.constrain_upper_bound.value = True
            module.upper_bound.value = 10

        measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
        expected = 10
        data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
        self.assertAlmostEqual(data, expected)
