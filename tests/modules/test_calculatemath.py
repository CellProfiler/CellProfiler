"""test_calculatemath.py - Test the CalculateMath module"""

import numpy as np

import cellprofiler.measurement
from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.calculatemath as C

OUTPUT_MEASUREMENTS = "outputmeasurements"
MATH_OUTPUT_MEASUREMENTS = "_".join(("Math", OUTPUT_MEASUREMENTS))
OBJECT = ["object%d" % i for i in range(2)]


def run_workspace(
    self,
    operation,
    m1_is_image_measurement,
    m1_data,
    m2_is_image_measurement,
    m2_data,
    setup_fn=None,
):
    """Create and run a workspace, returning the measurements

    m<n>_is_image_measurement - true for an image measurement, false
                                for object
    m<n>_data - either a single value or an array
    setup_fn - this gets called with the module before running
    """
    module = C.CalculateMath()
    module.operation.value = operation
    measurements = cpmeas.Measurements()
    for i, operand, is_image_measurement, data in (
        (0, module.operands[0], m1_is_image_measurement, m1_data),
        (1, module.operands[1], m2_is_image_measurement, m2_data),
    ):
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
    workspace = cpw.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        cpo.ObjectSet(),
        measurements,
        image_set_list,
    )
    if setup_fn is not None:
        setup_fn(module, workspace)
    module.run(workspace)
    return measurements


def test_add_image_image(self):
    measurements = self.run_workspace(C.O_ADD, True, 2, True, 2)
    assert measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    for i in range(2):
        assert not measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - 4), 7) == 0


def test_add_image_object(self):
    """Add an image measurement to each of several object measurements"""
    measurements = self.run_workspace(C.O_ADD, True, 2, False, np.array([1, 4, 9]))
    assert not measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    assert np.all(data == np.array([3, 6, 11]))


def test_add_object_image(self):
    """Add an image measurement to each of several object measurements (reverse)"""
    measurements = self.run_workspace(C.O_ADD, False, np.array([1, 4, 9]), True, 2)
    assert not measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert np.all(data == np.array([3, 6, 11]))


def test_add_premultiply(self):
    def fn(module, workspace):
        module.operands[0].multiplicand.value = 2
        module.operands[1].multiplicand.value = 3

    measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
    expected = 2 * 5 + 3 * 7
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0


def test_add_pre_exponentiate(self):
    def fn(module, workspace):
        module.operands[0].exponent.value = 2
        module.operands[1].exponent.value = 3

    measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
    expected = 5 ** 2 + 7 ** 3
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0


def test_add_postmultiply(self):
    def fn(module, workspace):
        module.final_multiplicand.value = 3

    measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
    expected = (5 + 7) * 3
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0


def test_add_postexponentiate(self):
    def fn(module, workspace):
        module.final_exponent.value = 3

    measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
    expected = (5 + 7) ** 3
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0


def test_add_log(self):
    def fn(module, workspace):
        module.wants_log.value = True

    measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
    expected = np.log10(5 + 7)
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0


def test_add_object_object(self):
    measurements = self.run_workspace(
        C.O_ADD, False, np.array([1, 2, 3]), False, np.array([1, 4, 9])
    )
    assert not measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    for i in range(2):
        assert measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
        data = measurements.get_current_measurement(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
        assert np.all(data == np.array([2, 6, 12]))


def test_subtract(self):
    measurements = self.run_workspace(C.O_SUBTRACT, True, 7, True, 5)
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - 2), 7) == 0


def test_multiply(self):
    measurements = self.run_workspace(C.O_MULTIPLY, True, 7, True, 5)
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - 35), 7) == 0


def test_divide(self):
    measurements = self.run_workspace(C.O_DIVIDE, True, 35, True, 5)
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - 7), 7) == 0


def test_measurement_columns_image(self):
    module = C.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    for operand in module.operands:
        operand.operand_choice.value = C.MC_IMAGE
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == cpmeas.IMAGE
    assert columns[0][1] == MATH_OUTPUT_MEASUREMENTS
    assert columns[0][2] == cpmeas.COLTYPE_FLOAT
    assert module.get_categories(None, cpmeas.IMAGE)[0] == "Math"
    assert module.get_measurements(None, cpmeas.IMAGE, "Math")[0] == OUTPUT_MEASUREMENTS


def test_measurement_columns_image_object(self):
    module = C.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    module.operands[0].operand_choice.value = C.MC_IMAGE
    module.operands[1].operand_choice.value = C.MC_OBJECT
    module.operands[1].operand_objects.value = OBJECT[1]
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == OBJECT[1]
    assert columns[0][1] == MATH_OUTPUT_MEASUREMENTS
    assert columns[0][2] == cpmeas.COLTYPE_FLOAT
    assert module.get_categories(None, OBJECT[1])[0] == "Math"
    assert module.get_measurements(None, OBJECT[1], "Math")[0] == OUTPUT_MEASUREMENTS
    assert len(module.get_categories(None, cpmeas.IMAGE)) == 0


def test_measurement_columns_object_image(self):
    module = C.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    module.operands[0].operand_choice.value = C.MC_OBJECT
    module.operands[1].operand_choice.value = C.MC_IMAGE
    module.operands[0].operand_objects.value = OBJECT[0]
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == OBJECT[0]
    assert columns[0][1] == MATH_OUTPUT_MEASUREMENTS
    assert columns[0][2] == cpmeas.COLTYPE_FLOAT
    assert module.get_categories(None, OBJECT[0])[0] == "Math"
    assert module.get_measurements(None, OBJECT[0], "Math")[0] == OUTPUT_MEASUREMENTS
    assert len(module.get_categories(None, cpmeas.IMAGE)) == 0


def test_measurement_columns_object_object(self):
    module = C.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    module.operands[0].operand_choice.value = C.MC_OBJECT
    module.operands[1].operand_choice.value = C.MC_OBJECT
    module.operands[0].operand_objects.value = OBJECT[0]
    module.operands[1].operand_objects.value = OBJECT[1]
    columns = list(module.get_measurement_columns(None))
    assert len(columns) == 2
    if columns[0][0] == OBJECT[1]:
        columns = [columns[1], columns[0]]
    for i in range(2):
        assert columns[i][0] == OBJECT[i]
        assert columns[i][1] == MATH_OUTPUT_MEASUREMENTS
        assert columns[i][2] == cpmeas.COLTYPE_FLOAT
        assert module.get_categories(None, OBJECT[i])[0] == "Math"
        assert (
            module.get_measurements(None, OBJECT[i], "Math")[0] == OUTPUT_MEASUREMENTS
        )
    assert len(module.get_categories(None, cpmeas.IMAGE)) == 0


def test_add_object_object_same(self):
    """Regression test: add two measurements from the same object

    The bug was that the measurement gets added twice
    """

    def fn(module, workspace):
        module.operands[1].operand_objects.value = OBJECT[0]
        module.operands[1].operand_measurement.value = "measurement0"

    measurements = self.run_workspace(
        C.O_ADD, False, np.array([5, 6]), False, np.array([-1, -1]), fn
    )
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert len(data) == 2
    assert round(abs(data[0] - 10), 7) == 0
    assert round(abs(data[1] - 12), 7) == 0


def test_img_379(self):
    """Regression test for IMG-379, divide by zero"""

    measurements = self.run_workspace(C.O_DIVIDE, True, 35, True, 0)
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert np.isnan(data)

    measurements = self.run_workspace(
        C.O_DIVIDE, False, np.array([1.0]), False, np.array([0.0])
    )
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert len(data) == 1
    assert np.isnan(data[0])


def test_none_operation(self):
    # In this case, just multiply the array by a constant
    def fn(module, workspace):
        module.operands[0].multiplicand.value = 2

    measurements = self.run_workspace(
        C.O_NONE, False, np.array([1, 2, 3]), False, np.array([1, 4, 9]), fn
    )
    assert not measurements.has_feature(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    # There should be only one operand and a measurement for that operand only
    assert len(OBJECT), 1
    assert measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    # Check the operation result
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data[0] - 2), 7) == 0
    assert round(abs(data[1] - 4), 7) == 0
    assert round(abs(data[2] - 6), 7) == 0


def test_img_919(self):
    """Regression test: one measurement, but both operands are from same object

    The bug was that the measurement gets added twice. It was fixed in run
    but not in get_measurement_columns
    """

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
    assert columns[0][0] == OBJECT[0]
    assert len(columns) == 1


def test_img_1566(self):
    """Regression test: different numbers of objects"""
    r = np.random.RandomState(1566)
    o0 = [
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 1, 2, 2, 3]),
        np.array([1, 2, 4, 5]),
        np.array([1, 1, 1, 1]),
    ]
    o1 = [
        np.array([1, 1, 2, 2, 3]),
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 1, 1, 1]),
        np.array([1, 2, 4, 5]),
    ]
    in0 = [
        np.array([0, 1, 2, 3, 4], float),
        np.array([2, 4, 8], float),
        np.array([0, 1, 2, 3, 4], float),
        np.array([5], float),
    ]
    in1 = [
        np.array([2, 4, 8], float),
        np.array([0, 1, 2, 3, 4], float),
        np.array([5], float),
        np.array([0, 1, 2, 3, 4], float),
    ]

    expected0 = [
        np.array([2, 3, 6, 7, 12]),
        np.array([2.5, 6.5, 12]),
        np.array([5, 6, np.nan, 8, 9]),
        np.array([7]),
    ]
    expected1 = [
        np.array([2.5, 6.5, 12]),
        np.array([2, 3, 6, 7, 12]),
        np.array([7]),
        np.array([5, 6, np.nan, 8, 9]),
    ]
    for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):
        for flip in (False, True):

            def setup_fn(module, workspace, oo0=oo0, oo1=oo1, flip=flip):
                m = workspace.measurements
                assert isinstance(m, cpmeas.Measurements)
                if not flip:
                    m.add_relate_measurement(
                        1,
                        cellprofiler.measurement.R_PARENT,
                        OBJECT[0],
                        OBJECT[1],
                        np.ones(len(oo0), int),
                        oo0,
                        np.ones(len(oo1), int),
                        oo1,
                    )
                else:
                    m.add_relate_measurement(
                        1,
                        cellprofiler.measurement.R_PARENT,
                        OBJECT[1],
                        OBJECT[0],
                        np.ones(len(oo0), int),
                        oo1,
                        np.ones(len(oo1), int),
                        oo0,
                    )

            measurements = self.run_workspace(C.O_ADD, False, ii0, False, ii1, setup_fn)
            data = measurements.get_current_measurement(
                OBJECT[0], MATH_OUTPUT_MEASUREMENTS
            )
            np.testing.assert_almost_equal(e0, data)
            data = measurements.get_current_measurement(
                OBJECT[1], MATH_OUTPUT_MEASUREMENTS
            )
            np.testing.assert_almost_equal(e1, data)


def test_02_different_image_sets(self):
    #
    # Relationship code was matching object numbers from any object
    # set to any other
    #
    r = np.random.RandomState(100102)
    o0 = [
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 1, 2, 2, 3]),
        np.array([1, 2, 4, 5]),
        np.array([1, 1, 1, 1]),
    ]
    o1 = [
        np.array([1, 1, 2, 2, 3]),
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 1, 1, 1]),
        np.array([1, 2, 4, 5]),
    ]
    in0 = [
        np.array([0, 1, 2, 3, 4], float),
        np.array([2, 4, 8], float),
        np.array([0, 1, 2, 3, 4], float),
        np.array([5], float),
    ]
    in1 = [
        np.array([2, 4, 8], float),
        np.array([0, 1, 2, 3, 4], float),
        np.array([5], float),
        np.array([0, 1, 2, 3, 4], float),
    ]

    expected0 = [
        np.array([2, 3, 6, 7, 12]),
        np.array([2.5, 6.5, 12]),
        np.array([5, 6, np.nan, 8, 9]),
        np.array([7]),
    ]
    expected1 = [
        np.array([2.5, 6.5, 12]),
        np.array([2, 3, 6, 7, 12]),
        np.array([7]),
        np.array([5, 6, np.nan, 8, 9]),
    ]
    for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):

        def setup_fn(module, workspace, oo0=oo0, oo1=oo1):
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            m.add_relate_measurement(
                1,
                cellprofiler.measurement.R_PARENT,
                OBJECT[0],
                OBJECT[1],
                np.ones(len(oo0), int),
                oo0,
                np.ones(len(oo1), int),
                oo1,
            )
            for i1, i2 in ((1, 2), (2, 1), (2, 2)):
                m.add_relate_measurement(
                    1,
                    cellprofiler.measurement.R_PARENT,
                    OBJECT[0],
                    OBJECT[1],
                    np.ones(len(oo0), int) * i1,
                    r.permutation(oo0),
                    np.ones(len(oo1), int) * i2,
                    oo1,
                )

        measurements = self.run_workspace(C.O_ADD, False, ii0, False, ii1, setup_fn)
        data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
        np.testing.assert_almost_equal(e0, data)
        data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
        np.testing.assert_almost_equal(e1, data)


def test_issue_422(self):
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
    assert len(c) == 1
    assert c[0][0] == OBJECT[0]
    assert c[0][1] == MATH_OUTPUT_MEASUREMENTS

    assert len(module.get_categories(None, OBJECT[0])) == 1
    assert len(module.get_categories(None, OBJECT[1])) == 0

    assert len(module.get_measurements(None, OBJECT[0], C.C_MATH)) == 1
    assert len(module.get_measurements(None, OBJECT[1], C.C_MATH)) == 0


def test_postadd(self):
    """Test whether the addend is added to the result"""

    def fn(module, workspace):
        module.final_addend.value = 1.5

    measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
    expected = (5 + 7) + 1.5
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0


def test_constrain_lower(self):
    """Test whether the lower bound option works"""

    def fn(module, workspace):
        module.constrain_lower_bound.value = True
        module.lower_bound.value = 0

    measurements = self.run_workspace(C.O_SUBTRACT, True, 5, True, 7, fn)
    expected = 0
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0


def test_constrain_upper(self):
    """Test whether the upper bound option works"""

    def fn(module, workspace):
        module.constrain_upper_bound.value = True
        module.upper_bound.value = 10

    measurements = self.run_workspace(C.O_ADD, True, 5, True, 7, fn)
    expected = 10
    data = measurements.get_current_measurement(cpmeas.IMAGE, MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data - expected), 7) == 0
