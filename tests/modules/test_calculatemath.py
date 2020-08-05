import numpy

import cellprofiler_core.image
import cellprofiler_core.measurement
import cellprofiler_core.measurement
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT, R_PARENT


import cellprofiler.modules.calculatemath
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.workspace

OUTPUT_MEASUREMENTS = "outputmeasurements"
MATH_OUTPUT_MEASUREMENTS = "_".join(("Math", OUTPUT_MEASUREMENTS))
OBJECT = ["object%d" % i for i in range(2)]


def run_workspace(
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
    module = cellprofiler.modules.calculatemath.CalculateMath()
    module.operation.value = operation
    measurements = cellprofiler_core.measurement.Measurements()
    for i, operand, is_image_measurement, data in (
        (0, module.operands[0], m1_is_image_measurement, m1_data),
        (1, module.operands[1], m2_is_image_measurement, m2_data),
    ):
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
    module.rounding.value = "Not rounded"
    module.rounding_digit.value = 0
    pipeline = cellprofiler_core.pipeline.Pipeline()
    image_set_list = cellprofiler_core.image.ImageSetList()
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set_list.get_image_set(0),
        cellprofiler_core.object.ObjectSet(),
        measurements,
        image_set_list,
    )
    if setup_fn is not None:
        setup_fn(module, workspace)
    module.run(workspace)
    return measurements


def test_add_image_image():
    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 2, True, 2
    )
    assert measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    for i in range(2):
        assert not measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - 4), 7) == 0


def test_add_image_object():
    """Add an image measurement to each of several object measurements"""
    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 2, False, numpy.array([1, 4, 9])
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    assert numpy.all(data == numpy.array([3, 6, 11]))


def test_add_object_image():
    """Add an image measurement to each of several object measurements (reverse)"""
    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, False, numpy.array([1, 4, 9]), True, 2
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert numpy.all(data == numpy.array([3, 6, 11]))


def test_add_premultiply():
    def fn(module, workspace):
        module.operands[0].multiplicand.value = 2
        module.operands[1].multiplicand.value = 3

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn
    )
    expected = 2 * 5 + 3 * 7
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_add_pre_exponentiate():
    def fn(module, workspace):
        module.operands[0].exponent.value = 2
        module.operands[1].exponent.value = 3

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn
    )
    expected = 5 ** 2 + 7 ** 3
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_add_postmultiply():
    def fn(module, workspace):
        module.final_multiplicand.value = 3

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn
    )
    expected = (5 + 7) * 3
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_add_postexponentiate():
    def fn(module, workspace):
        module.final_exponent.value = 3

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn
    )
    expected = (5 + 7) ** 3
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_add_log():
    def fn(module, workspace):
        module.wants_log.value = True

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn
    )
    expected = numpy.log10(5 + 7)
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_add_object_object():
    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD,
        False,
        numpy.array([1, 2, 3]),
        False,
        numpy.array([1, 4, 9]),
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    for i in range(2):
        assert measurements.has_feature(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
        data = measurements.get_current_measurement(OBJECT[i], MATH_OUTPUT_MEASUREMENTS)
        assert numpy.all(data == numpy.array([2, 6, 12]))


def test_subtract():
    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_SUBTRACT, True, 7, True, 5
    )
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - 2), 7) == 0


def test_multiply():
    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_MULTIPLY, True, 7, True, 5
    )
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - 35), 7) == 0


def test_divide():
    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_DIVIDE, True, 35, True, 5
    )
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - 7), 7) == 0


def test_measurement_columns_image():
    module = cellprofiler.modules.calculatemath.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    for operand in module.operands:
        operand.operand_choice.value = cellprofiler.modules.calculatemath.MC_IMAGE
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == "Image"
    assert columns[0][1] == MATH_OUTPUT_MEASUREMENTS
    assert columns[0][2] == COLTYPE_FLOAT
    assert module.get_categories(None, "Image")[0] == "Math"
    assert (
        module.get_measurements(None, "Image", "Math")[0]
        == OUTPUT_MEASUREMENTS
    )


def test_measurement_columns_image_object():
    module = cellprofiler.modules.calculatemath.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    module.operands[
        0
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_IMAGE
    module.operands[
        1
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.operands[1].operand_objects.value = OBJECT[1]
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == OBJECT[1]
    assert columns[0][1] == MATH_OUTPUT_MEASUREMENTS
    assert columns[0][2] == COLTYPE_FLOAT
    assert module.get_categories(None, OBJECT[1])[0] == "Math"
    assert module.get_measurements(None, OBJECT[1], "Math")[0] == OUTPUT_MEASUREMENTS
    assert len(module.get_categories(None, "Image")) == 0


def test_measurement_columns_object_image():
    module = cellprofiler.modules.calculatemath.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    module.operands[
        0
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.operands[
        1
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_IMAGE
    module.operands[0].operand_objects.value = OBJECT[0]
    columns = module.get_measurement_columns(None)
    assert len(columns) == 1
    assert columns[0][0] == OBJECT[0]
    assert columns[0][1] == MATH_OUTPUT_MEASUREMENTS
    assert columns[0][2] == COLTYPE_FLOAT
    assert module.get_categories(None, OBJECT[0])[0] == "Math"
    assert module.get_measurements(None, OBJECT[0], "Math")[0] == OUTPUT_MEASUREMENTS
    assert len(module.get_categories(None, "Image")) == 0


def test_measurement_columns_object_object():
    module = cellprofiler.modules.calculatemath.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    module.operands[
        0
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.operands[
        1
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.operands[0].operand_objects.value = OBJECT[0]
    module.operands[1].operand_objects.value = OBJECT[1]
    columns = list(module.get_measurement_columns(None))
    assert len(columns) == 2
    if columns[0][0] == OBJECT[1]:
        columns = [columns[1], columns[0]]
    for i in range(2):
        assert columns[i][0] == OBJECT[i]
        assert columns[i][1] == MATH_OUTPUT_MEASUREMENTS
        assert columns[i][2] == COLTYPE_FLOAT
        assert module.get_categories(None, OBJECT[i])[0] == "Math"
        assert (
            module.get_measurements(None, OBJECT[i], "Math")[0] == OUTPUT_MEASUREMENTS
        )
    assert len(module.get_categories(None, "Image")) == 0


def test_add_object_object_same():
    """Regression test: add two measurements from the same object

    The bug was that the measurement gets added twice
    """

    def fn(module, workspace):
        module.operands[1].operand_objects.value = OBJECT[0]
        module.operands[1].operand_measurement.value = "measurement0"

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD,
        False,
        numpy.array([5, 6]),
        False,
        numpy.array([-1, -1]),
        fn,
    )
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert len(data) == 2
    assert round(abs(data[0] - 10), 7) == 0
    assert round(abs(data[1] - 12), 7) == 0


def test_img_379():
    """Regression test for IMG-379, divide by zero"""

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_DIVIDE, True, 35, True, 0
    )
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert numpy.isnan(data)

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_DIVIDE,
        False,
        numpy.array([1.0]),
        False,
        numpy.array([0.0]),
    )
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert len(data) == 1
    assert numpy.isnan(data[0])


def test_none_operation():
    # In this case, just multiply the array by a constant
    def fn(module, workspace):
        module.operands[0].multiplicand.value = 2

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_NONE,
        False,
        numpy.array([1, 2, 3]),
        False,
        numpy.array([1, 4, 9]),
        fn,
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    # There should be only one operand and a measurement for that operand only
    assert len(OBJECT), 1
    assert measurements.has_feature(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    # Check the operation result
    data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
    assert round(abs(data[0] - 2), 7) == 0
    assert round(abs(data[1] - 4), 7) == 0
    assert round(abs(data[2] - 6), 7) == 0


def test_img_919():
    """Regression test: one measurement, but both operands are from same object

    The bug was that the measurement gets added twice. It was fixed in run
    but not in get_measurement_columns
    """

    def fn(module):
        module.operands[1].operand_objects.value = OBJECT[0]
        module.operands[1].operand_measurement.value = "measurement0"

    module = cellprofiler.modules.calculatemath.CalculateMath()
    module.output_feature_name.value = OUTPUT_MEASUREMENTS
    module.operands[
        0
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.operands[
        1
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.operands[0].operand_objects.value = OBJECT[0]
    module.operands[1].operand_objects.value = OBJECT[0]
    columns = module.get_measurement_columns(None)
    assert columns[0][0] == OBJECT[0]
    assert len(columns) == 1


def test_img_1566():
    """Regression test: different numbers of objects"""
    r = numpy.random.RandomState(1566)
    o0 = [
        numpy.array([1, 2, 3, 4, 5]),
        numpy.array([1, 1, 2, 2, 3]),
        numpy.array([1, 2, 4, 5]),
        numpy.array([1, 1, 1, 1]),
    ]
    o1 = [
        numpy.array([1, 1, 2, 2, 3]),
        numpy.array([1, 2, 3, 4, 5]),
        numpy.array([1, 1, 1, 1]),
        numpy.array([1, 2, 4, 5]),
    ]
    in0 = [
        numpy.array([0, 1, 2, 3, 4], float),
        numpy.array([2, 4, 8], float),
        numpy.array([0, 1, 2, 3, 4], float),
        numpy.array([5], float),
    ]
    in1 = [
        numpy.array([2, 4, 8], float),
        numpy.array([0, 1, 2, 3, 4], float),
        numpy.array([5], float),
        numpy.array([0, 1, 2, 3, 4], float),
    ]

    expected0 = [
        numpy.array([2, 3, 6, 7, 12]),
        numpy.array([2.5, 6.5, 12]),
        numpy.array([5, 6, numpy.nan, 8, 9]),
        numpy.array([7]),
    ]
    expected1 = [
        numpy.array([2.5, 6.5, 12]),
        numpy.array([2, 3, 6, 7, 12]),
        numpy.array([7]),
        numpy.array([5, 6, numpy.nan, 8, 9]),
    ]
    for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):
        for flip in (False, True):

            def setup_fn(module, workspace, oo0=oo0, oo1=oo1, flip=flip):
                m = workspace.measurements
                assert isinstance(m, cellprofiler_core.measurement.Measurements)
                if not flip:
                    m.add_relate_measurement(
                        1,
                        R_PARENT,
                        OBJECT[0],
                        OBJECT[1],
                        numpy.ones(len(oo0), int),
                        oo0,
                        numpy.ones(len(oo1), int),
                        oo1,
                    )
                else:
                    m.add_relate_measurement(
                        1,
                        R_PARENT,
                        OBJECT[1],
                        OBJECT[0],
                        numpy.ones(len(oo0), int),
                        oo1,
                        numpy.ones(len(oo1), int),
                        oo0,
                    )

            measurements = run_workspace(
                cellprofiler.modules.calculatemath.O_ADD,
                False,
                ii0,
                False,
                ii1,
                setup_fn,
            )
            data = measurements.get_current_measurement(
                OBJECT[0], MATH_OUTPUT_MEASUREMENTS
            )
            numpy.testing.assert_almost_equal(e0, data)
            data = measurements.get_current_measurement(
                OBJECT[1], MATH_OUTPUT_MEASUREMENTS
            )
            numpy.testing.assert_almost_equal(e1, data)


def test_02_different_image_sets():
    #
    # Relationship code was matching object numbers from any object
    # set to any other
    #
    r = numpy.random.RandomState(100102)
    o0 = [
        numpy.array([1, 2, 3, 4, 5]),
        numpy.array([1, 1, 2, 2, 3]),
        numpy.array([1, 2, 4, 5]),
        numpy.array([1, 1, 1, 1]),
    ]
    o1 = [
        numpy.array([1, 1, 2, 2, 3]),
        numpy.array([1, 2, 3, 4, 5]),
        numpy.array([1, 1, 1, 1]),
        numpy.array([1, 2, 4, 5]),
    ]
    in0 = [
        numpy.array([0, 1, 2, 3, 4], float),
        numpy.array([2, 4, 8], float),
        numpy.array([0, 1, 2, 3, 4], float),
        numpy.array([5], float),
    ]
    in1 = [
        numpy.array([2, 4, 8], float),
        numpy.array([0, 1, 2, 3, 4], float),
        numpy.array([5], float),
        numpy.array([0, 1, 2, 3, 4], float),
    ]

    expected0 = [
        numpy.array([2, 3, 6, 7, 12]),
        numpy.array([2.5, 6.5, 12]),
        numpy.array([5, 6, numpy.nan, 8, 9]),
        numpy.array([7]),
    ]
    expected1 = [
        numpy.array([2.5, 6.5, 12]),
        numpy.array([2, 3, 6, 7, 12]),
        numpy.array([7]),
        numpy.array([5, 6, numpy.nan, 8, 9]),
    ]
    for oo0, oo1, ii0, ii1, e0, e1 in zip(o0, o1, in0, in1, expected0, expected1):

        def setup_fn(module, workspace, oo0=oo0, oo1=oo1):
            m = workspace.measurements
            assert isinstance(m,cellprofiler_core.measurement.Measurements)
            m.add_relate_measurement(
                1,
                R_PARENT,
                OBJECT[0],
                OBJECT[1],
                numpy.ones(len(oo0), int),
                oo0,
                numpy.ones(len(oo1), int),
                oo1,
            )
            for i1, i2 in ((1, 2), (2, 1), (2, 2)):
                m.add_relate_measurement(
                    1,
                    R_PARENT,
                    OBJECT[0],
                    OBJECT[1],
                    numpy.ones(len(oo0), int) * i1,
                    r.permutation(oo0),
                    numpy.ones(len(oo1), int) * i2,
                    oo1,
                )

        measurements = run_workspace(
            cellprofiler.modules.calculatemath.O_ADD, False, ii0, False, ii1, setup_fn
        )
        data = measurements.get_current_measurement(OBJECT[0], MATH_OUTPUT_MEASUREMENTS)
        numpy.testing.assert_almost_equal(e0, data)
        data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
        numpy.testing.assert_almost_equal(e1, data)


def test_issue_422():
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
    module.operands[
        0
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.operands[
        1
    ].operand_choice.value = cellprofiler.modules.calculatemath.MC_OBJECT
    module.output_feature_name.value = OUTPUT_MEASUREMENTS

    c = module.get_measurement_columns(None)
    assert len(c) == 1
    assert c[0][0] == OBJECT[0]
    assert c[0][1] == MATH_OUTPUT_MEASUREMENTS

    assert len(module.get_categories(None, OBJECT[0])) == 1
    assert len(module.get_categories(None, OBJECT[1])) == 0

    assert (
        len(
            module.get_measurements(
                None, OBJECT[0], cellprofiler.modules.calculatemath.C_MATH
            )
        )
        == 1
    )
    assert (
        len(
            module.get_measurements(
                None, OBJECT[1], cellprofiler.modules.calculatemath.C_MATH
            )
        )
        == 0
    )


def test_postadd():
    """Test whether the addend is added to the result"""

    def fn(module, workspace):
        module.final_addend.value = 1.5

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn
    )
    expected = (5 + 7) + 1.5
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_constrain_lower():
    """Test whether the lower bound option works"""

    def fn(module, workspace):
        module.constrain_lower_bound.value = True
        module.lower_bound.value = 0

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_SUBTRACT, True, 5, True, 7, fn
    )
    expected = 0
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_constrain_upper():
    """Test whether the upper bound option works"""

    def fn(module, workspace):
        module.constrain_upper_bound.value = True
        module.upper_bound.value = 10

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD, True, 5, True, 7, fn
    )
    expected = 10
    data = measurements.get_current_measurement(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert round(abs(data - expected), 7) == 0


def test_round_digit_1():
    """Test if rounding to the first decimal place works"""

    def fn(module, workspace):
        module.rounding.value = cellprofiler.modules.calculatemath.ROUNDING[1]
        module.rounding_digit.value = 1

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD,
        True,
        2.1,
        False,
        numpy.array([1, 4, 9]),
        fn,
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    numpy.testing.assert_almost_equal(data, numpy.array([3.1, 6.1, 11.1]))


def test_round_digit_0():
    """Test if rounding to the zeroth decimal place works"""

    def fn(module, workspace):
        module.rounding.value = cellprofiler.modules.calculatemath.ROUNDING[1]
        module.rounding_digit.value = 0

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD,
        True,
        2.1,
        False,
        numpy.array([1, 4, 9]),
        fn,
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    numpy.testing.assert_almost_equal(data, numpy.array([3, 6, 11]))


def test_round_floor():
    """Test if floor rounding works"""

    def fn(module, workspace):
        module.rounding.value = cellprofiler.modules.calculatemath.ROUNDING[2]

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD,
        True,
        2.1,
        False,
        numpy.array([1, 4, 9]),
        fn,
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    numpy.testing.assert_almost_equal(data, numpy.array([3, 6, 11]))


def test_round_ceil():
    """Test if ceiling rounding works"""

    def fn(module, workspace):
        module.rounding.value = cellprofiler.modules.calculatemath.ROUNDING[3]

    measurements = run_workspace(
        cellprofiler.modules.calculatemath.O_ADD,
        True,
        2.1,
        False,
        numpy.array([1, 4, 9]),
        fn,
    )
    assert not measurements.has_feature(
        "Image", MATH_OUTPUT_MEASUREMENTS
    )
    assert measurements.has_feature(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    data = measurements.get_current_measurement(OBJECT[1], MATH_OUTPUT_MEASUREMENTS)
    numpy.testing.assert_almost_equal(data, numpy.array([4, 7, 12]))
