import os
import tempfile

import numpy

import cellprofiler_core.measurement
import cellprofiler.modules.loadimages
import cellprofiler.modules.mergeoutputfiles
import cellprofiler.pipeline


def execute_merge_files(mm):
    input_files = []
    output_fd, output_file = tempfile.mkstemp(".mat")
    pipeline = cellprofiler.pipeline.Pipeline()
    li = cellprofiler.modules.loadimages.LoadImages()
    li.module_num = 1
    pipeline.add_module(li)

    for m in mm:
        input_fd, input_file = tempfile.mkstemp(".mat")
        pipeline.save_measurements(input_file, m)
        input_files.append((input_fd, input_file))

    cellprofiler.modules.mergeoutputfiles.MergeOutputFiles.merge_files(
        output_file, [x[1] for x in input_files]
    )
    m = cellprofiler_core.measurement.load_measurements(output_file)
    os.close(output_fd)
    os.remove(output_file)
    for fd, filename in input_files:
        os.close(fd)
        os.remove(filename)
    return m


def write_image_measurements(m, feature, image_count):
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i in range(image_count):
        if i > 0:
            m.next_image_set(i + 1)
        m.add_image_measurement(feature, numpy.random.uniform())


def write_object_measurements(m, object_name, feature, object_counts):
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    for i, count in enumerate(object_counts):
        object_measurements = numpy.random.uniform(size=i)
        m.add_measurement(
            object_name, feature, object_measurements, image_set_number=i + 1
        )


def write_experiment_measurement(m, feature):
    assert isinstance(m, cellprofiler_core.measurement.Measurements)
    m.add_experiment_measurement(feature, numpy.random.uniform())


def test_nothing():
    """Make sure merge_files doesn't crash if no inputs"""
    cellprofiler.modules.mergeoutputfiles.MergeOutputFiles.merge_files("nope", [])


def test_one():
    """Test "merging" one file"""
    numpy.random.seed(11)
    m = cellprofiler_core.measurement.Measurements()
    write_image_measurements(m, "foo", 5)
    write_object_measurements(m, "myobjects", "bar", [3, 6, 2, 9, 16])
    write_experiment_measurement(m, "baz")
    result = execute_merge_files([m])
    assert (
        round(
            abs(
                result.get_experiment_measurement("baz")
                - m.get_experiment_measurement("baz")
            ),
            7,
        )
        == 0
    )
    ro = result.get_all_measurements("myobjects", "bar")
    mo = m.get_all_measurements("myobjects", "bar")
    for i in range(5):
        assert (
            round(
                abs(
                    result.get_all_measurements(cellprofiler_core.measurement.IMAGE, "foo")[
                        i
                    ]
                    - m.get_all_measurements(cellprofiler_core.measurement.IMAGE, "foo")[i]
                ),
                7,
            )
            == 0
        )
        assert len(ro[i]) == len(mo[i])
        numpy.testing.assert_almost_equal(ro[i], mo[i])


def test_two():
    numpy.random.seed(12)
    mm = []
    for i in range(2):
        m = cellprofiler_core.measurement.Measurements()
        write_image_measurements(m, "foo", 5)
        write_object_measurements(m, "myobjects", "bar", [3, 6, 2, 9, 16])
        write_experiment_measurement(m, "baz")
        mm.append(m)
    result = execute_merge_files(mm)
    assert (
        round(
            abs(
                result.get_experiment_measurement("baz")
                - mm[0].get_experiment_measurement("baz")
            ),
            7,
        )
        == 0
    )
    ro = result.get_all_measurements("myobjects", "bar")
    moo = [m.get_all_measurements("myobjects", "bar") for m in mm]
    for i in range(5):
        for j in range(2):
            numpy.testing.assert_almost_equal(ro[i + j * 5], moo[j][i])
        assert len(ro[i + j * 5]) == len(moo[j][i])
        numpy.testing.assert_almost_equal(ro[i + j * 5], moo[j][i])


def test_different_measurements():
    numpy.random.seed(13)
    mm = []
    for i in range(2):
        m = cellprofiler_core.measurement.Measurements()
        write_image_measurements(m, "foo", 5)
        write_object_measurements(m, "myobjects", "bar%d" % i, [3, 6, 2, 9, 16])
        write_experiment_measurement(m, "baz")
        mm.append(m)
    result = execute_merge_files(mm)
    assert (
        round(
            abs(
                result.get_experiment_measurement("baz")
                - mm[0].get_experiment_measurement("baz")
            ),
            7,
        )
        == 0
    )
    for imgidx in range(10):
        imgnum = imgidx + 1
        if imgidx < 5:
            ro = result.get_measurement("myobjects", "bar0", imgnum)
            mo = mm[0].get_measurement("myobjects", "bar0", imgnum)
            numpy.testing.assert_almost_equal(ro, mo)
            assert len(result.get_measurement("myobjects", "bar1", imgnum)) == 0
        else:
            ro = result.get_measurement("myobjects", "bar1", imgnum)
            mo = mm[1].get_measurement("myobjects", "bar1", imgnum - 5)
            numpy.testing.assert_almost_equal(ro, mo)
            assert len(result.get_measurement("myobjects", "bar0", imgnum)) == 0


def test_different_objects():
    numpy.random.seed(13)
    mm = []
    for i in range(2):
        m = cellprofiler_core.measurement.Measurements()
        write_image_measurements(m, "foo", 5)
        write_object_measurements(m, "myobjects%d" % i, "bar", [3, 6, 2, 9, 16])
        write_experiment_measurement(m, "baz")
        mm.append(m)
    result = execute_merge_files(mm)
    assert (
        round(
            abs(
                result.get_experiment_measurement("baz")
                - mm[0].get_experiment_measurement("baz")
            ),
            7,
        )
        == 0
    )
    for imgidx in range(10):
        imgnum = imgidx + 1
        if imgidx < 5:
            ro = result.get_measurement("myobjects0", "bar", imgnum)
            mo = mm[0].get_measurement("myobjects0", "bar", imgnum)
            numpy.testing.assert_almost_equal(ro, mo)
            assert len(result.get_measurement("myobjects1", "bar", imgnum)) == 0
        else:
            ro = result.get_measurement("myobjects1", "bar", imgnum)
            mo = mm[1].get_measurement("myobjects1", "bar", imgnum - 5)
            numpy.testing.assert_almost_equal(ro, mo)
            assert len(result.get_measurement("myobjects0", "bar", imgnum)) == 0
