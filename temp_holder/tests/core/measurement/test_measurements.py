import base64
import functools
import os
import tempfile
import zlib
import unittest

import numpy
import six.moves

import cellprofiler_core.constants.measurement
import cellprofiler_core.measurement

OBJECT_NAME = "myobjects"
FEATURE_NAME = "feature"


class TestMeasurements:
    def test_00_00_init(self):
        x = cellprofiler_core.measurement.Measurements()

    def test_00_01_wrap_unwrap(self):
        test = ["foo", "foo\\", "foo\\u0384", "foo\\u0384"]
        # numpy.object_
        test += numpy.array(test, object).tolist()
        for case in test:
            result = cellprofiler_core.measurement.Measurements.unwrap_string(
                cellprofiler_core.measurement.Measurements.wrap_string(case)
            )

            assert result == case

    def test_01_01_image_number_is_zero(self):
        x = cellprofiler_core.measurement.Measurements()
        assert x.image_set_number == 1

    def test_01_01_next_image(self):
        x = cellprofiler_core.measurement.Measurements()
        x.next_image_set()
        assert x.image_set_number == 2

    def test_02_01_add_image_measurement(self):
        x = cellprofiler_core.measurement.Measurements()
        x.add_measurement("Image", "Feature", "Value")
        assert x.get_current_measurement("Image", "Feature") == "Value"
        assert "Image" in x.get_object_names()
        assert "Feature" in x.get_feature_names("Image")

    def test_02_01b_add_image_measurement_arrayinterface(self):
        x = cellprofiler_core.measurement.Measurements()
        x["Image", "Feature"] = "Value"
        assert x["Image", "Feature"] == "Value"
        assert "Image" in x.get_object_names()
        assert "Feature" in x.get_feature_names("Image")

    def test_02_02_add_object_measurement(self):
        x = cellprofiler_core.measurement.Measurements()
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m)
        assert (x.get_current_measurement("Nuclei", "Feature") == m).all
        assert "Nuclei" in x.get_object_names()
        assert "Feature" in x.get_feature_names("Nuclei")

    def test_02_02b_add_object_measurement_arrayinterface(self):
        x = cellprofiler_core.measurement.Measurements()
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x["Nuclei", "Feature"] = m
        assert (x["Nuclei", "Feature"] == m).all
        assert "Nuclei" in x.get_object_names()
        assert "Feature" in x.get_feature_names("Nuclei")

    def test_02_03_add_two_measurements(self):
        x = cellprofiler_core.measurement.Measurements()
        x.add_measurement("Image", "Feature", "Value")
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m)
        assert x.get_current_measurement("Image", "Feature") == "Value"
        assert (x.get_current_measurement("Nuclei", "Feature") == m).all()
        assert "Image" in x.get_object_names()
        assert "Nuclei" in x.get_object_names()
        assert "Feature" in x.get_feature_names("Image")

    def test_02_03b_add_two_measurements_arrayinterface(self):
        x = cellprofiler_core.measurement.Measurements()
        x["Image", "Feature"] = "Value"
        numpy.random.seed(0)
        m = numpy.random.rand(10)
        x["Nuclei", "Feature"] = m
        assert x["Image", "Feature"] == "Value"
        assert (x["Nuclei", "Feature"] == m).all()
        assert "Image" in x.get_object_names()
        assert "Nuclei" in x.get_object_names()
        assert "Feature" in x.get_feature_names("Image")

    def test_02_04_add_two_measurements_to_object(self):
        x = cellprofiler_core.measurement.Measurements()
        x.add_measurement("Image", "Feature1", "Value1")
        x.add_measurement("Image", "Feature2", "Value2")
        assert x.get_current_measurement("Image", "Feature1") == "Value1"
        assert x.get_current_measurement("Image", "Feature2") == "Value2"
        assert "Image" in x.get_object_names()
        assert "Feature1" in x.get_feature_names("Image")
        assert "Feature2" in x.get_feature_names("Image")

    def test_02_04b_add_two_measurements_to_object_arrayinterface(self):
        x = cellprofiler_core.measurement.Measurements()
        x["Image", "Feature1"] = "Value1"
        x["Image", "Feature2"] = "Value2"
        assert x["Image", "Feature1"] == "Value1"
        assert x["Image", "Feature2"] == "Value2"
        assert "Image" in x.get_object_names()
        assert "Feature1" in x.get_feature_names("Image")
        assert "Feature2" in x.get_feature_names("Image")

    def test_03_03_MultipleImageSets(self):
        numpy.random.seed(0)
        x = cellprofiler_core.measurement.Measurements()
        x.add_measurement("Image", "Feature", "Value1")
        m1 = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m1)
        x.next_image_set()
        x.add_measurement("Image", "Feature", "Value2")
        m2 = numpy.random.rand(10)
        x.add_measurement("Nuclei", "Feature", m2)
        assert x.get_current_measurement("Image", "Feature") == "Value2"
        assert (x.get_current_measurement("Nuclei", "Feature") == m2).all()
        for a, b in zip(
            x.get_all_measurements("Image", "Feature"), ["Value1", "Value2"]
        ):
            assert a == b
        for a, b in zip(x.get_all_measurements("Nuclei", "Feature"), [m1, m2]):
            assert (a == b).all()

    def test_03_03b_MultipleImageSets_arrayinterface(self):
        numpy.random.seed(0)
        x = cellprofiler_core.measurement.Measurements()
        x["Image", "Feature"] = "Value1"
        m1 = numpy.random.rand(10)
        x["Nuclei", "Feature"] = m1
        x.next_image_set()
        x["Image", "Feature"] = "Value2"
        m2 = numpy.random.rand(10)
        x["Nuclei", "Feature"] = m2
        assert x["Image", "Feature"] == "Value2"
        assert (x["Nuclei", "Feature"] == m2).all()
        for a, b in zip(x["Image", "Feature", :], ["Value1", "Value2"]):
            assert a == b
        for a, b in zip(x["Nuclei", "Feature", :], [m1, m2]):
            assert (a == b).all()

    def test_04_01_get_all_image_measurements_float(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(41)
        vals = r.uniform(size=100)
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Feature",
                vals[image_number - 1],
                image_set_number=image_number,
            )
        result = m.get_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE, "Feature"
        )
        numpy.testing.assert_equal(result, vals)

    def test_04_01b_get_all_image_measurements_float_arrayinterface(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(41)
        vals = r.uniform(size=100)
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m[
                cellprofiler_core.constants.measurement.IMAGE, "Feature", image_number
            ] = vals[image_number - 1]
        result = m[cellprofiler_core.constants.measurement.IMAGE, "Feature", :]
        numpy.testing.assert_equal(result, vals)
        for image_number in bad_order:
            result = m[
                cellprofiler_core.constants.measurement.IMAGE, "Feature", image_number
            ]
            numpy.testing.assert_equal(result, vals[image_number - 1])

    def test_04_02_get_all_image_measurements_string(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(42)
        vals = r.uniform(size=100)
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Feature",
                six.text_type(vals[image_number - 1]),
                image_set_number=image_number,
            )
        result = m.get_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE, "Feature"
        )
        assert all([r == six.text_type(v) for r, v in zip(result, vals)])

    def test_04_02b_get_all_image_measurements_string_arrayinterface(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(42)
        vals = r.uniform(size=100)
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m[
                cellprofiler_core.constants.measurement.IMAGE, "Feature", image_number
            ] = six.text_type(vals[image_number - 1])
        result = m[cellprofiler_core.constants.measurement.IMAGE, "Feature", :]
        assert all([r == six.text_type(v) for r, v in zip(result, vals)])

    def test_04_03_get_all_image_measurements_unicode(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(42)
        vals = ["\\u2211" + str(r.uniform()) for _ in range(100)]
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Feature",
                vals[image_number - 1],
                image_set_number=image_number,
            )
        result = m.get_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE, "Feature"
        )
        assert all([r == six.text_type(v) for r, v in zip(result, vals)])

    def test_04_03b_get_all_image_measurements_unicode_arrayinterface(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(42)
        vals = ["\\u2211" + str(r.uniform()) for _ in range(100)]
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m[
                cellprofiler_core.constants.measurement.IMAGE, "Feature", image_number
            ] = vals[image_number - 1]
        result = m[cellprofiler_core.constants.measurement.IMAGE, "Feature", :]
        assert all([r == six.text_type(v) for r, v in zip(result, vals)])

    def test_04_04_get_all_object_measurements(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(42)
        vals = [r.uniform(size=r.randint(10, 100)) for _ in range(100)]
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m.add_measurement(
                OBJECT_NAME,
                "Feature",
                vals[image_number - 1],
                image_set_number=image_number,
            )
        result = m.get_all_measurements(OBJECT_NAME, "Feature")
        assert all(
            [numpy.all(r == v) and len(r) == len(v) for r, v in zip(result, vals)]
        )

    def test_04_04b_get_all_object_measurements_arrayinterface(self):
        r = numpy.random.RandomState()
        m = cellprofiler_core.measurement.Measurements()
        r.seed(42)
        vals = [r.uniform(size=r.randint(10, 100)) for _ in range(100)]
        bad_order = r.permutation(numpy.arange(1, 101))
        for image_number in bad_order:
            m[OBJECT_NAME, "Feature", image_number] = vals[image_number - 1]
        result = m[OBJECT_NAME, "Feature", :]
        assert all(
            [numpy.all(r == v) and len(r) == len(v) for r, v in zip(result, vals)]
        )

    # def test_04_05_get_many_string_measurements(self):
    #     #
    #     # Add string measurements that are likely to break things, then
    #     # read them back in one shot
    #     #
    #     test = [u"foo", u"foo\\", u"foo\\u0384", u"foo\u0384"]
    #     test = test + [x.encode('utf-8') for x in test]
    #     test.append(None)
    #     expected = [x if isinstance(x, unicode)
    #                 or x is None
    #                 else x.decode('utf-8')
    #                 for x in test]
    #     m = cpmeas.Measurements()
    #     for i, v in enumerate(test):
    #         m[cpmeas.IMAGE, "Feature", i+1] = v
    #
    #     result = m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)]
    #     self.assertSequenceEqual(expected, result)

    # def test_04_06_set_many_string_measurements(self):
    #     #
    #     # Add string measurements all at once
    #     #
    #     test = [u"foo", u"foo\\", u"foo\\u0384", u"foo\u0384"]
    #     test = test + [x.encode('utf-8') for x in test]
    #     test.append(None)
    #     expected = [x if isinstance(x, unicode)
    #                 or x is None
    #                 else x.decode('utf-8')
    #                 for x in test]
    #     m = cpmeas.Measurements()
    #     m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)] = test
    #
    #     result = m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)]
    #     self.assertSequenceEqual(expected, result)

    # def test_04_07_set_many_numeric_measurements(self):
    #     test = [1.5, np.NaN, 3.0]
    #     m = cpmeas.Measurements()
    #     m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)] = test
    #
    #     result = m[cpmeas.IMAGE, "Feature", range(1, len(test)+1)]
    #     np.testing.assert_array_equal(test, result)

    def test_04_08_set_one_blob_measurement(self):
        r = numpy.random.RandomState(408)
        test = r.randint(0, 255, 10).astype(numpy.uint8)
        m = cellprofiler_core.measurement.Measurements()
        m[
            cellprofiler_core.constants.measurement.IMAGE, "Feature", 1, numpy.uint8
        ] = test
        numpy.testing.assert_array_equal(
            test, m[cellprofiler_core.constants.measurement.IMAGE, "Feature", 1]
        )

    def test_04_09_set_many_blob_measurements(self):
        #
        # This is a regression test which ran into the exception
        # "ValueError: setting an array element with a sequence"
        # when CP attempted to execute something like:
        # np.array([np.nan, np.zeros(5)])
        #
        r = numpy.random.RandomState(408)
        test = [None, r.randint(0, 255, 10).astype(numpy.uint8)]
        m = cellprofiler_core.measurement.Measurements()
        image_numbers = numpy.arange(1, len(test) + 1)
        m[
            cellprofiler_core.constants.measurement.IMAGE,
            "Feature",
            image_numbers,
            numpy.uint8,
        ] = test
        result = m[
            cellprofiler_core.constants.measurement.IMAGE, "Feature", image_numbers
        ]
        assert result[0] is None
        numpy.testing.assert_array_equal(test[1], result[1])

    def test_05_01_test_has_current_measurements(self):
        x = cellprofiler_core.measurement.Measurements()
        assert not x.has_current_measurements("Image", "Feature")

    def test_05_02_test_has_current_measurements(self):
        x = cellprofiler_core.measurement.Measurements()
        x.add_measurement("Image", "OtherFeature", "Value")
        assert not x.has_current_measurements("Image", "Feature")

    def test_05_02b_test_has_current_measurements_arrayinterface(self):
        x = cellprofiler_core.measurement.Measurements()
        x["Image", "OtherFeature"] = "Value"
        assert not x.has_current_measurements("Image", "Feature")

    def test_05_03_test_has_current_measurements(self):
        x = cellprofiler_core.measurement.Measurements()
        x.add_measurement("Image", "Feature", "Value")
        assert x.has_current_measurements("Image", "Feature")

    def test_05_03b_test_has_current_measurements_arrayinterface(self):
        x = cellprofiler_core.measurement.Measurements()
        x["Image", "Feature"] = "Value"
        assert x.has_current_measurements("Image", "Feature")

    def test_06_00_00_dont_apply_metadata(self):
        x = cellprofiler_core.measurement.Measurements()
        value = "P12345"
        expected = "pre_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = "pre_post"
        assert x.apply_metadata(pattern) == expected

    def test_06_00_00b_dont_apply_metadata_arrayinterface(self):
        x = cellprofiler_core.measurement.Measurements()
        value = "P12345"
        expected = "pre_post"
        x["Image", "Metadata_Plate"] = value
        pattern = "pre_post"
        assert x.apply_metadata(pattern) == expected

    def test_06_00_01_dont_apply_metadata_with_slash(self):
        x = cellprofiler_core.measurement.Measurements()
        value = "P12345"
        expected = "pre\\post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = "pre\\\\post"
        assert x.apply_metadata(pattern) == expected

    def test_06_01_apply_metadata(self):
        x = cellprofiler_core.measurement.Measurements()
        value = "P12345"
        expected = "pre_" + value + "_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"pre_\g<Plate>_post"
        assert x.apply_metadata(pattern) == expected

    def test_06_02_apply_metadata_with_slash(self):
        x = cellprofiler_core.measurement.Measurements()
        value = "P12345"
        expected = "\\" + value + "_post"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"\\\g<Plate>_post"
        assert x.apply_metadata(pattern) == expected

    def test_06_03_apply_metadata_with_two_slashes(self):
        """Regression test of img-1144"""
        x = cellprofiler_core.measurement.Measurements()
        plate = "P12345"
        well = "A01"
        expected = "\\" + plate + "\\" + well
        x.add_measurement("Image", "Metadata_Plate", plate)
        x.add_measurement("Image", "Metadata_Well", well)
        pattern = r"\\\g<Plate>\\\g<Well>"
        assert x.apply_metadata(pattern) == expected

    def test_06_04_apply_metadata_when_user_messes_with_your_head(self):
        x = cellprofiler_core.measurement.Measurements()
        value = "P12345"
        expected = r"\g<Plate>"
        x.add_measurement("Image", "Metadata_Plate", value)
        pattern = r"\\g<Plate>"
        assert x.apply_metadata(pattern) == expected

    def test_06_05_apply_metadata_twice(self):
        """Regression test of img-1144 (second part)"""
        x = cellprofiler_core.measurement.Measurements()
        plate = "P12345"
        well = "A01"
        expected = plate + "_" + well
        x.add_measurement("Image", "Metadata_Plate", plate)
        x.add_measurement("Image", "Metadata_Well", well)
        pattern = r"\g<Plate>_\g<Well>"
        assert x.apply_metadata(pattern) == expected

    def test_06_06_apply_series_and_frame_metadata(self):
        x = cellprofiler_core.measurement.Measurements()
        x[
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.C_SERIES + "_DNA",
        ] = 1
        x[
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.C_SERIES + "_DNAIllum",
        ] = 0
        x[
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.C_FRAME + "_DNA",
        ] = 2
        x[
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.C_FRAME + "_DNAIllum",
        ] = 0
        pattern = r"\g<%s>_\g<%s>" % (
            cellprofiler_core.constants.measurement.C_SERIES,
            cellprofiler_core.constants.measurement.C_FRAME,
        )
        assert x.apply_metadata(pattern) == "1_2"

    def test_07_01_copy(self):
        x = cellprofiler_core.measurement.Measurements()
        r = numpy.random.RandomState()
        r.seed(71)
        areas = [r.randint(100, 200, size=r.randint(100, 200)) for _ in range(12)]

        for i in range(12):
            x.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Metadata_Well",
                "A%02d" % (i + 1),
                image_set_number=(i + 1),
            )
            x.add_measurement(
                OBJECT_NAME, "AreaShape_Area", areas[i], image_set_number=(i + 1)
            )

        y = cellprofiler_core.measurement.Measurements(copy=x)
        for i in range(12):
            assert y.get_measurement(
                cellprofiler_core.constants.measurement.IMAGE, "Metadata_Well", (i + 1)
            ) == "A%02d" % (i + 1)
            values = y.get_measurement(OBJECT_NAME, "AreaShape_Area", (i + 1))
            numpy.testing.assert_equal(values, areas[i])

    def test_09_01_group_by_metadata(self):
        m = cellprofiler_core.measurement.Measurements()
        r = numpy.random.RandomState()
        r.seed(91)
        aa = [None] * 100
        bb = [None] * 100
        for image_number in r.permutation(numpy.arange(1, 101)):
            a = r.randint(1, 3)
            aa[image_number - 1] = a
            b = "A%02d" % r.randint(1, 12)
            bb[image_number - 1] = b
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Metadata_A",
                a,
                image_set_number=image_number,
            )
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Metadata_B",
                b,
                image_set_number=image_number,
            )
        result = m.group_by_metadata(["A", "B"])
        for d in result:
            for image_number in d.image_numbers:
                assert d["A"] == aa[image_number - 1]
                assert d["B"] == bb[image_number - 1]

    def test_09_02_get_groupings(self):
        m = cellprofiler_core.measurement.Measurements()
        r = numpy.random.RandomState()
        r.seed(91)
        aa = [None] * 100
        bb = [None] * 100
        for image_number in r.permutation(numpy.arange(1, 101)):
            a = r.randint(1, 3)
            aa[image_number - 1] = a
            b = "A%02d" % r.randint(1, 12)
            bb[image_number - 1] = b
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Metadata_A",
                a,
                image_set_number=image_number,
            )
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                "Metadata_B",
                b,
                image_set_number=image_number,
            )
        result = m.get_groupings(["Metadata_A", "Metadata_B"])
        for d, image_numbers in result:
            for image_number in image_numbers:
                assert d["Metadata_A"] == six.text_type(aa[image_number - 1])
                assert d["Metadata_B"] == six.text_type(bb[image_number - 1])

    def test_10_01_remove_image_measurement(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "M",
            "Hello",
            image_set_number=1,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "M",
            "World",
            image_set_number=2,
        )
        m.remove_measurement(cellprofiler_core.constants.measurement.IMAGE, "M", 1)
        assert (
            m.get_measurement(cellprofiler_core.constants.measurement.IMAGE, "M", 1)
            is None
        )
        assert (
            m.get_measurement(cellprofiler_core.constants.measurement.IMAGE, "M", 2)
            == "World"
        )

    def test_10_02_remove_object_measurement(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, "M", numpy.arange(5), image_set_number=1)
        m.add_measurement(OBJECT_NAME, "M", numpy.arange(7), image_set_number=2)
        m.remove_measurement(OBJECT_NAME, "M", 1)
        assert len(m.get_measurement(OBJECT_NAME, "M", 1)) == 0
        numpy.testing.assert_equal(
            m.get_measurement(OBJECT_NAME, "M", 2), numpy.arange(7)
        )

    def test_10_03_remove_image_number(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "M",
            "Hello",
            image_set_number=1,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "M",
            "World",
            image_set_number=2,
        )
        numpy.testing.assert_equal(
            numpy.array(m.get_image_numbers()), numpy.arange(1, 3)
        )
        m.remove_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.IMAGE_NUMBER,
            1,
        )
        numpy.testing.assert_equal(numpy.array(m.get_image_numbers()), numpy.array([2]))

    def test_11_00_match_metadata_by_order_nil(self):
        m = cellprofiler_core.measurement.Measurements()
        result = m.match_metadata(
            ("Metadata_foo", "Metadata_bar"), (numpy.zeros(3), numpy.zeros(3))
        )
        assert len(result) == 3
        assert all([len(x) == 1 and x[0] == i + 1 for i, x in enumerate(result)])

    def test_11_01_match_metadata_by_order(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_foo",
            "Hello",
            image_set_number=1,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_foo",
            "Hello",
            image_set_number=2,
        )
        result = m.match_metadata(("Metadata_bar",), (numpy.zeros(2),))
        assert len(result) == 2
        assert all([len(x) == 1 and x[0] == i + 1 for i, x in enumerate(result)])

    def test_11_02_match_metadata_equal_length(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_foo",
            "Hello",
            image_set_number=1,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_bar",
            "World",
            image_set_number=1,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_foo",
            "Goodbye",
            image_set_number=2,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_bar",
            "Phobos",
            image_set_number=2,
        )
        result = m.match_metadata(
            ("Metadata_foo", "Metadata_bar"),
            (("Goodbye", "Hello"), ("Phobos", "World")),
        )
        assert len(result) == 2
        assert len(result[0]) == 1
        assert result[0][0] == 2
        assert len(result[1]) == 1
        assert result[1][0] == 1

    def test_11_03_match_metadata_different_length(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_foo",
            "Hello",
            image_set_number=1,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_bar",
            "World",
            image_set_number=1,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_foo",
            "Goodbye",
            image_set_number=2,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_bar",
            "Phobos",
            image_set_number=2,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_foo",
            "Hello",
            image_set_number=3,
        )
        m.add_measurement(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_bar",
            "Phobos",
            image_set_number=3,
        )
        result = m.match_metadata(("Metadata_foo",), (("Goodbye", "Hello"),))
        assert len(result) == 2
        assert len(result[0]) == 1
        assert result[0][0] == 2
        assert len(result[1]) == 2
        assert result[1][0] == 1
        assert result[1][1] == 3

    def test_15_01_get_object_names(self):
        """Test the get_object_names() function"""
        m = cellprofiler_core.measurement.Measurements()
        try:
            m.add_measurement(OBJECT_NAME, "Foo", numpy.zeros(3))
            object_names = m.get_object_names()
            assert len(object_names) == 2
            assert OBJECT_NAME in object_names
            assert cellprofiler_core.constants.measurement.IMAGE in object_names
        finally:
            del m

    def test_15_02_get_object_names_relationships(self):
        """Regression test - don't return Relationships"""
        m = cellprofiler_core.measurement.Measurements()
        try:
            m.add_image_measurement(
                cellprofiler_core.constants.measurement.GROUP_NUMBER, 1
            )
            m.add_image_measurement(
                cellprofiler_core.constants.measurement.GROUP_INDEX, 0
            )
            m.add_measurement(OBJECT_NAME, "Foo", numpy.zeros(3))
            m.add_relate_measurement(
                1,
                "Foo",
                OBJECT_NAME,
                OBJECT_NAME,
                numpy.zeros(3),
                numpy.zeros(3),
                numpy.zeros(3),
                numpy.zeros(3),
            )
            object_names = m.get_object_names()
            assert len(object_names) == 2
            assert OBJECT_NAME in object_names
            assert cellprofiler_core.constants.measurement.IMAGE in object_names
        finally:
            del m

    def test_15_03_relate_no_objects(self):
        # regression test of issue #886 - add_relate_measurement called
        #                                 with no objects
        m = cellprofiler_core.measurement.Measurements()
        m.add_image_measurement(cellprofiler_core.constants.measurement.GROUP_NUMBER, 1)
        m.add_image_measurement(cellprofiler_core.constants.measurement.GROUP_INDEX, 0)
        m.add_measurement(OBJECT_NAME, "Foo", numpy.zeros(3))
        m.add_relate_measurement(
            1,
            "Foo",
            OBJECT_NAME,
            OBJECT_NAME,
            numpy.zeros(3),
            numpy.zeros(3),
            numpy.zeros(3),
            numpy.zeros(3),
        )
        m.add_relate_measurement(
            1,
            "Foo",
            OBJECT_NAME,
            OBJECT_NAME,
            numpy.zeros(0),
            numpy.zeros(0),
            numpy.zeros(0),
            numpy.zeros(0),
        )

    def test_16_01_get_feature_names(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            m.add_measurement(OBJECT_NAME, "Foo", numpy.zeros(3))
            feature_names = m.get_feature_names(OBJECT_NAME)
            assert len(feature_names) == 1
            assert "Foo" in feature_names
        finally:
            del m

    def test_16_01b_get_feature_names_arrayinterface(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            m[OBJECT_NAME, "Foo"] = numpy.zeros(3)
            feature_names = m.get_feature_names(OBJECT_NAME)
            assert len(feature_names) == 1
            assert "Foo" in feature_names
        finally:
            del m

    def test_17_01_aggregate_measurements(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            values = numpy.arange(5).astype(float)
            m.add_measurement(OBJECT_NAME, "Foo", values)
            d = m.compute_aggregate_measurements(1)
            for agg_name, expected in (
                (cellprofiler_core.constants.measurement.AGG_MEAN, numpy.mean(values)),
                (
                    cellprofiler_core.constants.measurement.AGG_MEDIAN,
                    numpy.median(values),
                ),
                (
                    cellprofiler_core.constants.measurement.AGG_STD_DEV,
                    numpy.std(values),
                ),
            ):
                feature = "%s_%s_Foo" % (agg_name, OBJECT_NAME)
                assert feature in d
                assert round(abs(d[feature] - expected), 7) == 0
        finally:
            del m

    def test_17_02_aggregate_measurements_with_relate(self):
        """regression test of img-1554"""
        m = cellprofiler_core.measurement.Measurements()
        try:
            values = numpy.arange(5).astype(float)
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.GROUP_NUMBER,
                1,
                image_set_number=1,
            )
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.GROUP_NUMBER,
                1,
                image_set_number=2,
            )
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.GROUP_INDEX,
                1,
                image_set_number=1,
            )
            m.add_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                cellprofiler_core.constants.measurement.GROUP_INDEX,
                2,
                image_set_number=1,
            )
            m.add_measurement(OBJECT_NAME, "Foo", values, image_set_number=1)
            m.add_measurement(OBJECT_NAME, "Foo", values, image_set_number=2)
            m.add_relate_measurement(
                1,
                "R",
                "A1",
                "A2",
                numpy.array([1, 1, 1, 1, 1], int),
                numpy.array([1, 2, 3, 4, 5], int),
                numpy.array([2, 2, 2, 2, 2], int),
                numpy.array([5, 4, 3, 2, 1], int),
            )
            d = m.compute_aggregate_measurements(1)
            for agg_name, expected in (
                (cellprofiler_core.constants.measurement.AGG_MEAN, numpy.mean(values)),
                (
                    cellprofiler_core.constants.measurement.AGG_MEDIAN,
                    numpy.median(values),
                ),
                (
                    cellprofiler_core.constants.measurement.AGG_STD_DEV,
                    numpy.std(values),
                ),
            ):
                feature = "%s_%s_Foo" % (agg_name, OBJECT_NAME)
                assert feature in d
                assert round(abs(d[feature] - expected), 7) == 0
        finally:
            del m

    def test_18_01_test_add_all_measurements_string(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            values = ["Foo", "Bar", "Baz"]
            m.add_all_measurements(
                cellprofiler_core.constants.measurement.IMAGE, FEATURE_NAME, values
            )
            for i, expected in enumerate(values):
                value = m.get_measurement(
                    cellprofiler_core.constants.measurement.IMAGE,
                    FEATURE_NAME,
                    image_set_number=i + 1,
                )
                assert expected == value
        finally:
            del m

    def test_18_02_test_add_all_measurements_unicode(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            values = ["Foo", "Bar", "Baz", "-\\u221E < \\u221E"]
            m.add_all_measurements(
                cellprofiler_core.constants.measurement.IMAGE, FEATURE_NAME, values
            )
            for i, expected in enumerate(values):
                value = m.get_measurement(
                    cellprofiler_core.constants.measurement.IMAGE,
                    FEATURE_NAME,
                    image_set_number=i + 1,
                )
                assert expected == value
        finally:
            del m

    def test_18_03_test_add_all_measurements_number(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            r = numpy.random.RandomState()
            r.seed(1803)
            values = r.randint(0, 10, size=5)
            m.add_all_measurements(
                cellprofiler_core.constants.measurement.IMAGE, FEATURE_NAME, values
            )
            for i, expected in enumerate(values):
                value = m.get_measurement(
                    cellprofiler_core.constants.measurement.IMAGE,
                    FEATURE_NAME,
                    image_set_number=i + 1,
                )
                assert expected == value
        finally:
            del m

    def test_18_04_test_add_all_measurements_nulls(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            values = ["Foo", "Bar", None, "Baz", None, "-\\u221E < \\u221E"]
            m.add_all_measurements(
                cellprofiler_core.constants.measurement.IMAGE, FEATURE_NAME, values
            )
            for i, expected in enumerate(values):
                value = m.get_measurement(
                    cellprofiler_core.constants.measurement.IMAGE,
                    FEATURE_NAME,
                    image_set_number=i + 1,
                )
                if expected is None:
                    assert value is None
                else:
                    assert expected == value
        finally:
            del m

    def test_18_05_test_add_all_per_object_measurements(self):
        m = cellprofiler_core.measurement.Measurements()
        try:
            r = numpy.random.RandomState()
            r.seed(1803)
            values = [
                r.uniform(size=5),
                numpy.zeros(0),
                r.uniform(size=7),
                numpy.zeros(0),
                r.uniform(size=9),
                None,
                r.uniform(size=10),
            ]
            m.add_all_measurements(OBJECT_NAME, FEATURE_NAME, values)
            for i, expected in enumerate(values):
                value = m.get_measurement(
                    OBJECT_NAME, FEATURE_NAME, image_set_number=i + 1
                )
                if expected is None:
                    assert len(value) == 0
                else:
                    numpy.testing.assert_almost_equal(expected, value)
        finally:
            del m

    def test_19_01_load_image_sets(self):
        expected_features = [
            cellprofiler_core.constants.measurement.GROUP_NUMBER,
            cellprofiler_core.constants.measurement.GROUP_INDEX,
            "URL_DNA",
            "PathName_DNA",
            "FileName_DNA",
        ]
        expected_values = [
            [1, 1, "file://foo/bar.tif", "/foo", "bar.tif"],
            [1, 2, "file://bar/foo.tif", "/bar", "foo.tif"],
            [2, 1, "file://baz/foobar.tif", "/baz", "foobar.tif"],
        ]

        data = """"%s","%s","URL_DNA","PathName_DNA","FileName_DNA"
1,1,"file://foo/bar.tif","/foo","bar.tif"
1,2,"file://bar/foo.tif","/bar","foo.tif"
2,1,"file://baz/foobar.tif","/baz","foobar.tif"
""" % (
            cellprofiler_core.constants.measurement.GROUP_NUMBER,
            cellprofiler_core.constants.measurement.GROUP_INDEX,
        )
        m = cellprofiler_core.measurement.Measurements()
        try:
            m.load_image_sets(six.moves.StringIO(data))
            features = m.get_feature_names(
                cellprofiler_core.constants.measurement.IMAGE
            )

            unittest.TestCase().assertCountEqual(features, expected_features)

            for i, row_values in enumerate(expected_values):
                image_number = i + 1
                for value, feature_name in zip(row_values, expected_features):
                    assert value == m.get_measurement(
                        cellprofiler_core.constants.measurement.IMAGE,
                        feature_name,
                        image_set_number=image_number,
                    )
        finally:
            m.close()

    # FIXME: wxPython 4 PR
    def test_19_02_write_and_load_image_sets(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.GROUP_NUMBER,
            [1, 1, 2],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            cellprofiler_core.constants.measurement.GROUP_INDEX,
            [1, 2, 1],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            "URL_DNA",
            ["file://foo/bar.tif", "file://bar/foo.tif", "file://baz/foobar.tif"],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            "PathName_DNA",
            ["/foo", "/bar", "/baz"],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            "FileName_DNA",
            ["bar.tif", "foo.tif", "foobar.tif"],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_test",
            ['quotetest"', "backslashtest\\", "unicodeescapetest\\u0384"],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_testunicode",
            ['quotetest"', "backslashtest\\", "unicodeescapetest\u0384"],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            "Metadata_testnull",
            ["Something", None, "SomethingElse"],
        )
        m.add_all_measurements(
            cellprofiler_core.constants.measurement.IMAGE,
            "Dont_copy",
            ["do", "not", "copy"],
        )
        fd = six.moves.StringIO()
        m.write_image_sets(fd)
        fd.seek(0)
        mdest = cellprofiler_core.measurement.Measurements()
        mdest.load_image_sets(fd)
        expected_features = [
            feature_name
            for feature_name in m.get_feature_names(
                cellprofiler_core.constants.measurement.IMAGE
            )
            if feature_name != "Dont_copy"
        ]
        unittest.TestCase().assertCountEqual(
            expected_features,
            mdest.get_feature_names(cellprofiler_core.constants.measurement.IMAGE),
        )
        image_numbers = m.get_image_numbers()
        for feature_name in expected_features:
            src = m.get_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                feature_name,
                image_numbers,
            )
            dest = mdest.get_measurement(
                cellprofiler_core.constants.measurement.IMAGE,
                feature_name,
                image_numbers,
            )
            unittest.TestCase().assertCountEqual(list(src), list(dest))

    def test_19_03_delete_tempfile(self):
        m = cellprofiler_core.measurement.Measurements()
        filename = m.hdf5_dict.filename
        del m
        assert not os.path.exists(filename)

    def test_19_04_dont_delete_file(self):
        fd, filename = tempfile.mkstemp(suffix=".h5")
        m = cellprofiler_core.measurement.Measurements(filename=filename)
        os.close(fd)
        del m
        assert os.path.exists(filename)
        os.unlink(filename)

    def test_20_01_add_one_relationship_measurement(self):
        m = cellprofiler_core.measurement.Measurements()
        r = numpy.random.RandomState()
        r.seed(2001)
        image_numbers1, object_numbers1 = [x.flatten() for x in numpy.mgrid[1:4, 1:10]]
        order = r.permutation(len(image_numbers1))
        image_numbers2, object_numbers2 = [
            x[order] for x in (image_numbers1, object_numbers1)
        ]

        m.add_relate_measurement(
            1,
            "Foo",
            "O1",
            "O2",
            image_numbers1,
            object_numbers1,
            image_numbers2,
            object_numbers2,
        )
        rg = m.get_relationship_groups()
        assert len(rg) == 1
        assert rg[0].module_number == 1
        assert rg[0].relationship == "Foo"
        assert rg[0].object_name1 == "O1"
        assert rg[0].object_name2 == "O2"
        r = m.get_relationships(1, "Foo", "O1", "O2")
        ri1, ro1, ri2, ro2 = [
            r[key]
            for key in (
                cellprofiler_core.constants.measurement.R_FIRST_IMAGE_NUMBER,
                cellprofiler_core.constants.measurement.R_FIRST_OBJECT_NUMBER,
                cellprofiler_core.constants.measurement.R_SECOND_IMAGE_NUMBER,
                cellprofiler_core.constants.measurement.R_SECOND_OBJECT_NUMBER,
            )
        ]
        order = numpy.lexsort((ro1, ri1))
        numpy.testing.assert_array_equal(image_numbers1, ri1[order])
        numpy.testing.assert_array_equal(object_numbers1, ro1[order])
        numpy.testing.assert_array_equal(image_numbers2, ri2[order])
        numpy.testing.assert_array_equal(object_numbers2, ro2[order])

    def test_20_02_add_two_sets_of_relationships(self):
        m = cellprofiler_core.measurement.Measurements()
        r = numpy.random.RandomState()
        r.seed(2002)
        image_numbers1, object_numbers1 = [x.flatten() for x in numpy.mgrid[1:4, 1:10]]
        order = r.permutation(len(image_numbers1))
        image_numbers2, object_numbers2 = [
            x[order] for x in (image_numbers1, object_numbers1)
        ]

        split = int(len(image_numbers1) / 2)
        m.add_relate_measurement(
            1,
            "Foo",
            "O1",
            "O2",
            image_numbers1[:split],
            object_numbers1[:split],
            image_numbers2[:split],
            object_numbers2[:split],
        )
        m.add_relate_measurement(
            1,
            "Foo",
            "O1",
            "O2",
            image_numbers1[split:],
            object_numbers1[split:],
            image_numbers2[split:],
            object_numbers2[split:],
        )
        r = m.get_relationships(1, "Foo", "O1", "O2")
        ri1, ro1, ri2, ro2 = [
            r[key]
            for key in (
                cellprofiler_core.constants.measurement.R_FIRST_IMAGE_NUMBER,
                cellprofiler_core.constants.measurement.R_FIRST_OBJECT_NUMBER,
                cellprofiler_core.constants.measurement.R_SECOND_IMAGE_NUMBER,
                cellprofiler_core.constants.measurement.R_SECOND_OBJECT_NUMBER,
            )
        ]
        order = numpy.lexsort((ro1, ri1))
        numpy.testing.assert_array_equal(image_numbers1, ri1[order])
        numpy.testing.assert_array_equal(object_numbers1, ro1[order])
        numpy.testing.assert_array_equal(image_numbers2, ri2[order])
        numpy.testing.assert_array_equal(object_numbers2, ro2[order])

    # def test_20_03_add_many_different_relationships(self):
    #     m = cpmeas.Measurements()
    #     r = np.random.RandomState()
    #     r.seed(2003)
    #     image_numbers1, object_numbers1 = [
    #         x.flatten() for x in np.mgrid[1:4, 1:10]]
    #     module_numbers = [1, 2]
    #     relationship_names = ["Foo", "Bar"]
    #     first_object_names = ["Nuclei", "Cells"]
    #     second_object_names = ["Alice", "Bob"]
    #     d = {}
    #     midxs, ridxs, on1idxs, on2idxs = [
    #         x.flatten() for x in np.mgrid[0:2, 0:2, 0:2, 0:2]]
    #     for midx, ridx, on1idx, on2idx in zip(midxs, ridxs, on1idxs, on2idxs):
    #         key = (module_numbers[midx], relationship_names[ridx],
    #                first_object_names[on1idx], second_object_names[on2idx])
    #         order = r.permutation(len(image_numbers1))
    #         image_numbers2, object_numbers2 = [
    #             x[order] for x in image_numbers1, object_numbers1]
    #         d[key] = (image_numbers2, object_numbers2)
    #         m.add_relate_measurement(key[0], key[1], key[2], key[3],
    #                                  image_numbers1, object_numbers1,
    #                                  image_numbers2, object_numbers2)
    #
    #     rg = [(x.module_number, x.relationship, x.object_name1, x.object_name2)
    #           for x in m.get_relationship_groups()]
    #     self.assertCountEqual(d.keys(), rg)
    #
    #     for key in d:
    #         image_numbers2, object_numbers2 = d[key]
    #         r = m.get_relationships(key[0], key[1], key[2], key[3])
    #         ri1, ro1, ri2, ro2 = [
    #             r[key] for key in
    #             cpmeas.R_FIRST_IMAGE_NUMBER, cpmeas.R_FIRST_OBJECT_NUMBER,
    #             cpmeas.R_SECOND_IMAGE_NUMBER, cpmeas.R_SECOND_OBJECT_NUMBER]
    #         order = np.lexsort((ro1, ri1))
    #         np.testing.assert_array_equal(image_numbers1, ri1[order])
    #         np.testing.assert_array_equal(object_numbers1, ro1[order])
    #         np.testing.assert_array_equal(image_numbers2, ri2[order])
    #         np.testing.assert_array_equal(object_numbers2, ro2[order])

    def test_20_04_saved_relationships(self):
        #
        # Test closing and reopening a measurements file with
        # relationships.
        #
        fd, filename = tempfile.mkstemp(suffix=".h5")
        m = cellprofiler_core.measurement.Measurements(filename=filename)
        os.close(fd)
        try:
            r = numpy.random.RandomState()
            r.seed(2004)
            image_numbers1, object_numbers1 = [
                x.flatten() for x in numpy.mgrid[1:4, 1:10]
            ]
            module_numbers = [1, 2]
            relationship_names = ["Foo", "Bar"]
            first_object_names = ["Nuclei", "Cells"]
            second_object_names = ["Alice", "Bob"]
            d = {}
            midxs, ridxs, on1idxs, on2idxs = [
                x.flatten() for x in numpy.mgrid[0:2, 0:2, 0:2, 0:2]
            ]
            for midx, ridx, on1idx, on2idx in zip(midxs, ridxs, on1idxs, on2idxs):
                key = (
                    module_numbers[midx],
                    relationship_names[ridx],
                    first_object_names[on1idx],
                    second_object_names[on2idx],
                )
                order = r.permutation(len(image_numbers1))
                image_numbers2, object_numbers2 = [
                    x[order] for x in (image_numbers1, object_numbers1)
                ]
                d[key] = (image_numbers2, object_numbers2)
                m.add_relate_measurement(
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    image_numbers1,
                    object_numbers1,
                    image_numbers2,
                    object_numbers2,
                )

            m.close()
            m = cellprofiler_core.measurement.Measurements(filename=filename, mode="r")

            rg = [
                (x.module_number, x.relationship, x.object_name1, x.object_name2)
                for x in m.get_relationship_groups()
            ]

            unittest.TestCase().assertCountEqual(list(d.keys()), rg)

            for key in d:
                image_numbers2, object_numbers2 = d[key]
                r = m.get_relationships(key[0], key[1], key[2], key[3])
                ri1, ro1, ri2, ro2 = [
                    r[key]
                    for key in (
                        cellprofiler_core.constants.measurement.R_FIRST_IMAGE_NUMBER,
                        cellprofiler_core.constants.measurement.R_FIRST_OBJECT_NUMBER,
                        cellprofiler_core.constants.measurement.R_SECOND_IMAGE_NUMBER,
                        cellprofiler_core.constants.measurement.R_SECOND_OBJECT_NUMBER,
                    )
                ]
                order = numpy.lexsort((ro1, ri1))
                numpy.testing.assert_array_equal(image_numbers1, ri1[order])
                numpy.testing.assert_array_equal(object_numbers1, ro1[order])
                numpy.testing.assert_array_equal(image_numbers2, ri2[order])
                numpy.testing.assert_array_equal(object_numbers2, ro2[order])
        finally:
            m.close()
            os.unlink(filename)

    def test_20_05_copy_relationships(self):
        m1 = cellprofiler_core.measurement.Measurements()
        m2 = cellprofiler_core.measurement.Measurements()
        r = numpy.random.RandomState()
        r.seed(2005)
        image_numbers1, object_numbers1 = [x.flatten() for x in numpy.mgrid[1:4, 1:10]]
        image_numbers2 = r.permutation(image_numbers1)
        object_numbers2 = r.permutation(object_numbers1)
        m1.add_relate_measurement(
            1,
            "Foo",
            "O1",
            "O2",
            image_numbers1,
            object_numbers1,
            image_numbers2,
            object_numbers2,
        )
        m2.copy_relationships(m1)
        rg = m2.get_relationship_groups()
        assert len(rg) == 1
        assert rg[0].module_number == 1
        assert rg[0].relationship == "Foo"
        assert rg[0].object_name1 == "O1"
        assert rg[0].object_name2 == "O2"
        r = m2.get_relationships(1, "Foo", "O1", "O2")
        ri1, ro1, ri2, ro2 = [
            r[key]
            for key in (
                cellprofiler_core.constants.measurement.R_FIRST_IMAGE_NUMBER,
                cellprofiler_core.constants.measurement.R_FIRST_OBJECT_NUMBER,
                cellprofiler_core.constants.measurement.R_SECOND_IMAGE_NUMBER,
                cellprofiler_core.constants.measurement.R_SECOND_OBJECT_NUMBER,
            )
        ]
        order = numpy.lexsort((ro1, ri1))
        numpy.testing.assert_array_equal(image_numbers1, ri1[order])
        numpy.testing.assert_array_equal(object_numbers1, ro1[order])
        numpy.testing.assert_array_equal(image_numbers2, ri2[order])
        numpy.testing.assert_array_equal(object_numbers2, ro2[order])

    def test_20_06_get_relationship_range(self):
        #
        # Test writing and reading relationships with a variety of ranges
        # over the whole extent of the storage
        #
        m = cellprofiler_core.measurement.Measurements()
        r = numpy.random.RandomState()
        r.seed(2005)
        image_numbers1, image_numbers2 = r.randint(1, 1001, (2, 4000))
        object_numbers1, object_numbers2 = r.randint(1, 10, (2, 4000))
        for i in range(0, 4000, 500):
            m.add_relate_measurement(
                1,
                "Foo",
                "O1",
                "O2",
                *[
                    x[i : (i + 500)]
                    for x in (
                        image_numbers1,
                        object_numbers1,
                        image_numbers2,
                        object_numbers2,
                    )
                ],
            )

        for _ in range(50):
            image_numbers = r.randint(1, 1001, 3)
            result = m.get_relationships(1, "Foo", "O1", "O2", image_numbers)
            ri1, ro1, ri2, ro2 = [
                result[key]
                for key in (
                    cellprofiler_core.constants.measurement.R_FIRST_IMAGE_NUMBER,
                    cellprofiler_core.constants.measurement.R_FIRST_OBJECT_NUMBER,
                    cellprofiler_core.constants.measurement.R_SECOND_IMAGE_NUMBER,
                    cellprofiler_core.constants.measurement.R_SECOND_OBJECT_NUMBER,
                )
            ]
            rorder = numpy.lexsort((ro2, ri2, ro1, ri1))
            i, j = [x.flatten() for x in numpy.mgrid[0:2, 0:3]]
            mask = functools.reduce(
                numpy.logical_or,
                [
                    (image_numbers1 if ii == 0 else image_numbers2) == image_numbers[jj]
                    for ii, jj in zip(i, j)
                ],
            )
            ei1, eo1, ei2, eo2 = [
                x[mask]
                for x in (
                    image_numbers1,
                    object_numbers1,
                    image_numbers2,
                    object_numbers2,
                )
            ]
            eorder = numpy.lexsort((eo2, ei2, eo1, ei1))
            numpy.testing.assert_array_equal(ri1[rorder], ei1[eorder])
            numpy.testing.assert_array_equal(ri2[rorder], ei2[eorder])
            numpy.testing.assert_array_equal(ro1[rorder], eo1[eorder])
            numpy.testing.assert_array_equal(ro2[rorder], eo2[eorder])
