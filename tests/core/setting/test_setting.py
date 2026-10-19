import unittest

import pytest

from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.filter import Filter, FilterPredicate
from cellprofiler_core.setting.filter._filter import LITERAL_PREDICATE, OR_PREDICATE
from cellprofiler_core.setting.range import (
    IntegerOrUnboundedRange,
    FloatRange,
    IntegerRange,
)
from cellprofiler_core.setting.text import Float, Integer


class TestIntegerSetting(unittest.TestCase):
    def test_01_01_default(self):
        s = Integer("foo", value=5)
        assert s == 5
        assert s.value_text == "5"
        s.test_valid(None)

    def test_01_02_set_value(self):
        s = Integer("foo", value=5)
        for test_case in ("06", "-1"):
            s.value_text = test_case
            assert s == int(test_case)
            assert s.value_text == test_case
            s.test_valid(None)

    def test_01_03_set_bad(self):
        s = Integer("foo", value=5)
        s.value_text = "bad"
        assert s == 5
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_02_01_good_min(self):
        s = Integer("foo", value=5, minval=0)
        for test_case in ("0", "1"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = Integer("foo", value=5, minval=0)
        s.value_text = "-1"
        assert s == 5
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_02_03_good_max(self):
        s = Integer("foo", value=5, maxval=10)
        for test_case in ("9", "10"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = Integer("foo", value=5, maxval=10)
        s.value_text = "11"
        assert s.value == 5
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_03_01_no_metadata(self):
        with self.assertRaises(ValueError):
            Integer("foo", value=5, maxval=10, metadata=True)


class TestFloatSetting(unittest.TestCase):
    def test_01_01_default(self):
        for value in (5, "5.0"):
            s = Float("foo", value=value)
            assert s == 5
            assert s.value_text == "5.0"
            s.test_valid(None)

    def test_01_02_set_value(self):
        s = Float("foo", value=5)
        for test_case in ("6.00", "-1.75"):
            s.value_text = test_case
            assert s == float(test_case)
            assert s.value_text == test_case
            s.test_valid(None)

    def test_01_03_set_bad(self):
        s = Float("foo", value=5)
        s.value_text = "bad"
        assert s == 5
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_01_04_set_text_value(self):
        s = Float("foo", value=5)
        s.value = "6.00"
        assert s == 6
        assert s.value_text == "6.00"

    def test_02_01_good_min(self):
        s = Float("foo", value=5, minval=0)
        for test_case in ("0", "1"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = Float("foo", value=5, minval=0)
        s.value_text = "-1"
        assert s == 5
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_02_03_good_max(self):
        s = Float("foo", value=5, maxval=10)
        for test_case in ("9", "10"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = Float("foo", value=5, maxval=10)
        s.value_text = "11"
        assert s == 5
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()


class TestIntegerRange(unittest.TestCase):
    def test_01_01_default(self):
        s = IntegerRange("foo", (1, 15))
        assert s.min == 1
        assert s.max == 15
        assert s.min_text == "1"
        assert s.max_text == "15"
        s.test_valid(None)

    def test_01_02_set_min(self):
        s = IntegerRange("foo", (1, 15))
        s.value_text = s.compose_min_text("02")
        assert s.min == 2
        assert s.max == 15
        assert s.min_text == "02"
        assert s.max_text == "15"
        s.test_valid(None)

    def test_01_03_set_min_bad(self):
        s = IntegerRange("foo", (1, 15))
        s.value_text = s.compose_min_text("a2")
        assert s.min == 1
        assert s.max == 15
        assert s.min_text == "a2"
        assert s.max_text == "15"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_01_04_set_max(self):
        s = IntegerRange("foo", (1, 15))
        s.value_text = s.compose_max_text("016")
        assert s.min == 1
        assert s.max == 16
        assert s.min_text == "1"
        assert s.max_text == "016"
        s.test_valid(None)

    def test_01_05_set_max_bad(self):
        s = IntegerRange("foo", (1, 15))
        s.value_text = s.compose_max_text("a2")
        assert s.min == 1
        assert s.max == 15
        assert s.min_text == "1"
        assert s.max_text == "a2"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_02_01_good_min(self):
        s = IntegerRange("foo", (1, 15), minval=0)
        for test_case in ("2", "0"):
            s.value_text = s.compose_min_text(test_case)
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = IntegerRange("foo", (1, 15), minval=0)
        s.value_text = s.compose_min_text("-1")
        assert s.min == 0
        assert s.min_text == "-1"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_02_03_good_max(self):
        s = IntegerRange("foo", (1, 15), maxval=20)
        for test_case in ("18", "20"):
            s.value_text = s.compose_max_text(test_case)
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = IntegerRange("foo", (1, 15), maxval=20)
        s.value_text = s.compose_max_text("21")
        assert s.max == 20
        assert s.max_text == "21"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()


class TestFloatRange(unittest.TestCase):
    def test_01_01_default(self):
        s = FloatRange("foo", (1, 15))
        assert s.min == 1
        assert s.max == 15
        assert s.min_text == "1.0"
        assert s.max_text == "15.0"
        s.test_valid(None)

    def test_01_02_set_min(self):
        s = FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("2.10")
        assert s.min == 2.1
        assert s.max == 15
        assert s.min_text == "2.10"
        assert s.max_text == "15.0"
        s.test_valid(None)

    def test_01_03_set_min_bad(self):
        s = FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("a2")
        assert s.min == 1
        assert s.max == 15
        assert s.min_text == "a2"
        assert s.max_text == "15.0"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_01_04_set_max(self):
        s = FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("016")
        assert s.min == 1
        assert s.max == 16
        assert s.min_text == "1.0"
        assert s.max_text == "016"
        s.test_valid(None)

    def test_01_05_set_max_bad(self):
        s = FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("a2")
        assert s.min == 1
        assert s.max == 15
        assert s.min_text == "1.0"
        assert s.max_text == "a2"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_02_01_good_min(self):
        s = FloatRange("foo", (1, 15), minval=0)
        for test_case in ("2", "0"):
            s.value_text = s.compose_min_text(test_case)
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = FloatRange("foo", (1, 15), minval=0)
        s.value_text = s.compose_min_text("-1")
        assert s.min == 0
        assert s.min_text == "-1"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()

    def test_02_03_good_max(self):
        s = FloatRange("foo", (1, 15), maxval=20)
        for test_case in ("18", "20.00"):
            s.value_text = s.compose_max_text(test_case)
            assert s.max == float(test_case)
            assert s.max_text == test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = FloatRange("foo", (1, 15), maxval=20)
        s.value_text = s.compose_max_text("21")
        assert s.max == 20
        assert s.max_text == "21"
        with pytest.raises(ValidationError):
            (lambda: s.test_valid(None))()


class TestIntegerOrUnboundedRange(unittest.TestCase):
    def test_01_01_default(self):
        for (
            minval,
            maxval,
            expected_min,
            expected_min_text,
            expected_max,
            expected_max_text,
            expected_unbounded_max,
            expected_abs,
        ) in (
            (0, "end", 0, "0", "end", "end", True, True),
            ("begin", 15, 0, "begin", 15, "15", False, True),
            (0, -15, 0, "0", -15, "15", False, False),
            (0, "-" + "end", 0, "0", "end", "end", True, False),
        ):
            s = IntegerOrUnboundedRange("foo", (minval, maxval))
            assert s.min == expected_min
            assert s.max == expected_max
            assert s.display_min == expected_min_text
            assert s.display_max == expected_max_text
            assert s.is_abs() == expected_abs
            s.test_valid(None)

    def test_01_02_set_min(self):
        s = IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_min_text("01")
        assert s.min == 1
        assert s.display_min == "01"

    def test_01_03_set_max(self):
        s = IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_max_text("015")
        assert s.max == 15
        assert s.display_max == "015"

    def test_01_04_set_end(self):
        s = IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_max_text("end")
        assert s.max == "end"

    def test_01_05_set_abs(self):
        s = IntegerOrUnboundedRange("foo", (0, -15))
        assert not s.is_abs()
        s.value_text = s.compose_abs()
        assert s.is_abs()
        assert s.max == 15

    def test_01_06_set_abs_end(self):
        s = IntegerOrUnboundedRange("foo", (0, "-" + "end"))
        assert not s.is_abs()
        s.value_text = s.compose_abs()
        assert s.is_abs()
        assert s.max == "end"

    def test_01_07_set_rel(self):
        s = IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_rel()
        assert not s.is_abs()
        assert s.max == -15

    def test_01_06_set_rel_end(self):
        s = IntegerOrUnboundedRange("foo", (0, "end"))
        s.value_text = s.compose_rel()
        assert not s.is_abs()
        assert s.max == "end"


class TestFilterSetting(unittest.TestCase):
    def test_01_01_simple(self):
        filters = [FilterPredicate("foo", "Foo", lambda a: a == "x", [])]
        f = Filter("", filters, "foo")
        assert f.evaluate("x")
        assert not f.evaluate("y")

    def test_01_02_compound(self):
        f2 = FilterPredicate("bar", "Bar", lambda: "y", [])
        f1 = FilterPredicate("foo", "Foo", lambda a, b: a == b(), [f2])
        f = Filter("", [f1], "foo bar")
        assert not f.evaluate("x")
        assert f.evaluate("y")

    def test_01_03_literal(self):
        f1 = FilterPredicate("foo", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)
        f = Filter("", [f1], 'foo "x"')
        assert f.evaluate("x")
        assert not f.evaluate("y")
        f = Filter("", [f1], 'foo "y"')
        assert f.evaluate("y")
        assert not f.evaluate("x")

    def test_01_04_escaped_literal(self):
        f1 = FilterPredicate("foo", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)
        f = Filter("", [f1], 'foo "\\\\"')
        assert f.evaluate("\\")
        assert not f.evaluate("/")

    def test_01_05_literal_with_quote(self):
        f1 = FilterPredicate("foo", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)
        f = Filter("", [f1], 'foo "\\""')
        assert f.evaluate('"')
        assert not f.evaluate("/")

    def test_01_06_parentheses(self):
        f1 = FilterPredicate("eq", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)
        f2 = FilterPredicate("ne", "Bar", lambda a, b: a != b, [LITERAL_PREDICATE],)
        f = Filter("", [f1, f2], 'and (eq "x") (ne "y")')
        assert f.evaluate("x")
        assert not f.evaluate("y")
        assert not f.evaluate("z")

    def test_01_07_or(self):
        f1 = FilterPredicate("eq", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)

        f = Filter("", [f1], 'or (eq "x") (eq "y")')
        assert f.evaluate("x")
        assert f.evaluate("y")
        assert not f.evaluate("z")

    def test_02_01_build_one(self):
        f1 = FilterPredicate("foo", "Foo", lambda a: a == "foo", [])
        f = Filter("", [f1])
        f.build([f1])
        assert f.value == "foo"

    def test_02_02_build_literal(self):
        f1 = FilterPredicate("foo", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)
        f = Filter("", [f1])
        f.build([f1, "bar"])
        assert f.value == 'foo "bar"'

    def test_02_03_build_nested(self):
        f1 = FilterPredicate("foo", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)
        f = Filter("", [f1])
        f.build([OR_PREDICATE, [f1, "bar"], [f1, "baz"]])
        assert f.value == 'or (foo "bar") (foo "baz")'

    def test_02_04_build_escaped_literal(self):
        f1 = FilterPredicate("foo", "Foo", lambda a, b: a == b, [LITERAL_PREDICATE],)
        f = Filter("", [f1])
        f.build([f1, '"12\\'])
        assert f.value == 'foo "\\"12\\\\"'
        tokens = f.parse()
        assert tokens[1] == '"12\\'

    def test_02_05_build_escaped_symbol(self):
        ugly = '(\\")'
        expected = '\\(\\\\\\"\\)'
        f1 = FilterPredicate(ugly, "Foo", lambda a, b: a == b, [])
        f = Filter("", [f1])
        f.build([f1])
        assert f.value == '\\(\\\\"\\)'

    def test_02_06_parse_escaped_symbol(self):
        ugly = '(\\")'
        encoded_ugly = '\\(\\\\\\"\\)'
        f1 = FilterPredicate(ugly, "Foo", lambda a, b: a == b, [])
        f = Filter("", [f1], encoded_ugly)
        result = f.parse()
        assert len(result) == 1
        assert result[0].symbol == ugly
