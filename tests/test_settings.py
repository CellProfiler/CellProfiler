import unittest

import nucleus.setting


class TestIntegerSetting(unittest.TestCase):
    def test_01_01_default(self):
        s = nucleus.setting.Integer("foo", value=5)
        self.assertEqual(s, 5)
        self.assertEqual(s.value_text, "5")
        s.test_valid(None)

    def test_01_02_set_value(self):
        s = nucleus.setting.Integer("foo", value=5)
        for test_case in ("06", "-1"):
            s.value_text = test_case
            self.assertEqual(s, int(test_case))
            self.assertEqual(s.value_text, test_case)
            s.test_valid(None)

    def test_01_03_set_bad(self):
        s = nucleus.setting.Integer("foo", value=5)
        s.value_text = "bad"
        self.assertEqual(s, 5)
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_02_01_good_min(self):
        s = nucleus.setting.Integer("foo", value=5, minval=0)
        for test_case in ("0", "1"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = nucleus.setting.Integer("foo", value=5, minval=0)
        s.value_text = "-1"
        self.assertEqual(s, 5)
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_02_03_good_max(self):
        s = nucleus.setting.Integer("foo", value=5, maxval=10)
        for test_case in ("9", "10"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = nucleus.setting.Integer("foo", value=5, maxval=10)
        s.value_text = "11"
        self.assertEqual(s.value, 5)
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )


class TestFloatSetting(unittest.TestCase):
    def test_01_01_default(self):
        for value in (5, "5.0"):
            s = nucleus.setting.Float("foo", value=value)
            self.assertEqual(s, 5)
            self.assertEqual(s.value_text, "5.0")
            s.test_valid(None)

    def test_01_02_set_value(self):
        s = nucleus.setting.Float("foo", value=5)
        for test_case in ("6.00", "-1.75"):
            s.value_text = test_case
            self.assertEqual(s, float(test_case))
            self.assertEqual(s.value_text, test_case)
            s.test_valid(None)

    def test_01_03_set_bad(self):
        s = nucleus.setting.Float("foo", value=5)
        s.value_text = "bad"
        self.assertEqual(s, 5)
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_01_04_set_text_value(self):
        s = nucleus.setting.Float("foo", value=5)
        s.value = "6.00"
        self.assertEqual(s, 6)
        self.assertEqual(s.value_text, "6.00")

    def test_02_01_good_min(self):
        s = nucleus.setting.Float("foo", value=5, minval=0)
        for test_case in ("0", "1"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = nucleus.setting.Float("foo", value=5, minval=0)
        s.value_text = "-1"
        self.assertEqual(s, 5)
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_02_03_good_max(self):
        s = nucleus.setting.Float("foo", value=5, maxval=10)
        for test_case in ("9", "10"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = nucleus.setting.Float("foo", value=5, maxval=10)
        s.value_text = "11"
        self.assertEqual(s, 5)
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )


class TestIntegerRange(unittest.TestCase):
    def test_01_01_default(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15))
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "1")
        self.assertEqual(s.max_text, "15")
        s.test_valid(None)

    def test_01_02_set_min(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_min_text("02")
        self.assertEqual(s.min, 2)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "02")
        self.assertEqual(s.max_text, "15")
        s.test_valid(None)

    def test_01_03_set_min_bad(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_min_text("a2")
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "a2")
        self.assertEqual(s.max_text, "15")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_01_04_set_max(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_max_text("016")
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 16)
        self.assertEqual(s.min_text, "1")
        self.assertEqual(s.max_text, "016")
        s.test_valid(None)

    def test_01_05_set_max_bad(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_max_text("a2")
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "1")
        self.assertEqual(s.max_text, "a2")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_02_01_good_min(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15), minval=0)
        for test_case in ("2", "0"):
            s.value_text = s.compose_min_text(test_case)
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15), minval=0)
        s.value_text = s.compose_min_text("-1")
        self.assertEqual(s.min, 0)
        self.assertEqual(s.min_text, "-1")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_02_03_good_max(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15), maxval=20)
        for test_case in ("18", "20"):
            s.value_text = s.compose_max_text(test_case)
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = nucleus.setting.IntegerRange("foo", (1, 15), maxval=20)
        s.value_text = s.compose_max_text("21")
        self.assertEqual(s.max, 20)
        self.assertEqual(s.max_text, "21")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )


class TestFloatRange(unittest.TestCase):
    def test_01_01_default(self):
        s = nucleus.setting.FloatRange("foo", (1, 15))
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "1.0")
        self.assertEqual(s.max_text, "15.0")
        s.test_valid(None)

    def test_01_02_set_min(self):
        s = nucleus.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("2.10")
        self.assertEqual(s.min, 2.1)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "2.10")
        self.assertEqual(s.max_text, "15.0")
        s.test_valid(None)

    def test_01_03_set_min_bad(self):
        s = nucleus.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("a2")
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "a2")
        self.assertEqual(s.max_text, "15.0")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_01_04_set_max(self):
        s = nucleus.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("016")
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 16)
        self.assertEqual(s.min_text, "1.0")
        self.assertEqual(s.max_text, "016")
        s.test_valid(None)

    def test_01_05_set_max_bad(self):
        s = nucleus.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("a2")
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 15)
        self.assertEqual(s.min_text, "1.0")
        self.assertEqual(s.max_text, "a2")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_02_01_good_min(self):
        s = nucleus.setting.FloatRange("foo", (1, 15), minval=0)
        for test_case in ("2", "0"):
            s.value_text = s.compose_min_text(test_case)
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = nucleus.setting.FloatRange("foo", (1, 15), minval=0)
        s.value_text = s.compose_min_text("-1")
        self.assertEqual(s.min, 0)
        self.assertEqual(s.min_text, "-1")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )

    def test_02_03_good_max(self):
        s = nucleus.setting.FloatRange("foo", (1, 15), maxval=20)
        for test_case in ("18", "20.00"):
            s.value_text = s.compose_max_text(test_case)
            self.assertEqual(s.max, float(test_case))
            self.assertEqual(s.max_text, test_case)
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = nucleus.setting.FloatRange("foo", (1, 15), maxval=20)
        s.value_text = s.compose_max_text("21")
        self.assertEqual(s.max, 20)
        self.assertEqual(s.max_text, "21")
        self.assertRaises(
            nucleus.setting.ValidationError, (lambda: s.test_valid(None))
        )


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
            (
                0,
                nucleus.setting.END,
                0,
                "0",
                nucleus.setting.END,
                nucleus.setting.END,
                True,
                True,
            ),
            (
                nucleus.setting.BEGIN,
                15,
                0,
                nucleus.setting.BEGIN,
                15,
                "15",
                False,
                True,
            ),
            (0, -15, 0, "0", -15, "15", False, False),
            (
                0,
                "-" + nucleus.setting.END,
                0,
                "0",
                nucleus.setting.END,
                nucleus.setting.END,
                True,
                False,
            ),
        ):
            s = nucleus.setting.IntegerOrUnboundedRange("foo", (minval, maxval))
            self.assertEqual(s.min, expected_min)
            self.assertEqual(s.max, expected_max)
            self.assertEqual(s.display_min, expected_min_text)
            self.assertEqual(s.display_max, expected_max_text)
            self.assertEqual(s.is_abs(), expected_abs)
            s.test_valid(None)

    def test_01_02_set_min(self):
        s = nucleus.setting.IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_min_text("01")
        self.assertEqual(s.min, 1)
        self.assertEqual(s.display_min, "01")

    def test_01_03_set_max(self):
        s = nucleus.setting.IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_max_text("015")
        self.assertEqual(s.max, 15)
        self.assertEqual(s.display_max, "015")

    def test_01_04_set_end(self):
        s = nucleus.setting.IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_max_text(nucleus.setting.END)
        self.assertEqual(s.max, nucleus.setting.END)

    def test_01_05_set_abs(self):
        s = nucleus.setting.IntegerOrUnboundedRange("foo", (0, -15))
        self.assertFalse(s.is_abs())
        s.value_text = s.compose_abs()
        self.assertTrue(s.is_abs())
        self.assertEqual(s.max, 15)

    def test_01_06_set_abs_end(self):
        s = nucleus.setting.IntegerOrUnboundedRange(
            "foo", (0, "-" + nucleus.setting.END)
        )
        self.assertFalse(s.is_abs())
        s.value_text = s.compose_abs()
        self.assertTrue(s.is_abs())
        self.assertEqual(s.max, nucleus.setting.END)

    def test_01_07_set_rel(self):
        s = nucleus.setting.IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_rel()
        self.assertFalse(s.is_abs())
        self.assertEqual(s.max, -15)

    def test_01_06_set_rel_end(self):
        s = nucleus.setting.IntegerOrUnboundedRange(
            "foo", (0, nucleus.setting.END)
        )
        s.value_text = s.compose_rel()
        self.assertFalse(s.is_abs())
        self.assertEqual(s.max, nucleus.setting.END)


class TestFilterSetting(unittest.TestCase):
    def test_01_01_simple(self):
        filters = [
            nucleus.setting.Filter.FilterPredicate(
                "foo", "Foo", lambda a: a == "x", []
            )
        ]
        f = nucleus.setting.Filter("", filters, "foo")
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))

    def test_01_02_compound(self):
        f2 = nucleus.setting.Filter.FilterPredicate("bar", "Bar", lambda: "y", [])
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo", "Foo", lambda a, b: a == b(), [f2]
        )
        f = nucleus.setting.Filter("", [f1], "foo bar")
        self.assertFalse(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))

    def test_01_03_literal(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f = nucleus.setting.Filter("", [f1], 'foo "x"')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        f = nucleus.setting.Filter("", [f1], 'foo "y"')
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("x"))

    def test_01_04_escaped_literal(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f = nucleus.setting.Filter("", [f1], 'foo "\\\\"')
        self.assertTrue(f.evaluate("\\"))
        self.assertFalse(f.evaluate("/"))

    def test_01_05_literal_with_quote(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f = nucleus.setting.Filter("", [f1], 'foo "\\""')
        self.assertTrue(f.evaluate('"'))
        self.assertFalse(f.evaluate("/"))

    def test_01_06_parentheses(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "eq",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f2 = nucleus.setting.Filter.FilterPredicate(
            "ne",
            "Bar",
            lambda a, b: a != b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f = nucleus.setting.Filter("", [f1, f2], 'and (eq "x") (ne "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))

    def test_01_07_or(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "eq",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )

        f = nucleus.setting.Filter("", [f1], 'or (eq "x") (eq "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))

    def test_02_01_build_one(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo", "Foo", lambda a: a == "foo", []
        )
        f = nucleus.setting.Filter("", [f1])
        f.build([f1])
        self.assertEqual(f.value, "foo")

    def test_02_02_build_literal(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f = nucleus.setting.Filter("", [f1])
        f.build([f1, "bar"])
        self.assertEqual(f.value, 'foo "bar"')

    def test_02_03_build_nested(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f = nucleus.setting.Filter("", [f1])
        f.build([nucleus.setting.Filter.OR_PREDICATE, [f1, "bar"], [f1, "baz"]])
        self.assertEqual(f.value, 'or (foo "bar") (foo "baz")')

    def test_02_04_build_escaped_literal(self):
        f1 = nucleus.setting.Filter.FilterPredicate(
            "foo",
            "Foo",
            lambda a, b: a == b,
            [nucleus.setting.Filter.LITERAL_PREDICATE],
        )
        f = nucleus.setting.Filter("", [f1])
        f.build([f1, '"12\\'])
        self.assertEqual(f.value, 'foo "\\"12\\\\"')
        tokens = f.parse()
        self.assertEqual(tokens[1], '"12\\')

    def test_02_05_build_escaped_symbol(self):
        ugly = '(\\")'
        expected = '\\(\\\\\\"\\)'
        f1 = nucleus.setting.Filter.FilterPredicate(
            ugly, "Foo", lambda a, b: a == b, []
        )
        f = nucleus.setting.Filter("", [f1])
        f.build([f1])
        self.assertEqual(f.value, '\\(\\\\"\\)')

    def test_02_06_parse_escaped_symbol(self):
        ugly = '(\\")'
        encoded_ugly = '\\(\\\\\\"\\)'
        f1 = nucleus.setting.Filter.FilterPredicate(
            ugly, "Foo", lambda a, b: a == b, []
        )
        f = nucleus.setting.Filter("", [f1], encoded_ugly)
        result = f.parse()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].symbol, ugly)
