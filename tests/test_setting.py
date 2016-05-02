import unittest
import cellprofiler.setting


class TestIntegerSetting(unittest.TestCase):
    def test_01_01_default(self):
        s = cellprofiler.setting.Integer("foo", value=5)
        self.assertEqual(s, 5)
        self.assertEqual(s.value_text, "5")
        s.test_valid(None)

    def test_01_02_set_value(self):
        s = cellprofiler.setting.Integer("foo", value=5)
        for test_case in ("06", "-1"):
            s.value_text = test_case
            self.assertEqual(s, int(test_case))
            self.assertEqual(s.value_text, test_case)
            s.test_valid(None)

    def test_01_03_set_bad(self):
        s = cellprofiler.setting.Integer("foo", value=5)
        s.value_text = "bad"
        self.assertEqual(s, 5)
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_02_01_good_min(self):
        s = cellprofiler.setting.Integer("foo", value=5, minval=0)
        for test_case in ("0", "1"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = cellprofiler.setting.Integer("foo", value=5, minval=0)
        s.value_text = "-1"
        self.assertEquals(s, 5)
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_02_03_good_max(self):
        s = cellprofiler.setting.Integer("foo", value=5, maxval=10)
        for test_case in ("9", "10"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = cellprofiler.setting.Integer("foo", value=5, maxval=10)
        s.value_text = "11"
        self.assertEquals(s.value, 5)
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))


class TestFloatSetting(unittest.TestCase):
    def test_01_01_default(self):
        for value in (5, "5.0"):
            s = cellprofiler.setting.Float("foo", value=value)
            self.assertEqual(s, 5)
            self.assertEqual(s.value_text, "5.0")
            s.test_valid(None)

    def test_01_02_set_value(self):
        s = cellprofiler.setting.Float("foo", value=5)
        for test_case in ("6.00", "-1.75"):
            s.value_text = test_case
            self.assertEqual(s, float(test_case))
            self.assertEqual(s.value_text, test_case)
            s.test_valid(None)

    def test_01_03_set_bad(self):
        s = cellprofiler.setting.Float("foo", value=5)
        s.value_text = "bad"
        self.assertEqual(s, 5)
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_01_04_set_text_value(self):
        s = cellprofiler.setting.Float("foo", value=5)
        s.value = "6.00"
        self.assertEquals(s, 6)
        self.assertEquals(s.value_text, "6.00")

    def test_02_01_good_min(self):
        s = cellprofiler.setting.Float("foo", value=5, minval=0)
        for test_case in ("0", "1"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = cellprofiler.setting.Float("foo", value=5, minval=0)
        s.value_text = "-1"
        self.assertEquals(s, 5)
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_02_03_good_max(self):
        s = cellprofiler.setting.Float("foo", value=5, maxval=10)
        for test_case in ("9", "10"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = cellprofiler.setting.Float("foo", value=5, maxval=10)
        s.value_text = "11"
        self.assertEquals(s, 5)
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))


class TestIntegerRange(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.setting.IntegerRange("text", (1, 2), 1, 5)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(str(x), "1,2")
        self.assertEqual(x.min, 1)
        self.assertEqual(x.max, 2)
        x.test_valid(None)

    def test_01_01_assign_tuple(self):
        x = cellprofiler.setting.IntegerRange("text", (1, 2), 1, 5)
        x.value = (2, 5)
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_01_02_assign_string(self):
        x = cellprofiler.setting.IntegerRange("text", (1, 2), 1, 5)
        x.value = "2,5"
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_02_01_neg_min(self):
        x = cellprofiler.setting.IntegerRange("text", (1, 2), 1, 5)
        x.value = (0, 2)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_02_neg_max(self):
        x = cellprofiler.setting.IntegerRange("text", (1, 2), 1, 5)
        x.value = (1, 6)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_03_neg_order(self):
        x = cellprofiler.setting.IntegerRange("text", (1, 2), 1, 5)
        x.value = (2, 1)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_03_01_no_range(self):
        """Regression test a bug where the variable throws an exception if there is no range"""
        x = cellprofiler.setting.IntegerRange("text", (1, 2))
        x.test_valid(None)

    def test_01_01_default(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15))
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "1")
        self.assertEquals(s.max_text, "15")
        s.test_valid(None)

    def test_01_02_set_min(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_min_text("02")
        self.assertEquals(s.min, 2)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "02")
        self.assertEquals(s.max_text, "15")
        s.test_valid(None)

    def test_01_03_set_min_bad(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_min_text("a2")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "a2")
        self.assertEquals(s.max_text, "15")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_01_04_set_max(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_max_text("016")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 16)
        self.assertEquals(s.min_text, "1")
        self.assertEquals(s.max_text, "016")
        s.test_valid(None)

    def test_01_05_set_max_bad(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15))
        s.value_text = s.compose_max_text("a2")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "1")
        self.assertEquals(s.max_text, "a2")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_02_01_good_min(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15), minval=0)
        for test_case in ("2", "0"):
            s.value_text = s.compose_min_text(test_case)
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15), minval=0)
        s.value_text = s.compose_min_text("-1")
        self.assertEquals(s.min, 0)
        self.assertEquals(s.min_text, "-1")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_02_03_good_max(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15), maxval=20)
        for test_case in ("18", "20"):
            s.value_text = s.compose_max_text(test_case)
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = cellprofiler.setting.IntegerRange("foo", (1, 15), maxval=20)
        s.value_text = s.compose_max_text("21")
        self.assertEquals(s.max, 20)
        self.assertEquals(s.max_text, "21")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))


class TestFloatRange(unittest.TestCase):
    def test_01_01_default(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15))
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "1.0")
        self.assertEquals(s.max_text, "15.0")
        s.test_valid(None)

    def test_01_02_set_min(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("2.10")
        self.assertEquals(s.min, 2.1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "2.10")
        self.assertEquals(s.max_text, "15.0")
        s.test_valid(None)

    def test_01_03_set_min_bad(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("a2")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "a2")
        self.assertEquals(s.max_text, "15.0")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_01_04_set_max(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("016")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 16)
        self.assertEquals(s.min_text, "1.0")
        self.assertEquals(s.max_text, "016")
        s.test_valid(None)

    def test_01_05_set_max_bad(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("a2")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "1.0")
        self.assertEquals(s.max_text, "a2")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_02_01_good_min(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15), minval=0)
        for test_case in ("2", "0"):
            s.value_text = s.compose_min_text(test_case)
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15), minval=0)
        s.value_text = s.compose_min_text("-1")
        self.assertEquals(s.min, 0)
        self.assertEquals(s.min_text, "-1")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_02_03_good_max(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15), maxval=20)
        for test_case in ("18", "20.00"):
            s.value_text = s.compose_max_text(test_case)
            self.assertEquals(s.max, float(test_case))
            self.assertEquals(s.max_text, test_case)
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = cellprofiler.setting.FloatRange("foo", (1, 15), maxval=20)
        s.value_text = s.compose_max_text("21")
        self.assertEquals(s.max, 20)
        self.assertEquals(s.max_text, "21")
        self.assertRaises(cellprofiler.setting.ValidationError, (lambda: s.test_valid(None)))

    def test_00_00_init(self):
        x = cellprofiler.setting.FloatRange("text", (1, 2), 1, 5)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, (1, 2))
        self.assertEqual(x.min, 1)
        self.assertEqual(x.max, 2)
        x.test_valid(None)

    def test_01_01_assign_tuple(self):
        x = cellprofiler.setting.FloatRange("text", (1, 2), 1, 5)
        x.value = (2, 5)
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_01_02_assign_string(self):
        x = cellprofiler.setting.FloatRange("text", (1, 2), 1, 5)
        x.value = "2,5"
        self.assertEqual(x.min, 2)
        self.assertEqual(x.max, 5)
        x.test_valid(None)

    def test_02_01_neg_min(self):
        x = cellprofiler.setting.FloatRange("text", (1, 2), 1, 5)
        x.value = (0, 2)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_02_neg_max(self):
        x = cellprofiler.setting.FloatRange("text", (1, 2), 1, 5)
        x.value = (1, 6)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_02_03_neg_order(self):
        x = cellprofiler.setting.FloatRange("text", (1, 2), 1, 5)
        x.value = (2, 1)
        self.assertRaises(ValueError, x.test_valid, None)

    def test_03_01_no_range(self):
        """Regression test a bug where the variable throws an exception if there is no range"""
        x = cellprofiler.setting.FloatRange("text", (1, 2))
        x.test_valid(None)


class TestIntegerOrUnboundedRange(unittest.TestCase):
    def test_01_01_default(self):
        for minval, maxval, expected_min, expected_min_text, \
            expected_max, expected_max_text, expected_unbounded_max, \
            expected_abs in (
                (0, cellprofiler.setting.END, 0, "0", cellprofiler.setting.END, cellprofiler.setting.END, True, True),
                (cellprofiler.setting.BEGIN, 15, 0, cellprofiler.setting.BEGIN, 15, "15", False, True),
                (0, -15, 0, "0", -15, "15", False, False),
                (0, "-" + cellprofiler.setting.END, 0, "0", cellprofiler.setting.END, cellprofiler.setting.END, True, False)):
            s = cellprofiler.setting.IntegerOrUnboundedRange("foo", (minval, maxval))
            self.assertEquals(s.min, expected_min)
            self.assertEquals(s.max, expected_max)
            self.assertEquals(s.display_min, expected_min_text)
            self.assertEquals(s.display_max, expected_max_text)
            self.assertEquals(s.is_abs(), expected_abs)
            s.test_valid(None)

    def test_01_02_set_min(self):
        s = cellprofiler.setting.IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_min_text("01")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.display_min, "01")

    def test_01_03_set_max(self):
        s = cellprofiler.setting.IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_max_text("015")
        self.assertEquals(s.max, 15)
        self.assertEquals(s.display_max, "015")

    def test_01_04_set_end(self):
        s = cellprofiler.setting.IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_max_text(cellprofiler.setting.END)
        self.assertEquals(s.max, cellprofiler.setting.END)

    def test_01_05_set_abs(self):
        s = cellprofiler.setting.IntegerOrUnboundedRange("foo", (0, -15))
        self.assertFalse(s.is_abs())
        s.value_text = s.compose_abs()
        self.assertTrue(s.is_abs())
        self.assertEqual(s.max, 15)

    def test_01_06_set_abs_end(self):
        s = cellprofiler.setting.IntegerOrUnboundedRange("foo", (0, "-" + cellprofiler.setting.END))
        self.assertFalse(s.is_abs())
        s.value_text = s.compose_abs()
        self.assertTrue(s.is_abs())
        self.assertEqual(s.max, cellprofiler.setting.END)

    def test_01_07_set_rel(self):
        s = cellprofiler.setting.IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_rel()
        self.assertFalse(s.is_abs())
        self.assertEqual(s.max, -15)

    def test_01_06_set_rel_end(self):
        s = cellprofiler.setting.IntegerOrUnboundedRange("foo", (0, cellprofiler.setting.END))
        s.value_text = s.compose_rel()
        self.assertFalse(s.is_abs())
        self.assertEqual(s.max, cellprofiler.setting.END)


class TestFilterSetting(unittest.TestCase):
    def test_01_01_simple(self):
        filters = [cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a: a == "x", [])]
        f = cellprofiler.setting.Filter("", filters, "foo")
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))

    def test_01_02_compound(self):
        f2 = cellprofiler.setting.Filter.FilterPredicate("bar", "Bar", lambda: "y", [])
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b(), [f2])
        f = cellprofiler.setting.Filter("", [f1], "foo bar")
        self.assertFalse(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))

    def test_01_03_literal(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f = cellprofiler.setting.Filter("", [f1], 'foo "x"')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        f = cellprofiler.setting.Filter("", [f1], 'foo "y"')
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("x"))

    def test_01_04_escaped_literal(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f = cellprofiler.setting.Filter("", [f1], 'foo "\\\\"')
        self.assertTrue(f.evaluate("\\"))
        self.assertFalse(f.evaluate("/"))

    def test_01_05_literal_with_quote(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f = cellprofiler.setting.Filter("", [f1], 'foo "\\""')
        self.assertTrue(f.evaluate("\""))
        self.assertFalse(f.evaluate("/"))

    def test_01_06_parentheses(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("eq", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f2 = cellprofiler.setting.Filter.FilterPredicate("ne", "Bar", lambda a, b: a != b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f = cellprofiler.setting.Filter("", [f1, f2], 'and (eq "x") (ne "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))

    def test_01_07_or(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("eq", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])

        f = cellprofiler.setting.Filter("", [f1], 'or (eq "x") (eq "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))

    def test_02_01_build_one(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a: a == "foo", [])
        f = cellprofiler.setting.Filter("", [f1])
        f.build([f1])
        self.assertEqual(f.value, "foo")

    def test_02_02_build_literal(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f = cellprofiler.setting.Filter("", [f1])
        f.build([f1, "bar"])
        self.assertEqual(f.value, 'foo "bar"')

    def test_02_03_build_nested(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f = cellprofiler.setting.Filter("", [f1])
        f.build([cellprofiler.setting.Filter.OR_PREDICATE, [f1, "bar"], [f1, u"baz"]])
        self.assertEqual(f.value, 'or (foo "bar") (foo "baz")')

    def test_02_04_build_escaped_literal(self):
        f1 = cellprofiler.setting.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                                         [cellprofiler.setting.Filter.LITERAL_PREDICATE])
        f = cellprofiler.setting.Filter("", [f1])
        f.build([f1, '"12\\'])
        self.assertEqual(f.value, 'foo "\\"12\\\\"')
        tokens = f.parse()
        self.assertEqual(tokens[1], '"12\\')

    def test_02_05_build_escaped_symbol(self):
        ugly = '(\\")'
        expected = '\\(\\\\\\"\\)'
        f1 = cellprofiler.setting.Filter.FilterPredicate(ugly, "Foo", lambda a, b: a == b, [])
        f = cellprofiler.setting.Filter("", [f1])
        f.build([f1])
        self.assertEqual(f.value, '\\(\\\\\\"\\)')

    def test_02_06_parse_escaped_symbol(self):
        ugly = '(\\")'
        encoded_ugly = '\\(\\\\\\"\\)'
        f1 = cellprofiler.setting.Filter.FilterPredicate(ugly, "Foo", lambda a, b: a == b, [])
        f = cellprofiler.setting.Filter("", [f1], encoded_ugly)
        result = f.parse()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].symbol, ugly)


class TestVariable(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.setting.Setting("text", "value")
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "value")
        self.assertTrue(x.key())

    def test_01_01_equality(self):
        x = cellprofiler.setting.Setting("text", "value")
        self.assertTrue(x == "value")
        self.assertTrue(x != "text")
        self.assertFalse(x != "value")
        self.assertFalse(x == "text")
        self.assertEqual(x.value, "value")


class TestText(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.setting.Text("text", "value")
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "value")
        self.assertTrue(x.key())


class TestInteger(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.setting.Integer("text", 5)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, 5)

    def test_01_01_numeric_value(self):
        x = cellprofiler.setting.Integer("text", 5)
        self.assertTrue(isinstance(x.value, int))

    def test_01_02_equality(self):
        x = cellprofiler.setting.Integer("text", 5)
        self.assertTrue(x == 5)
        self.assertTrue(x.value == 5)
        self.assertFalse(x == 6)
        self.assertTrue(x != 6)

    def test_01_03_assign_str(self):
        x = cellprofiler.setting.Integer("text", 5)
        x.value = 6
        self.assertTrue(x == 6)

    def test_02_01_neg_assign_number(self):
        x = cellprofiler.setting.Integer("text", 5)
        x.set_value("foo")
        self.assertRaises(ValueError, x.test_valid, None)


class TestBinary(unittest.TestCase):
    def test_00_01_init_true(self):
        x = cellprofiler.setting.Binary("text", True)
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertTrue(x.value == True)
        self.assertTrue(x == True)
        self.assertTrue(x != False)
        self.assertFalse(x != True)
        self.assertFalse(x == False)

    def test_00_02_init_false(self):
        x = cellprofiler.setting.Binary("text", False)
        self.assertTrue(x.value == False)
        self.assertFalse(x == True)
        self.assertFalse(x != False)
        self.assertTrue(x != True)

    def test_01_01_set_true(self):
        x = cellprofiler.setting.Binary("text", False)
        x.value = True
        self.assertTrue(x.value == True)

    def test_01_02_set_false(self):
        x = cellprofiler.setting.Binary("text", True)
        x.value = False
        self.assertTrue(x.value == False)


class TestChoice(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.setting.Choice("text", ["choice"])
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "choice")
        self.assertEqual(len(x.choices), 1)
        self.assertEqual(x.choices[0], "choice")

    def test_01_01_assign(self):
        x = cellprofiler.setting.Choice("text", ["foo", "bar"], "bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")

    def test_02_01_neg_assign(self):
        x = cellprofiler.setting.Choice("text", ["choice"])
        x.set_value("foo")
        self.assertRaises(ValueError, x.test_valid, None)


class TestCustomChoice(unittest.TestCase):
    def test_00_00_init(self):
        x = cellprofiler.setting.CustomChoice("text", ["choice"])
        x.test_valid(None)
        self.assertEqual(x.text, "text")
        self.assertEqual(x.value, "choice")
        self.assertEqual(len(x.choices), 1)
        self.assertEqual(x.choices[0], "choice")

    def test_01_01_assign(self):
        x = cellprofiler.setting.CustomChoice("text", ["foo", "bar"], "bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")

    def test_01_02_assign_other(self):
        x = cellprofiler.setting.CustomChoice("text", ["foo", "bar"], "bar")
        x.value = "other"
        self.assertTrue(x == "other")
        self.assertEqual(len(x.choices), 3)
        self.assertEqual(x.choices[0], "other")


class TestDirectoryPath(unittest.TestCase):
    def setUp(self):
        #
        # Make three temporary directory structures
        #
        cpprefs.set_headless()
        self.directories = [tempfile.mkdtemp() for i in range(3)]
        for directory in self.directories:
            for i in range(3):
                os.mkdir(os.path.join(directory, str(i)))
                for j in range(3):
                    os.mkdir(os.path.join(directory, str(i), str(j)))
                    for k in range(3):
                        os.mkdir(os.path.join(directory, str(i), str(j), str(k)))
        cpprefs.set_default_image_directory(os.path.join(self.directories[0], "1"))
        cpprefs.set_default_output_directory(os.path.join(self.directories[1], "1"))
        self.root_directory = os.path.join(self.directories[2], "1")

    def tearDown(self):
        for directory in self.directories:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        os.rmdir(os.path.join(directory, str(i), str(j), str(k)))
                    os.rmdir(os.path.join(directory, str(i), str(j)))
                os.rmdir(os.path.join(directory, str(i)))
            os.rmdir(directory)

    def test_01_01_static_split_and_join(self):
        gibberish = "aqwura[oijs|fd"
        for dir_choice in cellprofiler.setting.DirectoryPath.DIR_ALL + [cellprofiler.setting.NO_FOLDER_NAME]:
            value = cellprofiler.setting.DirectoryPath.static_join_string(dir_choice, gibberish)
            out_dir_choice, custom_path = cellprofiler.setting.DirectoryPath.split_string(value)
            self.assertEqual(dir_choice, out_dir_choice)
            self.assertEqual(custom_path, gibberish)

    def test_01_02_split_and_join(self):
        gibberish = "aqwura[oijs|fd"
        for dir_choice in cellprofiler.setting.DirectoryPath.DIR_ALL + [cellprofiler.setting.NO_FOLDER_NAME]:
            s = cellprofiler.setting.DirectoryPath("whatever")
            value = s.join_parts(dir_choice, gibberish)
            out_dir_choice = s.dir_choice
            custom_path = s.custom_path
            self.assertEqual(dir_choice, out_dir_choice)
            self.assertEqual(custom_path, gibberish)

    def test_01_03_is_custom_choice(self):
        for dir_choice, expected in (
                (cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME, False),
                (cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME, True),
                (cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME, False),
                (cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME, True),
                (cellprofiler.setting.ABSOLUTE_FOLDER_NAME, True),
                (cellprofiler.setting.URL_FOLDER_NAME, True)):
            s = cellprofiler.setting.DirectoryPath("whatever")
            s.dir_choice = dir_choice
            self.assertEqual(s.is_custom_choice, expected)

    def test_01_04_get_parts_from_image_folder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(
                cpprefs.get_default_image_directory())
        self.assertEqual(dir_choice, cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME)

    def test_01_05_get_parts_from_output_folder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(cpprefs.get_default_output_directory())
        self.assertEqual(dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME)

    def test_01_06_get_parts_from_image_subfolder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(
                os.path.join(cpprefs.get_default_image_directory(), "1"))
        self.assertEqual(dir_choice, cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME)
        self.assertEqual(custom_path, "1")

    def test_01_07_get_parts_from_output_subfolder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(
                os.path.join(cpprefs.get_default_output_directory(), "2"))
        self.assertEqual(dir_choice, cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME)
        self.assertEqual(custom_path, "2")

    def test_01_08_get_parts_from_abspath(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        dir_choice, custom_path = s.get_parts_from_path(self.root_directory)
        self.assertEqual(dir_choice, cellprofiler.setting.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(custom_path, self.root_directory)

    def test_02_01_get_default_input_folder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME
        self.assertEqual(s.get_absolute_path(), cpprefs.get_default_image_directory())

    def test_02_02_get_default_output_folder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
        self.assertEqual(s.get_absolute_path(), cpprefs.get_default_output_directory())

    def test_02_03_get_input_subfolder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME
        s.custom_path = "2"
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(cpprefs.get_default_image_directory(), "2"))

    def test_02_04_get_output_subfolder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME
        s.custom_path = "0"
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(cpprefs.get_default_output_directory(), "0"))

    def test_02_05_get_absolute_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
        s.custom_path = os.path.join(self.root_directory, "..", "1", "2")
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(self.root_directory, "2"))

    def test_02_06_get_url(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.URL_FOLDER_NAME
        s.custom_path = "http://www.cellprofiler.org"
        self.assertEqual(s.get_absolute_path(),
                         "http://www.cellprofiler.org")

    def test_02_07_no_folder(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.NO_FOLDER_NAME
        s.custom_path = "gibberish"
        self.assertEqual(s.get_absolute_path(), '')

    def test_03_07_metadata(self):
        m = cpmeas.Measurement()
        m.add_image_measurement("Metadata_Path", "2")
        s = cellprofiler.setting.DirectoryPath("whatever", allow_metadata=True)
        for dir_choice, expected in (
                (cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME,
                 os.path.join(cpprefs.get_default_image_directory(), "0", "2")),
                (cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                 os.path.join(cpprefs.get_default_output_directory(), "0", "2")),
                (cellprofiler.setting.ABSOLUTE_FOLDER_NAME,
                 os.path.join(self.root_directory, "2")),
                (cellprofiler.setting.URL_FOLDER_NAME, "http://www.cellprofiler.org/2")):
            s.dir_choice = dir_choice
            if dir_choice in (cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME,
                              cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME):
                s.custom_path = "0" + os.path.sep.replace('\\', '\\\\') + "\\g<Path>"
            elif dir_choice == cellprofiler.setting.ABSOLUTE_FOLDER_NAME:
                s.custom_path = self.root_directory + os.path.sep.replace('\\', '\\\\') + "\\g<Path>"
            else:
                s.custom_path = "http://www.cellprofiler.org/\\g<Path>"
            self.assertEqual(s.get_absolute_path(m), expected)

    @staticmethod
    def fn_alter_path(path, **kwargs):
        '''Add "altered" to the path'''
        return path + "altered"

    def test_04_01_alter_input_folder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                cpprefs.get_default_image_directory())

    def test_04_02_alter_output_folder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                cpprefs.get_default_output_directory())

    def test_04_03_alter_input_subfolder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME
        s.custom_path = "2"

        def fn_alter_path(path, **kwargs):
            self.assertEqual(path, "2")
            return "3"

        s.alter_for_create_batch_files(fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                os.path.join(cpprefs.get_default_image_directory(), "3"))

    def test_04_04_alter_output_subfolder_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME
        s.custom_path = "0"

        def fn_alter_path(path, **kwargs):
            self.assertEqual(path, "0")
            return "5"

        s.alter_for_create_batch_files(fn_alter_path)
        self.assertEqual(
                s.get_absolute_path(),
                os.path.join(cpprefs.get_default_output_directory(), "5"))

    def test_04_05_alter_absolute_path(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.ABSOLUTE_FOLDER_NAME
        s.custom_path = os.path.join(self.root_directory, "..", "1", "2")
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(s.get_absolute_path(),
                         os.path.join(self.root_directory, "2altered"))

    def test_04_06_alter_url(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.URL_FOLDER_NAME
        s.custom_path = "http://www.cellprofiler.org"
        s.alter_for_create_batch_files(TestDirectoryPath.fn_alter_path)
        self.assertEqual(s.get_absolute_path(),
                         "http://www.cellprofiler.org")

    def test_04_07_no_folder(self):
        s = cellprofiler.setting.DirectoryPath("whatever")
        s.dir_choice = cellprofiler.setting.NO_FOLDER_NAME
        s.custom_path = "gibberish"
        self.assertEqual(s.get_absolute_path(), '')
