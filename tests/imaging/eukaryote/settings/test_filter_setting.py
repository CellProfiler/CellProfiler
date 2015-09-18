class TestFilterSetting(unittest.TestCase):
    def test_01_01_simple(self):
        filters = [cps.Filter.FilterPredicate("foo", "Foo", lambda a: a == "x", [])]
        f = cps.Filter("", filters, "foo")
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))

    def test_01_02_compound(self):
        f2 = cps.Filter.FilterPredicate("bar", "Bar", lambda: "y", [])
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b(), [f2])
        f = cps.Filter("", [f1], "foo bar")
        self.assertFalse(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))

    def test_01_03_literal(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1], 'foo "x"')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        f = cps.Filter("", [f1], 'foo "y"')
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("x"))

    def test_01_04_escaped_literal(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1], 'foo "\\\\"')
        self.assertTrue(f.evaluate("\\"))
        self.assertFalse(f.evaluate("/"))

    def test_01_05_literal_with_quote(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1], 'foo "\\""')
        self.assertTrue(f.evaluate("\""))
        self.assertFalse(f.evaluate("/"))

    def test_01_06_parentheses(self):
        f1 = cps.Filter.FilterPredicate("eq", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f2 = cps.Filter.FilterPredicate("ne", "Bar", lambda a, b: a != b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1, f2], 'and (eq "x") (ne "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertFalse(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))

    def test_01_07_or(self):
        f1 = cps.Filter.FilterPredicate("eq", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])

        f = cps.Filter("", [f1], 'or (eq "x") (eq "y")')
        self.assertTrue(f.evaluate("x"))
        self.assertTrue(f.evaluate("y"))
        self.assertFalse(f.evaluate("z"))

    def test_02_01_build_one(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a: a == "foo", [])
        f = cps.Filter("", [f1])
        f.build([f1])
        self.assertEqual(f.value, "foo")

    def test_02_02_build_literal(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1])
        f.build([f1, "bar"])
        self.assertEqual(f.value, 'foo "bar"')

    def test_02_03_build_nested(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1])
        f.build([cps.Filter.OR_PREDICATE, [f1, "bar"], [f1, u"baz"]])
        self.assertEqual(f.value, 'or (foo "bar") (foo "baz")')

    def test_02_04_build_escaped_literal(self):
        f1 = cps.Filter.FilterPredicate("foo", "Foo", lambda a, b: a == b,
                                        [cps.Filter.LITERAL_PREDICATE])
        f = cps.Filter("", [f1])
        f.build([f1, '"12\\'])
        self.assertEqual(f.value, 'foo "\\"12\\\\"')
        tokens = f.parse()
        self.assertEqual(tokens[1], '"12\\')

    def test_02_05_build_escaped_symbol(self):
        ugly = '(\\")'
        expected = '\\(\\\\\\"\\)'
        f1 = cps.Filter.FilterPredicate(ugly, "Foo", lambda a, b: a == b, [])
        f = cps.Filter("", [f1])
        f.build([f1])
        self.assertEqual(f.value, '\\(\\\\\\"\\)')

    def test_02_06_parse_escaped_symbol(self):
        ugly = '(\\")'
        encoded_ugly = '\\(\\\\\\"\\)'
        f1 = cps.Filter.FilterPredicate(ugly, "Foo", lambda a, b: a == b, [])
        f = cps.Filter("", [f1], encoded_ugly)
        result = f.parse()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].symbol, ugly)
