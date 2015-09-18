class TestIntegerOrUnboundedRange(unittest.TestCase):
    def test_01_01_default(self):
        for minval, maxval, expected_min, expected_min_text, \
            expected_max, expected_max_text, expected_unbounded_max, \
            expected_abs in (
                (0, cps.END, 0, "0", cps.END, cps.END, True, True),
                (cps.BEGIN, 15, 0, cps.BEGIN, 15, "15", False, True),
                (0, -15, 0, "0", -15, "15", False, False),
                (0, "-" + cps.END, 0, "0", cps.END, cps.END, True, False)):
            s = cps.IntegerOrUnboundedRange("foo", (minval, maxval))
            self.assertEquals(s.min, expected_min)
            self.assertEquals(s.max, expected_max)
            self.assertEquals(s.display_min, expected_min_text)
            self.assertEquals(s.display_max, expected_max_text)
            self.assertEquals(s.is_abs(), expected_abs)
            s.test_valid(None)

    def test_01_02_set_min(self):
        s = cps.IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_min_text("01")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.display_min, "01")

    def test_01_03_set_max(self):
        s = cps.IntegerOrUnboundedRange("foo")
        s.value_text = s.compose_max_text("015")
        self.assertEquals(s.max, 15)
        self.assertEquals(s.display_max, "015")

    def test_01_04_set_end(self):
        s = cps.IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_max_text(cps.END)
        self.assertEquals(s.max, cps.END)

    def test_01_05_set_abs(self):
        s = cps.IntegerOrUnboundedRange("foo", (0, -15))
        self.assertFalse(s.is_abs())
        s.value_text = s.compose_abs()
        self.assertTrue(s.is_abs())
        self.assertEqual(s.max, 15)

    def test_01_06_set_abs_end(self):
        s = cps.IntegerOrUnboundedRange("foo", (0, "-" + cps.END))
        self.assertFalse(s.is_abs())
        s.value_text = s.compose_abs()
        self.assertTrue(s.is_abs())
        self.assertEqual(s.max, cps.END)

    def test_01_07_set_rel(self):
        s = cps.IntegerOrUnboundedRange("foo", (0, 15))
        s.value_text = s.compose_rel()
        self.assertFalse(s.is_abs())
        self.assertEqual(s.max, -15)

    def test_01_06_set_rel_end(self):
        s = cps.IntegerOrUnboundedRange("foo", (0, cps.END))
        s.value_text = s.compose_rel()
        self.assertFalse(s.is_abs())
        self.assertEqual(s.max, cps.END)
