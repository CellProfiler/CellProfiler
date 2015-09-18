class TestFloatSetting(unittest.TestCase):
    def test_01_01_default(self):
        for value in (5, "5.0"):
            s = cps.Float("foo", value=value)
            self.assertEqual(s, 5)
            self.assertEqual(s.value_text, "5.0")
            s.test_valid(None)

    def test_01_02_set_value(self):
        s = cps.Float("foo", value=5)
        for test_case in ("6.00", "-1.75"):
            s.value_text = test_case
            self.assertEqual(s, float(test_case))
            self.assertEqual(s.value_text, test_case)
            s.test_valid(None)

    def test_01_03_set_bad(self):
        s = cps.Float("foo", value=5)
        s.value_text = "bad"
        self.assertEqual(s, 5)
        self.assertRaises(cps.ValidationError, (lambda: s.test_valid(None)))

    def test_01_04_set_text_value(self):
        s = cps.Float("foo", value=5)
        s.value = "6.00"
        self.assertEquals(s, 6)
        self.assertEquals(s.value_text, "6.00")

    def test_02_01_good_min(self):
        s = cps.Float("foo", value=5, minval=0)
        for test_case in ("0", "1"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = cps.Float("foo", value=5, minval=0)
        s.value_text = "-1"
        self.assertEquals(s, 5)
        self.assertRaises(cps.ValidationError, (lambda: s.test_valid(None)))

    def test_02_03_good_max(self):
        s = cps.Float("foo", value=5, maxval=10)
        for test_case in ("9", "10"):
            s.value_text = test_case
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = cps.Float("foo", value=5, maxval=10)
        s.value_text = "11"
        self.assertEquals(s, 5)
        self.assertRaises(cps.ValidationError, (lambda: s.test_valid(None)))
