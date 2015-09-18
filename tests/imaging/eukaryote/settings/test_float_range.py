class TestFloatRange(unittest.TestCase):
    def test_01_01_default(self):
        s = cps.FloatRange("foo", (1, 15))
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "1.0")
        self.assertEquals(s.max_text, "15.0")
        s.test_valid(None)

    def test_01_02_set_min(self):
        s = cps.FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("2.10")
        self.assertEquals(s.min, 2.1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "2.10")
        self.assertEquals(s.max_text, "15.0")
        s.test_valid(None)

    def test_01_03_set_min_bad(self):
        s = cps.FloatRange("foo", (1, 15))
        s.value_text = s.compose_min_text("a2")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "a2")
        self.assertEquals(s.max_text, "15.0")
        self.assertRaises(cps.ValidationError, (lambda: s.test_valid(None)))

    def test_01_04_set_max(self):
        s = cps.FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("016")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 16)
        self.assertEquals(s.min_text, "1.0")
        self.assertEquals(s.max_text, "016")
        s.test_valid(None)

    def test_01_05_set_max_bad(self):
        s = cps.FloatRange("foo", (1, 15))
        s.value_text = s.compose_max_text("a2")
        self.assertEquals(s.min, 1)
        self.assertEquals(s.max, 15)
        self.assertEquals(s.min_text, "1.0")
        self.assertEquals(s.max_text, "a2")
        self.assertRaises(cps.ValidationError, (lambda: s.test_valid(None)))

    def test_02_01_good_min(self):
        s = cps.FloatRange("foo", (1, 15), minval=0)
        for test_case in ("2", "0"):
            s.value_text = s.compose_min_text(test_case)
            s.test_valid(None)

    def test_02_02_bad_min(self):
        s = cps.FloatRange("foo", (1, 15), minval=0)
        s.value_text = s.compose_min_text("-1")
        self.assertEquals(s.min, 0)
        self.assertEquals(s.min_text, "-1")
        self.assertRaises(cps.ValidationError, (lambda: s.test_valid(None)))

    def test_02_03_good_max(self):
        s = cps.FloatRange("foo", (1, 15), maxval=20)
        for test_case in ("18", "20.00"):
            s.value_text = s.compose_max_text(test_case)
            self.assertEquals(s.max, float(test_case))
            self.assertEquals(s.max_text, test_case)
            s.test_valid(None)

    def test_02_04_bad_max(self):
        s = cps.FloatRange("foo", (1, 15), maxval=20)
        s.value_text = s.compose_max_text("21")
        self.assertEquals(s.max, 20)
        self.assertEquals(s.max_text, "21")
        self.assertRaises(cps.ValidationError, (lambda: s.test_valid(None)))
