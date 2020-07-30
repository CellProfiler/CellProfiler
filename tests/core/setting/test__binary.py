from cellprofiler_core.setting import Binary


class TestBinary:
    def test_set_value(self):
        setting = Binary("example", "yes")

        setting.set_value("no")

        assert not setting.value

    def test_get_value(self):
        setting = Binary("example", "yes")

        assert setting.get_value()

    def test_eq(self):
        pass

    def test_on_event_fired(self):
        pass
