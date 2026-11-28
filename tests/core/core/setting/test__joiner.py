import pytest

from cellprofiler_core.setting import Joiner, ValidationError


class TestJoiner:
    def test_parse(self):
        joins = [{"Images": "Well", "Plates": None}, {"Images": "Site", "Plates": "S"}]
        joiner = Joiner("Match metadata", value=Joiner.build_string(joins))
        assert joiner.parse() == joins

    def test_parse_python_2_strings(self):
        # Pipelines saved by older versions store the join with u"" strings
        joiner = Joiner("Match metadata", value="[{u'Images': u'Well', u'Plates': None}]")
        assert joiner.parse() == [{"Images": "Well", "Plates": None}]

    @pytest.mark.parametrize(
        "value",
        [
            "().__class__.__base__.__subclasses__()",
            "[x for x in ().__class__.__base__.__subclasses__()]",
            "(lambda: [])()",
            "[{'Images': 'Well'}",
        ],
    )
    def test_parse_rejects_code(self, value):
        # The value comes from a pipeline or project file, so it must not be
        # evaluated as Python (#5251)
        joiner = Joiner("Match metadata", value=value)
        with pytest.raises(ValueError):
            joiner.parse()

    def test_default(self):
        pass

    def test_build(self):
        pass

    def test_build_string(self):
        pass

    def test_test_valid(self):
        pass

    def test_test_valid_rejects_code(self):
        joiner = Joiner("Match metadata", value="().__class__.__base__.__subclasses__()")
        with pytest.raises(ValidationError):
            joiner.test_valid(None)
