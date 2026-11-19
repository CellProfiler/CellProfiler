import datetime
import os

import pkg_resources
import pytest

from cellprofiler.gui.utilities.icon import (
    DEFAULT_ICON_NAME,
    ICON_MONTH_ENV,
    SEASONAL_ICON_NAMES,
    seasonal_icon_name,
)

EXPECTED = {
    1: "CellebrateProfiler",
    2: "CellentinesProfiler",
    10: "SpellProfiler",
    11: "CellProfaller",
    12: "JingleCellProfiler",
}


@pytest.fixture(autouse=True)
def clear_override(monkeypatch):
    monkeypatch.delenv(ICON_MONTH_ENV, raising=False)


@pytest.mark.parametrize("month", range(1, 13))
def test_seasonal_icon_name(month):
    today = datetime.date(2026, month, 15)
    assert seasonal_icon_name(today) == EXPECTED.get(month, DEFAULT_ICON_NAME)


@pytest.mark.parametrize("override,expected", [("10", "SpellProfiler"), ("5", "CellProfiler")])
def test_env_override(monkeypatch, override, expected):
    monkeypatch.setenv(ICON_MONTH_ENV, override)
    assert seasonal_icon_name(datetime.date(2026, 12, 1)) == expected


@pytest.mark.parametrize("override", ["13", "0", "october"])
def test_invalid_env_override_ignored(monkeypatch, override):
    monkeypatch.setenv(ICON_MONTH_ENV, override)
    assert seasonal_icon_name(datetime.date(2026, 12, 1)) == "JingleCellProfiler"


@pytest.mark.parametrize(
    "name", [DEFAULT_ICON_NAME] + list(SEASONAL_ICON_NAMES.values())
)
@pytest.mark.parametrize("extension", [".png", ".ico", ".icns"])
def test_icon_assets_exist(name, extension):
    path = pkg_resources.resource_filename(
        "cellprofiler", os.path.join("data", "icons", name + extension)
    )
    assert os.path.isfile(path)
