import ntpath
from pathlib import PureWindowsPath

import pytest
from cellprofiler.gui.help import content


@pytest.mark.parametrize(
    "working_directory, expected",
    [
        ("C:/work", "D:/CellProfiler/cellprofiler/data/images/UnmixColors.png"),
        ("D:/CellProfiler", "cellprofiler/data/images/UnmixColors.png"),
    ],
)
def test_image_resource_windows_paths(monkeypatch, working_directory, expected):
    windows_relpath = ntpath.relpath
    monkeypatch.setattr(
        content.importlib.resources,
        "files",
        lambda package: PureWindowsPath("D:/CellProfiler/cellprofiler"),
    )
    monkeypatch.setattr(
        content.os.path,
        "relpath",
        lambda path: windows_relpath(path, working_directory),
    )
    assert content.image_resource("UnmixColors.png") == expected


def test_image_resource_documentation_path(monkeypatch, tmp_path):
    documentation_directory = tmp_path / "docs"
    documentation_directory.mkdir()
    monkeypatch.chdir(documentation_directory)
    monkeypatch.setattr(
        content.importlib.resources, "files", lambda package: tmp_path / "cellprofiler"
    )

    assert content.image_resource("UnmixColors.png") == "../images/UnmixColors.png"
