import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy
import pytest
import tifffile

import cellprofiler.modules.runimagejmacro
from cellprofiler_core.image import ImageSetList
from cellprofiler_core.preferences import ABSOLUTE_FOLDER_NAME


@pytest.mark.parametrize("launcher", ["ImageJ-macosx", "fiji-macos"])
def test_run_macos_app(tmp_path, monkeypatch, launcher):
    executable = tmp_path / "Fiji.app" / "Contents" / "MacOS" / launcher
    executable.parent.mkdir(parents=True)
    executable.touch()

    run_macro(tmp_path, monkeypatch, "Fiji.app", executable)


def test_run_macos_app_prefers_legacy_launcher(tmp_path, monkeypatch):
    executable = tmp_path / "Fiji.app" / "Contents" / "MacOS" / "ImageJ-macosx"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (executable.parent / "fiji-macos").touch()

    run_macro(tmp_path, monkeypatch, "Fiji.app", executable)


def test_run_executable(tmp_path, monkeypatch):
    executable = tmp_path / "ImageJ-win64.exe"
    executable.touch()

    run_macro(tmp_path, monkeypatch, executable.name, executable)


def run_macro(tmp_path, monkeypatch, filename, executable):
    module = cellprofiler.modules.runimagejmacro.RunImageJMacro()
    module.executable_directory.join_parts(ABSOLUTE_FOLDER_NAME, str(tmp_path))
    module.executable_file.value = filename
    module.macro_directory.join_parts(ABSOLUTE_FOLDER_NAME, str(tmp_path))
    module.macro_file.value = "macro.ijm"
    module.image_groups_in.clear()
    workspace = SimpleNamespace(
        pipeline=SimpleNamespace(test_mode=False),
        image_set=ImageSetList().get_image_set(0),
        display_data=SimpleNamespace(),
    )
    monkeypatch.setattr(
        cellprofiler.modules.runimagejmacro,
        "get_default_output_directory",
        lambda: str(tmp_path),
    )
    pixels = numpy.arange(16, dtype=numpy.uint8).reshape(4, 4)

    def fake_run(cmd, **kwargs):
        tempdir = next(tmp_path.glob("runimagejmacro_*"))
        tifffile.imwrite(tempdir / module.image_groups_out[0].input_filename.value, pixels)
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    run = Mock(side_effect=fake_run)
    monkeypatch.setattr(cellprofiler.modules.runimagejmacro.subprocess, "run", run)

    module.run(workspace)

    run.assert_called_once()
    cmd = run.call_args.args[0]
    assert Path(cmd[0]) == executable
    assert cmd[1:4] == ["--headless", "console", "--run"]
    assert Path(cmd[4]) == tmp_path / "macro.ijm"
    numpy.testing.assert_array_equal(
        workspace.image_set.get_image(module.image_groups_out[0].image_name.value).pixel_data,
        pixels,
    )
    assert not list(tmp_path.glob("runimagejmacro_*"))
