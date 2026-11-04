import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("test_path", ["tests", "tests/core"])
def test_test_startup_is_headless(tmp_path, test_path):
    tests = tmp_path / "tests"
    core = tests / "core"
    core.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "conftest.py", tests / "conftest.py")
    (core / "test_headless.py").write_text(
        "from cellprofiler_core.preferences import get_headless\n"
        "from cellprofiler_core.constants.reader import AVAILABLE_READERS\n"
        "assert get_headless(), 'Tests must be headless before collection'\n"
        "assert 'Bio-Formats' in AVAILABLE_READERS\n"
        "def test_image_set_matching_error():\n"
        "    from cellprofiler_core.modules.namesandtypes import NamesAndTypes\n"
        "    assert NamesAndTypes.handle_error_messages(\n"
        "        [('too many matches', 'Column0', None)])\n"
    )
    script = """
import builtins
import sys

original_import = builtins.__import__

def without_wx(name, *args, **kwargs):
    if name == 'wx' or name.startswith('wx.'):
        raise AssertionError('Test startup must not import wx')
    return original_import(name, *args, **kwargs)

builtins.__import__ = without_wx
import pytest
raise SystemExit(pytest.main([sys.argv[1], '-q', '--confcutdir=tests']))
"""
    run_pytest(tmp_path, test_path, script)


@pytest.mark.parametrize(
    "subpackage,package", [("library", "cellprofiler_library"), ("centriole", "centriole")]
)
def test_test_startup_without_core(tmp_path, subpackage, package):
    tests = tmp_path / "tests"
    suite = tests / subpackage
    suite.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "conftest.py", tests / "conftest.py")
    (suite / "test_package.py").write_text(
        f"import {package}\n"
        "def test_package_is_importable():\n"
        f"    assert {package} is not None\n"
    )
    script = """
import builtins
import sys

sys.modules['cellprofiler_core'] = None
original_import = builtins.__import__

def without_core(name, *args, **kwargs):
    if name == 'cellprofiler_core' or name.startswith('cellprofiler_core.'):
        raise AssertionError('Isolated package tests must not import Core')
    return original_import(name, *args, **kwargs)

builtins.__import__ = without_core
import pytest
raise SystemExit(pytest.main([sys.argv[1], '-q', '--confcutdir=tests']))
"""
    run_pytest(tmp_path, f"tests/{subpackage}", script)


def run_pytest(tmp_path, test_path, script):
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, test_path],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
