import os
import subprocess
import sys
import textwrap

from cellprofiler.gui.pipeline import Pipeline

# Right single quotation mark, em dash and the combining sequence suggested in
# #4879. Written as escapes so that this file is pure ASCII.
NOTE = "cells\u2019 flow \u2014 y\u0316\u0320\u034d\u0318\u0347\u0357\u030f\u033d\u030e\u035e"

# The script is plain ASCII so that it survives an ASCII locale. The note is
# passed in as an ASCII literal and rebuilt inside the child process.
SCRIPT = textwrap.dedent(
    """
    import sys

    from cellprofiler.gui.pipeline import Pipeline
    from cellprofiler.modules.crop import Crop
    from cellprofiler_core.modules.images import Images

    pipeline = Pipeline()
    images = Images()
    images.set_module_num(1)
    pipeline.add_module(images)
    crop = Crop()
    crop.set_module_num(2)
    crop.notes = [{note}]
    pipeline.add_module(crop)
    pipeline.save(sys.argv[1])
    """
).format(note=ascii(NOTE))


def test_save_non_ascii_note_with_ascii_locale(tmp_path):
    """Saving must not depend on the platform default encoding (#4879, #5123)"""
    path = tmp_path / "pipeline.cppipe"
    path.write_text("existing pipeline", encoding="utf-8")

    env = dict(os.environ)
    # GTK cannot open a display in the C locale, so the child runs headless.
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env.update(
        LC_ALL="C",
        LANG="C",
        PYTHONCOERCECLOCALE="0",
        PYTHONUTF8="0",
        PYTHONIOENCODING="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(path)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr

    pipeline = Pipeline()
    pipeline.load(str(path))
    assert pipeline.modules()[-1].notes == [NOTE]
