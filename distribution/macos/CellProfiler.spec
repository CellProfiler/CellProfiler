import importlib.metadata
import importlib.util
import os
import plistlib
import sys
from pathlib import Path

from packaging.version import Version
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    conda_support,
    copy_metadata,
)

spec_dir = Path(SPECPATH)
codesign_identity = os.environ.get("CP_CODESIGN_IDENTITY")
java_home = Path(os.environ.get("JAVA_HOME", Path(sys.prefix) / "lib/jvm")).resolve()
java_jars = Path(os.environ.get("CP_JAVA_JARS", spec_dir / "build/java-jars"))
if not (java_home / "lib/server/libjvm.dylib").is_file():
    raise RuntimeError(f"JAVA_HOME does not contain a macOS JVM: {java_home}")
if not list(java_jars.glob("*.jar")):
    raise RuntimeError("Run prepare_java.py before building the application")

datas = []
for package in ("cellprofiler", "cellprofiler_core", "cellprofiler_library", "skimage"):
    datas += collect_data_files(package)
datas += collect_data_files("cellprofiler.modules.plugins", include_py_files=True)
datas += copy_metadata("cellprofiler", recursive=True)
datas += copy_metadata("appose")
datas += [(str(java_jars), "java-jars")]

binaries = []
for path in conda_support.files("mysql-client", dependencies=False):
    if path.parent.as_posix() == "lib/plugin" and path.suffix == ".so":
        binaries.append((str(path.locate()), "mysql/plugin"))

hiddenimports = ["cellprofiler.__main__", "appose.python_worker", "pywt._extensions._cwt"]
for package in (
    "cellprofiler.modules", "cellprofiler_core.modules", "cellprofiler_core.readers",
    "scipy", "skimage", "sklearn", "sentry_sdk.integrations",
):
    hiddenimports += collect_submodules(package, filter=lambda name: ".tests" not in name)

version = Version(importlib.metadata.version("cellprofiler")).base_version
info_plist = plistlib.loads(
    (spec_dir / "Info.plist.template").read_text().replace("{{CP_VERSION}}", version).encode()
)
a = Analysis(
    [str(spec_dir / "CellProfiler.py")],
    pathex=[
        str(Path(importlib.util.find_spec(package).origin).parent.parent)
        for package in ("cellprofiler", "cellprofiler_core", "cellprofiler_library", "centriole")
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    runtime_hooks=[str(spec_dir / "runtime_hook.py")],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "sphinx"],
)
a.datas += [
    (str(Path("jre") / path.relative_to(java_home)), str(path), "DATA")
    for path in sorted(java_home.rglob("*"))
    if path.is_file()
]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cp",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    entitlements_file=str(spec_dir / "entitlements.plist"),
    codesign_identity=codesign_identity,
)
coll = COLLECT(exe, a.binaries, a.datas, name="CellProfiler")
app = BUNDLE(
    coll,
    name="CellProfiler.app",
    icon=str(spec_dir / "CellProfiler.icns"),
    bundle_identifier="org.cellprofiler.CellProfiler",
    version=version,
    info_plist=info_plist,
    codesign_identity=codesign_identity,
    entitlements_file=str(spec_dir / "entitlements.plist"),
)
