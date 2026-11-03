# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_delvewheel_libs_directory,
    collect_submodules,
    copy_metadata,
)

spec_dir = Path(SPECPATH)
java_home = Path(os.environ["JAVA_HOME"]).resolve()
java_jars = Path(os.environ.get("CP_JAVA_JARS", spec_dir / "build/java-jars")).resolve()
if not (java_home / "bin/server/jvm.dll").is_file():
    raise RuntimeError(f"JAVA_HOME does not contain a Windows JVM: {java_home}")
if not list(java_jars.glob("*.jar")):
    raise RuntimeError("Run prepare_java.py before building the application")
source_roots = {
    package: Path(importlib.util.find_spec(package).origin).parent.parent
    for package in ("cellprofiler", "cellprofiler_core", "cellprofiler_library", "centriole")
}

binaries = []

block_cipher = None

datas = []

for package in ("cellprofiler", "cellprofiler_core", "cellprofiler_library", "skimage"):
    datas += collect_data_files(package)
datas += collect_data_files("cellprofiler.modules.plugins", include_py_files=True)
datas += [(str(java_jars), "java-jars")]
datas, binaries = collect_delvewheel_libs_directory("mahotas", datas=datas, binaries=binaries)
datas += collect_data_files("skimage.io._plugins")
# for skimage/feature/orb_descriptor_positions.txt
datas += collect_data_files("skimage.feature")

datas += [
    (str(source_roots["cellprofiler"] / "cellprofiler/data/images/*"), "cellprofiler/data/images"),
    (str(source_roots["cellprofiler"] / "cellprofiler/data/icons/*"), "cellprofiler/data/icons")
]

# needed for packages getting version at runtime via importlib.metadata
# https://pyinstaller.org/en/stable/hooks.html#PyInstaller.utils.hooks.copy_metadata
copied_metadata = copy_metadata("cellprofiler", recursive=True)

datas += copied_metadata

hiddenimports = []

hiddenimports += collect_submodules('centriole')
hiddenimports += collect_submodules('cellprofiler.modules')
hiddenimports += collect_submodules('cellprofiler_core.modules')
hiddenimports += collect_submodules('cellprofiler_core.readers')
hiddenimports += collect_submodules('cellprofiler.utilities')

hiddenimports += collect_submodules("scipy", filter=lambda name: ".tests" not in name)
hiddenimports += collect_submodules("scipy.special", filter=lambda name: ".tests" not in name)
hiddenimports += collect_submodules('skimage.io._plugins')
hiddenimports += collect_submodules("skimage.feature")
hiddenimports += collect_submodules("skimage.filters")
hiddenimports += collect_submodules("sklearn", filter=lambda name: ".tests" not in name)
hiddenimports += collect_submodules("sentry_sdk")
hiddenimports += collect_submodules("sentry_sdk.integrations")
hiddenimports += collect_submodules("sentry_sdk.integrations.modules")
hiddenimports += collect_submodules("sentry_sdk.integrations.threading")
hiddenimports += collect_submodules("sentry_sdk.integrations.stdlib")
hiddenimports += collect_submodules("sentry_sdk.integrations.excepthook")


hiddenimports += [
    "scipy._lib.messagestream",
    "pywt._extensions._cwt",
    "sklearn.utils.sparsetools",
    "sentry_sdk",
    "sentry_sdk.integrations.excepthook",
    "sentry_sdk.integrations.stdlib",
    "sentry_sdk.integrations.modules",
    "sentry_sdk.integrations.threading",
]

a = Analysis([str(spec_dir / 'CellProfiler.py')],
             binaries=binaries,
             cipher=block_cipher,
             datas=datas,
             excludes=[],
             hiddenimports=hiddenimports,
             hookspath=[],
             noarchive=False,
             pathex=[str(root) for root in source_roots.values()],
             runtime_hooks=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False)

a.datas += [
    (str(Path("jre") / path.relative_to(java_home)), str(path), "DATA")
    for path in sorted(java_home.rglob("*"))
    if path.is_file()
]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='CellProfiler',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          icon=str(source_roots['cellprofiler'] / 'cellprofiler/data/icons/CellProfiler.ico'),
          console=True)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='CellProfiler')
