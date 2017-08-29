# -*- mode: python -*-

import os.path

import PyInstaller.utils.hooks

datas = []

datas += PyInstaller.utils.hooks.collect_data_files("bioformats")
datas += PyInstaller.utils.hooks.collect_data_files("cellprofiler")
datas += PyInstaller.utils.hooks.collect_data_files("javabridge")
datas += PyInstaller.utils.hooks.collect_data_files("prokaryote")
datas += PyInstaller.utils.hooks.collect_data_files("skimage.io._plugins")

datas += [("cellprofiler/data/images/*", "cellprofiler/data/images")]

excludes = []

excludes += [
    "botocore",
    "PyQt5",
    "sphinx",
    "tcl",
    "tk"
]

hiddenimports = []

hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')

options = [("v", None, "OPTION"), ("W ignore", None, "OPTION")]

block_cipher = None

hiddenimports += [
    "imageio",
    "prokaryote",
    "pywt._extensions._cwt",
    "zmq",
    "zmq.backend.cython"
]

a = Analysis(
    [
        "CellProfiler.py"
    ],
    binaries=[],
    cipher=block_cipher,
    datas=datas,
    excludes=[
        "zmq.libzmq"
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    pathex=[
        "."
    ],
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False
)

a.binaries = [x for x in a.binaries if not x[0].startswith("libzmq.pyd")]

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    console=True,
    debug=False,
    exclude_binaries=True,
    icon="./cellprofiler/data/images/CellProfilerIcon.icns",
    name="cp",
    strip=False,
    upx=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="CellProfiler",
    strip=False,
    upx=True
)

app = BUNDLE(
    coll,
    bundle_identifier=None,
    icon="./cellprofiler/data/images/CellProfilerIcon.icns",
    name="CellProfiler.app"
)
