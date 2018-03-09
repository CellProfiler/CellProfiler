# -*- mode: python -*-

import os
import os.path

import PyInstaller.compat
import PyInstaller.utils.hooks

binaries = []

block_cipher = None

datas = []

datas += PyInstaller.utils.hooks.collect_data_files("bioformats")
datas += PyInstaller.utils.hooks.collect_data_files("cellprofiler")
datas += PyInstaller.utils.hooks.collect_data_files("javabridge")
datas += PyInstaller.utils.hooks.collect_data_files("prokaryote")
datas += PyInstaller.utils.hooks.collect_data_files("skimage.io._plugins")

datas += [
    ("CellProfiler/cellprofiler/data/images/*", "cellprofiler/data/images"),
    ("CellProfiler/cellprofiler/data/icons/*", "cellprofiler/data/icons")
]

hiddenimports = []

hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')

hiddenimports += [
    "scipy._lib.messagestream",
    "pywt._extensions._cwt"
]

a = Analysis(
    [
        'CellProfiler.py'
    ],
    binaries=binaries,
    cipher=block_cipher,
    datas=datas,
    excludes=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    pathex=[
        'CellProfiler'
    ],
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    console=False,
    debug=False,
    icon="./CellProfiler/cellprofiler/data/icons/CellProfiler.ico",
    name="CellProfiler",
    strip=False,
    upx=True
)
