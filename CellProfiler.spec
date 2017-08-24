# -*- mode: python -*-

import glob
import os.path

import cellprofiler
import PyInstaller.utils.hooks
import bioformats
import javabridge
import prokaryote

datas = PyInstaller.utils.hooks.collect_data_files("skimage.io._plugins")

hiddenimports = PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')

options = [('v', None, 'OPTION'), ('W ignore', None, 'OPTION')]

block_cipher = None

pattern = os.path.join(os.path.dirname(cellprofiler.__file__), "modules", "*.py")

for pathname in glob.glob(pattern):
    name, _ = os.path.splitext(os.path.basename(pathname))

    module = "cellprofiler.modules." + name

    hiddenimports.append(module)

hiddenimports += [
    "imageio",
    "prokaryote",
    "pywt._extensions._cwt",
    "zmq",
    "zmq.backend.cython"
]

a = Analysis(
    [
        'CellProfiler.py'
    ],
    binaries=[],
    cipher=block_cipher,
    datas=datas + [
        (os.path.dirname(bioformats.__file__), "bioformats"),
        (os.path.dirname(cellprofiler.__file__), "cellprofiler"),
        (os.path.dirname(javabridge.__file__), "javabridge"),
        (os.path.dirname(prokaryote.__file__), "prokaryote")
    ],
    excludes=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    pathex=[
        '.'
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
    # options,
    console=False,
    debug=False,
    exclude_binaries=True,
    icon=os.path.join("cellprofiler", "data", "CellProfilerIcon.ico"),
    name='CellProfiler-App',
    strip=False,
    upx=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='CellProfiler',
    strip=False,
    upx=True
)

app = BUNDLE(
    coll,
    bundle_identifier=None,
    icon=None,
    name='CellProfiler.app'
)
