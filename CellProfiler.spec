# -*- mode: python -*-

import glob
import os.path

import PyInstaller.utils.hooks
import bioformats
import javabridge
import prokaryote

datas = PyInstaller.utils.hooks.collect_data_files("skimage.io._plugins")

hiddenimports = PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')

options = [('v', None, 'OPTION'), ('W ignore', None, 'OPTION')]

block_cipher = None

for pathname in glob.glob("cellprofiler/modules/*.py"):
  name = os.path.splitext(os.path.basename(pathname))[0]

  hiddenimport = "cellprofiler.modules." + name

  hiddenimports.append(hiddenimport)

hiddenimports += [
  "pywt._extensions._cwt"
]

a = Analysis(
  [
    'CellProfiler.py'
  ],
  binaries=[],
  cipher=block_cipher,
  datas=datas + [
    ('cellprofiler', 'cellprofiler'),
    (os.path.dirname(prokaryote.__file__), "prokaryote"),
    (os.path.dirname(bioformats.__file__), "bioformats"),
    (os.path.dirname(javabridge.__file__), "javabridge")
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
