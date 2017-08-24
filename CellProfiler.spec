# -*- mode: python -*-

import glob
import os.path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("skimage.io._plugins")
hiddenimports = collect_submodules('skimage.io._plugins')

options = [ ('v', None, 'OPTION'), ('W ignore', None, 'OPTION') ]

block_cipher = None

modules = ["cellprofiler.modules." + os.path.splitext(os.path.basename(pathname))[0] for pathname in glob.glob("cellprofiler/modules/*.py")]

a = Analysis(
  [
    'CellProfiler.py'
  ],
  binaries=[],
  cipher=block_cipher,
  datas=datas + [
    ('cellprofiler', 'cellprofiler')
  ],
  excludes=[],
  hiddenimports=hiddenimports + modules + [
    "PIL",
    "imread",
    "libtiff",
    "skimage.io",
    "imageio",
    "prokaryote",
    "pywt._extensions._cwt",
    "python-bioformats",
    "PyMySQL",
    "MySQL-python"
  ],
  hookspath=[],
  pathex=[
    '/Users/agoodman/Documents/com/github/CellProfiler/CellProfiler'
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

# pyinstaller -y CellProfiler.spec
# cp /usr/local/Cellar/libpng/1.6.31/lib/libpng16.16.dylib dist/CellProfiler.app/Contents/MacOS
# cp -R /usr/local/lib/python2.7/site-packages/prokaryote dist/CellProfiler.app/Contents/MacOS
# rm dist/CellProfiler.app/Contents/MacOS/libwx_*
# cp -R /usr/local/lib/python2.7/site-packages/prokaryote dist/CellProfiler.app/Contents/MacOS
# cp -R /usr/local/lib/python2.7/site-packages/bioformats dist/CellProfiler.app/Contents/MacOS
# cp -R /usr/local/lib/python2.7/site-packages/javabridge dist/CellProfiler.app/Contents/MacOS
# ./dist/CellProfiler.app/Contents/MacOS/CellProfiler-App 
