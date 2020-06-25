# -*- mode: python ; coding: utf-8 -*-

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
    ("../../CellProfiler/cellprofiler/data/images/*", "cellprofiler/data/images"),
    ("../../CellProfiler/cellprofiler/data/icons/*", "cellprofiler/data/icons")
]

hiddenimports = []

hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler_core.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.utilities')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')

hiddenimports += [
    "scipy._lib.messagestream",
    "pywt._extensions._cwt",
    "sklearn.utils.sparsetools"
]

a = Analysis(['CellProfiler.py'],
             pathex=['CellProfiler'],
             binaries=binaries,
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
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
		  icon='../../CellProfiler/cellprofiler/data/icons/CellProfiler.ico',
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='CellProfiler')
