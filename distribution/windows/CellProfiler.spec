# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

binaries = []

block_cipher = None

datas = []

datas += collect_data_files("cellprofiler")
datas += collect_data_files("skimage.io._plugins")
# for skimage/feature/orb_descriptor_positions.txt
datas += collect_data_files("skimage.feature")

datas += [
    ("../../src/frontend/cellprofiler/data/images/*", "cellprofiler/data/images"),
    ("../../src/frontend/cellprofiler/data/icons/*", "cellprofiler/data/icons")
]

# needed for packages getting version at runtime via importlib.metadata
# https://pyinstaller.org/en/stable/hooks.html#PyInstaller.utils.hooks.copy_metadata
copied_metadata = copy_metadata('scyjava')

datas += copied_metadata

hiddenimports = []

hiddenimports += collect_submodules('cellprofiler.modules')
hiddenimports += collect_submodules('cellprofiler_core.modules')
hiddenimports += collect_submodules('cellprofiler_core.readers')
hiddenimports += collect_submodules('cellprofiler.utilities')

hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("scipy.special")
hiddenimports += collect_submodules('skimage.io._plugins')
hiddenimports += collect_submodules("skimage.feature")
hiddenimports += collect_submodules("skimage.filters")
hiddenimports += collect_submodules("sklearn")
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

a = Analysis(['CellProfiler.py'],
             binaries=binaries,
             cipher=block_cipher,
             datas=datas,
             excludes=[],
             hiddenimports=hiddenimports,
             hookspath=[],
             noarchive=False,
             pathex=['../../src/frontend', '../../src/subpackages/core', '../../src/subpackages/library'],
             runtime_hooks=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False)

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
          icon='../../src/frontend/cellprofiler/data/icons/CellProfiler.ico',
          console=True)

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='CellProfiler')
