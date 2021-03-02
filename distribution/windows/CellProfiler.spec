# -*- mode: python ; coding: utf-8 -*-

import os
import os.path
import pkgutil

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
    ("../../cellprofiler/data/images/*", "cellprofiler/data/images"),
    ("../../cellprofiler/data/icons/*", "cellprofiler/data/icons")
]

hiddenimports = []

for module_name in list(pkgutil.iter_modules()):
    if module_name[1].startswith("omero_"):
        hiddenimports.append(module_name[1])
hiddenimports += PyInstaller.utils.hooks.collect_submodules("Ice")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("IceImport")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("omero")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("omero.all")

hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler_core.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.utilities')
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy.special")
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.feature")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.filters")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.filters.rank")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations.modules")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations.threading")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations.stdlib")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations.excepthook")


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
		  icon='../../cellprofiler/data/icons/CellProfiler.ico',
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='CellProfiler')
