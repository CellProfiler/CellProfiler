# -*- mode: python -*-

import os.path
import pkgutil

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
    ("./CellProfiler/cellprofiler/data/images/*", "cellprofiler/data/images"),
    ("./CellProfiler/cellprofiler/data/icons/*", "cellprofiler/data/icons"),
]

for subdir, dirs, files in os.walk(os.environ["JAVA_HOME"]):
    if 'Contents/' in subdir:
        if len(subdir.split('Contents/')) >1:
            _, subdir_split = subdir.split('Contents/')
            for file in files:
                datas += [(os.path.join(subdir, file), subdir_split)]

hiddenimports = []

for module_name in list(pkgutil.iter_modules()):
    if module_name[1].startswith("omero_"):
        hiddenimports.append(module_name[1])
hiddenimports += PyInstaller.utils.hooks.collect_submodules("Ice")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("IceImport")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("omero")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("omero.all")

hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy.core")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("pandas")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy.special")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("wx")
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.gui')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler_core.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.feature")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.filters")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.filters.rank")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations.stdlib")

hiddenimports += [
    "pkg_resources.py2_warn",
    "pywt._extensions._cwt",
    "sentry_sdk.integrations.excepthook",
    "sentry_sdk.integrations.stdlib",
    "sentry_sdk.integrations.modules",
    "sentry_sdk.integrations.threading",
]

excludes = []

excludes += [
    "SimpleITK",
    "pyamg",
    "sphinx",
    "whoosh",
    "glib",
    "PyQt5.QtGui",
    "PyQt5.QtCore",
    "PyQt4.QtGui",
    "PyQt4.QtCore",
    "PySide.QtGui",
    "PySide.QtCore",
    "astropy",
    "pandas",
    "PyQt5",
    "PyQt4",
    "PySide",
    "PySide2",
    "gtk",
    "FixTk",
    "tcl",
    "tk",
    "_tkinter",
    "tkinter",
    "Tkinter"
]

a = Analysis(
    [
        'CellProfiler.py'
    ],
    binaries=binaries,
    cipher=block_cipher,
    datas=datas,
    excludes=excludes,
    hiddenimports=hiddenimports,
    hookspath=[],
    pathex=[
        'CellProfiler'
    ],
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False
)

libpng_pathname = PyInstaller.utils.hooks.get_homebrew_path("libpng")
libpng_pathname = os.path.join(libpng_pathname, "lib", "libpng16.16.dylib")

java_pathname = os.path.join(os.environ["JAVA_HOME"], "lib/libjava.dylib")

a.binaries += [
    ("libpng16.16.dylib", libpng_pathname, "BINARY"),
    ("libjava.dylib", java_pathname, "BINARY")
]

exclude_binaries = [
    ('libpng16.16.dylib', '/usr/local/lib/python3.8/site-packages/matplotlib/.dylibs/libpng16.16.dylib', 'BINARY'),
]

a.binaries = [binary for binary in a.binaries if binary not in exclude_binaries]

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cp",
    debug=True,
    strip=False,
    upx=True,
    console=True
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    icon="./CellProfiler/cellprofiler/data/icons/CellProfiler.icns",
    name="CellProfiler.app"
)

app = BUNDLE(
    coll,
    name="CellProfiler.app",
    icon="./CellProfiler/cellprofiler/data/icons/CellProfiler.icns",
    bundle_identifier=None
)
