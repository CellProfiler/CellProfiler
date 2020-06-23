# -*- mode: python -*-

import os.path

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

hiddenimports = []

hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy.core")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("pandas")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy.special")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("wx")
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.gui')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')

hiddenimports += [
    "pkg_resources.py2_warn",
    "pywt._extensions._cwt"
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

java_pathname = os.path.join(os.environ["JAVA_HOME"], "jre/lib/server/libjvm.dylib")

a.binaries += [
    ("libpng16.16.dylib", libpng_pathname, "BINARY"),
    # ("libjvm.dylib", java_pathname, "BINARY")
]

exclude_binaries = [
    ('libpng16.16.dylib', '/usr/local/lib/python2.7/site-packages/matplotlib/.dylibs/libpng16.16.dylib', 'BINARY'),
    # ('libwx_osx_cocoau_webview-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_webview-3.0.dylib', 'BINARY'),
    # ('libwx_osx_cocoau_html-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_html-3.0.dylib', 'BINARY'),
    # ('libwx_osx_cocoau_xrc-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_xrc-3.0.dylib', 'BINARY'),
    # ('libwx_osx_cocoau_core-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_core-3.0.dylib', 'BINARY'),
    # ('libwx_osx_cocoau_adv-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_adv-3.0.dylib', 'BINARY'),
    # ('libwx_osx_cocoau_qa-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_qa-3.0.dylib', 'BINARY'),
    # ('libwx_baseu_xml-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_baseu_xml-3.0.dylib', 'BINARY'),
    # ('libwx_baseu_net-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_baseu_net-3.0.dylib', 'BINARY'),
    # ('libwx_baseu-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_baseu-3.0.dylib', 'BINARY'),
    # ('libwx_osx_cocoau_stc-3.0.dylib', '/usr/local/opt/wxmac/lib/libwx_osx_cocoau_stc-3.0.dylib', 'BINARY')
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
