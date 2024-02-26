# -*- mode: python -*-

import os.path

import PyInstaller.utils.hooks

from cellprofiler import __version__ as cp_version

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
    ("../../cellprofiler/data/icons/*", "cellprofiler/data/icons"),
]

# Handle symlinks more gracefully.
source = os.path.realpath(os.environ["JAVA_HOME"])

for subdir, dirs, files in os.walk(source):
    if 'Contents/' in subdir:
        if len(subdir.split('Contents/')) >1:
            _, subdir_split = subdir.split('Contents/')
            for file in files:
                datas += [(os.path.join(subdir, file), subdir_split)]

hiddenimports = []

hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numpy.core")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("pandas")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("scipy.special")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("wx")
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.gui')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('cellprofiler_core.modules')
hiddenimports += PyInstaller.utils.hooks.collect_submodules('skimage.io._plugins')
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.feature")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.filters")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("skimage.filters.rank")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sklearn")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("sentry_sdk.integrations.stdlib")
hiddenimports += PyInstaller.utils.hooks.collect_submodules("numcodecs")

hiddenimports += [
    "pkg_resources.py2_warn",
    "pywt._extensions._cwt",
    "sentry_sdk.integrations.excepthook",
    "sentry_sdk.integrations.stdlib",
    "sentry_sdk.integrations.modules",
    "sentry_sdk.integrations.threading",
]

print(f"De-duplicating {len(hiddenimports)} hidden imports...")
hiddenimports = list(set(hiddenimports))
print(f"...Complete! {len(hiddenimports)} found")

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
        'CellProfilerLauncher.py'
    ],
    binaries=binaries,
    cipher=block_cipher,
    datas=datas,
    excludes=excludes,
    hiddenimports=hiddenimports,
    hookspath=[],
    pathex=[
        'CellProfiler', '.', '../core'
    ],
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False
)

libpng_pathname = PyInstaller.utils.hooks.get_homebrew_path("libpng")
libpng_pathname = os.path.join(libpng_pathname, "lib", "libpng16.16.dylib")

java_pathname = os.path.join(os.environ["JAVA_HOME"], "lib/libjava.dylib")

manual_binaries = [
    ("libpng16.16.dylib", libpng_pathname, "BINARY"),
    ("libjava.dylib", java_pathname, "BINARY")
]

binaries_to_exclude = set([binary[0] for binary in manual_binaries])


def check_binary(x):
    name, path, bintype = x
    if name in binaries_to_exclude:
        print(f"Removing {name} at {path} ({bintype})")
        return False
    return True


a.binaries = [b for b in a.binaries if check_binary(b)]

a.binaries += manual_binaries

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# entitlements file must be provided as absolute path going into codesign.
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cellprofilerapp",
    debug=True,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=os.path.join(os.getcwd(),
                                   'distribution/macos/entitlements.plist')
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    icon="../../cellprofiler/data/icons/CellProfiler.icns",
    name="CellProfiler"
)

app = BUNDLE(
    coll,
    name="CellProfiler+AI.app",
    icon="../../cellprofiler/data/icons/CellProfiler.icns",
    bundle_identifier=None,
    version=cp_version,
    info_plist={
        'CFBundleDevelopmentRegion': 'English',
        'CFBundleExecutable': 'MacOS/cellprofilerapp',
        'CFBundleIdentifier': 'org.cellprofiler.CellProfiler',
        'CFBundleVersion': cp_version,
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeExtensions': ['cppipe'],
                'CFBundleTypeName': 'CellProfiler pipeline',
                'CFBundleTypeIconFile': 'CellProfiler.icns',
                'CFBundleTypeRole': 'Editor',
                },
            {
                'CFBundleTypeExtensions': ['cpproj'],
                'CFBundleTypeName': 'CellProfiler project',
                'CFBundleTypeIconFile': 'CellProfiler.icns',
                'CFBundleTypeRole': 'Editor',
                }
            ],
        'LSApplicationCategoryType': '',
        'LSBackgroundOnly': False,
        'LSEnvironment': {
            'JAVA_HOME': './Contents',
        },
        'NSHighResolutionCapable': True,
        'NSPrincipalClass': 'NSApplication'
    },
)
