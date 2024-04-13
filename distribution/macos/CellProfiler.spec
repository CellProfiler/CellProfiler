# -*- mode: python -*-

import os.path

import importlib.resources

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, get_homebrew_path, copy_metadata
from PyInstaller.compat import is_pure_conda

binaries = []

block_cipher = None

datas = []

datas += collect_data_files("cellprofiler")
datas += collect_data_files("skimage.io._plugins")
# for skimage/feature/orb_descriptor_positions.txt
datas += collect_data_files("skimage.feature")

datas += [
    ("../../src/frontend/cellprofiler/data/images/*", "cellprofiler/data/images"),
    ("../../src/frontend/cellprofiler/data/icons/*", "cellprofiler/data/icons"),
]

for subdir, dirs, files in os.walk(os.environ["JAVA_HOME"]):
    if 'Contents/' in subdir:
        if len(subdir.split('Contents/')) >1:
            _, subdir_split = subdir.split('Contents/')
            for file in files:
                datas += [(os.path.join(subdir, file), subdir_split)]

# needed for packages getting version at runtime via importlib.metadata
# https://pyinstaller.org/en/stable/hooks.html#PyInstaller.utils.hooks.copy_metadata
copied_metadata = copy_metadata('scyjava')

datas += copied_metadata

hiddenimports = []

hiddenimports += collect_submodules('cellprofiler.modules')
hiddenimports += collect_submodules('cellprofiler_core.modules')
hiddenimports += collect_submodules('cellprofiler_core.readers')

hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("numpy.core")
hiddenimports += collect_submodules("pandas")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("scipy.special")
hiddenimports += collect_submodules("wx")
hiddenimports += collect_submodules('skimage.io._plugins')
hiddenimports += collect_submodules("skimage.feature")
hiddenimports += collect_submodules("skimage.filters")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("sentry_sdk")
hiddenimports += collect_submodules("sentry_sdk.integrations")
hiddenimports += collect_submodules("sentry_sdk.integrations.stdlib")

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
    ['CellProfiler.py'],
    binaries=binaries,
    cipher=block_cipher,
    datas=datas,
    excludes=excludes,
    hiddenimports=hiddenimports,
    hookspath=[],
    pathex=['../../src/frontend', '../../src/subpackages/core', '../../src/subpackages/library'],
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False
)

libpng_pathname = get_homebrew_path("libpng")
libpng_pathname = os.path.join(libpng_pathname, "lib", "libpng16.16.dylib")

java_pathname = os.path.join(os.environ["JAVA_HOME"], "lib/libjava.dylib")

a.binaries += [
    ("libpng16.16.dylib", libpng_pathname, "BINARY"),
    ("libjava.dylib", java_pathname, "BINARY")
]

exclude_binaries = [
    ('libpng16.16.dylib', str(importlib.resources.files('matplotlib') / '.dylibs/libpng16.16.dylib'), 'BINARY'),
]

# in conda, numpy 1.24 stores its dylibs in <conda_path>/envs/<env_name>/lib/libgfortran.5.dylib
# scipy stores in <conda_path>/envs/C<env_name>/lib/python3.9/site-packages/scipy/.dylibs/libgfortran.5.dylib
# causing scipy's (incorrectly specified architecture) build to be used instead of numpys (because of some internal pyintaller reason, I guess)
# correct for that here
if is_pure_conda:
    from PyInstaller.utils.hooks import conda_support

    if conda_support.distributions.get('numpy') is not None:
        numpy_dyn_libs = [d[0] for d in conda_support.collect_dynamic_libs("numpy", dependencies=True)]
        libgfortran_path = list(filter(lambda d: 'libgfortran.5.dylib' in d, numpy_dyn_libs34j))
        libgfortran_path = libgfortran_path[0]
        # in python 3.9 specifically, it's compiled for arm64 (`lipo -info`), remove it
        exclude_binaries += [('scipy/.dylibs/libgfortran.5.dylib', str(importlib.resources.files('scipy') / '.dylibs/libgfortran.5.dylib'), 'BINARY')]
        # and replace it with the one numpy uses, which is x86_64
        a.binaries += [('scipy/.dylibs/libgfortran.5.dylib', libgfortran_path, 'BINARY')]

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
    icon="../../src/frontend/cellprofiler/data/icons/CellProfiler.icns",
    name="CellProfiler.app"
)

app = BUNDLE(
    coll,
    name="CellProfiler.app",
    icon="../../src/frontend/cellprofiler/data/icons/CellProfiler.icns",
    bundle_identifier=None
)
