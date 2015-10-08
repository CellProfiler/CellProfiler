# -*- mode: python -*-

import glob

def data(prefix, *filenames, **kw):
    import os

    def datafile(path, strip_path=True):
        parts = path.split('/')
        path = name = os.path.join(*parts)
        if strip_path:
            name = os.path.basename(path)
        return prefix + name, path, 'DATA'

    strip_path = kw.get('strip_path', True)
    return TOC(
        datafile(filename, strip_path=strip_path)
        for filename in filenames
        if os.path.isfile(filename))

def get_pandas_path():
    import pandas
    pandas_path = pandas.__path__[0]
    return pandas_path

artwork = data("./artwork/", *glob.glob("./artwork/*"))

JARS = data("./imagej/jars/", *glob.glob("./imagej/jars/*"))

modules = data("./cellprofiler/modules/", *glob.glob("./cellprofiler/modules/*"))

block_cipher = None

a = Analysis(
    [
        'CellProfiler.py'
    ],
    binaries=None,
    cipher=block_cipher,
    datas=None,
    excludes=None,
    hiddenimports=[
        'cellh5',
        'cellprofiler.utilities.rules',
        'centrosome.bg_compensate',
        'centrosome.fastemd',
        'centrosome.haralick',
        'centrosome.lapjv',
        'centrosome.propagate',
        'centrosome.radial_power_spectrum',
        'centrosome.watershed',
        'centrosome.watershed',
        'centrosome.zernike',
        'contrib.english',
        'imagej.imageplus',
        'imagej.imageprocessor',
        'imagej.macros',
        'scipy.fftpack',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'sklearn.utils.sparsetools._graph_validation',
        'sklearn.utils.sparsetools._graph_tools',
        'sklearn.utils.lgamma',
        'sklearn.utils.weight_vector',
        'sklearn.tree._utils',
        'typedefs',
    ],
    hookspath=None,
    pathex=[
        './CellProfiler'
    ],
    runtime_hooks=None,
    win_no_prefer_redirects=None,
    win_private_assemblies=None,
)

dict_tree = Tree(get_pandas_path(), prefix='pandas', excludes=["*.pyc"])
a.datas += dict_tree
a.binaries = filter(lambda x: 'pandas' not in x[0], a.binaries)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='CellProfilerApp',
    debug=False,
    strip=None,
    upx=True,
    console=False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    artwork,
    JARS,
    modules,
    strip=None,
    upx=True,
    name='CellProfiler'
)

app = BUNDLE(
    coll,
    bundle_identifier="org.cellprofiler.CellProfiler",
    icon="./artwork/CellProfilerIcon.png",
    name='CellProfiler.app',
)
