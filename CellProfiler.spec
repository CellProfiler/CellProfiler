# -*- mode: python -*-

block_cipher = None

import PyInstaller.utils.hooks

cellprofiler_modules = PyInstaller.utils.hooks.collect_submodules('cellprofiler.modules')
pywt_extensions = PyInstaller.utils.hooks.collect_submodules('pywt._extensions')
zmq_backend = PyInstaller.utils.hooks.collect_submodules('zmq.backend')

a = Analysis(['CellProfiler.py'],
             pathex=['C:\\Users\\Public\\Documents\\CellProfiler',
                     'C:\\Python27\\Lib\\site-packages'],
             binaries=None,
             datas=[('C:\\Users\\Public\\Documents\\CellProfiler\\artwork', 'artwork'),
                    ('C:\\Users\\Public\\Documents\\CellProfiler\\cellprofiler\\VERSION', 'cellprofiler'),
                    ('C:\\Python27\\Lib\\site-packages\\prokaryote\\prokaryote.jar', 'prokaryote'),
                    ('C:\\Python27\\Lib\\site-packages\\javabridge\\jars', 'javabridge\\jars')],
             hiddenimports=cellprofiler_modules + pywt_extensions + zmq_backend,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=True,
             win_private_assemblies=True,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='CellProfiler',
          debug=True,
          strip=False,
          upx=True,
          console=True )
