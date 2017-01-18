# -*- mode: python -*-

block_cipher = None


a = Analysis(['-', 'v', 'CellProfiler.py'],
             pathex=['C:\\Users\\Public\\Documents\\CellProfiler'],
             binaries=None,
             datas=None,
             hiddenimports=[],
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
          name='-',
          debug=True,
          strip=False,
          upx=True,
          console=True )
