# -*- mode: python ; coding: utf-8 -*-
"""Build the independent GPL repair helper before Meshropractor.spec."""
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

if sys.platform == 'win32':
    windows = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    os.environ['PATH'] = os.pathsep.join((str(Path(sys.executable).parent), sys.base_prefix,
                                        str(windows / 'System32'), str(windows)))
datas, binaries, hiddenimports = collect_all('pymeshfix')
datas += [('licenses/repair-engine', 'licenses'), ('src/repair_engine_cli.py', 'source')]
a = Analysis(['src/repair_engine_cli.py'], pathex=['src'], binaries=binaries,
             datas=datas, hiddenimports=hiddenimports, hookspath=[], hooksconfig={},
             runtime_hooks=[], excludes=['PySide6', 'PyQt5', 'PyQt6', 'pyvista', 'matplotlib', 'torch', 'open3d'],
             noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MeshRepairEngine',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='MeshRepairEngine')
