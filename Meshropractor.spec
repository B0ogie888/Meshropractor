# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

# Do not bundle unrelated DLLs from tools injected into the shell PATH (e.g. Poppler's
# ICU DLL has the same filename as Windows ICU but incompatible exported symbols).
# Package hooks provide their own library directories for Qt, Torch, OCP and Open3D.
if sys.platform == 'win32':
    windows = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    os.environ['PATH'] = os.pathsep.join((str(Path(sys.executable).parent), sys.base_prefix,
                                        str(windows / 'System32'), str(windows)))
from PyInstaller.utils.hooks import collect_all, collect_delvewheel_libs_directory

datas = [('assets', 'assets'), ('VERSION', '.')]
repair_engine = Path('dist/MeshRepairEngine')
helper_name = 'MeshRepairEngine.exe' if sys.platform == 'win32' else 'MeshRepairEngine'
if not (repair_engine / helper_name).is_file():
    raise RuntimeError('Build the repair helper first: python -m PyInstaller --noconfirm RepairEngine.spec')
datas.append((str(repair_engine), 'repair_engine'))
binaries = []
hiddenimports = []
tmp_ret = collect_all('OCP')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
if sys.platform == 'win32':
    datas, binaries = collect_delvewheel_libs_directory(
        'OCP', libdir_name='cadquery_ocp_novtk.libs', datas=datas, binaries=binaries)
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('open3d')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyvista')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('rtree')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('manifold3d')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/Meshropractor.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pymeshlab', 'pymeshfix'],
    noarchive=False,
    optimize=0,
)
if sys.platform == 'linux':
    # DRI drivers belong to the target OS and can require a newer C++ ABI.
    # The Debian package supplies libstdc++6 as a system dependency.
    a.binaries = [entry for entry in a.binaries
                  if Path(entry[0]).name != 'libstdc++.so.6']
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Meshropractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/logo.ico'] if sys.platform == 'win32' else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Meshropractor',
)
