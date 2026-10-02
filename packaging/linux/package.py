"""Assemble a Debian package from a Linux PyInstaller onedir distribution."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = (
    'libc6 (>= 2.36)', 'libstdc++6 (>= 12)', 'libgl1', 'libegl1', 'libopengl0',
    'libglx-mesa0', 'libegl-mesa0', 'libgl1-mesa-dri',
    'libglib2.0-0', 'libgomp1', 'libx11-6', 'libxext6', 'libxrender1',
    'libxfixes3', 'libxrandr2', 'libxcursor1', 'libxi6', 'libxkbcommon0',
    'libxkbcommon-x11-0', 'libxcb1', 'libxcb-cursor0', 'libxcb-icccm4',
    'libxcb-image0', 'libxcb-keysyms1', 'libxcb-randr0', 'libxcb-render0',
    'libxcb-render-util0', 'libxcb-shape0', 'libxcb-shm0', 'libxcb-sync1',
    'libxcb-xfixes0', 'libxcb-xkb1', 'libfontconfig1', 'libfreetype6',
    'libdbus-1-3', 'libnss3', 'libasound2 | libasound2t64', 'fonts-dejavu-core',
)


def main():
    if sys.platform != 'linux':
        raise SystemExit('Build this package on Linux, not from a Windows distribution.')
    bundle, output = map(lambda p: Path(p).resolve(), sys.argv[1:3])
    version = (ROOT / 'VERSION').read_text().strip()
    if not (bundle / 'Meshropractor').is_file():
        raise SystemExit('Linux application executable is missing')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='meshropractor-deb-') as temporary:
        staging = Path(temporary)
        # TemporaryDirectory starts as 0700; a package root must be traversable.
        staging.chmod(0o755)
        shutil.copytree(bundle, staging / 'opt/meshropractor', symlinks=True)
        launcher = staging / 'usr/bin/meshropractor'
        launcher.parent.mkdir(parents=True)
        launcher.write_text('#!/bin/sh\n'
                            '# VTK uses X11; Wayland sessions need XWayland.\n'
                            'export QT_QPA_PLATFORM=xcb\n'
                            'exec /opt/meshropractor/Meshropractor "$@"\n')
        launcher.chmod(0o755)
        desktop = staging / 'usr/share/applications/meshropractor.desktop'
        desktop.parent.mkdir(parents=True)
        desktop.write_text('[Desktop Entry]\nType=Application\nName=Meshropractor\n'
                           'Comment=3D print preparation and deformation compensation\n'
                           'Exec=meshropractor\nIcon=meshropractor\nTerminal=false\n'
                           'Categories=Graphics;3DGraphics;Engineering;\nStartupNotify=true\n')
        icon = staging / 'usr/share/pixmaps/meshropractor.png'
        icon.parent.mkdir(parents=True)
        shutil.copy2(ROOT / 'assets/logo.png', icon)
        docs = staging / 'usr/share/doc/meshropractor'
        docs.mkdir(parents=True)
        shutil.copytree(ROOT / 'licenses/repair-engine', docs / 'repair-engine')
        control = staging / 'DEBIAN/control'
        control.parent.mkdir(parents=True)
        size = sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()) // 1024
        control.write_text(f'Package: meshropractor\nVersion: {version}\nArchitecture: amd64\n'
                           'Maintainer: Meshropractor Team <theboogie888@gmail.com>\n'
                           f'Installed-Size: {size}\nDepends: {", ".join(RUNTIME)}\n'
                           'Recommends: xwayland\nSection: graphics\nPriority: optional\n'
                           'Homepage: https://github.com/B0ogie888/Meshropractor\n'
                           'Description: 3D printing preparation and deformation compensation\n'
                           ' Native STEP/BREP, mesh repair, slicing preparation and scan comparison.\n')
        package = output / f'meshropractor_{version}_amd64.deb'
        subprocess.run(['dpkg-deb', '--root-owner-group', '-Zxz', '-z3', '--build',
                        str(staging), str(package)], check=True)
    with package.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    (output / (package.name + '.sha256')).write_text(f'{digest}  {package.name}\n')
    print(f'PACKAGE_OK: {package} ({package.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
