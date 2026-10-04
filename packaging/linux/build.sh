#!/bin/bash
set -euo pipefail
# Sources are mounted read-only; Linux outputs never overwrite the Windows build.
cd /build
build_version=$(tr -d '\r\n' </workspace/VERSION)
cp -a /workspace/src /workspace/assets /workspace/licenses /workspace/scripts .
cp /workspace/VERSION /workspace/Meshropractor.spec /workspace/RepairEngine.spec .
python -m PyInstaller --noconfirm RepairEngine.spec
python -m PyInstaller --noconfirm Meshropractor.spec
xvfb-run -a -s '-screen 0 1600x1000x24' env QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
    OMP_NUM_THREADS=2 dist/Meshropractor/Meshropractor --self-test "/out/build-validation-${build_version}"
python /workspace/packaging/linux/package.py /build/dist/Meshropractor /out
