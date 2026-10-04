#!/bin/bash
set -euo pipefail
# Run only inside a disposable Debian/Ubuntu container, as root.
package=${1:?path to .deb}
report=${2:?writable output directory}
test -e /.dockerenv || { echo 'Use a disposable Docker container'; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends "$package" xvfb xauth xdotool
! command -v python3 || { echo 'Validation image must not have Python installed'; exit 1; }
id mesh-test >/dev/null 2>&1 || useradd -m -s /bin/bash mesh-test
mkdir -p "$report"
chmod 777 "$report"
runuser -u mesh-test -- xvfb-run -a -s '-screen 0 1600x1000x24' \
    env LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=2 meshropractor --self-test "$report"
# Test the ordinary desktop startup too, including XDG file logging.
runuser -u mesh-test -- xvfb-run -a -s '-screen 0 1600x1000x24' \
    env LIBGL_ALWAYS_SOFTWARE=1 OMP_NUM_THREADS=2 bash -c '
    meshropractor & pid=$!
    trap "kill $pid 2>/dev/null || true" EXIT
    for attempt in $(seq 1 30); do
        if xdotool search --onlyvisible --name "^Meshropractor (—|-)" >/dev/null; then
            test -s "$HOME/.local/state/Meshropractor/logs/Meshropractor.log"
            echo DESKTOP_STARTUP_OK
            exit 0
        fi
        kill -0 "$pid" 2>/dev/null || exit 1
        sleep 1
    done
    echo Desktop window did not open
    exit 1'
dpkg-query -W -f='${Package} ${Version} ${Architecture}\n' meshropractor
echo CLEAN_INSTALL_OK
