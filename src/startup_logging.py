"""Keep diagnostics available when a desktop starts the GUI without a console."""
import faulthandler
import logging
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime


def configure_windowed_logging(directory=None):
    """Replace missing pythonw/PyInstaller streams before native imports.

    An existing terminal, IDE or redirected stream is deliberately preserved.
    The real file stream also supports ``fileno``/``buffer`` used by libraries.
    """
    desktop_linux = (sys.platform == 'linux' and getattr(sys, 'frozen', False)
                     and '--self-test' not in sys.argv)
    if sys.stdout is not None and sys.stderr is not None and not desktop_linux:
        return None

    if directory is not None:
        primary = Path(directory)
    elif sys.platform == 'linux':
        primary = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state') / 'Meshropractor/logs'
    else:
        primary = (Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData/Local')
                   / 'Meshropractor/logs')
    fallback = Path(tempfile.gettempdir()) / 'Meshropractor' / 'logs'
    log_file = None
    header = (f'\n--- Meshropractor startup {datetime.now().isoformat(timespec="seconds")} '
              f'(PID {os.getpid()}) ---\n')
    for folder in (primary, fallback):
        stream = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            candidate = folder / 'Meshropractor.log'
            # Keep the previous log instead of allowing diagnostics to grow forever.
            if candidate.exists() and candidate.stat().st_size >= 5 * 1024 * 1024:
                candidate.replace(candidate.with_suffix('.log.1'))
            stream = candidate.open('a', encoding='utf-8', errors='backslashreplace', buffering=1)
            # Opening can succeed on a full disk; test a real write before using it.
            stream.write(header)
            stream.flush()
            log_file = candidate
            break
        except OSError:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            continue
    else:
        # A read-only/full profile must not prevent the GUI from starting.
        stream = open(os.devnull, 'w', encoding='utf-8', buffering=1)

    if sys.stdout is None or desktop_linux:
        sys.stdout = stream
    if sys.stderr is None or desktop_linux:
        sys.stderr = stream
    logging.basicConfig(stream=sys.stderr,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        faulthandler.enable(file=stream)
    except (OSError, RuntimeError):
        pass
    return log_file
