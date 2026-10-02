"""Windowed startup diagnostics without launching an installer or building an EXE."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StartupLoggingTests(unittest.TestCase):
    def run_script(self, folder, body):
        script = ('import sys\nfrom pathlib import Path\n'
                  f'sys.path.insert(0, {str(ROOT / "src")!r})\n'
                  'from startup_logging import configure_windowed_logging\n'
                  f'folder = Path({str(folder)!r})\n' + body)
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, encoding='utf-8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_windowed_streams_write_unicode_errors_and_native_file_descriptors(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_script(folder, '''
import logging
sys.stdout = sys.stderr = None
assert configure_windowed_logging(folder) == folder / 'Meshropractor.log'
assert sys.stdout.fileno() >= 0
assert sys.stderr.buffer is not None
print('Проверка запуска без консоли', flush=True)
try:
    raise ValueError('Ошибка геометрии')
except ValueError:
    logging.exception('Background failure')
''')
            text = (Path(folder) / 'Meshropractor.log').read_text(encoding='utf-8')
            self.assertIn('Проверка запуска без консоли', text)
            self.assertIn('ValueError: Ошибка геометрии', text)
            self.assertIn('Traceback', text)

    def test_existing_console_streams_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_script(folder, '''
original = sys.stdout, sys.stderr
assert configure_windowed_logging(folder) is None
assert (sys.stdout, sys.stderr) == original
print('console preserved')
''')
            self.assertIn('console preserved', result.stdout)
            self.assertFalse((Path(folder) / 'Meshropractor.log').exists())

    def test_frozen_linux_desktop_logs_to_xdg_state_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_script(folder, '''
import os
os.environ['XDG_STATE_HOME'] = str(folder)
sys.platform = 'linux'
sys.frozen = True
assert configure_windowed_logging() == folder / 'Meshropractor/logs/Meshropractor.log'
print('desktop launch diagnostics')
''')
            self.assertEqual(result.stdout, '')
            self.assertIn('desktop launch diagnostics',
                          (Path(folder) / 'Meshropractor/logs/Meshropractor.log').read_text())

    def test_only_the_missing_stream_is_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_script(folder, '''
original = sys.stdout
sys.stderr = None
configure_windowed_logging(folder)
assert sys.stdout is original
sys.stderr.write('error output\\n')
''')
            self.assertIn('error output', (Path(folder) / 'Meshropractor.log').read_text())

    def test_large_log_rotates_at_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'Meshropractor.log'
            log.write_bytes(b'x' * (5 * 1024 * 1024))
            self.run_script(folder, '''
sys.stdout = sys.stderr = None
configure_windowed_logging(folder)
print('new session')
''')
            self.assertEqual(log.with_suffix('.log.1').stat().st_size, 5 * 1024 * 1024)
            self.assertIn('new session', log.read_text())
            self.assertLess(log.stat().st_size, 1024)

    def test_unavailable_profile_uses_temp_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_script(folder, '''
import startup_logging
blocked = folder / 'file_instead_of_directory'
blocked.write_text('keep')
startup_logging.tempfile.gettempdir = lambda: str(folder)
sys.stdout = sys.stderr = None
result = configure_windowed_logging(blocked)
assert result == folder / 'Meshropractor' / 'logs' / 'Meshropractor.log'
print('fallback works')
''')
            text = (Path(folder) / 'Meshropractor' / 'logs' / 'Meshropractor.log').read_text()
            self.assertIn('fallback works', text)

    def test_failed_first_write_uses_temp_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_script(folder, '''
import io
import startup_logging
from unittest.mock import patch
class FullDisk(io.StringIO):
    def write(self, value):
        raise OSError('Disk full')
failed = FullDisk()
original_open = Path.open
primary = folder / 'profile'
def open_log(path, *args, **kwargs):
    if path.parent == primary:
        return failed
    return original_open(path, *args, **kwargs)
startup_logging.tempfile.gettempdir = lambda: str(folder)
sys.stdout = sys.stderr = None
with patch.object(Path, 'open', open_log):
    result = configure_windowed_logging(primary)
assert result == folder / 'Meshropractor' / 'logs' / 'Meshropractor.log'
assert failed.closed
print('write fallback works')
''')
            text = (Path(folder) / 'Meshropractor' / 'logs' / 'Meshropractor.log').read_text()
            self.assertIn('write fallback works', text)


if __name__ == '__main__':
    unittest.main()
