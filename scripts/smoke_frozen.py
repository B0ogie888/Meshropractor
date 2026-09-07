"""Launch our packaged GUI from an unrelated working directory, then close normally."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import tempfile
import time

root = Path(__file__).resolve().parents[1]
exe = root / 'dist' / 'Meshropractor' / 'Meshropractor.exe'
log = root / 'output' / 'frozen-startup.log'
user32 = ctypes.windll.user32
callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]

with tempfile.TemporaryDirectory() as directory, log.open('wb') as stream:
    process = subprocess.Popen([str(exe)], cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        found = []
        @callback_type
        def check(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == process.pid and user32.IsWindowVisible(hwnd):
                title = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd, title, len(title))
                if title.value.startswith(('Meshropractor - ', 'Meshropractor — ')):
                    found.append(hwnd)
            return True
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not found:
            if process.poll() is not None: raise RuntimeError(f'Application exited during startup: {process.returncode}; see {log}')
            user32.EnumWindows(check, 0)
            time.sleep(.25)
        if not found: raise RuntimeError(f'Main window did not appear; see {log}')
        time.sleep(2)
        if process.poll() is not None: raise RuntimeError('Application failed after showing the window')
        user32.PostMessageW(found[0], 0x0010, 0, 0)  # WM_CLOSE, no forced shutdown on success
        code = process.wait(timeout=20)
        if code != 0: raise RuntimeError(f'Application exit code: {code}; see {log}')
        print('FROZEN_STARTUP_OK: main window opened outside the project folder and closed normally')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
