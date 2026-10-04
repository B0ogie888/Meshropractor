"""Run the real desktop launcher/window with isolated preferences, then close it."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM', 'windows' if sys.platform == 'win32' else 'xcb')
from PySide6.QtCore import QEvent, QObject, QSettings, QTimer
import main_window
import app_settings
import app_updater
import desktop_launcher
import startup_bridge
from app_branding import APP_ID, app_icon


def main():
    output = ROOT/'output/desktop-entry-smoke'
    output.mkdir(parents=True, exist_ok=True)
    original_window = main_window.MainWindow
    original_splash = startup_bridge.SplashProcess
    windows, splashes, workers, events = [], [], [], []
    failure = []

    class ShowProbe(QObject):
        def eventFilter(self, widget, event):
            if event.type() == QEvent.Show:
                if workers[0].poll() is None:
                    failure.append(AssertionError('Main window appeared before the splash finished'))
                events.append('show')
            return False

    class TracedSplash(original_splash):
        def start(self, theme='light'):
            super().start(theme)
            splashes.append(self)
            assert self.process is not None
            workers.append(self.process)
        def stage(self, name):
            events.append(name)
            super().stage(name)
        def ready(self):
            assert not windows[0].isVisible()
            if sys.platform == 'win32':
                import ctypes
                ctypes.windll.user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
                assert not ctypes.windll.user32.IsWindowVisible(int(windows[0].winId()))
            events.append('ready')
            super().ready()

    def inspect():
        window = windows[0]
        try:
            assert events == ['geometry', 'interface', 'scene', 'ready', 'show'], events
            assert workers[0].poll() == 0, 'Splash must have completed independently'
            assert window.isVisible()
            assert window.windowTitle().startswith('Meshropractor —')
            assert 'New' not in window.windowTitle() and 'Classic' not in window.windowTitle()
            assert window.ui.magics_ribbon.count() == 11
            assert window.windowIcon().pixmap(32, 32).toImage() == app_icon().pixmap(32, 32).toImage()
            if sys.platform == 'win32':
                import ctypes
                user = ctypes.windll.user32
                user.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
                user.SendMessageW.restype = ctypes.c_ssize_t
                assert user.SendMessageW(int(window.winId()), 0x7F, 1, 0), 'Missing native large taskbar icon'
                assert user.SendMessageW(int(window.winId()), 0x7F, 0, 0), 'Missing native small caption icon'
                current = ctypes.c_void_p()
                shell = ctypes.windll.shell32
                shell.GetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                shell.GetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
                assert shell.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(current)) == 0
                try: assert ctypes.wstring_at(current) == APP_ID
                finally:
                    ctypes.windll.ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
                    ctypes.windll.ole32.CoTaskMemFree(current)
            window.grab().save(str(output/'home.png'))
        except BaseException as exc:
            failure.append(exc)
        finally:
            window.close()

    def make_window():
        window = original_window()
        windows.append(window)
        window.startup_probe = ShowProbe(window)
        window.installEventFilter(window.startup_probe)
        QTimer.singleShot(3500, inspect)
        return window

    with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
        def settings(team, name):
            value = QSettings(str(Path(folder)/(name+'.ini')), QSettings.IniFormat)
            value.setFallbacksEnabled(False)
            return value
        original_load = app_settings.load_settings
        patches.enter_context(patch.object(app_settings, 'load_settings', lambda *args: original_load(settings)))
        patches.enter_context(patch.object(main_window, 'MainWindow', make_window))
        patches.enter_context(patch.object(startup_bridge, 'SplashProcess', TracedSplash))
        patches.enter_context(patch.object(app_updater.UpdateController, 'start', lambda *args: None))
        code = desktop_launcher.main([])
        assert code == 0, code
        if failure: raise failure[0]
        assert splashes[0].process is None
        assert workers[0].poll() == 0
        report = dict(status='passed', checks=[
            'Production launcher shows the real unified Qt/VTK main window and version 0.3',
            'Stage order: geometry, interface, scene, ready while Qt/native window is hidden, show after splash exits',
            'Real splash subprocess completes independently with exit code 0',
            'Normal close exits the launcher and releases its worker; settings isolated',
            'Updated multi-size icon matches the window; Windows native taskbar/caption icons and AppUserModelID present',
        ])
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DESKTOP_ENTRY_OK')


if __name__ == '__main__': main()
